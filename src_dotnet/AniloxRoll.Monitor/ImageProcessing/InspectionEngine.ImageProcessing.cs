using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;
using AniloxRoll.Monitor.Core.Interop;
using AOI.SDK.Core.Models;
using AOI.SDK.Utils;

namespace AniloxRoll.Monitor.Core.Services
{
    public partial class InspectionEngine
    {
        public TimedResult<InspectionData> LoadThumbnailOnly(string filePath, int targetThumbWidth)
        {
            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                // [IO] 使用 CoreCV_FastReadBMP 直接讀入 Pinned Memory，
                // 比 GDI+ new Bitmap() 快，且為後續 DMA 傳輸做好準備。
                stopwatch.Start();
                bool readSuccess = NativeMethods.CoreCV_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                int thumbH = (int)((float)h / w * targetThumbWidth);

                // [GPU] 在 GPU 上縮圖，對應 de24f715 的 PICoater_RunThumbnail_GPU 設計。
                // _inputBuffer / _thumbnailBuffer 均為 Pinned Memory，H<->D 走 DMA 加速。
                stopwatch.Restart();
                int ret = NativeMethods.CoreCV_Resize_GPU(
                    _inputBuffer, w, h,
                    _thumbnailBuffer, targetThumbWidth, thumbH);
                stopwatch.Stop();
                long gpuTime = stopwatch.ElapsedMilliseconds;

                if (ret != 0) throw new InvalidOperationException($"GPU Resize Error: {ret}");

                // [BMP] 從 Pinned Memory 建立 Bitmap（直接 MemoryCopy，無額外 Marshal 開銷）
                stopwatch.Restart();
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                var data = new InspectionData { Image = bitmap, MuraCurveMean = null };
                return (data, ioTime, gpuTime, bmpTime);
            });
        }

        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth, float hessianFactor)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                // [IO] 使用 CoreCV_FastReadBMP 讀入 Pinned Memory
                stopwatch.Start();
                bool readSuccess = NativeMethods.CoreCV_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                // [GPU] CUDA 全尺寸檢測 Pipeline (背景去除 + Ridge 增強)
                stopwatch.Restart();
                _aoiService.ProcessImage(new AoiProcessRequest
                {
                    Input = new AoiProcessRequest.InputImage
                    {
                        Width = w,
                        Height = h,
                        Data = _inputBuffer,
                        Stream = IntPtr.Zero
                    },
                    Output = new AoiProcessRequest.OutputBuffers
                    {
                        BackgroundData = IntPtr.Zero,
                        MuraData = _muraBuffer,
                        RidgeData = _ridgeBuffer,
                        MuraCurveMean = _curveMeanBuffer,
                        MuraCurveMax = _curveMaxBuffer,
                        Stream = IntPtr.Zero
                    },
                    Params = new AoiProcessRequest.AlgorithmParams
                    {
                        BgSigmaFactor = InspectionEngineConfig.DefaultBgSigma,
                        RidgeSigma = InspectionEngineConfig.DefaultRidgeSigma,
                        HessianMaxFactor = hessianFactor,
                        RidgeMode = InspectionEngineConfig.DefaultRidgeMode
                    }
                });
                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;

                // [BMP + GPU 縮圖] 對應 de24f715 的 PICoater_Run_WithThumb 設計：
                // 在 GPU 上將 ridge 結果縮圖，再建立 Bitmap 供縮圖牆顯示。
                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);

                int resizeRet = NativeMethods.CoreCV_Resize_GPU(
                    _ridgeBuffer, w, h,
                    _thumbnailBuffer, targetThumbWidth, thumbH);

                if (resizeRet != 0) throw new InvalidOperationException($"GPU Resize Error: {resizeRet}");

                Bitmap thumb = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);

                float[] curveMean = new float[w];
                Marshal.Copy(_curveMeanBuffer, curveMean, 0, w);

                float[] curveMax = new float[w];
                Marshal.Copy(_curveMaxBuffer, curveMax, 0, w);

                var data = new InspectionData
                {
                    Image = thumb,
                    MuraCurveMean = curveMean,
                    MuraCurveMax = curveMax
                };

                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                return (data, ioTime, algoTime, bmpTime);
            });
        }

        public InspectionData RunInspectionFullRes(string filePath, bool isProcessedMode, float hessianFactor)
        {
            if (_isDisposed) return null;
            if (!File.Exists(filePath)) return null;

            lock (_lock)
            {
                var swTotal = Stopwatch.StartNew();
                var sw = Stopwatch.StartNew();

                bool readSuccess = NativeMethods.CoreCV_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, (int)_imgBufferSize);
                long ioMs = sw.ElapsedMilliseconds;

                if (!readSuccess) return null;

                Bitmap bmp;
                float[] curveMean = null;
                float[] curveMax = null;
                long gpuMs = 0, bmpMs = 0, copyMs = 0;

                if (isProcessedMode)
                {
                    sw.Restart();
                    _aoiService.ProcessImage(new AoiProcessRequest
                    {
                        Input = new AoiProcessRequest.InputImage
                        {
                            Width = w,
                            Height = h,
                            Data = _inputBuffer,
                            Stream = IntPtr.Zero
                        },
                        Output = new AoiProcessRequest.OutputBuffers
                        {
                            BackgroundData = IntPtr.Zero,
                            MuraData = _muraBuffer,
                            RidgeData = _ridgeBuffer,
                            MuraCurveMean = _curveMeanBuffer,
                            MuraCurveMax = _curveMaxBuffer,
                            Stream = IntPtr.Zero
                        },
                        Params = new AoiProcessRequest.AlgorithmParams
                        {
                            BgSigmaFactor = InspectionEngineConfig.DefaultBgSigma,
                            RidgeSigma = InspectionEngineConfig.DefaultRidgeSigma,
                            HessianMaxFactor = hessianFactor,
                            RidgeMode = InspectionEngineConfig.DefaultRidgeMode
                        }
                    });
                    gpuMs = sw.ElapsedMilliseconds;

                    sw.Restart();
                    bmp = ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h, flipY: false);
                    bmpMs = sw.ElapsedMilliseconds;

                    sw.Restart();
                    curveMean = new float[w];
                    curveMax = new float[w];
                    Marshal.Copy(_curveMeanBuffer, curveMean, 0, w);
                    Marshal.Copy(_curveMaxBuffer, curveMax, 0, w);
                    copyMs = sw.ElapsedMilliseconds;
                }
                else
                {
                    sw.Restart();
                    bmp = ImageUtils.Create8bppBitmap(_inputBuffer, w, h, flipY: false);
                    bmpMs = sw.ElapsedMilliseconds;
                }

                Console.WriteLine(
                    $"[FullRes] mode={isProcessedMode,-5} | " +
                    $"IO={ioMs,4}ms | GPU={gpuMs,4}ms | BMP={bmpMs,4}ms | Copy={copyMs,3}ms | " +
                    $"Total={swTotal.ElapsedMilliseconds,5}ms  ({w}x{h})");

                return new InspectionData
                {
                    Image = bmp,
                    MuraCurveMean = curveMean,
                    MuraCurveMax = curveMax
                };
            }
        }

    }
}
