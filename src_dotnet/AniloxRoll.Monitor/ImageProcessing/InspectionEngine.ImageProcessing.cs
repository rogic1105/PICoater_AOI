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
            // 新格式：直接從 JPEG 讀取縮圖，無需 GPU
            if (filePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return LoadThumbnailFromJpeg(filePath, targetThumbWidth);

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

            // 新格式：從預存 JPEG + .bin 載入，無需 GPU
            if (filePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return LoadProcessedThumbnailFromJpeg(filePath, targetThumbWidth);

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

            // 新格式：預先存好的 JPEG + .bin，不需要 GPU，直接載入
            if (filePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return LoadFromPrecomputedFiles(filePath, isProcessedMode);

            // 舊格式：BMP 讀取 + 視需要跑 GPU Pipeline（向下相容）
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
                    MuraCurveMax = curveMax,
                    IsCompressedJpeg = false,
                    ScaleFactor = 1
                };
            }
        }

        /// <summary>從 _raw.jpg 讀取縮圖（不含處理結果），用於批次縮圖牆。</summary>
        private static TimedResult<InspectionData> LoadThumbnailFromJpeg(string rawJpgPath, int targetThumbWidth)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                Bitmap srcBmp;
                using (var ms = new MemoryStream(File.ReadAllBytes(rawJpgPath)))
                    srcBmp = new Bitmap(ms);
                long ioMs = sw.ElapsedMilliseconds;

                sw.Restart();
                int thumbH = (int)((float)srcBmp.Height / srcBmp.Width * targetThumbWidth);
                var thumb  = new Bitmap(targetThumbWidth, thumbH);
                using (var g = Graphics.FromImage(thumb))
                    g.DrawImage(srcBmp, 0, 0, targetThumbWidth, thumbH);
                srcBmp.Dispose();
                long bmpMs = sw.ElapsedMilliseconds;

                return new TimedResult<InspectionData>(
                    new InspectionData { Image = thumb }, ioMs, 0, bmpMs);
            }
            catch { return new TimedResult<InspectionData>(); }
        }

        /// <summary>從 _proc.jpg + .bin 讀取縮圖與曲線，用於批次縮圖牆（處理模式）。</summary>
        private static TimedResult<InspectionData> LoadProcessedThumbnailFromJpeg(string rawJpgPath, int targetThumbWidth)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                string baseNoSuffix = rawJpgPath.Substring(0, rawJpgPath.Length - "_raw.jpg".Length);
                string procJpgPath  = baseNoSuffix + "_proc.jpg";
                string imgPath      = File.Exists(procJpgPath) ? procJpgPath : rawJpgPath;

                Bitmap srcBmp;
                using (var ms = new MemoryStream(File.ReadAllBytes(imgPath)))
                    srcBmp = new Bitmap(ms);
                long ioMs = sw.ElapsedMilliseconds;

                sw.Restart();
                int thumbH = (int)((float)srcBmp.Height / srcBmp.Width * targetThumbWidth);
                var thumb  = new Bitmap(targetThumbWidth, thumbH);
                using (var g = Graphics.FromImage(thumb))
                    g.DrawImage(srcBmp, 0, 0, targetThumbWidth, thumbH);
                srcBmp.Dispose();

                float[] curveMean = LoadCurveBin(baseNoSuffix + "_mean.bin");
                float[] curveMax  = LoadCurveBin(baseNoSuffix + "_max.bin");
                long bmpMs = sw.ElapsedMilliseconds;

                return new TimedResult<InspectionData>(
                    new InspectionData { Image = thumb, MuraCurveMean = curveMean, MuraCurveMax = curveMax },
                    ioMs, 0, bmpMs);
            }
            catch { return new TimedResult<InspectionData>(); }
        }

        /// <summary>
        /// 載入預先存好的新格式檔案（_raw.jpg / _proc.jpg / _mean.bin / _max.bin），無需 GPU。
        /// </summary>
        private static InspectionData LoadFromPrecomputedFiles(string rawJpgPath, bool isProcessedMode)
        {
            var swTotal = Stopwatch.StartNew();

            string baseNoSuffix = rawJpgPath.Substring(0, rawJpgPath.Length - "_raw.jpg".Length);
            string procJpgPath  = baseNoSuffix + "_proc.jpg";
            string meanBinPath  = baseNoSuffix + "_mean.bin";
            string maxBinPath   = baseNoSuffix + "_max.bin";

            string imgPath = (isProcessedMode && File.Exists(procJpgPath)) ? procJpgPath : rawJpgPath;

            Bitmap bmp;
            try
            {
                // 用 MemoryStream 載入避免 GDI+ 鎖住檔案
                byte[] bytes = File.ReadAllBytes(imgPath);
                using (var ms = new MemoryStream(bytes))
                    bmp = new Bitmap(ms);
            }
            catch { return null; }

            float[] curveMean = isProcessedMode ? LoadCurveBin(meanBinPath) : null;
            float[] curveMax  = isProcessedMode ? LoadCurveBin(maxBinPath)  : null;

            Console.WriteLine(
                $"[FullRes-New] mode={isProcessedMode,-5} | Total={swTotal.ElapsedMilliseconds,4}ms  ({bmp.Width}x{bmp.Height})");

            int scaleFactor = curveMean != null && bmp.Width > 0
                ? Math.Max(1, (int)Math.Round((double)curveMean.Length / bmp.Width))
                : ReadScaleFactorFromBin(meanBinPath);

            return new InspectionData
            {
                Image = bmp,
                MuraCurveMean = curveMean,
                MuraCurveMax = curveMax,
                IsCompressedJpeg = true,
                ScaleFactor = scaleFactor
            };
        }

        /// <summary>僅讀取 .bin 標頭中的 scale_factor，不載入整個陣列。</summary>
        private static int ReadScaleFactorFromBin(string path)
        {
            if (!File.Exists(path)) return 1;
            try
            {
                using (var br = new BinaryReader(File.OpenRead(path)))
                {
                    byte[] magic = br.ReadBytes(4);
                    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F')
                        return 1;
                    br.ReadInt32();   // version
                    float sf = br.ReadSingle();
                    return Math.Max(1, (int)Math.Round(sf));
                }
            }
            catch { return 1; }
        }

        /// <summary>
        /// 讀取 .bin 曲線檔案。格式：magic(4) + version(4) + scale_factor(4f) + array_length(4) + float[]
        /// </summary>
        private static float[] LoadCurveBin(string path)
        {
            if (!File.Exists(path)) return null;
            try
            {
                using (var br = new BinaryReader(File.OpenRead(path)))
                {
                    byte[] magic = br.ReadBytes(4);
                    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F')
                        return null;
                    br.ReadInt32();    // version
                    br.ReadSingle();   // scale_factor（保存供日後使用）
                    int len = br.ReadInt32();
                    float[] arr = new float[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = br.ReadSingle();
                    return arr;
                }
            }
            catch { return null; }
        }

    }
}
