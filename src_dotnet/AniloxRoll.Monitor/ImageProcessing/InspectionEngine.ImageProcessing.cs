using System;
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
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                NativeMethods.PICoater_Initialize(_handle, w, h);

                int ret = NativeMethods.PICoater_RunThumbnail_GPU(
                    _handle, _inputBuffer, targetThumbWidth, _thumbnailBuffer,
                    out int realW, out int realH
                );

                stopwatch.Stop();
                long gpuTime = stopwatch.ElapsedMilliseconds;

                if (ret != 0) throw new Exception($"GPU Resize Error: {ret}");

                stopwatch.Restart();

                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, realW, realH);

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
                stopwatch.Start();
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                NativeMethods.PICoater_Initialize(_handle, w, h);

                int thumbH = (int)((float)h / w * targetThumbWidth);

                int ret = NativeMethods.PICoater_Run_WithThumb(
                    _handle, _inputBuffer, _thumbnailBuffer, targetThumbWidth, thumbH,
                    _curveMeanBuffer, _curveMaxBuffer,
                    InspectionEngineConfig.DefaultBgSigma,
                    InspectionEngineConfig.DefaultRidgeSigma,
                    hessianFactor,
                    InspectionEngineConfig.DefaultRidgeMode
                );

                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;
                if (ret != 0) throw new Exception($"Algo Error: {ret}");

                stopwatch.Restart();

                var thumb = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);

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
                bool readSuccess = NativeMethods.PICoater_FastReadBMP(
                     filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                if (!readSuccess) return null;

                Bitmap bmp = null;
                float[] curveMean = null;
                float[] curveMax = null;

                if (isProcessedMode)
                {
                    NativeMethods.PICoater_Initialize(_handle, w, h);
                    int ret = NativeMethods.PICoater_Run(
                        _handle, _inputBuffer, IntPtr.Zero, IntPtr.Zero, _ridgeBuffer, IntPtr.Zero, IntPtr.Zero,
                        InspectionEngineConfig.DefaultBgSigma,
                        InspectionEngineConfig.DefaultRidgeSigma,
                        hessianFactor,
                        InspectionEngineConfig.DefaultRidgeMode
                    );
                    if (ret != 0) return null;

                    bmp = ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h, flipY: false);

                    curveMean = new float[w];
                    curveMax = new float[w];

                    NativeMethods.PICoater_Run(
                        _handle, _inputBuffer, IntPtr.Zero, IntPtr.Zero, _ridgeBuffer,
                        _curveMeanBuffer, _curveMaxBuffer,
                        InspectionEngineConfig.DefaultBgSigma,
                        InspectionEngineConfig.DefaultRidgeSigma,
                        hessianFactor,
                        InspectionEngineConfig.DefaultRidgeMode
                    );

                    Marshal.Copy(_curveMeanBuffer, curveMean, 0, w);
                    Marshal.Copy(_curveMaxBuffer, curveMax, 0, w);
                }
                else
                {
                    bmp = ImageUtils.Create8bppBitmap(_inputBuffer, w, h, flipY: false);
                }

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
