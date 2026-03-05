using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;
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
                bool readSuccess = LoadGrayscaleImageToBuffer(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                int thumbH = (int)((float)h / w * targetThumbWidth);

                stopwatch.Restart();
                using (Bitmap source = ImageUtils.Create8bppBitmap(_inputBuffer, w, h))
                using (Bitmap thumb = new Bitmap(source, new Size(targetThumbWidth, thumbH)))
                {
                    stopwatch.Stop();
                    long gpuTime = stopwatch.ElapsedMilliseconds;

                    stopwatch.Restart();
                    var data = new InspectionData { Image = (Bitmap)thumb.Clone(), MuraCurveMean = null };
                    stopwatch.Stop();
                    long bmpTime = stopwatch.ElapsedMilliseconds;
                    return (data, ioTime, gpuTime, bmpTime);
                }
            });
        }

        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth, float hessianFactor)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            return ExecuteTimedOperation<InspectionData>(filePath, (stopwatch) =>
            {
                stopwatch.Start();
                bool readSuccess = LoadGrayscaleImageToBuffer(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                stopwatch.Stop();
                long ioTime = stopwatch.ElapsedMilliseconds;

                if (!readSuccess) return (null, ioTime, 0, 0);

                stopwatch.Restart();
                _aoiService.ProcessImage(
                    w,
                    h,
                    _inputBuffer,
                    IntPtr.Zero,
                    _muraBuffer,
                    _ridgeBuffer,
                    _curveMeanBuffer,
                    _curveMaxBuffer,
                    InspectionEngineConfig.DefaultBgSigma,
                    InspectionEngineConfig.DefaultRidgeSigma,
                    hessianFactor,
                    InspectionEngineConfig.DefaultRidgeMode,
                    IntPtr.Zero);
                stopwatch.Stop();
                long algoTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                int thumbH = (int)((float)h / w * targetThumbWidth);
                Bitmap thumb;
                using (Bitmap source = ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h))
                {
                    thumb = new Bitmap(source, new Size(targetThumbWidth, thumbH));
                }

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
                bool readSuccess = LoadGrayscaleImageToBuffer(
                    filePath, out int w, out int h, _inputBuffer, _imgBufferSize);
                if (!readSuccess) return null;

                Bitmap bmp;
                float[] curveMean = null;
                float[] curveMax = null;

                if (isProcessedMode)
                {
                    _aoiService.ProcessImage(
                        w,
                        h,
                        _inputBuffer,
                        IntPtr.Zero,
                        _muraBuffer,
                        _ridgeBuffer,
                        _curveMeanBuffer,
                        _curveMaxBuffer,
                        InspectionEngineConfig.DefaultBgSigma,
                        InspectionEngineConfig.DefaultRidgeSigma,
                        hessianFactor,
                        InspectionEngineConfig.DefaultRidgeMode,
                        IntPtr.Zero);

                    bmp = ImageUtils.Create8bppBitmap(_ridgeBuffer, w, h, flipY: false);

                    curveMean = new float[w];
                    curveMax = new float[w];
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

        private static bool LoadGrayscaleImageToBuffer(
            string filePath,
            out int width,
            out int height,
            IntPtr destination,
            ulong destinationSize)
        {
            width = 0;
            height = 0;
            if (!File.Exists(filePath) || destination == IntPtr.Zero)
            {
                return false;
            }

            using (Bitmap bitmap = new Bitmap(filePath))
            {
                width = bitmap.Width;
                height = bitmap.Height;
                int required = width * height;
                if ((ulong)required > destinationSize)
                {
                    return false;
                }

                byte[] grayscale = new byte[required];
                Rectangle rect = new Rectangle(0, 0, width, height);
                BitmapData data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, bitmap.PixelFormat);
                try
                {
                    int bytes = Math.Abs(data.Stride) * height;
                    byte[] raw = new byte[bytes];
                    Marshal.Copy(data.Scan0, raw, 0, bytes);

                    int bitsPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat);
                    int bytesPerPixel = Math.Max(1, bitsPerPixel / 8);

                    for (int y = 0; y < height; y++)
                    {
                        int srcRow = y * Math.Abs(data.Stride);
                        int dstRow = y * width;

                        for (int x = 0; x < width; x++)
                        {
                            if (bytesPerPixel == 1)
                            {
                                grayscale[dstRow + x] = raw[srcRow + x];
                            }
                            else
                            {
                                int srcIndex = srcRow + (x * bytesPerPixel);
                                if (srcIndex + 2 >= raw.Length)
                                {
                                    grayscale[dstRow + x] = 0;
                                    continue;
                                }

                                byte b = raw[srcIndex + 0];
                                byte g = raw[srcIndex + 1];
                                byte r = raw[srcIndex + 2];
                                grayscale[dstRow + x] = (byte)((r * 299 + g * 587 + b * 114) / 1000);
                            }
                        }
                    }
                }
                finally
                {
                    bitmap.UnlockBits(data);
                }

                Marshal.Copy(grayscale, 0, destination, grayscale.Length);
            }

            return true;
        }
    }
}
