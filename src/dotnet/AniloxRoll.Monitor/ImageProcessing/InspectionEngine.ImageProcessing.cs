using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;
using AniloxRoll.Monitor.Core.Interop;
using TanukiCv.Core.Models;
using TanukiCv.Utils;

namespace AniloxRoll.Monitor.Core.Services
{
    public partial class InspectionEngine
    {
        /// <summary>
        /// BMP 拼接專用：CoreCV_FastReadBMP + GPU resize 縮 scale 倍，回傳 Bitmap。
        /// 比 GDI+ new Bitmap(path) 快約 10x（繞過 GDI+，DMA 傳輸）。
        /// 使用 _muraBuffer 作為 resize 目標（MaxWidth×MaxHeight，不受 ThumbnailBufferSize 限制）。
        /// </summary>
        public Bitmap LoadBitmapAtScale(string path, int scale)
        {
            lock (_lock)
            {
                bool ok = NativeMethods.CoreCV_FastReadBMP(
                    path, out int w, out int h, _inputBuffer, (int)_imgBufferSize);
                if (!ok) return null;
                int dstW = Math.Max(1, w / scale);
                int dstH = Math.Max(1, h / scale);
                int ret = NativeMethods.CoreCV_Resize_GPU(_inputBuffer, w, h, _muraBuffer, dstW, dstH);
                if (ret != 0) return null;
                // CoreCV_Resize_GPU 輸出為 bottom-up（Y 軸反轉），需 flipY: true 補正；
                // 直接從 _inputBuffer 建立 bitmap（RunInspectionFullRes 路徑）不經過 GPU resize，
                // 保持 top-down，故用 flipY: false。
                return ImageUtils.Create8bppBitmap(_muraBuffer, dstW, dstH, flipY: true);
            }
        }

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
                // CoreCV_Resize_GPU 輸出為 bottom-up，此應用刻意保持翻轉（取像時序）
                stopwatch.Restart();
                var bitmap = ImageUtils.Create8bppBitmap(_thumbnailBuffer, targetThumbWidth, thumbH);
                stopwatch.Stop();
                long bmpTime = stopwatch.ElapsedMilliseconds;

                var data = new InspectionData { Image = bitmap, MuraCurveMean = null };
                return (data, ioTime, gpuTime, bmpTime);
            });
        }

        public TimedResult<InspectionData> ProcessImage(string filePath, int targetThumbWidth, float hessianFactor,
            string ridgeMode = null)
        {
            if (_isDisposed) throw new ObjectDisposedException(nameof(InspectionEngine));

            if (filePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return LoadProcessedThumbnailFromJpeg(filePath, targetThumbWidth);

            return new TimedResult<InspectionData>();
        }

        public InspectionData RunInspectionFullRes(string filePath, bool isProcessedMode, float hessianFactor,
            string ridgeMode = null, string ridgeDirection = "v")
        {
            if (_isDisposed) return null;
            if (!File.Exists(filePath)) return null;

            if (filePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return LoadFromPrecomputedFiles(filePath, isProcessedMode, ridgeDirection);

            return null;
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
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[InspectionEngine.LoadRawThumbnailFromJpeg] {ex.GetType().Name}: {ex.Message}"); return new TimedResult<InspectionData>(); }
        }

        /// <summary>從 _proc_v.jpg + .bin 讀取縮圖與曲線，用於批次縮圖牆（處理模式）。</summary>
        private static TimedResult<InspectionData> LoadProcessedThumbnailFromJpeg(string rawJpgPath, int targetThumbWidth)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                string baseNoSuffix = rawJpgPath.Substring(0, rawJpgPath.Length - "_raw.jpg".Length);
                string procJpgPath  = baseNoSuffix + "_proc_v.jpg";
                if (!File.Exists(procJpgPath)) procJpgPath = baseNoSuffix + "_proc.jpg"; // 向後相容
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

                float[] curveMean    = LoadCurveBinCompat(baseNoSuffix, "_mean_v.bin", "_mean.bin");
                float[] curveMax     = LoadCurveBinCompat(baseNoSuffix, "_max_v.bin", "_max.bin");
                float[] rowCurveMean = LoadCurveBinCompat(baseNoSuffix, "_mean_h.bin", "_row_mean.bin");
                float[] rowCurveMax  = LoadCurveBinCompat(baseNoSuffix, "_max_h.bin", "_row_max.bin");
                long bmpMs = sw.ElapsedMilliseconds;

                return new TimedResult<InspectionData>(
                    new InspectionData
                    {
                        Image = thumb,
                        MuraCurveMean = curveMean,
                        MuraCurveMax = curveMax,
                        MuraRowCurveMean = rowCurveMean,
                        MuraRowCurveMax = rowCurveMax
                    },
                    ioMs, 0, bmpMs);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[InspectionEngine.LoadProcessedThumbnailFromJpeg] {ex.GetType().Name}: {ex.Message}"); return new TimedResult<InspectionData>(); }
        }

        /// <summary>
        /// 載入預先存好的新格式檔案（_raw.jpg / _proc_v.jpg / _proc_h.jpg / .bin），無需 GPU。
        /// ridgeDirection: "v" = vertical, "h" = horizontal（控制處理模式顯示哪張）。
        /// </summary>
        private static InspectionData LoadFromPrecomputedFiles(string rawJpgPath, bool isProcessedMode,
            string ridgeDirection = "v")
        {
            var swTotal = Stopwatch.StartNew();

            string baseNoSuffix = rawJpgPath.Substring(0, rawJpgPath.Length - "_raw.jpg".Length);
            // 依方向選擇處理圖：_proc_v.jpg 或 _proc_h.jpg
            string procSuffix = (ridgeDirection == "h") ? "_proc_h.jpg" : "_proc_v.jpg";
            string procJpgPath  = File.Exists(baseNoSuffix + procSuffix)
                ? baseNoSuffix + procSuffix
                : baseNoSuffix + "_proc.jpg"; // 向後相容
            string meanBinPath  = ResolveCompatPath(baseNoSuffix, "_mean_v.bin", "_mean.bin");
            string maxBinPath   = ResolveCompatPath(baseNoSuffix, "_max_v.bin", "_max.bin");

            string imgPath = (isProcessedMode && File.Exists(procJpgPath)) ? procJpgPath : rawJpgPath;

            Bitmap bmp;
            try
            {
                // 用 MemoryStream 載入避免 GDI+ 鎖住檔案
                byte[] bytes = File.ReadAllBytes(imgPath);
                using (var ms = new MemoryStream(bytes))
                    bmp = new Bitmap(ms);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[InspectionEngine.LoadFromPrecomputedFiles] {ex.GetType().Name}: {ex.Message}"); return null; }

            float[] curveMean    = LoadCurveBin(meanBinPath);
            float[] curveMax     = LoadCurveBin(maxBinPath);
            float[] rowCurveMean = LoadCurveBinCompat(baseNoSuffix, "_mean_h.bin", "_row_mean.bin");
            float[] rowCurveMax  = LoadCurveBinCompat(baseNoSuffix, "_max_h.bin", "_row_max.bin");

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
                MuraRowCurveMean = rowCurveMean,
                MuraRowCurveMax = rowCurveMax,
                IsCompressedJpeg = true,
                ScaleFactor = scaleFactor
            };
        }

        private void RunGpuPipeline(int w, int h, float hessianFactor, string ridgeMode = null)
        {
            _aoiService.ProcessImage(new AoiProcessRequest
            {
                Input = new AoiProcessRequest.InputImage
                {
                    Width  = w,
                    Height = h,
                    Data   = _inputBuffer,
                    Stream = IntPtr.Zero
                },
                Output = new AoiProcessRequest.OutputBuffers
                {
                    BackgroundData   = IntPtr.Zero,
                    MuraData         = _muraBuffer,
                    RidgeData        = _ridgeBuffer,
                    MuraCurveMean    = _curveMeanBuffer,
                    MuraCurveMax     = _curveMaxBuffer,
                    MuraRowCurveMean = _curveRowMeanBuffer,
                    MuraRowCurveMax  = _curveRowMaxBuffer,
                    Stream           = IntPtr.Zero
                },
                Params = new AoiProcessRequest.AlgorithmParams
                {
                    BgSigmaFactor    = InspectionEngineConfig.DefaultBgSigma,
                    RidgeSigma       = InspectionEngineConfig.DefaultRidgeSigma,
                    HessianMaxFactor = hessianFactor,
                    RidgeMode        = "vertical+horizontal"  // 永遠計算雙方向，確保 V/H 皆可存檔
                }
            });
        }

        /// <summary>
        /// BMP 拼接處理模式：讀 BMP → GPU pipeline → resize 縮 scale 倍，回傳處理後 Bitmap。
        /// 曲線保持全解析度（用於 MergeCurves），同時存 .bin 供下次原圖模式讀取。
        /// </summary>
        public Bitmap ProcessBmpAtScale(string path, int scale, float hessianFactor,
            out float[] curveMean, out float[] curveMax, string ridgeMode = null)
        {
            curveMean = null;
            curveMax  = null;
            if (_isDisposed) return null;

            lock (_lock)
            {
                bool ok = NativeMethods.CoreCV_FastReadBMP(
                    path, out int w, out int h, _inputBuffer, (int)_imgBufferSize);
                if (!ok) return null;

                // Pipeline 永遠跑 "vertical+horizontal"，一律產生 V/H 曲線與圖片
                RunGpuPipeline(w, h, hessianFactor, ridgeMode);

                string basePath       = Path.Combine(Path.GetDirectoryName(path),
                                           Path.GetFileNameWithoutExtension(path));

                // Col curves（vertical 方向）
                curveMean = new float[w];
                curveMax  = new float[w];
                Marshal.Copy(_curveMeanBuffer, curveMean, 0, w);
                Marshal.Copy(_curveMaxBuffer,  curveMax,  0, w);
                string meanBinPath = basePath + "_mean_v.bin";
                string maxBinPath  = basePath + "_max_v.bin";
                if (!File.Exists(meanBinPath)) SaveCurveBin(curveMean, 1, meanBinPath);
                if (!File.Exists(maxBinPath))  SaveCurveBin(curveMax,  1, maxBinPath);

                // Row curves（horizontal 方向）
                float[] rowMean = new float[h];
                float[] rowMax  = new float[h];
                Marshal.Copy(_curveRowMeanBuffer, rowMean, 0, h);
                Marshal.Copy(_curveRowMaxBuffer,  rowMax,  0, h);
                string rowMeanBinPath = basePath + "_mean_h.bin";
                string rowMaxBinPath  = basePath + "_max_h.bin";
                if (!File.Exists(rowMeanBinPath)) SaveCurveBin(rowMean, 1, rowMeanBinPath);
                if (!File.Exists(rowMaxBinPath))  SaveCurveBin(rowMax,  1, rowMaxBinPath);

                int dstW = Math.Max(1, w / scale);
                int dstH = Math.Max(1, h / scale);

                // _muraBuffer = horizontal ridge → _proc_h.jpg（先存，resize 前 _muraBuffer 還沒被覆蓋）
                string procHPath = basePath + "_proc_h.jpg";
                if (!File.Exists(procHPath))
                {
                    int retH = NativeMethods.CoreCV_Resize_GPU(_muraBuffer, w, h, _inputBuffer, dstW, dstH);
                    if (retH == 0)
                    {
                        using (var bmpH = ImageUtils.Create8bppBitmap(_inputBuffer, dstW, dstH, flipY: true))
                        using (var bmp24 = new Bitmap(dstW, dstH, PixelFormat.Format24bppRgb))
                        using (var g = Graphics.FromImage(bmp24))
                        {
                            g.DrawImage(bmpH, 0, 0, dstW, dstH);
                            SaveBitmapAsJpeg(bmp24, procHPath, 90);
                        }
                    }
                }

                // _ridgeBuffer = vertical ridge → resize 至 _muraBuffer 作為回傳 Bitmap + 存 _proc_v.jpg
                int ret = NativeMethods.CoreCV_Resize_GPU(_ridgeBuffer, w, h, _muraBuffer, dstW, dstH);
                if (ret != 0) return null;

                string procVPath = basePath + "_proc_v.jpg";
                if (!File.Exists(procVPath))
                {
                    using (var bmpProc = ImageUtils.Create8bppBitmap(_muraBuffer, dstW, dstH, flipY: true))
                    using (var bmp24 = new Bitmap(dstW, dstH, PixelFormat.Format24bppRgb))
                    using (var g = Graphics.FromImage(bmp24))
                    {
                        g.DrawImage(bmpProc, 0, 0, dstW, dstH);
                        SaveBitmapAsJpeg(bmp24, procVPath, 90);
                    }
                }

                return ImageUtils.Create8bppBitmap(_muraBuffer, dstW, dstH, flipY: true);
            }
        }

        private static void SaveBitmapAsJpeg(Bitmap bmp, string path, int quality)
        {
            try
            {
                ImageCodecInfo codec = null;
                foreach (var c in ImageCodecInfo.GetImageEncoders())
                    if (c.FormatID == ImageFormat.Jpeg.Guid) { codec = c; break; }
                if (codec == null) { bmp.Save(path); return; }
                using (var ep = new EncoderParameters(1))
                {
                    ep.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
                    bmp.Save(path, codec, ep);
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[InspectionEngine.SaveBitmapAsJpeg] {ex.Message}");
            }
        }

        /// <summary>將 float[] 曲線寫成 MCBF .bin（scaleForHeader=1 代表全解析度 BMP）。</summary>
        private static void SaveCurveBin(float[] data, int scaleForHeader, string path)
        {
            if (data == null || data.Length == 0) return;
            try
            {
                using (var bw = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
                {
                    bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                    bw.Write(1);                       // version
                    bw.Write((float)scaleForHeader);   // scale_factor
                    bw.Write(data.Length);             // array_length
                    foreach (float v in data) bw.Write(v);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[InspectionEngine.SaveCurveBin] {ex.Message}");
            }
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
        internal static float[] LoadCurveBin(string path)
        {
            if (!File.Exists(path)) return null;
            try
            {
                using (var br = new BinaryReader(File.OpenRead(path)))
                {
                    byte[] magic = br.ReadBytes(4);
                    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F')
                        return null;
                    int version = br.ReadInt32();
                    br.ReadSingle();   // scale_factor
                    if (version >= 2)
                    {
                        br.ReadInt32();  // lightLevel (light controller 0-255)
                        br.ReadSingle(); // exposureUs
                    }
                    int len = br.ReadInt32();
                    float[] arr = new float[len];
                    for (int i = 0; i < len; i++)
                        arr[i] = br.ReadSingle();
                    return arr;
                }
            }
            catch { return null; }
        }

        internal static (int Light, float ExposureUs)? ReadBgBinMeta(string path)
        {
            if (!File.Exists(path)) return null;
            try
            {
                using (var br = new BinaryReader(File.OpenRead(path)))
                {
                    byte[] magic = br.ReadBytes(4);
                    if (magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F') return null;
                    if (br.ReadInt32() < 2) return null; // version
                    br.ReadSingle();                     // scale_factor
                    int light = br.ReadInt32();
                    float exposureUs = br.ReadSingle();
                    return (light, exposureUs);
                }
            }
            catch { return null; }
        }

        /// <summary>向後相容：優先讀新命名，不存在則嘗試舊命名。</summary>
        private static float[] LoadCurveBinCompat(string baseNoSuffix, string newSuffix, string oldSuffix)
        {
            string path = baseNoSuffix + newSuffix;
            if (File.Exists(path)) return LoadCurveBin(path);
            return LoadCurveBin(baseNoSuffix + oldSuffix);
        }

        /// <summary>向後相容：優先回傳新命名路徑，不存在則回傳舊命名路徑。</summary>
        private static string ResolveCompatPath(string baseNoSuffix, string newSuffix, string oldSuffix)
        {
            string path = baseNoSuffix + newSuffix;
            return File.Exists(path) ? path : baseNoSuffix + oldSuffix;
        }

    }
}
