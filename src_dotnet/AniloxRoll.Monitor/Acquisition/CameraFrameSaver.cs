using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using AOI.SDK.Utils;

namespace AniloxRoll.Monitor.Core.Camera
{
    /// <summary>
    /// 取像存檔與資源監控。從 AniloxCamera 提取出的純 I/O 邏輯。
    /// GPU 資料提取（lock + native buffer）仍留在 AniloxCamera，
    /// 提取完的 byte[]/float[] 透過 <see cref="CaptureContext"/> 傳入。
    /// </summary>
    public class CameraFrameSaver
    {
        // ── 每相機統計 ──────────────────────────────────────────────────────

        /// <summary>最近一幀存檔總大小（bytes，含 raw + proc + bin）。</summary>
        public long LastSaveBytesTotal { get; private set; }
        /// <summary>本次 session 累計存檔大小（bytes）。</summary>
        public long SessionSaveBytes { get; private set; }
        /// <summary>本次 session 累計存檔幀數。</summary>
        public long SessionFrameCount { get; private set; }

        // ── 存檔（Task.Run 內呼叫）──────────────────────────────────────────

        /// <summary>
        /// 背景執行緒存檔主入口。將 CaptureContext 中的 byte[]/float[] 寫入檔案，
        /// 更新統計，寫 resource log，最後呼叫 OnResult 回呼。
        /// </summary>
        public void SaveCapture(CaptureContext ctx)
        {
            Directory.CreateDirectory(ctx.SaveDir);

            SaveJpegFromBytes(ctx.RawBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                Path.Combine(ctx.SaveDir, ctx.BaseName + "_raw.jpg"), ctx.JpgQuality);

            if (ctx.ProcVBytes != null)
                SaveJpegFromBytes(ctx.ProcVBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_proc_v.jpg"), ctx.JpgQuality);

            if (ctx.ProcHBytes != null)
                SaveJpegFromBytes(ctx.ProcHBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_proc_h.jpg"), ctx.JpgQuality);

            if (ctx.MeanArr != null)
            {
                SaveCurveBinFromArray(ctx.MeanArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_mean_v.bin"));
                SaveCurveBinFromArray(ctx.MaxArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_max_v.bin"));
            }

            if (ctx.RowMeanArr != null)
            {
                SaveCurveBinFromArray(ctx.RowMeanArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_mean_h.bin"));
                SaveCurveBinFromArray(ctx.RowMaxArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + "_max_h.bin"));
            }

            // 計算本幀存檔總大小（排除 .bmp 原圖）
            long frameBytes = 0;
            foreach (var f in Directory.GetFiles(ctx.SaveDir, ctx.BaseName + "*"))
            {
                if (f.EndsWith(".bmp", StringComparison.OrdinalIgnoreCase)) continue;
                frameBytes += new FileInfo(f).Length;
            }
            LastSaveBytesTotal = frameBytes;
            SessionSaveBytes += frameBytes;
            SessionFrameCount++;

            // Resource log: 寫入 CSV
            long ramMB = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
            AppendResourceLog(ctx.CameraId, ctx.OrigWidth, ctx.OrigHeight,
                ctx.GpuTimeMs, frameBytes, SessionSaveBytes, SessionFrameCount, ramMB);

            ctx.OnResult?.Invoke(ctx.CameraId, ctx.BaseName, ctx.MeanPeak, ctx.MaxPeak);
        }

        // ── JPEG 存檔 ───────────────────────────────────────────────────────

        [ThreadStatic] private static Bitmap _reuseBmp24;
        [ThreadStatic] private static int _reuseBmp24W, _reuseBmp24H;
        [ThreadStatic] private static ImageCodecInfo _jpegCodec;

        private static ImageCodecInfo GetJpegEncoder()
        {
            if (_jpegCodec != null) return _jpegCodec;
            foreach (var c in ImageCodecInfo.GetImageEncoders())
                if (c.MimeType == "image/jpeg") { _jpegCodec = c; return c; }
            return null;
        }

        /// <summary>
        /// 將 8-bit 灰階 byte[] 存成 JPEG。
        /// GDI+ JPEG encoder 需要 24bpp，透過 GCHandle pin + Graphics.DrawImage 轉換。
        /// bmp24 使用 ThreadStatic 重用，避免每幀分配 LOH 觸發 Gen2 GC。
        /// </summary>
        internal static void SaveJpegFromBytes(byte[] data, int w, int h, string path, int quality)
        {
            var gch = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                using (var bmp8 = ImageUtils.Create8bppBitmap(gch.AddrOfPinnedObject(), w, h))
                {
                    if (_reuseBmp24 == null || _reuseBmp24W != w || _reuseBmp24H != h)
                    {
                        _reuseBmp24?.Dispose();
                        _reuseBmp24 = new Bitmap(w, h, PixelFormat.Format24bppRgb);
                        _reuseBmp24W = w;
                        _reuseBmp24H = h;
                    }
                    using (var g = Graphics.FromImage(_reuseBmp24))
                        g.DrawImage(bmp8, 0, 0, w, h);

                    var codec = GetJpegEncoder();
                    if (codec == null) { _reuseBmp24.Save(path); return; }

                    using (var ep = new EncoderParameters(1))
                    {
                        ep.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
                        _reuseBmp24.Save(path, codec, ep);
                    }
                }
            }
            finally
            {
                gch.Free();
            }
        }

        // ── Curve .bin 存檔 ─────────────────────────────────────────────────

        /// <summary>
        /// 將 float[] 曲線資料寫成自描述 .bin 格式。
        /// Header: magic(4)"MCBF" + version(4)=1 + scale_factor(4f) + array_length(4) + float[]
        /// </summary>
        internal static void SaveCurveBinFromArray(float[] arr, int scaleForHeader, string path)
        {
            if (arr == null || arr.Length == 0) return;
            using (var bw = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(1);                        // version
                bw.Write((float)scaleForHeader);    // scale_factor
                bw.Write(arr.Length);               // array_length
                for (int i = 0; i < arr.Length; i++)
                    bw.Write(arr[i]);
            }
        }

        // ══════════════════════════════════════════════════════════════════════
        // Resource Log（靜態，所有相機共用一份 CSV）
        // ══════════════════════════════════════════════════════════════════════

        private static readonly object _resourceLogLock = new object();
        private static string _resourceLogPath;
        private static bool _resourceLogInitialized;

        // CPU% 計算用
        private static TimeSpan _lastCpuTime;
        private static DateTime _lastCpuSample;
        private static int _cpuCoreCount;

        // UI 狀態回呼：由 Form 注入，回傳 "Live=T,Review=F,Stitch=Global" 格式
        public static Func<string> GetUiStateCallback { get; set; }

        // VRAM 查詢用
        private static long _cachedVramMB;
        private static DateTime _lastVramQuery;
        private static readonly TimeSpan VramQueryInterval = TimeSpan.FromSeconds(2);

        /// <summary>初始化 resource log 檔案（啟動時呼叫一次）。</summary>
        public static void InitResourceLog(string captureRootPath)
        {
            try
            {
                string baseDir = !string.IsNullOrEmpty(captureRootPath) && Directory.Exists(Path.GetPathRoot(captureRootPath))
                    ? captureRootPath
                    : AppDomain.CurrentDomain.BaseDirectory;
                string dir = Path.Combine(baseDir, "logs");
                Directory.CreateDirectory(dir);
                _resourceLogPath = Path.Combine(dir, $"resource-monitor-{DateTime.Now:yyyyMMdd-HHmmss}.csv");

                var proc = System.Diagnostics.Process.GetCurrentProcess();
                _lastCpuTime = proc.TotalProcessorTime;
                _lastCpuSample = DateTime.UtcNow;
                _cpuCoreCount = Environment.ProcessorCount;

                lock (_resourceLogLock)
                {
                    using (var sw = new StreamWriter(_resourceLogPath, append: false))
                    {
                        sw.WriteLine("Timestamp,Mode,CamId,Width,Height,RawMB,ProcessMs,SaveKB,SessionGB,SessionFrames,RamMB,CpuPct,VramMB,Live,Review,StitchMode");
                    }
                }

                _resourceLogInitialized = true;
                System.Diagnostics.Trace.WriteLine($"[ResourceLog] 已建立: {_resourceLogPath}");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[ResourceLog] 初始化失敗: {ex.Message}");
            }
        }

        private static void AppendResourceLog(int camId, int w, int h, long processMs, long saveBytes,
            long sessionBytes, long sessionFrames, long ramMB)
        {
            WriteResourceLine("Grab", camId, w, h, processMs, saveBytes, sessionBytes, sessionFrames, ramMB);
        }

        /// <summary>Review 操作的 resource log。
        /// mode: "Stitch"/"Single"/"Global"；camCount: 載入相機數；imgCount: 總影像數。</summary>
        public static void AppendReviewResourceLog(string mode, int camCount, int imgCount,
            int w, int h, long loadMs)
        {
            long ramMB = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
            WriteResourceLine(mode, camCount, w, h, loadMs, 0, 0, imgCount, ramMB);
        }

        private static double GetCpuPercent()
        {
            try
            {
                var proc = System.Diagnostics.Process.GetCurrentProcess();
                var now = DateTime.UtcNow;
                var cpuTime = proc.TotalProcessorTime;
                double elapsed = (now - _lastCpuSample).TotalMilliseconds;
                if (elapsed < 10) return 0;
                double cpuUsed = (cpuTime - _lastCpuTime).TotalMilliseconds;
                _lastCpuTime = cpuTime;
                _lastCpuSample = now;
                return cpuUsed / elapsed / _cpuCoreCount * 100.0;
            }
            catch { return 0; }
        }

        private static long GetVramMB()
        {
            try
            {
                if ((DateTime.UtcNow - _lastVramQuery) < VramQueryInterval)
                    return _cachedVramMB;

                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=memory.used --format=csv,noheader,nounits",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                using (var p = System.Diagnostics.Process.Start(psi))
                {
                    string output = p.StandardOutput.ReadToEnd().Trim();
                    p.WaitForExit(2000);
                    if (long.TryParse(output, out long mb))
                        _cachedVramMB = mb;
                }
                _lastVramQuery = DateTime.UtcNow;
                return _cachedVramMB;
            }
            catch { return _cachedVramMB; }
        }

        private static void WriteResourceLine(string mode, int id, int w, int h, long processMs, long saveBytes,
            long sessionBytes, long frames, long ramMB)
        {
            if (!_resourceLogInitialized) return;
            try
            {
                double cpuPct = GetCpuPercent();
                long vramMB = GetVramMB();
                string uiState = "";
                try { uiState = GetUiStateCallback?.Invoke() ?? ""; } catch { }
                lock (_resourceLogLock)
                {
                    using (var sw = new StreamWriter(_resourceLogPath, append: true))
                    {
                        sw.WriteLine($"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff},{mode},{id},{w},{h}," +
                            $"{(long)w * h / (1024.0 * 1024):F1},{processMs},{saveBytes / 1024.0:F0}," +
                            $"{sessionBytes / (1024.0 * 1024 * 1024):F3},{frames},{ramMB}," +
                            $"{cpuPct:F1},{vramMB},{uiState}");
                    }
                }
            }
            catch { /* log 失敗不影響主程式 */ }
        }
    }

    /// <summary>
    /// TrySaveCapture 的資料快照。GPU 提取後建立，傳給 CameraFrameSaver 在背景執行緒存檔。
    /// </summary>
    public struct CaptureContext
    {
        public byte[] RawBytes;
        public byte[] ProcVBytes;
        public byte[] ProcHBytes;
        public float[] MeanArr;
        public float[] MaxArr;
        public float[] RowMeanArr;
        public float[] RowMaxArr;
        public int ResizeWidth;
        public int ResizeHeight;
        public int JpgQuality;
        public int ScaleForHeader;
        public string SaveDir;
        public string BaseName;
        public int CameraId;
        public int OrigWidth;
        public int OrigHeight;
        public float MeanPeak;
        public float MaxPeak;
        public long GpuTimeMs;
        public Action<int, string, float, float> OnResult;
    }
}
