using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Services;
using TanukiCv.Utils;

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
                Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.RawJpg), ctx.JpgQuality);

            if (ctx.ProcVBytes != null)
                SaveJpegFromBytes(ctx.ProcVBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.ProcV), ctx.JpgQuality);

            if (ctx.ProcHBytes != null)
                SaveJpegFromBytes(ctx.ProcHBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.ProcH), ctx.JpgQuality);

            if (ctx.MeanArr != null)
            {
                SaveCurveBinFromArray(ctx.MeanArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MeanV));
                SaveCurveBinFromArray(ctx.MaxArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MaxV));
            }

            if (ctx.RowMeanArr != null)
            {
                SaveCurveBinFromArray(ctx.RowMeanArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MeanH));
                SaveCurveBinFromArray(ctx.RowMaxArr, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MaxH));
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

            // 通知遠端複製佇列
            if (ctx.OnFilesSaved != null)
            {
                var savedFiles = Directory.GetFiles(ctx.SaveDir, ctx.BaseName + "*");
                ctx.OnFilesSaved(savedFiles);
            }

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

        /// <summary>初始化 resource log 檔案（啟動時呼叫一次）。
        /// 傳入完整的 logs 目錄路徑（如 D:\Anilox\Logs），呼叫端負責 AniloxRoot fallback 與目錄建立。</summary>
        public static void InitResourceLog(string logsPath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(logsPath))
                {
                    System.Diagnostics.Trace.WriteLine("[ResourceLog] logsPath 空，初始化跳過");
                    return;
                }
                string dir = logsPath;
                Directory.CreateDirectory(dir);

                // 啟動時把「昨天以前」的多個 resource-monitor-yyyyMMdd-HHmmss.csv 合成
                // 以日為單位的 resource-monitor-yyyyMMdd.csv（今天的檔不動，可能還在被別的程式寫入）
                MergeOldResourceLogs(dir);

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

        /// <summary>
        /// 將 logs/ 內「昨天以前」的 resource-monitor-yyyyMMdd-HHmmss.csv 小檔，
        /// 按日合併成 resource-monitor-yyyyMMdd.csv。合併成功後刪除原始小檔。
        /// 排除「今天」的檔（程式可能還在寫）。
        /// </summary>
        private static void MergeOldResourceLogs(string logDir)
        {
            const string prefix = "resource-monitor-";
            if (!Directory.Exists(logDir)) return;

            string today = DateTime.Now.ToString("yyyyMMdd");

            // Group: yyyyMMdd → 該日的所有小檔（含 HHmmss 的）
            var grouped = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<string>>();
            foreach (var file in Directory.GetFiles(logDir, prefix + "*.csv"))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                if (!name.StartsWith(prefix)) continue;
                string rest = name.Substring(prefix.Length);
                if (rest.Length <= 8) continue;                 // "yyyyMMdd" 是已合併目標檔，跳過
                if (rest.Length < 9 || rest[8] != '-') continue; // 需為 yyyyMMdd-HHmmss 形式
                string datePart = rest.Substring(0, 8);
                if (!System.Text.RegularExpressions.Regex.IsMatch(datePart, @"^\d{8}$")) continue;
                if (datePart == today) continue;                 // 今天的不動

                if (!grouped.TryGetValue(datePart, out var list))
                {
                    list = new System.Collections.Generic.List<string>();
                    grouped[datePart] = list;
                }
                list.Add(file);
            }

            foreach (var kv in grouped)
            {
                string datePart = kv.Key;
                var files = kv.Value;
                files.Sort();  // 按檔名升序 = 時間升序（HHmmss）

                string targetPath = Path.Combine(logDir, $"{prefix}{datePart}.csv");
                bool targetExisted = File.Exists(targetPath);

                try
                {
                    using (var sw = new StreamWriter(targetPath, append: true))
                    {
                        foreach (var src in files)
                        {
                            bool firstLine = true;
                            using (var sr = new StreamReader(src))
                            {
                                string line;
                                while ((line = sr.ReadLine()) != null)
                                {
                                    if (firstLine)
                                    {
                                        firstLine = false;
                                        if (!targetExisted)
                                        {
                                            sw.WriteLine(line);  // 第一個檔的 header 寫入
                                            targetExisted = true;
                                        }
                                        // 已寫過 header → 跳過後續檔的 header
                                    }
                                    else
                                    {
                                        sw.WriteLine(line);
                                    }
                                }
                            }
                        }
                    }

                    // 合併成功 → 刪除原始小檔
                    int deleted = 0;
                    foreach (var src in files)
                    {
                        try { File.Delete(src); deleted++; }
                        catch (Exception ex) { System.Diagnostics.Trace.WriteLine($"[ResourceLog] Delete {Path.GetFileName(src)} failed: {ex.Message}"); }
                    }
                    System.Diagnostics.Trace.WriteLine($"[ResourceLog] Merged {deleted} files → {Path.GetFileName(targetPath)}");
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine($"[ResourceLog] Merge {datePart} failed: {ex.Message}");
                }
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
        /// <summary>存檔完成後回呼，傳入已儲存的檔案路徑陣列（供遠端複製佇列用）。</summary>
        public Action<string[]> OnFilesSaved;
    }
}
