using System;
using System.Collections.Generic;
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

            if (!string.IsNullOrWhiteSpace(ctx.GrabId))
            {
                SaveCaptureArchive(ctx);
                return;
            }

            SaveJpegFromBytes(ctx.RawBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.RawJpg), ctx.JpgQuality);

            if (ctx.ProcCBytes != null)
                SaveJpegFromBytes(ctx.ProcCBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.ProcC), ctx.JpgQuality);

            if (ctx.ProcRBytes != null)
                SaveJpegFromBytes(ctx.ProcRBytes, ctx.ResizeWidth, ctx.ResizeHeight,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.ProcR), ctx.JpgQuality);

            if (ctx.MeanC != null)
            {
                SaveCurveBinFromArray(ctx.MeanC, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MeanC));
                SaveCurveBinFromArray(ctx.MaxC, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MaxC));
            }

            if (ctx.MeanR != null)
            {
                SaveCurveBinFromArray(ctx.MeanR, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MeanR));
                SaveCurveBinFromArray(ctx.MaxR, ctx.ScaleForHeader,
                    Path.Combine(ctx.SaveDir, ctx.BaseName + CaptureFileNaming.MaxR));
            }

            // 每幀硬體 frame-start tick 側車（回顧用 tick 就近對位補黑：cam 各自獨立掉幀位置不同，
            // seq 會歪、檔名軟體戳不知道實際掉哪幀；唯有同板 tick 可精準對齊）。
            if (ctx.FrameStartTicks > 0)
                AppendTickSidecar(ctx.SaveDir, ctx.BaseName, ctx.FrameStartTicks);

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
                if (ctx.FrameStartTicks > 0)
                {
                    string tickSidecar = Path.Combine(ctx.SaveDir, TickSidecarName);
                    if (File.Exists(tickSidecar))
                    {
                        Array.Resize(ref savedFiles, savedFiles.Length + 1);
                        savedFiles[savedFiles.Length - 1] = tickSidecar;
                    }
                }
                ctx.OnFilesSaved(savedFiles);
            }

            float maxCMean = ComputeCurveMeanNormalized(ctx.MaxC);
            float meanRPeak = ComputeCurvePeakNormalized(ctx.MeanR);
            float maxRPeak = ComputeCurvePeakNormalized(ctx.MaxR);
            ctx.OnResult?.Invoke(ctx.GrabId, ctx.CameraId, ctx.BaseName,
                ctx.MeanPeak, ctx.MaxPeak, maxCMean, meanRPeak, maxRPeak);
        }

        private void SaveCaptureArchive(CaptureContext ctx)
        {
            string archivePath = Path.Combine(
                ctx.SaveDir, ctx.GrabId + CaptureArchiveStore.Extension);
            var assets = new List<CaptureArchiveAsset>(7)
            {
                new CaptureArchiveAsset
                {
                    Kind = CaptureAssetKind.RawJpeg,
                    Data = EncodeJpegFromBytes(
                        ctx.RawBytes, ctx.ResizeWidth, ctx.ResizeHeight, ctx.JpgQuality)
                }
            };
            AddJpegAsset(assets, CaptureAssetKind.ProcessedColumnJpeg, ctx.ProcCBytes, ctx);
            AddJpegAsset(assets, CaptureAssetKind.ProcessedRowJpeg, ctx.ProcRBytes, ctx);
            AddCurveAsset(assets, CaptureAssetKind.MeanColumnCurve, ctx.MeanC, ctx.ScaleForHeader);
            AddCurveAsset(assets, CaptureAssetKind.MaxColumnCurve, ctx.MaxC, ctx.ScaleForHeader);
            AddCurveAsset(assets, CaptureAssetKind.MeanRowCurve, ctx.MeanR, ctx.ScaleForHeader);
            AddCurveAsset(assets, CaptureAssetKind.MaxRowCurve, ctx.MaxR, ctx.ScaleForHeader);

            long frameBytes = CaptureArchiveStore.AppendFrame(
                archivePath, ctx.GrabId, ctx.BaseName, ctx.CameraId,
                ctx.FrameStartTicks, assets);
            LastSaveBytesTotal = frameBytes;
            SessionSaveBytes += frameBytes;
            SessionFrameCount++;

            long ramMB = System.Diagnostics.Process.GetCurrentProcess().WorkingSet64 / (1024 * 1024);
            AppendResourceLog(ctx.CameraId, ctx.OrigWidth, ctx.OrigHeight,
                ctx.GpuTimeMs, frameBytes, SessionSaveBytes, SessionFrameCount, ramMB);

            FlowTrace.Log(
                $"capture archive append grab={ctx.GrabId} cam={ctx.CameraId} " +
                $"frame={ctx.BaseName} assets={assets.Count} bytes={frameBytes}");
            ctx.OnFilesSaved?.Invoke(new[] { archivePath });

            float maxCMean = ComputeCurveMeanNormalized(ctx.MaxC);
            float meanRPeak = ComputeCurvePeakNormalized(ctx.MeanR);
            float maxRPeak = ComputeCurvePeakNormalized(ctx.MaxR);
            ctx.OnResult?.Invoke(ctx.GrabId, ctx.CameraId, ctx.BaseName,
                ctx.MeanPeak, ctx.MaxPeak, maxCMean, meanRPeak, maxRPeak);
        }

        private static void AddJpegAsset(
            List<CaptureArchiveAsset> assets,
            CaptureAssetKind kind,
            byte[] data,
            CaptureContext ctx)
        {
            if (data == null || data.Length == 0) return;
            assets.Add(new CaptureArchiveAsset
            {
                Kind = kind,
                Data = EncodeJpegFromBytes(
                    data, ctx.ResizeWidth, ctx.ResizeHeight, ctx.JpgQuality)
            });
        }

        private static void AddCurveAsset(
            List<CaptureArchiveAsset> assets,
            CaptureAssetKind kind,
            float[] curve,
            int scaleForHeader)
        {
            byte[] data = EncodeCurveBin(curve, scaleForHeader);
            if (data == null) return;
            assets.Add(new CaptureArchiveAsset { Kind = kind, Data = data });
        }

        internal static float ComputeCurveMeanNormalized(float[] curve)
        {
            if (curve == null || curve.Length == 0) return float.NaN;

            double sum = 0;
            for (int i = 0; i < curve.Length; i++)
                sum += curve[i];
            return (float)(sum / curve.Length / 255.0);
        }

        internal static float ComputeCurvePeakNormalized(float[] curve)
        {
            if (curve == null || curve.Length == 0) return float.NaN;
            float peak = curve[0];
            for (int i = 1; i < curve.Length; i++)
                if (curve[i] > peak) peak = curve[i];
            return peak / 255f;
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
            File.WriteAllBytes(path, EncodeJpegFromBytes(data, w, h, quality));
        }

        internal static byte[] EncodeJpegFromBytes(byte[] data, int w, int h, int quality)
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
                    using (var output = new MemoryStream())
                    {
                        if (codec == null)
                        {
                            _reuseBmp24.Save(output, ImageFormat.Jpeg);
                            return output.ToArray();
                        }

                        using (var ep = new EncoderParameters(1))
                        {
                            ep.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
                            _reuseBmp24.Save(output, codec, ep);
                        }
                        return output.ToArray();
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
            byte[] data = EncodeCurveBin(arr, scaleForHeader);
            if (data != null) File.WriteAllBytes(path, data);
        }

        internal static byte[] EncodeCurveBin(float[] arr, int scaleForHeader)
        {
            if (arr == null || arr.Length == 0) return null;
            using (var output = new MemoryStream(16 + arr.Length * sizeof(float)))
            using (var bw = new BinaryWriter(output))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(1);                        // version
                bw.Write((float)scaleForHeader);    // scale_factor
                bw.Write(arr.Length);               // array_length
                for (int i = 0; i < arr.Length; i++)
                    bw.Write(arr[i]);
                bw.Flush();
                return output.ToArray();
            }
        }

        // ── Tick 側車（每影像資料夾一份 _ticks.csv：「baseName,ticks」）──────────

        /// <summary>tick 側車檔名（與當日影像同資料夾；回顧載入時讀回對位）。</summary>
        public const string TickSidecarName = "_ticks.csv";
        private static readonly object _tickSidecarLock = new object();

        /// <summary>append 一列「baseName,ticks」到當日資料夾的 _ticks.csv。
        /// 多相機背景執行緒共寫 → static lock。失敗不影響存檔主流程。</summary>
        internal static void AppendTickSidecar(string saveDir, string baseName, long ticks)
        {
            try
            {
                string path = Path.Combine(saveDir, TickSidecarName);
                lock (_tickSidecarLock) { File.AppendAllText(path, baseName + "," + ticks + "\r\n"); }
            }
            catch { /* 側車失敗不影響主存檔 */ }
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
        public byte[] ProcCBytes;
        public byte[] ProcRBytes;
        public float[] MeanC;
        public float[] MaxC;
        public float[] MeanR;
        public float[] MaxR;
        public int ResizeWidth;
        public int ResizeHeight;
        public int JpgQuality;
        public int ScaleForHeader;
        public string SaveDir;
        public string BaseName;
        public string GrabId;
        public int CameraId;
        public int OrigWidth;
        public int OrigHeight;
        public float MeanPeak;
        public float MaxPeak;
        public long GpuTimeMs;
        /// <summary>本幀 frame-start 硬體時戳（Data Latch ticks）。0＝未取得。
        /// 寫進 _ticks.csv 側車，供回顧用「tick 就近對位」精準補黑（免疫 seq 歪掉/軟體戳抖動）。</summary>
        public long FrameStartTicks;
        public Action<string, int, string, float, float, float, float, float> OnResult;
        /// <summary>存檔完成後回呼，傳入已儲存的檔案路徑陣列（供遠端複製佇列用）。</summary>
        public Action<string[]> OnFilesSaved;
    }
}
