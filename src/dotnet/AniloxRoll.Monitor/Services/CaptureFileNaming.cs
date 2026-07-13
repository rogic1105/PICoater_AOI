using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 擷取檔案命名規則的單一真相。寫端（CameraFrameSaver / InspectionEngine 存檔）
    /// 與讀端（回顧 / 統計 / 合圖 / 縮圖）共用同一套 suffix —— 改命名格式只改這裡。
    ///
    /// 一張擷取（base = "{yyyyMMdd_HHmmss.fff}-{camId}"）對應的產出：
    ///   {base}_raw.jpg / _proc_v.jpg / _proc_h.jpg / _mean_c.bin / _max_c.bin / _mean_r.bin / _max_r.bin
    /// 上一代曲線格式（_mean_v/_max_v/_mean_h/_max_h）與更早格式僅供讀端 fallback。
    ///
    /// 本類只統一「檔名字串」；fallback 的「載入行為」（new ?? legacy）仍由各 caller 決定，
    /// 因為不同 caller 對「new 存在但讀失敗」的處理略有差異，統一會改變 edge case。
    /// </summary>
    public static class CaptureFileNaming
    {
        // ── 新格式 suffix ──────────────────────────────────────────────────
        public const string RawJpg = "_raw.jpg";
        public const string ProcV  = "_proc_v.jpg";
        public const string ProcH  = "_proc_h.jpg";
        public const string MeanC  = "_mean_c.bin";
        public const string MaxC   = "_max_c.bin";
        public const string MeanR  = "_mean_r.bin";
        public const string MaxR   = "_max_r.bin";

        // ── 舊格式 suffix（向後相容 fallback）──────────────────────────────
        public const string MeanCPrevious = "_mean_v.bin";
        public const string MaxCPrevious  = "_max_v.bin";
        public const string MeanRPrevious = "_mean_h.bin";
        public const string MaxRPrevious  = "_max_h.bin";
        public const string ProcLegacy  = "_proc.jpg";
        public const string MeanCLegacy = "_mean.bin";
        public const string MaxCLegacy  = "_max.bin";
        public const string MeanRLegacy = "_row_mean.bin";
        public const string MaxRLegacy  = "_row_max.bin";

        // ── glob ─────────────────────────────────────────────────────────
        public const string RawJpgGlob = "*" + RawJpg;

        // ── 背景 .bin（每相機一張，依影像寬度區分）──────────────────────
        /// <summary>背景檔名：bg_{width}_{camId}.bin</summary>
        public static string BgBin(int width, int camId) => $"bg_{width}_{camId}.bin";
        /// <summary>背景檔 glob（全部）：bg_*.bin</summary>
        public const string BgGlob = "bg_*.bin";
        /// <summary>背景檔 glob（指定相機，不分寬度）：bg_*_{camId}.bin</summary>
        public static string BgGlobForCam(int camId) => $"bg_*_{camId}.bin";

        // ── _raw.jpg 後綴判斷 / 去除 ─────────────────────────────────────
        public static bool IsRawJpg(string path) =>
            path != null && path.EndsWith(RawJpg, StringComparison.OrdinalIgnoreCase);

        /// <summary>去掉 "_raw.jpg" 後綴取得 base（caller 須先確認是 raw.jpg）。</summary>
        public static string StripRawJpg(string rawJpgPath) =>
            rawJpgPath.Substring(0, rawJpgPath.Length - RawJpg.Length);

        /// <summary>影像路徑 → 同組檔的 base：_raw.jpg 去後綴；其餘去副檔名（dir + 檔名）。</summary>
        public static string BaseFromImagePath(string imagePath)
        {
            if (IsRawJpg(imagePath)) return StripRawJpg(imagePath);
            return Path.Combine(
                Path.GetDirectoryName(imagePath),
                Path.GetFileNameWithoutExtension(imagePath));
        }

        // ── 方向（"h" / 其餘視為 "v"）選 suffix ─────────────────────────
        public static string ProcSuffix(string dir) => dir == "h" ? ProcH : ProcV;
        public static string MeanSuffix(string dir) => dir == "h" ? MeanR : MeanC;
        public static string MaxSuffix(string dir)  => dir == "h" ? MaxR : MaxC;

        public static string ResolveMeanC(string baseNoSuffix) =>
            ResolveExisting(baseNoSuffix, MeanC, MeanCPrevious, MeanCLegacy);

        public static string ResolveMaxC(string baseNoSuffix) =>
            ResolveExisting(baseNoSuffix, MaxC, MaxCPrevious, MaxCLegacy);

        public static string ResolveMeanR(string baseNoSuffix) =>
            ResolveExisting(baseNoSuffix, MeanR, MeanRPrevious, MeanRLegacy);

        public static string ResolveMaxR(string baseNoSuffix) =>
            ResolveExisting(baseNoSuffix, MaxR, MaxRPrevious, MaxRLegacy);

        private static string ResolveExisting(
            string baseNoSuffix, string current, string previous, string legacy)
        {
            string path = baseNoSuffix + current;
            if (File.Exists(path)) return path;
            path = baseNoSuffix + previous;
            if (File.Exists(path)) return path;
            return baseNoSuffix + legacy;
        }

        /// <summary>解析處理圖路徑：新 v/h 命名存在則用之，否則回傳舊命名 _proc.jpg（不保證存在）。</summary>
        public static string ResolveProcJpg(string baseNoSuffix, string dir)
        {
            string p = baseNoSuffix + ProcSuffix(dir);
            return File.Exists(p) ? p : baseNoSuffix + ProcLegacy;
        }
    }
}
