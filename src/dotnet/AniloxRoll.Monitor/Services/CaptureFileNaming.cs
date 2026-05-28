using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 擷取檔案命名規則的單一真相。寫端（CameraFrameSaver / InspectionEngine 存檔）
    /// 與讀端（回顧 / 統計 / 合圖 / 縮圖）共用同一套 suffix —— 改命名格式只改這裡。
    ///
    /// 一張擷取（base = "{yyyyMMdd_HHmmss.fff}-{camId}"）對應的產出：
    ///   {base}_raw.jpg / _proc_v.jpg / _proc_h.jpg / _mean_v.bin / _max_v.bin / _mean_h.bin / _max_h.bin
    /// 舊格式（_proc.jpg / _mean.bin / _max.bin / _row_mean.bin / _row_max.bin）僅供讀端 fallback。
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
        public const string MeanV  = "_mean_v.bin";
        public const string MaxV   = "_max_v.bin";
        public const string MeanH  = "_mean_h.bin";
        public const string MaxH   = "_max_h.bin";

        // ── 舊格式 suffix（向後相容 fallback）──────────────────────────────
        public const string ProcLegacy  = "_proc.jpg";
        public const string MeanVLegacy = "_mean.bin";
        public const string MaxVLegacy  = "_max.bin";
        public const string MeanHLegacy = "_row_mean.bin";
        public const string MaxHLegacy  = "_row_max.bin";

        // ── glob ─────────────────────────────────────────────────────────
        public const string RawJpgGlob = "*" + RawJpg;

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
        public static string MeanSuffix(string dir) => dir == "h" ? MeanH : MeanV;
        public static string MaxSuffix(string dir)  => dir == "h" ? MaxH : MaxV;

        /// <summary>解析處理圖路徑：新 v/h 命名存在則用之，否則回傳舊命名 _proc.jpg（不保證存在）。</summary>
        public static string ResolveProcJpg(string baseNoSuffix, string dir)
        {
            string p = baseNoSuffix + ProcSuffix(dir);
            return File.Exists(p) ? p : baseNoSuffix + ProcLegacy;
        }
    }
}
