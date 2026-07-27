using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 擷取資料的日期階層儲存路徑單一真相。寫端（InspectionLogService）與讀端
    /// （統計與資料 repositories）共用 —— 改目錄結構只改這裡。
    ///
    /// 結構：
    ///   {root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv   ← 每日檢測 CSV（檔，位於 yyyyMM 資料夾）
    ///   {root}\{yyyy}\{yyyyMM}\{yyyyMMdd}\      ← 該日影像資料夾
    ///
    /// "yyyy"/"yyyyMM"/"yyyyMMdd" 為純數字格式，culture-invariant。
    /// </summary>
    public static class CaptureStoragePaths
    {
        private const string LegacyCaptureDirectoryName = "Captures_pack";

        /// <summary>每日 CSV 路徑：{root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv</summary>
        public static string DailyCsv(string root, DateTime d) =>
            Path.Combine(root, d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd") + ".csv");

        /// <summary>該日影像資料夾：{root}\{yyyy}\{yyyyMM}\{yyyyMMdd}</summary>
        public static string DateImageDir(string root, DateTime d) =>
            Path.Combine(root, d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd"));

        /// <summary>該日影像資料夾（從 "yyyyMMdd…" 字串，須 ≥ 8 字元）：取前 4/6/8 字元組階層。</summary>
        public static string DateImageDir(string root, string yyyymmdd) =>
            Path.Combine(root, yyyymmdd.Substring(0, 4), yyyymmdd.Substring(0, 6), yyyymmdd.Substring(0, 8));

        /// <summary>
        /// Rebuildable per-grab curve summary. The original curve bins remain the source of truth.
        /// Keeping the summary under the capture date makes retention remove both together.
        /// </summary>
        public static string GrabCurveSummary(string root, DateTime captureDate, string grabId) =>
            Path.Combine(DateImageDir(root, captureDate), "_curve_summary", grabId + ".mcsf");

        /// <summary>One appendable archive per grab, stored beside that day's capture outputs.</summary>
        public static string GrabArchive(string root, DateTime captureDate, string grabId) =>
            Path.Combine(DateImageDir(root, captureDate), grabId + CaptureArchiveStore.Extension);

        /// <summary>
        /// Keeps review/report readers on the configured packed-capture root while preserving
        /// intentionally selected external data roots.
        /// </summary>
        public static string ResolveSelectedDataRoot(string selectedRoot, string configuredRoot)
        {
            if (string.IsNullOrWhiteSpace(selectedRoot))
                return configuredRoot ?? string.Empty;
            if (string.IsNullOrWhiteSpace(configuredRoot))
                return selectedRoot;

            try
            {
                string selectedFull = NormalizeForComparison(selectedRoot);
                string configuredFull = NormalizeForComparison(configuredRoot);
                string configuredParent = Path.GetDirectoryName(configuredFull);
                if (string.Equals(selectedFull, configuredFull, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(selectedFull, configuredParent, StringComparison.OrdinalIgnoreCase))
                    return configuredRoot;

                string legacyRoot = string.IsNullOrEmpty(configuredParent)
                    ? string.Empty
                    : NormalizeForComparison(
                        Path.Combine(configuredParent, LegacyCaptureDirectoryName));
                return string.Equals(selectedFull, legacyRoot, StringComparison.OrdinalIgnoreCase)
                    ? configuredRoot
                    : selectedRoot;
            }
            catch (Exception)
            {
                return selectedRoot;
            }
        }

        public static string UpgradeLegacyPackedRoot(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return path ?? string.Empty;

            try
            {
                string fullPath = NormalizeForComparison(path);
                if (!string.Equals(
                    Path.GetFileName(fullPath),
                    LegacyCaptureDirectoryName,
                    StringComparison.OrdinalIgnoreCase))
                    return path;

                string parent = Path.GetDirectoryName(fullPath);
                return string.IsNullOrEmpty(parent)
                    ? path
                    : Path.Combine(parent, "Captures");
            }
            catch (Exception)
            {
                return path;
            }
        }

        private static string NormalizeForComparison(string path) =>
            Path.GetFullPath(path).TrimEnd(
                Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar);
    }
}
