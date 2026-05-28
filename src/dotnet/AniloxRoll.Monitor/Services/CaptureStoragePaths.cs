using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 擷取資料的日期階層儲存路徑單一真相。寫端（InspectionLogService）與讀端
    /// （InspectionStatisticsService）共用 —— 改目錄結構只改這裡。
    ///
    /// 結構：
    ///   {root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv   ← 每日檢測 CSV（檔，位於 yyyyMM 資料夾）
    ///   {root}\{yyyy}\{yyyyMM}\{yyyyMMdd}\      ← 該日影像資料夾
    ///
    /// "yyyy"/"yyyyMM"/"yyyyMMdd" 為純數字格式，culture-invariant。
    /// </summary>
    public static class CaptureStoragePaths
    {
        /// <summary>每日 CSV 路徑：{root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv</summary>
        public static string DailyCsv(string root, DateTime d) =>
            Path.Combine(root, d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd") + ".csv");

        /// <summary>該日影像資料夾：{root}\{yyyy}\{yyyyMM}\{yyyyMMdd}</summary>
        public static string DateImageDir(string root, DateTime d) =>
            Path.Combine(root, d.ToString("yyyy"), d.ToString("yyyyMM"), d.ToString("yyyyMMdd"));

        /// <summary>該日影像資料夾（從 "yyyyMMdd…" 字串，須 ≥ 8 字元）：取前 4/6/8 字元組階層。</summary>
        public static string DateImageDir(string root, string yyyymmdd) =>
            Path.Combine(root, yyyymmdd.Substring(0, 4), yyyymmdd.Substring(0, 6), yyyymmdd.Substring(0, 8));
    }
}
