using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// DcfPath 路徑解析：所有 dcf path 進 MIL 前呼這個 helper，
    /// 把 InspectionDefaults.DcfPath 的相對路徑（Config\Radient_Config.dcf）
    /// 結合 exe 目錄成絕對路徑。Path.Combine 對絕對路徑天然兼容。
    /// </summary>
    internal static class DcfPathHelper
    {
        public static string Resolve(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return path;
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, path);
        }
    }
}
