using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// AcquisitionSettings 儲存層。
    /// 讀寫 Config\acquisition-settings.json（與 exe 同目錄）。
    /// 對應 tabPageCamera（TrackBar）的所有取像參數。
    /// </summary>
    public static class AcquisitionSettingsStore
    {
        private static string FullConfigPath =>
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\acquisition-settings.json");

        public static AcquisitionSettings Load()
        {
            var settings = SettingsStoreHelper.LoadJsonFile(
                FullConfigPath,
                ParseJson,
                () => { var d = new AcquisitionSettings(); d.Validate(); return d; });

            if (settings != null) settings.Validate();

            // 檔案不存在時自動建立預設檔
            if (!File.Exists(FullConfigPath))
                Save(settings);

            return settings;
        }

        public static void Save(AcquisitionSettings settings)
        {
            if (settings == null) settings = new AcquisitionSettings();
            settings.Validate();
            SettingsStoreHelper.SaveJsonFile(FullConfigPath, SerializeJson(settings),
                nameof(AcquisitionSettingsStore));
        }

        // ── 序列化 ─────────────────────────────────────────────────────────

        private static string SerializeJson(AcquisitionSettings s)
        {
            return "{\n" +
                $"  \"CameraGrabHeight\": {IntArrayToJson(s.CameraGrabHeight)},\n" +
                $"  \"CameraExposureTimeUs\": {DoubleArrayToJson(s.CameraExposureTimeUs)},\n" +
                $"  \"CameraLineRateHz\": {DoubleArrayToJson(s.CameraLineRateHz)}\n" +
                "}";
        }

        private static string IntArrayToJson(int[] arr)
            => arr == null ? "[]" : "[" + string.Join(", ", arr) + "]";

        private static string DoubleArrayToJson(double[] arr)
            => arr == null ? "[]" : "[" + string.Join(", ", arr.Select(v => v.ToString("G"))) + "]";

        // ── 解析 ──────────────────────────────────────────────────────────

        private static AcquisitionSettings ParseJson(string json)
        {
            var result = new AcquisitionSettings();

            var ht = ParseIntArray(json, "CameraGrabHeight");
            if (ht != null) result.CameraGrabHeight = ht;

            var exp = ParseDoubleArray(json, "CameraExposureTimeUs");
            if (exp != null) result.CameraExposureTimeUs = exp;

            var lr = ParseDoubleArray(json, "CameraLineRateHz");
            if (lr != null) result.CameraLineRateHz = lr;

            return result;
        }

        private static readonly Regex _arrayPattern =
            new Regex("\"([^\"]+)\"\\s*:\\s*\\[([^\\]]*)\\]", RegexOptions.Singleline);

        private static int[] ParseIntArray(string json, string key)
        {
            foreach (Match m in _arrayPattern.Matches(json))
            {
                if (m.Groups[1].Value != key) continue;
                string body = m.Groups[2].Value.Trim();
                if (string.IsNullOrEmpty(body)) return new int[0];
                try
                {
                    return body.Split(',')
                               .Select(t => int.Parse(t.Trim()))
                               .ToArray();
                }
                catch { return null; }
            }
            return null;
        }

        private static double[] ParseDoubleArray(string json, string key)
        {
            foreach (Match m in _arrayPattern.Matches(json))
            {
                if (m.Groups[1].Value != key) continue;
                string body = m.Groups[2].Value.Trim();
                if (string.IsNullOrEmpty(body)) return new double[0];
                try
                {
                    return body.Split(',')
                               .Select(t => double.Parse(t.Trim(),
                                   System.Globalization.CultureInfo.InvariantCulture))
                               .ToArray();
                }
                catch { return null; }
            }
            return null;
        }
    }
}
