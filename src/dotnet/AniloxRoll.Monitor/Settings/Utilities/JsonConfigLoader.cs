using System;
using System.IO;
using System.Text;
using System.Web.Script.Serialization;

namespace AniloxRoll.Monitor.Core.Data
{
    internal static class JsonConfigLoader
    {
        public static T LoadOrDefault<T>(string relativePath, T fallback) where T : class
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string fullPath = Path.Combine(baseDir, relativePath);
            return SettingsStoreHelper.LoadJsonFile(
                fullPath,
                json => new JavaScriptSerializer().Deserialize<T>(json),
                () => fallback);
        }

        /// <summary>簡單 JSON 格式化（縮排 2 空格），供各 Store 共用。</summary>
        public static string FormatJson(string json)
        {
            var sb = new StringBuilder();
            int indent = 0;
            bool inString = false;
            foreach (char c in json)
            {
                if (c == '"') inString = !inString;
                if (inString) { sb.Append(c); continue; }
                switch (c)
                {
                    case '{': case '[':
                        sb.Append(c); sb.Append('\n'); sb.Append(new string(' ', ++indent * 2));
                        break;
                    case '}': case ']':
                        sb.Append('\n'); sb.Append(new string(' ', --indent * 2)); sb.Append(c);
                        break;
                    case ',':
                        sb.Append(c); sb.Append('\n'); sb.Append(new string(' ', indent * 2));
                        break;
                    case ':':
                        sb.Append(": ");
                        break;
                    default:
                        sb.Append(c);
                        break;
                }
            }
            return sb.ToString();
        }

        public static void SaveJson<T>(string fullPath, T obj) where T : class
        {
            string json = new JavaScriptSerializer().Serialize(obj);
            SettingsStoreHelper.SaveJsonFile(
                fullPath, FormatJson(json), typeof(T).Name);
        }
    }
}
