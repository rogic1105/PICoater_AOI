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
            try
            {
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string fullPath = Path.Combine(baseDir, relativePath);

                if (!File.Exists(fullPath)) return fallback;

                string json = File.ReadAllText(fullPath, Encoding.UTF8);
                if (string.IsNullOrWhiteSpace(json)) return fallback;

                JavaScriptSerializer serializer = new JavaScriptSerializer();
                var result = serializer.Deserialize<T>(json);
                return result ?? fallback;
            }
            catch
            {
                return fallback;
            }
        }
    }
}
