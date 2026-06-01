using System;
using System.Collections.Generic;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 系統層設定（硬體/設備拓樸）。
    /// </summary>
    public class SystemSettings
    {
        public List<CameraHardwareConfig> CameraDevices { get; set; } = new List<CameraHardwareConfig>();

        public static SystemSettings CreateDefault()
        {
            SystemSettings fallback = new SystemSettings { CameraDevices = SystemDefaults.NewCameraDevices() };
            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\system-settings.json");
            var loaded = JsonConfigLoader.LoadOrDefault(@"Config\system-settings.json", fallback);
            bool useDefault = loaded.CameraDevices == null || loaded.CameraDevices.Count == 0;
            // 檔案不存在時自動建立預設檔
            if (!File.Exists(fullPath))
                JsonConfigLoader.SaveJson(fullPath, fallback);
            return useDefault ? fallback : loaded;
        }
    }
}
