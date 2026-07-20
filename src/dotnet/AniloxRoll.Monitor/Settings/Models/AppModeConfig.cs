using System;
using System.ComponentModel;
using System.IO;
using StorageBridge.Core;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum MachineRole
    {
        [Description("檢測模式")] Inspection,
        [Description("儲存模式")] Storage
    }

    /// <summary>
    /// 讀取 Config\app-mode.json 決定本機角色。
    /// 檔案不存在時自動建立預設檔（與 Inspection / Acquisition / System 統一 fallback 行為）。
    /// </summary>
    public class AppModeConfig
    {
        public MachineRole Role { get; set; } = AppModeDefaults.Role;

        /// <summary>Storage 模式：本機 Config 子目錄路徑（供 CleanupFlagWatcher 監看 cleanup-request.flag）。</summary>
        public string StorageMachineConfigFolder { get; set; } = AppModeDefaults.StorageMachineConfigFolder;

        /// <summary>Storage 模式：循環儲存的根目錄；空字串時 fallback 至 CaptureRootPath。</summary>
        public string StorageMachineDataPath { get; set; } = AppModeDefaults.StorageMachineDataPath;

        /// <summary>
        /// Storage 部署的預留空間 bootstrap；執行時與 PropertyGrid 的 LocalMinFreeGB 同步。
        /// </summary>
        public int StorageMinFreeGB { get; set; } = AppModeDefaults.StorageMinFreeGB;

        public static AppModeConfig Load()
        {
            string fullPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\app-mode.json");
            var loaded = JsonConfigLoader.LoadOrDefault(@"Config\app-mode.json", new AppModeConfig());
            if (loaded.StorageMinFreeGB <= 0)
                loaded.StorageMinFreeGB = AppModeDefaults.StorageMinFreeGB;
            if (!File.Exists(fullPath))
                JsonConfigLoader.SaveJson(fullPath, loaded);
            return loaded;
        }

        public void Save()
        {
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"Config\app-mode.json");
            JsonConfigLoader.SaveJson(path, this);
        }
    }
}
