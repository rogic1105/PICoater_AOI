using System.ComponentModel;

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
    /// 檔案不存在或讀取失敗時預設為 Inspection。
    /// </summary>
    public class AppModeConfig
    {
        public MachineRole Role { get; set; } = MachineRole.Inspection;

        /// <summary>Storage 模式：AniloxConfig 共用資料夾的本機路徑（供 CleanupFlagWatcher 使用）。</summary>
        public string LocalConfigFolder { get; set; } = @"D:\AniloxConfig";

        /// <summary>Storage 模式：循環儲存的根目錄（例如 D:\AniloxStorage）。空字串時 fallback 至 CaptureRootPath。</summary>
        public string StorageFolderPath { get; set; } = string.Empty;

        public static AppModeConfig Load()
            => JsonConfigLoader.LoadOrDefault(@"Config\app-mode.json", new AppModeConfig());

        public void Save()
        {
            string path = System.IO.Path.Combine(
                System.AppDomain.CurrentDomain.BaseDirectory, @"Config\app-mode.json");
            JsonConfigLoader.SaveJson(path, this);
        }
    }
}
