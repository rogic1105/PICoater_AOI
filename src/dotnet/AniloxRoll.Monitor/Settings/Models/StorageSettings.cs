using System.ComponentModel;
using System.IO;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("存檔")]     public bool EnableAutoCapture { get; set; } = InspectionDefaults.EnableAutoCapture;
        [DisplayName("存原圖")]   public bool SaveOriginalBmp { get; set; } = InspectionDefaults.SaveOriginalBmp;

        /// <summary>Anilox 資料根目錄（如 D:\Anilox），Captures/Bg/Logs 為其子目錄。
        /// 註：DCF 不在此，改跟 exe 走（exe\Config\Radient_Config.dcf）。</summary>
        [DisplayName("Anilox 根目錄")] public string AniloxRootPath { get; set; } = InspectionDefaults.AniloxRootPath;

        // 子目錄路徑均由 AniloxRootPath 推算（PropertyGrid 不顯示，避免使用者改錯）
        [Browsable(false)] public string CaptureRootPath => Path.Combine(AniloxRootPath, "Captures");
        [Browsable(false)] public string BackgroundPath  => Path.Combine(AniloxRootPath, "Bg");
        [Browsable(false)] public string LogsPath        => Path.Combine(AniloxRootPath, "Logs");

        [DisplayName("本地預留磁碟空間")] public int LocalMinFreeGB { get; set; } = InspectionDefaults.LocalMinFreeGB;
        [DisplayName("遠端路徑")]         public string RemotePath       { get; set; } = InspectionDefaults.RemotePath;
        [DisplayName("遠端 Config 路徑")] public string RemoteConfigPath { get; set; } = InspectionDefaults.RemoteConfigPath;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(AniloxRootPath)) AniloxRootPath = InspectionDefaults.AniloxRootPath;
            if (LocalMinFreeGB < 1) LocalMinFreeGB = InspectionDefaults.LocalMinFreeGB;
            if (RemotePath == null) RemotePath = InspectionDefaults.RemotePath;
            if (RemoteConfigPath == null) RemoteConfigPath = InspectionDefaults.RemoteConfigPath;
        }

        public override string ToString() => "Storage";
    }
}
