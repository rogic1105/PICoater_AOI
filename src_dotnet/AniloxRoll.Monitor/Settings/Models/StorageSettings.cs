using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("存檔")]       public bool EnableAutoCapture { get; set; } = InspectionDefaults.EnableAutoCapture;
        [DisplayName("存原圖")]     public bool SaveOriginalBmp { get; set; } = InspectionDefaults.SaveOriginalBmp;
        [DisplayName("存圖目錄")]   public string CaptureRootPath { get; set; } = InspectionDefaults.CaptureRootPath;
        [DisplayName("存背景目錄")] public string BackgroundPath { get; set; } = InspectionDefaults.BackgroundPath;

        [DisplayName("本地預留磁碟空間")] public int LocalMinFreeGB { get; set; } = InspectionDefaults.LocalMinFreeGB;
        [DisplayName("遠端路徑")]         public string RemotePath       { get; set; } = InspectionDefaults.RemotePath;
        [DisplayName("遠端 Config 路徑")] public string RemoteConfigPath { get; set; } = InspectionDefaults.RemoteConfigPath;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = InspectionDefaults.CaptureRootPath;
            if (string.IsNullOrWhiteSpace(BackgroundPath)) BackgroundPath = InspectionDefaults.BackgroundPath;
            if (LocalMinFreeGB < 1) LocalMinFreeGB = InspectionDefaults.LocalMinFreeGB;
            if (RemotePath == null) RemotePath = InspectionDefaults.RemotePath;
            if (RemoteConfigPath == null) RemoteConfigPath = InspectionDefaults.RemoteConfigPath;
        }

        public override string ToString() => "Storage";
    }
}
