using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("存檔")]       public bool EnableAutoCapture { get; set; } = true;
        [DisplayName("存原圖")]     public bool SaveOriginalBmp { get; set; } = false;
        [DisplayName("存圖目錄")]   public string CaptureRootPath { get; set; } = @"D:\AniloxCaptures";
        [DisplayName("存背景目錄")] public string BackgroundPath { get; set; } = @"D:\AniloxCaptures\bg";

        [DisplayName("本地預留磁碟空間")] public int LocalMinFreeGB { get; set; } = 100;
        [DisplayName("遠端路徑")]         public string RemotePath       { get; set; } = @"\\192.168.10.20\AniloxStorage";
        [DisplayName("遠端 Config 路徑")] public string RemoteConfigPath { get; set; } = @"\\192.168.10.20\AniloxConfig";

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = @"D:\AniloxCaptures";
            if (string.IsNullOrWhiteSpace(BackgroundPath)) BackgroundPath = @"D:\AniloxCaptures\bg";
            if (LocalMinFreeGB < 1) LocalMinFreeGB = 100;
            if (RemotePath == null) RemotePath = @"\\192.168.10.20\AniloxStorage";
            if (RemoteConfigPath == null) RemoteConfigPath = @"\\192.168.10.20\AniloxConfig";
        }

        public override string ToString() => "Storage";
    }
}
