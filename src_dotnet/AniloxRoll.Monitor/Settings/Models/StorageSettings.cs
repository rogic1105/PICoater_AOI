using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("存檔")]       public bool EnableAutoCapture { get; set; } = false;
        [DisplayName("存原圖")]     public bool SaveOriginalBmp { get; set; } = false;
        [DisplayName("存圖目錄")]   public string CaptureRootPath { get; set; } = @"D:\AniloxCaptures";
        [DisplayName("存背景目錄")] public string BackgroundPath { get; set; } = @"D:\AniloxCaptures\bg";

        [DisplayName("本地上限(GB)")]   public int LocalMaxGB { get; set; } = 500;
        [DisplayName("異常保護天數")]    public int FailProtectDays { get; set; } = 30;
        [DisplayName("遠端路徑")]       public string RemotePath { get; set; } = "";

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = @"D:\AniloxCaptures";
            if (string.IsNullOrWhiteSpace(BackgroundPath)) BackgroundPath = @"D:\AniloxCaptures\bg";
            if (LocalMaxGB < 1) LocalMaxGB = 500;
            if (FailProtectDays < 0) FailProtectDays = 30;
            if (RemotePath == null) RemotePath = "";
        }

        public override string ToString() => "Storage";
    }
}
