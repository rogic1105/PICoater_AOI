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

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = @"D:\AniloxCaptures";
            if (string.IsNullOrWhiteSpace(BackgroundPath)) BackgroundPath = @"D:\AniloxCaptures\bg";
        }

        public override string ToString() => "Storage";
    }
}
