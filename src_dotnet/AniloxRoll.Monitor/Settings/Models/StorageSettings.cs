using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("存檔")]     public bool EnableAutoCapture { get; set; } = false;
        [DisplayName("存檔目錄")] public string CaptureRootPath { get; set; } = @"D:\AniloxCaptures";
        [DisplayName("存原圖")]   public bool SaveOriginalBmp { get; set; } = false;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = @"D:\AniloxCaptures";
        }

        public override string ToString() => "Storage";
    }
}
