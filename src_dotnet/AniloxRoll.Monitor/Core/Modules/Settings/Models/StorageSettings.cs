using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class StorageSettings
    {
        [DisplayName("啟用即時截圖")] public bool EnableAutoCapture { get; set; } = false;
        [DisplayName("截圖根目錄")] public string CaptureRootPath { get; set; } = @"D:\AniloxCaptures";

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) CaptureRootPath = @"D:\AniloxCaptures";
        }

        public override string ToString() => "Storage";
    }
}
