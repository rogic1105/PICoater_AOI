using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum LogRecordingMode
    {
        [Description("日常運行（預設，檔案較小）")]
        Operational = 0,

        [Description("流程驗證（測試／驗收）")]
        FlowVerification = 1,

        [Description("完整診斷（除錯，檔案較大）")]
        FullDiagnostic = 2
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public sealed class LoggingSettings
    {
        public LogRecordingMode RecordingMode { get; set; } = InspectionDefaults.DefaultLogRecordingMode;
        public int RetentionHours { get; set; } = InspectionDefaults.LogRetentionHours;

        public void Validate()
        {
            if (!System.Enum.IsDefined(typeof(LogRecordingMode), RecordingMode))
                RecordingMode = InspectionDefaults.DefaultLogRecordingMode;
            if (RetentionHours < 1)
                RetentionHours = InspectionDefaults.LogRetentionHours;
        }

        public override string ToString() => "Logging";
    }
}
