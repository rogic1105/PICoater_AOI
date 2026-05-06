using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    public enum ChartScaleMode
    {
        [Description("自動")] Auto,
        [Description("產量")] Fixed
    }

    /// <summary>
    /// 合圖方式：垂直（每台相機多張垂直拼接，gallery 切換）、全域（水平合併為全域圖，GrabId 模式先垂直拼接再合併）。
    /// </summary>
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum StitchMode
    {
        [Description("垂直")] Vertical,
        [Description("全域")] Global
    }

    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class ChartSettings
    {
        [DisplayName("數量範圍")] public ChartScaleMode ScaleMode  { get; set; } = InspectionDefaults.ScaleMode;
        [DisplayName("月產量")][TypeConverter(typeof(LeftAlignNumericConverter))] public int YearlyYMax  { get; set; } = InspectionDefaults.YearlyYMax;
        [DisplayName("日產量")][TypeConverter(typeof(LeftAlignNumericConverter))] public int MonthlyYMax { get; set; } = InspectionDefaults.MonthlyYMax;
        [DisplayName("時產量")][TypeConverter(typeof(LeftAlignNumericConverter))] public int DailyYMax   { get; set; } = InspectionDefaults.DailyYMax;
        public void Validate()
        {
            if (YearlyYMax  <= 0) YearlyYMax  = InspectionDefaults.YearlyYMax;
            if (MonthlyYMax <= 0) MonthlyYMax = InspectionDefaults.MonthlyYMax;
            if (DailyYMax   <= 0) DailyYMax   = InspectionDefaults.DailyYMax;
        }

        public override string ToString() => "";
    }
}
