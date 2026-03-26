using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    public enum ChartScaleMode
    {
        [Description("自動")] Auto,
        [Description("產量")] Fixed
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ChartSettings
    {
        [DisplayName("數量範圍")] public ChartScaleMode ScaleMode { get; set; } = ChartScaleMode.Fixed;
        [DisplayName("月產量")]   public int YearlyYMax  { get; set; } = 60000;
        [DisplayName("日產量")]   public int MonthlyYMax { get; set; } = 2000;
        [DisplayName("時產量")]   public int DailyYMax   { get; set; } = 300;

        public void Validate()
        {
            if (YearlyYMax  <= 0) YearlyYMax  = 60000;
            if (MonthlyYMax <= 0) MonthlyYMax = 2000;
            if (DailyYMax   <= 0) DailyYMax   = 300;
        }

        public override string ToString() => "";
    }
}
