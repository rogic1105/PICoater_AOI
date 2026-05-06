using System;
using System.ComponentModel;
using System.Globalization;
using System.Linq;
using System.Reflection;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>讓 PropertyGrid 下拉顯示 [Description] 文字而非 enum 名稱。</summary>
    public class EnumDescriptionConverter : EnumConverter
    {
        public EnumDescriptionConverter(Type type) : base(type) { }

        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (destinationType == typeof(string) && value != null)
            {
                var fi = value.GetType().GetField(value.ToString());
                var attr = fi?.GetCustomAttribute<DescriptionAttribute>();
                if (attr != null) return attr.Description;
            }
            return base.ConvertTo(context, culture, value, destinationType);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string s)
            {
                foreach (var fi in EnumType.GetFields(BindingFlags.Public | BindingFlags.Static))
                {
                    var attr = fi.GetCustomAttribute<DescriptionAttribute>();
                    if (attr != null && attr.Description == s)
                        return fi.GetValue(null);
                }
            }
            return base.ConvertFrom(context, culture, value);
        }
    }

    /// <summary>去背演算法選項。</summary>
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum BackgroundAlgorithm
    {
        /// <summary>單張去背：每幀獨立計算 column mean 後減去。</summary>
        [Description("單張去背")] SingleFrameBgSub,
        /// <summary>標準去背：使用預存背景 row，raw - bg + 127 後再 Ridge。</summary>
        [Description("標準去背")] StandardBgSub
    }

    /// <summary>檢測方向：決定哪個方向的 Mura 超標才觸發 DO_MURA_DETECTED。</summary>
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum RidgeDirection
    {
        [Description("垂直")]   Vertical,
        [Description("水平")]   Horizontal,
        [Description("全部")]   Both
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("去背演算法")]    public BackgroundAlgorithm Algorithm { get; set; } = InspectionDefaults.Algorithm;
        [DisplayName("Ridge 方向")]    public RidgeDirection RidgeDir { get; set; } = InspectionDefaults.RidgeDir;
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = InspectionDefaults.HessianMaxFactor;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = InspectionDefaults.ErrorValueMean;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = InspectionDefaults.ErrorValueMax;
        [DisplayName("背景取樣秒數")]  public int BackgroundSampleSeconds { get; set; } = InspectionDefaults.BackgroundSampleSeconds;
        [DisplayName("A輪速度 (m/min)")]  public double AniloxRollSpeedMPerMin { get; set; } = InspectionDefaults.AniloxRollSpeedMPerMin;

        /// <summary>存檔縮小倍率。原圖寬高各除以此值後存成 JPEG。唯一預設值來源：InspectionEngineConfig.DefaultSaveResizeScale。</summary>
        [Browsable(false)] public int SaveResizeScale { get; set; } = InspectionEngineConfig.DefaultSaveResizeScale;

        /// <summary>JPEG 存檔品質（1–100）。唯一預設值來源：InspectionEngineConfig.DefaultSaveJpgQuality。</summary>
        [Browsable(false)] public int SaveJpgQuality  { get; set; } = InspectionEngineConfig.DefaultSaveJpgQuality;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = InspectionDefaults.HessianMaxFactor;
            if (ErrorValueMean <= 0) ErrorValueMean = InspectionDefaults.ErrorValueMean;
            if (ErrorValueMax <= 0) ErrorValueMax = InspectionDefaults.ErrorValueMax;
            if (BackgroundSampleSeconds < 1) BackgroundSampleSeconds = InspectionDefaults.BackgroundSampleSeconds;
            if (AniloxRollSpeedMPerMin <= 0) AniloxRollSpeedMPerMin = InspectionDefaults.AniloxRollSpeedMPerMin;
            if (SaveResizeScale <= 0) SaveResizeScale = InspectionEngineConfig.DefaultSaveResizeScale;
            if (SaveJpgQuality  < 1 || SaveJpgQuality > 100) SaveJpgQuality = InspectionEngineConfig.DefaultSaveJpgQuality;
        }

        /// <summary>將 RidgeDirection enum 轉為 native API 字串。</summary>
        public static string RidgeDirectionToNative(RidgeDirection dir)
        {
            switch (dir)
            {
                case RidgeDirection.Horizontal: return "horizontal";
                case RidgeDirection.Both:       return "vertical+horizontal";
                default:                        return "vertical";
            }
        }

        public override string ToString() => "Inspection Recipe";
    }
}
