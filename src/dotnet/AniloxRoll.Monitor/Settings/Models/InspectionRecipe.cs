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
        [Description("欄")]   Vertical,
        [Description("列")]   Horizontal,
        [Description("全部")]   Both
    }

    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum CaptureStopCondition
    {
        [Description("IO")]   IoSignal,
        [Description("時間")] Time,
        [Description("高度")] Height
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("去背演算法")]    public BackgroundAlgorithm Algorithm { get; set; } = InspectionDefaults.Algorithm;
        [DisplayName("Ridge 方向")]    public RidgeDirection RidgeDir { get; set; } = InspectionDefaults.RidgeDir;
        [DisplayName("Hessian Max Factor V")] public float HessianMaxFactorV { get; set; } = InspectionDefaults.HessianMaxFactorV;
        [DisplayName("Hessian Max Factor H")] public float HessianMaxFactorH { get; set; } = InspectionDefaults.HessianMaxFactorH;

        /// <summary>細線濾除 = ridge_sigma（2-1 Gaussian blur 的 sigma）。值越大→模糊越多→濾掉越多細線/雜訊→越不敏感。
        /// 唯一預設值來源：InspectionEngineConfig.DefaultRidgeSigma。走每幀 json 送進 native pipeline。</summary>
        [DisplayName("細線濾除")] public float RidgeSigma { get; set; } = InspectionEngineConfig.DefaultRidgeSigma;
        [DisplayName("Error Value Mean V")] public float ErrorValueMeanV { get; set; } = InspectionDefaults.ErrorValueMeanV;
        [DisplayName("Error Value Max V")]  public float ErrorValueMaxV  { get; set; } = InspectionDefaults.ErrorValueMaxV;
        [DisplayName("Error Value Mean H")] public float ErrorValueMeanH { get; set; } = InspectionDefaults.ErrorValueMeanH;
        [DisplayName("Error Value Max H")]  public float ErrorValueMaxH  { get; set; } = InspectionDefaults.ErrorValueMaxH;
        [DisplayName("背景採樣(秒)")] public int BackgroundSampleSeconds { get; set; } = InspectionDefaults.BackgroundSampleSeconds;
        [DisplayName("停止條件")] public CaptureStopCondition CaptureStopCondition { get; set; } = InspectionDefaults.DefaultCaptureStopCondition;
        [DisplayName("總時間(秒)")] public int GrabLimitSeconds { get; set; } = InspectionDefaults.GrabLimitSeconds;
        [DisplayName("A輪速度 (m/min)")]  public double AniloxRollSpeedMPerMin { get; set; } = InspectionDefaults.AniloxRollSpeedMPerMin;

        /// <summary>存檔縮小倍率。原圖寬高各除以此值後存成 JPEG。唯一預設值來源：InspectionEngineConfig.DefaultSaveResizeScale。</summary>
        [Browsable(false)] public int SaveResizeScale { get; set; } = InspectionEngineConfig.DefaultSaveResizeScale;

        /// <summary>JPEG 存檔品質（1–100）。唯一預設值來源：InspectionEngineConfig.DefaultSaveJpgQuality。</summary>
        [Browsable(false)] public int SaveJpgQuality  { get; set; } = InspectionEngineConfig.DefaultSaveJpgQuality;

        public void Validate()
        {
            if (HessianMaxFactorV <= 0) HessianMaxFactorV = InspectionDefaults.HessianMaxFactorV;
            if (HessianMaxFactorH <= 0) HessianMaxFactorH = InspectionDefaults.HessianMaxFactorH;
            if (RidgeSigma < 1f) RidgeSigma = InspectionEngineConfig.DefaultRidgeSigma;   // 過小=近乎不模糊；過大 ksize 爆
            else if (RidgeSigma > 50f) RidgeSigma = 50f;
            if (ErrorValueMeanV <= 0) ErrorValueMeanV = InspectionDefaults.ErrorValueMeanV;
            if (ErrorValueMaxV  <= 0) ErrorValueMaxV  = InspectionDefaults.ErrorValueMaxV;
            if (ErrorValueMeanH <= 0) ErrorValueMeanH = InspectionDefaults.ErrorValueMeanH;
            if (ErrorValueMaxH  <= 0) ErrorValueMaxH  = InspectionDefaults.ErrorValueMaxH;
            if (BackgroundSampleSeconds < 1) BackgroundSampleSeconds = InspectionDefaults.BackgroundSampleSeconds;
            if (GrabLimitSeconds < 1) GrabLimitSeconds = InspectionDefaults.GrabLimitSeconds;
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
