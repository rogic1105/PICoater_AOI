using System.ComponentModel;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>去背演算法選項。</summary>
    public enum BackgroundAlgorithm
    {
        /// <summary>單張去背：每幀獨立計算 column mean 後減去。</summary>
        [Description("單張去背")] SingleFrameBgSub,
        /// <summary>標準去背：使用預存背景 row，raw - bg + 127 後再 Ridge。</summary>
        [Description("標準去背")] StandardBgSub
    }

    /// <summary>Ridge 偵測方向。</summary>
    public enum RidgeDirection
    {
        [Description("Vertical")]             Vertical,
        [Description("Horizontal")]           Horizontal,
        [Description("Vertical+Horizontal")]  Both
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("去背演算法")]    public BackgroundAlgorithm Algorithm { get; set; } = BackgroundAlgorithm.SingleFrameBgSub;
        [DisplayName("Ridge 方向")]    public RidgeDirection RidgeDir { get; set; } = RidgeDirection.Vertical;
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = 1.0f;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = 0.3f;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = 0.5f;
        [DisplayName("背景取樣秒數")]  public int BackgroundSampleSeconds { get; set; } = 3;
        [DisplayName("A輪速度 (m/min)")]  public double AniloxRollSpeedMPerMin { get; set; } = 10.0;

        /// <summary>存檔縮小倍率。原圖寬高各除以此值後存成 JPEG。唯一預設值來源：InspectionEngineConfig.DefaultSaveResizeScale。</summary>
        [Browsable(false)] public int SaveResizeScale { get; set; } = InspectionEngineConfig.DefaultSaveResizeScale;

        /// <summary>JPEG 存檔品質（1–100）。唯一預設值來源：InspectionEngineConfig.DefaultSaveJpgQuality。</summary>
        [Browsable(false)] public int SaveJpgQuality  { get; set; } = InspectionEngineConfig.DefaultSaveJpgQuality;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = 1.0f;
            if (ErrorValueMean <= 0) ErrorValueMean = 0.3f;
            if (ErrorValueMax <= 0) ErrorValueMax = 0.5f;
            if (BackgroundSampleSeconds < 1) BackgroundSampleSeconds = 3;
            if (AniloxRollSpeedMPerMin <= 0) AniloxRollSpeedMPerMin = 10.0;
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
