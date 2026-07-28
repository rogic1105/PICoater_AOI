using System.ComponentModel;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>監控主畫面顯示方式：即時（CPU/bytes→bitmap，跟回顧畫布同源）/ 瀑布（全幅合圖每幀往下接、即時捲動；掉偵那欄補黑）。</summary>
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum MainDisplayMode
    {
        [Description("即時")] ImageCanvas,
        [Description("瀑布")] Waterfall
    }

    /// <summary>監控主畫面動態 LOD：關 / GPU（TanukiCv）/ CPU（GrayResizeCpu）。放大巨圖看細節用，顯示成本 ~180ms→~1ms。</summary>
    public enum LiveLodMode { Off, GPU, CPU }

    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum VerticalDisplayDirection
    {
        [Description("由下而上")] BottomToTop,
        [Description("由上而下")] TopToBottom
    }

    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum EnhanceHeatmapMode
    {
        [Description("關閉")] Off,
        [Description("冷色")] Cold,
        [Description("暖色")] Warm,
        [Description("藍黃紅")] BlueYellowRed,
        [Description("綠階")] Green
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get => StitchMode.Global; set { } } // 永遠 Global（選項退場；setter 吞掉舊 json 殘值）
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = InspectionDefaults.EnableMuraEnhance;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = InspectionDefaults.EnableReviewEnhance;
        [DisplayName("強化熱力圖")] public EnhanceHeatmapMode EnhanceHeatmap { get; set; } = InspectionDefaults.EnhanceHeatmap;
        [DisplayName("主畫面顯示")] public MainDisplayMode MainDisplay  { get; set; } = InspectionDefaults.MainDisplay;
        [DisplayName("上下方向")] public VerticalDisplayDirection VerticalDirection { get; set; } = InspectionDefaults.VerticalDirection;
        [DisplayName("動態LOD")]   public LiveLodMode LiveLod          { get; set; } = InspectionDefaults.LiveLod;
        [DisplayName("瀑布總高")]  public int       WaterfallTotalHeight { get; set; } = InspectionDefaults.WaterfallTotalHeight;
        [DisplayName("瀑布滿了")]  public WaterfallFullMode WaterfallFullMode { get; set; } = InspectionDefaults.WaterfallFullMode;

        public void Validate()
        {
            if (WaterfallTotalHeight < 1000) WaterfallTotalHeight = InspectionDefaults.WaterfallTotalHeight;
        }

        public IntensityColorMap ResolveColorMap(bool enhanced)
        {
            if (!enhanced) return IntensityColorMap.Grayscale;
            switch (EnhanceHeatmap)
            {
                case EnhanceHeatmapMode.Cold: return IntensityColorMap.HeatmapCold;
                case EnhanceHeatmapMode.Warm: return IntensityColorMap.HeatmapWarm;
                case EnhanceHeatmapMode.BlueYellowRed: return IntensityColorMap.HeatmapBlueYellowRed;
                case EnhanceHeatmapMode.Green: return IntensityColorMap.HeatmapGreen;
                default: return IntensityColorMap.Grayscale;
            }
        }

        public static string ColorMapFlowName(IntensityColorMap colorMap)
        {
            switch (colorMap)
            {
                case IntensityColorMap.HeatmapCold: return "cold";
                case IntensityColorMap.HeatmapWarm: return "warm";
                case IntensityColorMap.HeatmapBlueYellowRed: return "blue-yellow-red";
                case IntensityColorMap.HeatmapGreen: return "green";
                default: return "gray";
            }
        }

        public override string ToString() => "";
    }
}
