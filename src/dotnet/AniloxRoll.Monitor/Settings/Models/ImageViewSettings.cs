using System.ComponentModel;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>監控主畫面顯示方式：MIL 直繪（現狀）/ SmartCanvas（CPU/bytes→bitmap，跟回顧畫布同源）/
    /// Waterfall（瀑布圖：全幅合圖每幀往下接、即時捲動；掉偵那欄補黑，沿用回顧經驗）。</summary>
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum MainDisplayMode
    {
        [Description("MIL 直繪")]   MilDirect,
        [Description("SmartCanvas")] SmartCanvas,
        [Description("瀑布圖")]      Waterfall
    }

    /// <summary>監控主畫面動態 LOD：關 / GPU（TanukiCv）/ CPU（GrayResizeCpu）。放大巨圖看細節用，顯示成本 ~180ms→~1ms。</summary>
    public enum LiveLodMode { Off, GPU, CPU }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get => StitchMode.Global; set { } } // 永遠 Global（選項退場；setter 吞掉舊 json 殘值）
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = InspectionDefaults.EnableMuraEnhance;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = InspectionDefaults.EnableReviewEnhance;
        [DisplayName("主畫面顯示")] public MainDisplayMode MainDisplay  { get; set; } = InspectionDefaults.MainDisplay;
        [DisplayName("動態LOD")]   public LiveLodMode LiveLod          { get; set; } = InspectionDefaults.LiveLod;
        [DisplayName("瀑布總高")]  public int       WaterfallTotalHeight { get; set; } = InspectionDefaults.WaterfallTotalHeight;
        [DisplayName("瀑布滿了")]  public WaterfallFullMode WaterfallFullMode { get; set; } = InspectionDefaults.WaterfallFullMode;

        public void Validate()
        {
            if (WaterfallTotalHeight < 1000) WaterfallTotalHeight = InspectionDefaults.WaterfallTotalHeight;
        }

        public override string ToString() => "";
    }
}
