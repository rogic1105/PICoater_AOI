using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>監控主畫面顯示方式：MIL 直繪（現狀）vs SmartCanvas（CPU/bytes→bitmap，跟回顧畫布同源）。</summary>
    public enum MainDisplayMode { MilDirect, SmartCanvas }

    /// <summary>監控主畫面動態 LOD：關 / GPU（TanukiCv）/ CPU（GrayResizeCpu）。放大巨圖看細節用，顯示成本 ~180ms→~1ms。</summary>
    public enum LiveLodMode { Off, GPU, CPU }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get; set; } = InspectionDefaults.DefaultStitch;
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = InspectionDefaults.EnableMuraEnhance;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = InspectionDefaults.EnableReviewEnhance;
        [DisplayName("主畫面顯示")] public MainDisplayMode MainDisplay  { get; set; } = InspectionDefaults.MainDisplay;
        [DisplayName("動態LOD")]   public LiveLodMode LiveLod          { get; set; } = InspectionDefaults.LiveLod;

        public void Validate() { }

        public override string ToString() => "";
    }
}
