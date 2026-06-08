using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>監控主畫面顯示方式：MIL 直繪（現狀）vs SmartCanvas（CPU/bytes→bitmap，跟回顧畫布同源）。</summary>
    public enum MainDisplayMode { MilDirect, SmartCanvas }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get; set; } = InspectionDefaults.DefaultStitch;
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = InspectionDefaults.EnableMuraEnhance;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = InspectionDefaults.EnableReviewEnhance;
        [DisplayName("主畫面顯示")] public MainDisplayMode MainDisplay  { get; set; } = MainDisplayMode.MilDirect;

        public void Validate() { }

        public override string ToString() => "";
    }
}
