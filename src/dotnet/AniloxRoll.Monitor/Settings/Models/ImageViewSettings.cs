using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get; set; } = InspectionDefaults.DefaultStitch;
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = InspectionDefaults.EnableMuraEnhance;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = InspectionDefaults.EnableReviewEnhance;

        public void Validate() { }

        public override string ToString() => "";
    }
}
