using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")]  public StitchMode StitchMode       { get; set; } = StitchMode.Vertical;
        [DisplayName("監控強化")]  public bool       EnableMuraEnhance   { get; set; } = false;
        [DisplayName("回顧強化")]  public bool       EnableReviewEnhance { get; set; } = false;

        public void Validate() { }

        public override string ToString() => "";
    }
}
