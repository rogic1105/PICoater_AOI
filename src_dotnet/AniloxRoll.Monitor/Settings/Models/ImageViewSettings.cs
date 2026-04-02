using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class ImageViewSettings
    {
        [DisplayName("合圖方式")] public StitchMode StitchMode { get; set; } = StitchMode.Vertical;

        public void Validate() { }

        public override string ToString() => "";
    }
}
