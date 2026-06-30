using System.ComponentModel;

namespace TanukiCv.Controls
{
    [TypeConverter(typeof(EnumDescriptionConverter))]
    public enum WaterfallFullMode
    {
        [Description("重來")] Restart,
        [Description("循環")] Ring
    }
}
