using System;
using System.ComponentModel;
using System.Globalization;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>讓 PropertyGrid 以靠左字串渲染 int / float / double，
    /// 取代預設的靠右數值渲染。</summary>
    public class LeftAlignNumericConverter : TypeConverter
    {
        public override bool CanConvertTo(ITypeDescriptorContext context, Type destinationType)
            => destinationType == typeof(string) || base.CanConvertTo(context, destinationType);

        public override object ConvertTo(ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (destinationType == typeof(string) && value != null)
            {
                if (value is float  f) return f.ToString("G", CultureInfo.InvariantCulture);
                if (value is double d) return d.ToString("G", CultureInfo.InvariantCulture);
                return value.ToString();
            }
            return base.ConvertTo(context, culture, value, destinationType);
        }

        public override bool CanConvertFrom(ITypeDescriptorContext context, Type sourceType)
            => sourceType == typeof(string) || base.CanConvertFrom(context, sourceType);

        public override object ConvertFrom(ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string s)
            {
                s = s.Trim();
                var propType = context?.PropertyDescriptor?.PropertyType;
                if (propType == typeof(int)    && int.TryParse(s, out int i))                                                         return i;
                if (propType == typeof(float)  && float.TryParse (s, NumberStyles.Float, CultureInfo.InvariantCulture, out float  f)) return f;
                if (propType == typeof(double) && double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double d)) return d;
            }
            return base.ConvertFrom(context, culture, value);
        }
    }
}
