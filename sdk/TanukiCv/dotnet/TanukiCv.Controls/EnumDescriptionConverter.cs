using System;
using System.ComponentModel;

namespace TanukiCv.Controls
{
    public sealed class EnumDescriptionConverter : EnumConverter
    {
        public EnumDescriptionConverter(Type type) : base(type)
        {
        }

        public override object ConvertTo(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value, Type destinationType)
        {
            if (destinationType == typeof(string) && value != null)
            {
                var field = EnumType.GetField(value.ToString());
                var attrs = field?.GetCustomAttributes(typeof(DescriptionAttribute), false);
                if (attrs != null && attrs.Length > 0) return ((DescriptionAttribute)attrs[0]).Description;
            }

            return base.ConvertTo(context, culture, value, destinationType);
        }

        public override object ConvertFrom(ITypeDescriptorContext context, System.Globalization.CultureInfo culture, object value)
        {
            if (value is string text)
            {
                foreach (var field in EnumType.GetFields())
                {
                    var attrs = field.GetCustomAttributes(typeof(DescriptionAttribute), false);
                    if (attrs.Length > 0 && ((DescriptionAttribute)attrs[0]).Description == text)
                        return Enum.Parse(EnumType, field.Name);
                }
            }

            return base.ConvertFrom(context, culture, value);
        }
    }
}
