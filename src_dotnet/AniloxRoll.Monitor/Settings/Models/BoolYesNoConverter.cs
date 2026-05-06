using System;
using System.ComponentModel;
using System.Globalization;

namespace AniloxRoll.Monitor.Core.Data
{
    public class BoolYesNoConverter : BooleanConverter
    {
        private static readonly StandardValuesCollection _values =
            new StandardValuesCollection(new object[] { true, false });

        public override object ConvertTo(
            ITypeDescriptorContext context, CultureInfo culture, object value, Type destinationType)
        {
            if (destinationType == typeof(string) && value is bool b)
                return b ? "是" : "否";
            return base.ConvertTo(context, culture, value, destinationType);
        }

        public override object ConvertFrom(
            ITypeDescriptorContext context, CultureInfo culture, object value)
        {
            if (value is string s)
            {
                if (s == "是") return true;
                if (s == "否") return false;
            }
            return base.ConvertFrom(context, culture, value);
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
            => _values;
    }
}
