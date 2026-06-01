using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>擴展 ExpandableObjectConverter，對所有數值子屬性（int/float/double）
    /// 套用 LeftAlignNumericConverter，使 PropertyGrid 子層也靠左顯示。</summary>
    public class ExpandableLeftAlignConverter : ExpandableObjectConverter
    {
        public override PropertyDescriptorCollection GetProperties(
            ITypeDescriptorContext context, object value, Attribute[] attributes)
        {
            var orig = base.GetProperties(context, value, attributes);
            var result = new PropertyDescriptor[orig.Count];
            for (int i = 0; i < orig.Count; i++)
                result[i] = IsNumeric(orig[i].PropertyType)
                    ? new NumericLeftAlignDescriptor(orig[i])
                    : orig[i];
            return new PropertyDescriptorCollection(result);
        }

        static bool IsNumeric(Type t) =>
            t == typeof(int) || t == typeof(float) || t == typeof(double);

        // ── PropertyDescriptor wrapper ──────────────────────────────────────
        private sealed class NumericLeftAlignDescriptor : PropertyDescriptor
        {
            private readonly PropertyDescriptor _inner;
            private static readonly LeftAlignNumericConverter _conv = new LeftAlignNumericConverter();

            public NumericLeftAlignDescriptor(PropertyDescriptor inner)
                : base(inner) { _inner = inner; }

            public override TypeConverter Converter    => _conv;
            public override Type   ComponentType       => _inner.ComponentType;
            public override bool   IsReadOnly          => _inner.IsReadOnly;
            public override Type   PropertyType        => _inner.PropertyType;
            public override bool   CanResetValue(object c)             => _inner.CanResetValue(c);
            public override object GetValue(object c)                  => _inner.GetValue(c);
            public override void   ResetValue(object c)                => _inner.ResetValue(c);
            public override void   SetValue(object c, object v)        => _inner.SetValue(c, v);
            public override bool   ShouldSerializeValue(object c)      => _inner.ShouldSerializeValue(c);
        }
    }
}
