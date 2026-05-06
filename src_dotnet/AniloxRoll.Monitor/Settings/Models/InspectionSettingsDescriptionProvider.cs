using System;
using System.ComponentModel;
using System.Reflection;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 讓 PropertyGrid 的一般屬性在底部說明欄顯示該屬性的目前值。
    /// </summary>
    class InspectionSettingsDescriptionProvider : TypeDescriptionProvider
    {
        private readonly InspectionSettings _s;

        public InspectionSettingsDescriptionProvider(TypeDescriptionProvider parent, InspectionSettings s)
            : base(parent) { _s = s; }

        public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
            => new DynamicDescriptor(base.GetTypeDescriptor(objectType, instance), _s);

        // ── type descriptor ───────────────────────────────────────────────
        sealed class DynamicDescriptor : CustomTypeDescriptor
        {
            private readonly InspectionSettings _s;
            public DynamicDescriptor(ICustomTypeDescriptor parent, InspectionSettings s)
                : base(parent) { _s = s; }

            public override PropertyDescriptorCollection GetProperties()
                => Wrap(base.GetProperties());
            public override PropertyDescriptorCollection GetProperties(Attribute[] attributes)
                => Wrap(base.GetProperties(attributes));

            PropertyDescriptorCollection Wrap(PropertyDescriptorCollection props)
            {
                if (_s == null) return props;
                var arr = new PropertyDescriptor[props.Count];
                for (int i = 0; i < props.Count; i++)
                {
                    var p = props[i];
                    arr[i] = !p.Name.EndsWith("Header")
                        ? new ValueDescriptor(p, _s)
                        : p;
                }
                return new PropertyDescriptorCollection(arr);
            }
        }

        // ── PropertyDescriptor wrapper ─────────────────────────────────────
        sealed class ValueDescriptor : PropertyDescriptor
        {
            private readonly PropertyDescriptor _inner;
            private readonly InspectionSettings _s;

            public ValueDescriptor(PropertyDescriptor inner, InspectionSettings s)
                : base(inner) { _inner = inner; _s = s; }

            public override string Description
            {
                get
                {
                    try
                    {
                        var val = _inner.GetValue(_s);
                        if (val == null) return "";
                        if (val is bool b) return b ? "是" : "否";
                        if (val is Enum e) return EnumDesc(e);
                        return val.ToString();
                    }
                    catch { return ""; }
                }
            }

            static string EnumDesc(Enum value)
            {
                var field = value.GetType().GetField(value.ToString());
                if (field == null) return value.ToString();
                var attrs = (DescriptionAttribute[])field.GetCustomAttributes(typeof(DescriptionAttribute), false);
                return attrs.Length > 0 ? attrs[0].Description : value.ToString();
            }

            public override TypeConverter   Converter                  => _inner.Converter;
            public override Type            ComponentType              => _inner.ComponentType;
            public override bool            IsReadOnly                 => _inner.IsReadOnly;
            public override Type            PropertyType               => _inner.PropertyType;
            public override bool            CanResetValue(object c)    => _inner.CanResetValue(c);
            public override object          GetValue(object c)         => _inner.GetValue(c);
            public override void            ResetValue(object c)       => _inner.ResetValue(c);
            public override void            SetValue(object c, object v) => _inner.SetValue(c, v);
            public override bool            ShouldSerializeValue(object c) => _inner.ShouldSerializeValue(c);
        }
    }
}
