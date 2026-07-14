using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>提供常用 IO 位址下拉，同時保留其他 IP 的自由輸入。</summary>
    public sealed class IoIpTypeConverter : StringConverter
    {
        private static readonly StandardValuesCollection Values =
            new StandardValuesCollection(new object[]
            {
                "192.168.255.1",
                "127.0.0.1"
            });

        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context)
        {
            return false;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            return Values;
        }
    }

    /// <summary>提供實機與模擬器 Port 下拉，同時保留其他 Port 的自由輸入。</summary>
    public sealed class IoPortTypeConverter : Int32Converter
    {
        private static readonly StandardValuesCollection Values =
            new StandardValuesCollection(new object[] { 502, 1502 });

        public override bool GetStandardValuesSupported(ITypeDescriptorContext context)
        {
            return true;
        }

        public override bool GetStandardValuesExclusive(ITypeDescriptorContext context)
        {
            return false;
        }

        public override StandardValuesCollection GetStandardValues(ITypeDescriptorContext context)
        {
            return Values;
        }
    }
}
