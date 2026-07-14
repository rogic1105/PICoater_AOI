using System.ComponentModel;
using System.Linq;
using AniloxRoll.Monitor.Core.Data;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class IoEndpointTypeConverterTests
    {
        [Test]
        public void IoIp_OffersHardwareAndSimulatorAddresses_AndAllowsCustomInput()
        {
            TypeConverter converter = GetConverter(nameof(InspectionSettings.IoIp));

            Assert.That(converter.GetStandardValuesSupported(), Is.True);
            Assert.That(converter.GetStandardValuesExclusive(), Is.False);
            Assert.That(
                converter.GetStandardValues().Cast<string>(),
                Is.EquivalentTo(new[] { "192.168.255.1", "127.0.0.1" }));
            Assert.That(converter.ConvertFromInvariantString("10.20.30.40"), Is.EqualTo("10.20.30.40"));
        }

        [Test]
        public void IoPort_OffersHardwareAndSimulatorPorts_AndAllowsCustomInput()
        {
            TypeConverter converter = GetConverter(nameof(InspectionSettings.IoPort));

            Assert.That(converter.GetStandardValuesSupported(), Is.True);
            Assert.That(converter.GetStandardValuesExclusive(), Is.False);
            Assert.That(
                converter.GetStandardValues().Cast<int>(),
                Is.EquivalentTo(new[] { 502, 1502 }));
            Assert.That(converter.ConvertFromInvariantString("2502"), Is.EqualTo(2502));
        }

        private static TypeConverter GetConverter(string propertyName)
        {
            PropertyDescriptor property = TypeDescriptor
                .GetProperties(typeof(InspectionSettings))[propertyName];
            Assert.That(property, Is.Not.Null);
            return property.Converter;
        }
    }
}
