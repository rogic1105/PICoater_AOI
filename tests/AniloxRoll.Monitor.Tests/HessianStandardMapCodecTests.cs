using System;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class HessianStandardMapCodecTests
    {
        [Test]
        public void EncodeDecode_RoundTripsHalfBitsLosslessly()
        {
            byte[] source = HalfBytes(0x0000, 0x3800, 0x3c00, 0x4000, 0x4200, 0x4400);

            byte[] encoded = HessianStandardMapCodec.Encode(source, 3, 2);
            HessianStandardMapData decoded = HessianStandardMapCodec.Decode(encoded);

            Assert.That(decoded.Width, Is.EqualTo(3));
            Assert.That(decoded.Height, Is.EqualTo(2));
            CollectionAssert.AreEqual(source, decoded.HalfBytes);
        }

        [Test]
        public void ToGray8_AppliesLinearDisplayGainAfterLoading()
        {
            var map = new HessianStandardMapData
            {
                Width = 4,
                Height = 1,
                HalfBytes = HalfBytes(0x0000, 0x3800, 0x3c00, 0x4000)
            };

            CollectionAssert.AreEqual(
                new byte[] { 0, 64, 128, 255 },
                HessianStandardMapCodec.ToGray8(map, 0.5f));
            CollectionAssert.AreEqual(
                new byte[] { 0, 128, 255, 255 },
                HessianStandardMapCodec.ToGray8(map, 1.0f));
        }

        [Test]
        public void Encode_SmoothMapCompressesBelowRawHalfSize()
        {
            const int width = 512;
            const int height = 128;
            var source = new byte[width * height * 2];
            for (int i = 0; i < width * height; i++)
            {
                ushort value = (ushort)(0x3000 + (i % 32));
                source[i * 2] = (byte)value;
                source[i * 2 + 1] = (byte)(value >> 8);
            }

            byte[] encoded = HessianStandardMapCodec.Encode(source, width, height);

            Assert.That(encoded.Length, Is.LessThan(source.Length / 10));
        }

        [TestCase(0x0000, 0f)]
        [TestCase(0x3800, 0.5f)]
        [TestCase(0x3c00, 1f)]
        [TestCase(0x4000, 2f)]
        public void HalfToSingle_DecodesKnownValues(int bits, float expected)
        {
            Assert.That(HessianStandardMapCodec.HalfToSingle((ushort)bits), Is.EqualTo(expected));
        }

        private static byte[] HalfBytes(params ushort[] values)
        {
            var bytes = new byte[values.Length * 2];
            for (int i = 0; i < values.Length; i++)
            {
                bytes[i * 2] = (byte)values[i];
                bytes[i * 2 + 1] = (byte)(values[i] >> 8);
            }
            return bytes;
        }
    }
}
