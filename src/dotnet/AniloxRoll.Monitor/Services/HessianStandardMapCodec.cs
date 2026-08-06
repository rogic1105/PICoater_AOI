using System;
using System.IO;
using System.IO.Compression;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class HessianStandardMapData
    {
        public int Width { get; set; }
        public int Height { get; set; }
        public byte[] HalfBytes { get; set; }
    }

    /// <summary>
    /// Lossless container for pre-normalization Hessian responses. Values are IEEE binary16;
    /// byte shuffling groups exponent bytes before Deflate so smooth response maps compress well.
    /// </summary>
    internal static class HessianStandardMapCodec
    {
        private const int Magic = 0x314d5348; // HSM1
        private const byte Version = 1;
        private const byte SampleFormatHalf = 1;
        private const byte CompressionDeflateShuffled = 1;

        public static byte[] Encode(byte[] halfBytes, int width, int height)
        {
            int sampleCount = ValidateDimensions(width, height);
            if (halfBytes == null) throw new ArgumentNullException(nameof(halfBytes));
            if (halfBytes.Length != checked(sampleCount * 2))
                throw new ArgumentException("Half map byte count does not match dimensions.", nameof(halfBytes));

            byte[] shuffled = Shuffle(halfBytes, sampleCount);
            byte[] compressed;
            using (var payload = new MemoryStream())
            {
                using (var deflate = new DeflateStream(payload, CompressionLevel.Optimal, true))
                    deflate.Write(shuffled, 0, shuffled.Length);
                compressed = payload.ToArray();
            }

            using (var output = new MemoryStream(24 + compressed.Length))
            using (var writer = new BinaryWriter(output))
            {
                writer.Write(Magic);
                writer.Write(Version);
                writer.Write(SampleFormatHalf);
                writer.Write(CompressionDeflateShuffled);
                writer.Write((byte)0);
                writer.Write(width);
                writer.Write(height);
                writer.Write(halfBytes.Length);
                writer.Write(compressed.Length);
                writer.Write(compressed);
                return output.ToArray();
            }
        }

        public static HessianStandardMapData Decode(byte[] encoded)
        {
            if (encoded == null) throw new ArgumentNullException(nameof(encoded));
            using (var input = new MemoryStream(encoded, false))
            using (var reader = new BinaryReader(input))
            {
                if (input.Length < 24 || reader.ReadInt32() != Magic)
                    throw new InvalidDataException("Invalid Hessian standard-map header.");
                byte version = reader.ReadByte();
                byte sampleFormat = reader.ReadByte();
                byte compression = reader.ReadByte();
                reader.ReadByte();
                if (version != Version || sampleFormat != SampleFormatHalf ||
                    compression != CompressionDeflateShuffled)
                    throw new InvalidDataException("Unsupported Hessian standard-map format.");

                int width = reader.ReadInt32();
                int height = reader.ReadInt32();
                int rawLength = reader.ReadInt32();
                int compressedLength = reader.ReadInt32();
                int sampleCount = ValidateDimensions(width, height);
                if (rawLength != checked(sampleCount * 2) || compressedLength < 0 ||
                    compressedLength != input.Length - input.Position)
                    throw new InvalidDataException("Invalid Hessian standard-map lengths.");

                byte[] shuffled = new byte[rawLength];
                using (var compressed = new MemoryStream(reader.ReadBytes(compressedLength), false))
                using (var deflate = new DeflateStream(compressed, CompressionMode.Decompress))
                    ReadExactly(deflate, shuffled);

                return new HessianStandardMapData
                {
                    Width = width,
                    Height = height,
                    HalfBytes = Unshuffle(shuffled, sampleCount)
                };
            }
        }

        public static byte[] ToGray8(HessianStandardMapData map, float displayGain)
        {
            if (map == null || map.HalfBytes == null) throw new ArgumentNullException(nameof(map));
            if (!(displayGain > 0f)) throw new ArgumentOutOfRangeException(nameof(displayGain));
            int sampleCount = ValidateDimensions(map.Width, map.Height);
            if (map.HalfBytes.Length != sampleCount * 2)
                throw new InvalidDataException("Half map byte count does not match dimensions.");

            var gray = new byte[sampleCount];
            // The PropertyGrid normalization value is a display gain: increasing it must raise
            // both the Curve and the enhanced image linearly. The stored half map stays neutral.
            float scale = 255f * displayGain;
            for (int i = 0; i < sampleCount; i++)
            {
                ushort bits = (ushort)(map.HalfBytes[i * 2] | (map.HalfBytes[i * 2 + 1] << 8));
                float value = HalfToSingle(bits) * scale;
                if (float.IsNaN(value) || value <= 0f) gray[i] = 0;
                else if (value >= 255f) gray[i] = 255;
                else gray[i] = (byte)(value + 0.5f);
            }
            return gray;
        }

        internal static float HalfToSingle(ushort half)
        {
            uint sign = (uint)(half & 0x8000) << 16;
            uint exponent = (uint)(half >> 10) & 0x1f;
            uint mantissa = (uint)half & 0x03ff;
            uint bits;
            if (exponent == 0)
            {
                if (mantissa == 0) bits = sign;
                else
                {
                    int shift = 0;
                    while ((mantissa & 0x0400) == 0) { mantissa <<= 1; shift++; }
                    mantissa &= 0x03ff;
                    bits = sign | (uint)(113 - shift) << 23 | mantissa << 13;
                }
            }
            else if (exponent == 31)
                bits = sign | 0x7f800000u | mantissa << 13;
            else
                bits = sign | (exponent + 112) << 23 | mantissa << 13;
            return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
        }

        private static int ValidateDimensions(int width, int height)
        {
            if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException("dimensions");
            return checked(width * height);
        }

        private static byte[] Shuffle(byte[] source, int sampleCount)
        {
            var result = new byte[source.Length];
            for (int i = 0; i < sampleCount; i++)
            {
                result[i] = source[i * 2 + 1];
                result[sampleCount + i] = source[i * 2];
            }
            return result;
        }

        private static byte[] Unshuffle(byte[] source, int sampleCount)
        {
            var result = new byte[source.Length];
            for (int i = 0; i < sampleCount; i++)
            {
                result[i * 2 + 1] = source[i];
                result[i * 2] = source[sampleCount + i];
            }
            return result;
        }

        private static void ReadExactly(Stream stream, byte[] buffer)
        {
            int offset = 0;
            while (offset < buffer.Length)
            {
                int read = stream.Read(buffer, offset, buffer.Length - offset);
                if (read <= 0) throw new InvalidDataException("Truncated Hessian standard-map payload.");
                offset += read;
            }
            if (stream.ReadByte() >= 0)
                throw new InvalidDataException("Hessian standard-map payload is longer than declared.");
        }
    }
}
