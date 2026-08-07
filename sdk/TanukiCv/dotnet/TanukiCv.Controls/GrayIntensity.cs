using System;

namespace TanukiCv.Controls
{
    /// <summary>Applies a display-only gain to an 8-bit grayscale source.</summary>
    public static class GrayIntensity
    {
        public static bool IsNoOp(float scale)
            => scale <= 0f || Math.Abs(scale - 1f) < 0.0001f;

        public static byte Scale(byte value, float scale)
        {
            if (IsNoOp(scale)) return value;
            int scaled = (int)Math.Round(value * scale);
            return (byte)Math.Max(0, Math.Min(255, scaled));
        }

        public static byte[] ScaleCopy(byte[] source, float scale)
        {
            if (source == null) return null;
            var copy = new byte[source.Length];
            if (IsNoOp(scale))
            {
                Array.Copy(source, copy, source.Length);
                return copy;
            }

            for (int i = 0; i < source.Length; i++)
                copy[i] = Scale(source[i], scale);
            return copy;
        }

        public static void ScaleInPlace(byte[] data, float scale)
        {
            if (data == null || IsNoOp(scale)) return;
            for (int i = 0; i < data.Length; i++)
                data[i] = Scale(data[i], scale);
        }
    }
}
