using System;
using System.IO;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Reads the persisted MCBF curve format with one bulk payload read.</summary>
    internal static class CurveBinFile
    {
        private const int MaxCurveLength = 200000;

        [ThreadStatic]
        private static byte[] _payloadBuffer;

        public static float[] Load(string path)
        {
            if (string.IsNullOrEmpty(path)) return null;

            if (CaptureArchiveStore.IsVirtualPath(path))
                return Load(CaptureArchiveStore.ReadAllBytes(path));
            if (!File.Exists(path)) return null;

            try
            {
                using (var stream = new FileStream(
                    path, FileMode.Open, FileAccess.Read, FileShare.Read,
                    64 * 1024, FileOptions.SequentialScan))
                using (var reader = new BinaryReader(stream))
                {
                    if (stream.Length < 16) return null;
                    byte[] magic = reader.ReadBytes(4);
                    if (magic.Length != 4 ||
                        magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F')
                        return null;

                    int version = reader.ReadInt32();
                    reader.ReadSingle();
                    if (version >= 2)
                    {
                        reader.ReadInt32();
                        reader.ReadSingle();
                    }

                    int length = reader.ReadInt32();
                    if (length <= 0 || length > MaxCurveLength) return null;
                    int byteCount = checked(length * sizeof(float));
                    if (stream.Length - stream.Position < byteCount) return null;

                    byte[] payload = GetPayloadBuffer(byteCount);
                    int offset = 0;
                    while (offset < byteCount)
                    {
                        int read = stream.Read(payload, offset, byteCount - offset);
                        if (read == 0) return null;
                        offset += read;
                    }

                    var values = new float[length];
                    Buffer.BlockCopy(payload, 0, values, 0, byteCount);
                    return values;
                }
            }
            catch
            {
                return null;
            }
        }

        internal static float[] Load(byte[] fileBytes)
        {
            if (fileBytes == null || fileBytes.Length < 16) return null;
            try
            {
                using (var stream = new MemoryStream(fileBytes, false))
                using (var reader = new BinaryReader(stream))
                    return ReadPayload(stream, reader);
            }
            catch
            {
                return null;
            }
        }

        private static float[] ReadPayload(Stream stream, BinaryReader reader)
        {
            if (stream.Length < 16) return null;
            byte[] magic = reader.ReadBytes(4);
            if (magic.Length != 4 ||
                magic[0] != 'M' || magic[1] != 'C' || magic[2] != 'B' || magic[3] != 'F')
                return null;

            int version = reader.ReadInt32();
            reader.ReadSingle();
            if (version >= 2)
            {
                reader.ReadInt32();
                reader.ReadSingle();
            }

            int length = reader.ReadInt32();
            if (length <= 0 || length > MaxCurveLength) return null;
            int byteCount = checked(length * sizeof(float));
            if (stream.Length - stream.Position < byteCount) return null;

            byte[] payload = GetPayloadBuffer(byteCount);
            int offset = 0;
            while (offset < byteCount)
            {
                int read = stream.Read(payload, offset, byteCount - offset);
                if (read == 0) return null;
                offset += read;
            }

            var values = new float[length];
            Buffer.BlockCopy(payload, 0, values, 0, byteCount);
            return values;
        }

        private static byte[] GetPayloadBuffer(int byteCount)
        {
            if (_payloadBuffer == null || _payloadBuffer.Length < byteCount)
                _payloadBuffer = new byte[byteCount];
            return _payloadBuffer;
        }
    }
}
