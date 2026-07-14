using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>Raw per-camera CurveMean/CurveMax aggregates persisted for one grab.</summary>
    internal sealed class SingleGrabCurveSummary
    {
        public SingleGrabCurveSummary(float[][] mean, float[][] max, int captureCount)
        {
            Mean = mean ?? new float[0][];
            Max = max ?? new float[0][];
            CaptureCount = captureCount;
        }

        public float[][] Mean { get; }
        public float[][] Max { get; }
        public int CaptureCount { get; }
    }

    /// <summary>
    /// Versioned materialized view for report curves. A summary is valid only for the
    /// same grab time range and camera count; invalid or corrupt files are rebuilt from bins.
    /// </summary>
    internal static class SingleGrabCurveSummaryStore
    {
        private const int FormatVersion = 1;
        private const int AggregationVersion = 1;
        private const int MaxCameraCount = 64;
        private const int MaxCurveLength = 200000;
        private const int MaxGrabIdBytes = 64;
        private const int WriteIdleMs = 750;
        private const long MaxPendingBytes = 96L * 1024 * 1024;
        private static readonly byte[] Magic = { (byte)'M', (byte)'C', (byte)'S', (byte)'F' };
        private static readonly object WriteSync = new object();
        private static readonly Dictionary<string, PendingWrite> Pending =
            new Dictionary<string, PendingWrite>(StringComparer.OrdinalIgnoreCase);
        private static readonly LinkedList<string> PendingOrder = new LinkedList<string>();
        private static readonly HashSet<string> Completed =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        private static readonly Timer WriteTimer =
            new Timer(DrainPending, null, Timeout.Infinite, Timeout.Infinite);

        private static DateTime _lastReadActivityUtc = DateTime.MinValue;
        private static long _pendingBytes;
        private static bool _writerRunning;

        [ThreadStatic]
        private static byte[] _payloadBuffer;

        private sealed class PendingWrite
        {
            public string IdentityKey;
            public string Root;
            public GrabIdInfo Info;
            public int CameraCount;
            public SingleGrabCurveSummary Summary;
            public long EstimatedBytes;
            public LinkedListNode<string> Node;
        }

        /// <summary>Signals interactive curve IO so summary writes yield to foreground reads.</summary>
        public static void NotifyReadActivity()
        {
            lock (WriteSync)
            {
                _lastReadActivityUtc = DateTime.UtcNow;
                if (Pending.Count > 0 && !_writerRunning)
                    WriteTimer.Change(WriteIdleMs, Timeout.Infinite);
            }
        }

        /// <summary>
        /// Queues a complete summary for one idle background writer. Pending data is bounded;
        /// an oversized entry is rejected instead of increasing application memory without limit.
        /// </summary>
        public static bool QueueSave(
            string root, GrabIdInfo info, int cameraCount, SingleGrabCurveSummary summary)
        {
            if (!IsIdentityValid(root, info, cameraCount) || summary == null) return false;
            if (summary.Mean.Length != cameraCount || summary.Max.Length != cameraCount) return false;

            long estimatedBytes = EstimateBytes(summary.Mean) + EstimateBytes(summary.Max);
            if (estimatedBytes > MaxPendingBytes) return false;
            string identityKey = BuildIdentityKey(root, info, cameraCount);

            lock (WriteSync)
            {
                if (Completed.Contains(identityKey)) return true;

                if (Pending.TryGetValue(identityKey, out PendingWrite existing))
                {
                    _pendingBytes -= existing.EstimatedBytes;
                    existing.Summary = summary;
                    existing.EstimatedBytes = estimatedBytes;
                    _pendingBytes += estimatedBytes;
                    PendingOrder.Remove(existing.Node);
                    PendingOrder.AddLast(existing.Node);
                }
                else
                {
                    var node = new LinkedListNode<string>(identityKey);
                    var copy = new GrabIdInfo
                    {
                        GrabId = info.GrabId,
                        Earliest = info.Earliest,
                        Latest = info.Latest
                    };
                    Pending[identityKey] = new PendingWrite
                    {
                        IdentityKey = identityKey,
                        Root = root,
                        Info = copy,
                        CameraCount = cameraCount,
                        Summary = summary,
                        EstimatedBytes = estimatedBytes,
                        Node = node
                    };
                    PendingOrder.AddLast(node);
                    _pendingBytes += estimatedBytes;
                }

                while (_pendingBytes > MaxPendingBytes && PendingOrder.First != null)
                    RemovePending(PendingOrder.First.Value);

                WriteTimer.Change(WriteIdleMs, Timeout.Infinite);
                return Pending.ContainsKey(identityKey) || Completed.Contains(identityKey);
            }
        }

        public static bool TryLoad(
            string root, GrabIdInfo info, int cameraCount, out SingleGrabCurveSummary summary)
        {
            summary = null;
            if (!IsIdentityValid(root, info, cameraCount)) return false;

            string path = CaptureStoragePaths.GrabCurveSummary(root, info.Earliest, info.GrabId);
            if (!File.Exists(path)) return false;

            try
            {
                using (var stream = new FileStream(
                    path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete, 64 * 1024, FileOptions.SequentialScan))
                using (var reader = new BinaryReader(stream, Encoding.UTF8))
                {
                    if (!ReadAndMatchMagic(reader)) return false;
                    if (reader.ReadInt32() != FormatVersion) return false;
                    if (reader.ReadInt32() != AggregationVersion) return false;
                    if (!string.Equals(ReadString(reader), info.GrabId, StringComparison.Ordinal)) return false;
                    if (reader.ReadInt64() != info.Earliest.Ticks) return false;
                    if (reader.ReadInt64() != info.Latest.Ticks) return false;
                    if (reader.ReadInt32() != cameraCount) return false;

                    int captureCount = reader.ReadInt32();
                    if (captureCount < 0) return false;

                    var mean = new float[cameraCount][];
                    var max = new float[cameraCount][];
                    for (int i = 0; i < cameraCount; i++)
                    {
                        mean[i] = ReadCurve(reader, stream);
                        max[i] = ReadCurve(reader, stream);
                    }

                    summary = new SingleGrabCurveSummary(mean, max, captureCount);
                    return true;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[SingleGrabCurveSummaryStore.TryLoad] {path}: {ex.GetType().Name}: {ex.Message}");
                return false;
            }
        }

        public static bool TrySave(
            string root, GrabIdInfo info, int cameraCount, SingleGrabCurveSummary summary)
        {
            if (!IsIdentityValid(root, info, cameraCount) || summary == null) return false;
            if (summary.Mean.Length != cameraCount || summary.Max.Length != cameraCount) return false;

            string path = CaptureStoragePaths.GrabCurveSummary(root, info.Earliest, info.GrabId);
            string directory = Path.GetDirectoryName(path);
            string tempPath = path + "." + Guid.NewGuid().ToString("N") + ".tmp";

            try
            {
                Directory.CreateDirectory(directory);
                using (var stream = new FileStream(
                    tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None,
                    64 * 1024, FileOptions.SequentialScan))
                using (var writer = new BinaryWriter(stream, Encoding.UTF8))
                {
                    writer.Write(Magic);
                    writer.Write(FormatVersion);
                    writer.Write(AggregationVersion);
                    WriteString(writer, info.GrabId);
                    writer.Write(info.Earliest.Ticks);
                    writer.Write(info.Latest.Ticks);
                    writer.Write(cameraCount);
                    writer.Write(summary.CaptureCount);
                    for (int i = 0; i < cameraCount; i++)
                    {
                        WriteCurve(writer, summary.Mean[i]);
                        WriteCurve(writer, summary.Max[i]);
                    }
                    writer.Flush();
                    stream.Flush(true);
                }

                lock (WriteSync)
                {
                    if (File.Exists(path))
                        File.Replace(tempPath, path, null, true);
                    else
                        File.Move(tempPath, path);
                }
                return true;
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[SingleGrabCurveSummaryStore.TrySave] {path}: {ex.GetType().Name}: {ex.Message}");
                return false;
            }
            finally
            {
                try { if (File.Exists(tempPath)) File.Delete(tempPath); } catch { }
            }
        }

        private static bool IsIdentityValid(string root, GrabIdInfo info, int cameraCount) =>
            !string.IsNullOrWhiteSpace(root) && info != null &&
            !string.IsNullOrWhiteSpace(info.GrabId) &&
            info.Earliest != default(DateTime) && info.Latest != default(DateTime) &&
            cameraCount > 0 && cameraCount <= MaxCameraCount;

        private static void DrainPending(object state)
        {
            PendingWrite pending;
            lock (WriteSync)
            {
                if (_writerRunning || PendingOrder.First == null) return;

                double idleMs = (DateTime.UtcNow - _lastReadActivityUtc).TotalMilliseconds;
                if (idleMs < WriteIdleMs)
                {
                    WriteTimer.Change(
                        Math.Max(1, WriteIdleMs - (int)idleMs), Timeout.Infinite);
                    return;
                }

                string key = PendingOrder.First.Value;
                pending = Pending[key];
                RemovePending(key);
                _writerRunning = true;
            }

            var sw = Stopwatch.StartNew();
            bool saved = TrySave(
                pending.Root, pending.Info, pending.CameraCount, pending.Summary);
            FlowTrace.Log($"DT curve summary {pending.Info.GrabId} " +
                $"write={(saved ? "ok" : "failed")} captures={pending.Summary.CaptureCount} " +
                $"merged={pending.Summary.CaptureCount} ms={sw.ElapsedMilliseconds}");

            lock (WriteSync)
            {
                if (saved) Completed.Add(pending.IdentityKey);
                _writerRunning = false;
                if (PendingOrder.First != null)
                {
                    double idleMs = (DateTime.UtcNow - _lastReadActivityUtc).TotalMilliseconds;
                    int dueMs = idleMs >= WriteIdleMs
                        ? 1
                        : Math.Max(1, WriteIdleMs - (int)idleMs);
                    WriteTimer.Change(dueMs, Timeout.Infinite);
                }
            }
        }

        private static void RemovePending(string identityKey)
        {
            if (!Pending.TryGetValue(identityKey, out PendingWrite pending)) return;
            _pendingBytes -= pending.EstimatedBytes;
            PendingOrder.Remove(pending.Node);
            Pending.Remove(identityKey);
        }

        private static string BuildIdentityKey(string root, GrabIdInfo info, int cameraCount) =>
            Path.GetFullPath(root).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + "|" +
            info.GrabId + "|" + info.Earliest.Ticks + "|" + info.Latest.Ticks + "|" + cameraCount;

        private static long EstimateBytes(float[][] curves)
        {
            long bytes = 0;
            for (int i = 0; i < curves.Length; i++)
                if (curves[i] != null) bytes += (long)curves[i].Length * sizeof(float);
            return bytes;
        }

        private static bool ReadAndMatchMagic(BinaryReader reader)
        {
            byte[] value = reader.ReadBytes(Magic.Length);
            if (value.Length != Magic.Length) return false;
            for (int i = 0; i < Magic.Length; i++)
                if (value[i] != Magic[i]) return false;
            return true;
        }

        private static void WriteString(BinaryWriter writer, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value);
            if (bytes.Length > MaxGrabIdBytes) throw new InvalidDataException("Grab ID is too long.");
            writer.Write(bytes.Length);
            writer.Write(bytes);
        }

        private static string ReadString(BinaryReader reader)
        {
            int byteCount = reader.ReadInt32();
            if (byteCount <= 0 || byteCount > MaxGrabIdBytes)
                throw new InvalidDataException("Invalid grab ID length.");
            byte[] bytes = reader.ReadBytes(byteCount);
            if (bytes.Length != byteCount) throw new EndOfStreamException();
            return Encoding.UTF8.GetString(bytes);
        }

        private static void WriteCurve(BinaryWriter writer, float[] curve)
        {
            if (curve == null)
            {
                writer.Write(-1);
                return;
            }
            if (curve.Length > MaxCurveLength) throw new InvalidDataException("Curve is too long.");
            int byteCount = checked(curve.Length * sizeof(float));
            byte[] payload = GetPayloadBuffer(byteCount);
            Buffer.BlockCopy(curve, 0, payload, 0, byteCount);
            writer.Write(curve.Length);
            writer.Write(payload, 0, byteCount);
        }

        private static float[] ReadCurve(BinaryReader reader, Stream stream)
        {
            int length = reader.ReadInt32();
            if (length == -1) return null;
            if (length < 0 || length > MaxCurveLength)
                throw new InvalidDataException("Invalid curve length.");

            int byteCount = checked(length * sizeof(float));
            if (stream.Length - stream.Position < byteCount) throw new EndOfStreamException();
            byte[] payload = GetPayloadBuffer(byteCount);
            int offset = 0;
            while (offset < byteCount)
            {
                int read = stream.Read(payload, offset, byteCount - offset);
                if (read == 0) throw new EndOfStreamException();
                offset += read;
            }
            var curve = new float[length];
            Buffer.BlockCopy(payload, 0, curve, 0, byteCount);
            return curve;
        }

        private static byte[] GetPayloadBuffer(int byteCount)
        {
            if (_payloadBuffer == null || _payloadBuffer.Length < byteCount)
                _payloadBuffer = new byte[byteCount];
            return _payloadBuffer;
        }
    }
}
