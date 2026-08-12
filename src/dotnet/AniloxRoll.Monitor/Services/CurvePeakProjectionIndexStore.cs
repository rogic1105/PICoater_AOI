using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class CurvePeakProjectionEntry
    {
        public string GrabId { get; set; }
        public long EarliestTicks { get; set; }
        public long LatestTicks { get; set; }
        public string ConfigKey { get; set; }
        public ColumnCurvePeakRecord[] Columns { get; set; }
        public RowCurvePeakRecord Row { get; set; }

        public bool Matches(GrabIdInfo info, string configKey, int cameraCount)
        {
            return info != null &&
                string.Equals(GrabId, info.GrabId, StringComparison.Ordinal) &&
                EarliestTicks == info.Earliest.Ticks &&
                LatestTicks == info.Latest.Ticks &&
                string.Equals(ConfigKey ?? string.Empty, configKey ?? string.Empty,
                    StringComparison.Ordinal) &&
                Columns != null && Columns.Length == cameraCount;
        }
    }

    /// <summary>
    /// Stores the scalar peaks projected from visible merged curves. The original curves remain
    /// the source of truth; this index only avoids reopening every large per-grab summary.
    /// </summary>
    internal static class CurvePeakProjectionIndexStore
    {
        private const int FormatVersion = 1;
        private const int ProjectionVersion = 1;
        private const int MaxCameraCount = 64;
        private const int MaxEntryCount = 1000000;
        private const int MaxStringBytes = 4096;
        private const long MaxFileBytes = 256L * 1024 * 1024;
        private static readonly byte[] Magic =
            { (byte)'M', (byte)'C', (byte)'P', (byte)'I' };
        private static readonly object WriteSync = new object();

        public static Dictionary<string, CurvePeakProjectionEntry> LoadForGrabIds(
            string root,
            IList<GrabIdInfo> grabInfos,
            int cameraCount,
            out int dayCount)
        {
            var result = new Dictionary<string, CurvePeakProjectionEntry>(
                StringComparer.Ordinal);
            dayCount = 0;
            if (string.IsNullOrWhiteSpace(root) || grabInfos == null ||
                cameraCount <= 0 || cameraCount > MaxCameraCount)
                return result;

            var days = new HashSet<DateTime>();
            foreach (GrabIdInfo info in grabInfos)
            {
                if (info == null || info.Earliest == default(DateTime)) continue;
                days.Add(info.Earliest.Date);
            }

            foreach (DateTime day in days)
            {
                Dictionary<string, CurvePeakProjectionEntry> entries =
                    LoadDay(root, day, cameraCount);
                if (entries == null) continue;
                dayCount++;
                foreach (var pair in entries)
                    result[pair.Key] = pair.Value;
            }
            return result;
        }

        public static bool MergeSave(
            string root,
            IList<GrabIdInfo> grabInfos,
            IDictionary<string, CsvConfigSnapshot> configByGrabId,
            int cameraCount,
            ColumnCurvePeakIndexResult result)
        {
            if (string.IsNullOrWhiteSpace(root) || grabInfos == null || result == null ||
                cameraCount <= 0 || cameraCount > MaxCameraCount)
                return false;

            var infoByGrabId = new Dictionary<string, GrabIdInfo>(StringComparer.Ordinal);
            foreach (GrabIdInfo info in grabInfos)
            {
                if (info != null && !string.IsNullOrWhiteSpace(info.GrabId))
                    infoByGrabId[info.GrabId] = info;
            }

            var updatesByDay = new Dictionary<DateTime, List<CurvePeakProjectionEntry>>();
            var grabIds = new HashSet<string>(result.ByGrabId.Keys, StringComparer.Ordinal);
            foreach (string grabId in result.RowByGrabId.Keys)
                grabIds.Add(grabId);
            foreach (string grabId in grabIds)
            {
                if (!infoByGrabId.TryGetValue(grabId, out GrabIdInfo info)) continue;
                result.ByGrabId.TryGetValue(grabId, out ColumnCurvePeakRecord[] columns);
                result.RowByGrabId.TryGetValue(grabId, out RowCurvePeakRecord row);
                if (columns == null || columns.Length != cameraCount) continue;

                var entry = new CurvePeakProjectionEntry
                {
                    GrabId = grabId,
                    EarliestTicks = info.Earliest.Ticks,
                    LatestTicks = info.Latest.Ticks,
                    ConfigKey = GetConfigKey(configByGrabId, grabId),
                    Columns = columns,
                    Row = row
                };
                DateTime day = info.Earliest.Date;
                if (!updatesByDay.TryGetValue(
                    day, out List<CurvePeakProjectionEntry> updates))
                {
                    updates = new List<CurvePeakProjectionEntry>();
                    updatesByDay[day] = updates;
                }
                updates.Add(entry);
            }

            bool saved = updatesByDay.Count > 0;
            foreach (var pair in updatesByDay)
                saved &= MergeSaveDay(root, pair.Key, cameraCount, pair.Value);
            return saved;
        }

        public static string GetConfigKey(
            IDictionary<string, CsvConfigSnapshot> configByGrabId, string grabId)
        {
            if (configByGrabId != null &&
                configByGrabId.TryGetValue(grabId, out CsvConfigSnapshot config) &&
                config != null)
                return config.ContentKey;
            return string.Empty;
        }

        private static Dictionary<string, CurvePeakProjectionEntry> LoadDay(
            string root, DateTime day, int cameraCount)
        {
            string path = CaptureStoragePaths.DailyCurvePeakIndex(root, day);
            if (!File.Exists(path)) return null;
            try
            {
                var fileInfo = new FileInfo(path);
                if (fileInfo.Length <= 0 || fileInfo.Length > MaxFileBytes) return null;
                using (var stream = new FileStream(
                    path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete,
                    64 * 1024, FileOptions.SequentialScan))
                using (var reader = new BinaryReader(stream, Encoding.UTF8))
                {
                    if (!ReadAndMatchMagic(reader)) return null;
                    if (reader.ReadInt32() != FormatVersion) return null;
                    if (reader.ReadInt32() != ProjectionVersion) return null;
                    if (reader.ReadInt32() != cameraCount) return null;
                    int count = reader.ReadInt32();
                    if (count < 0 || count > MaxEntryCount) return null;

                    var result = new Dictionary<string, CurvePeakProjectionEntry>(
                        count, StringComparer.Ordinal);
                    for (int i = 0; i < count; i++)
                    {
                        var entry = new CurvePeakProjectionEntry
                        {
                            GrabId = ReadString(reader),
                            EarliestTicks = reader.ReadInt64(),
                            LatestTicks = reader.ReadInt64(),
                            ConfigKey = ReadString(reader),
                            Columns = new ColumnCurvePeakRecord[cameraCount]
                        };
                        for (int cameraIndex = 0; cameraIndex < cameraCount; cameraIndex++)
                        {
                            if (!reader.ReadBoolean()) continue;
                            entry.Columns[cameraIndex] = new ColumnCurvePeakRecord
                            {
                                GrabId = entry.GrabId,
                                CameraId = cameraIndex + 1,
                                CaptureHmV = reader.ReadSingle(),
                                RawMeanPeak = reader.ReadSingle(),
                                RawMaxPeak = reader.ReadSingle()
                            };
                        }
                        if (reader.ReadBoolean())
                        {
                            entry.Row = new RowCurvePeakRecord
                            {
                                GrabId = entry.GrabId,
                                CaptureHmV = reader.ReadSingle(),
                                RawMeanPeak = reader.ReadSingle(),
                                RawMaxPeak = reader.ReadSingle()
                            };
                        }
                        result[entry.GrabId] = entry;
                    }
                    return result;
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine(
                    $"[CurvePeakProjectionIndexStore.LoadDay] {path}: " +
                    $"{ex.GetType().Name}: {ex.Message}");
                return null;
            }
        }

        private static bool MergeSaveDay(
            string root,
            DateTime day,
            int cameraCount,
            IList<CurvePeakProjectionEntry> updates)
        {
            if (updates == null || updates.Count == 0) return false;
            string path = CaptureStoragePaths.DailyCurvePeakIndex(root, day);
            string directory = Path.GetDirectoryName(path);

            lock (WriteSync)
            {
                Dictionary<string, CurvePeakProjectionEntry> entries =
                    LoadDay(root, day, cameraCount) ??
                    new Dictionary<string, CurvePeakProjectionEntry>(StringComparer.Ordinal);
                foreach (CurvePeakProjectionEntry entry in updates)
                    entries[entry.GrabId] = entry;

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
                        writer.Write(ProjectionVersion);
                        writer.Write(cameraCount);
                        writer.Write(entries.Count);
                        foreach (CurvePeakProjectionEntry entry in entries.Values)
                            WriteEntry(writer, entry, cameraCount);
                        writer.Flush();
                        stream.Flush(true);
                    }
                    if (File.Exists(path))
                        File.Replace(tempPath, path, null, true);
                    else
                        File.Move(tempPath, path);
                    return true;
                }
                catch (Exception ex)
                {
                    Trace.WriteLine(
                        $"[CurvePeakProjectionIndexStore.MergeSaveDay] {path}: " +
                        $"{ex.GetType().Name}: {ex.Message}");
                }
                finally
                {
                    try { if (File.Exists(tempPath)) File.Delete(tempPath); } catch { }
                }
            }
            return false;
        }

        private static void WriteEntry(
            BinaryWriter writer, CurvePeakProjectionEntry entry, int cameraCount)
        {
            writer.Write(entry.GrabId ?? string.Empty);
            writer.Write(entry.EarliestTicks);
            writer.Write(entry.LatestTicks);
            writer.Write(entry.ConfigKey ?? string.Empty);
            for (int i = 0; i < cameraCount; i++)
            {
                ColumnCurvePeakRecord record = entry.Columns != null && i < entry.Columns.Length
                    ? entry.Columns[i]
                    : null;
                writer.Write(record != null);
                if (record == null) continue;
                writer.Write(record.CaptureHmV);
                writer.Write(record.RawMeanPeak);
                writer.Write(record.RawMaxPeak);
            }
            writer.Write(entry.Row != null);
            if (entry.Row == null) return;
            writer.Write(entry.Row.CaptureHmV);
            writer.Write(entry.Row.RawMeanPeak);
            writer.Write(entry.Row.RawMaxPeak);
        }

        private static bool ReadAndMatchMagic(BinaryReader reader)
        {
            byte[] actual = reader.ReadBytes(Magic.Length);
            if (actual.Length != Magic.Length) return false;
            for (int i = 0; i < Magic.Length; i++)
                if (actual[i] != Magic[i]) return false;
            return true;
        }

        private static string ReadString(BinaryReader reader)
        {
            string value = reader.ReadString();
            if (Encoding.UTF8.GetByteCount(value) > MaxStringBytes)
                throw new InvalidDataException("Curve peak index string exceeds the limit.");
            return value;
        }
    }
}
