using System;
using System.Collections.Generic;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// Owns the report detail and curve-peak indexes as one coherent state.
    /// The index coordinator controls scheduling; this owner controls index identity.
    /// </summary>
    internal sealed class ReportCurveVerdictIndex
    {
        private string _root = string.Empty;
        private bool _ready;
        private CurveVerdictSettingsSnapshot _settings;

        public Dictionary<string, GrabDetail> Details { get; } =
            new Dictionary<string, GrabDetail>(StringComparer.Ordinal);

        public Dictionary<string, ColumnCurvePeakRecord[]> ColumnPeaks { get; } =
            new Dictionary<string, ColumnCurvePeakRecord[]>(StringComparer.Ordinal);

        public Dictionary<string, RowCurvePeakRecord> RowPeaks { get; } =
            new Dictionary<string, RowCurvePeakRecord>(StringComparer.Ordinal);

        public bool IsCurrent(string root)
        {
            return _ready && string.Equals(
                _root, root, StringComparison.OrdinalIgnoreCase);
        }

        public bool IsVerdictCurrent(ThresholdContext threshold)
        {
            return _settings != null && _settings.Matches(threshold);
        }

        public void ReplaceDetails(
            string root,
            IEnumerable<KeyValuePair<string, GrabDetail>> details,
            ThresholdContext threshold)
        {
            string nextRoot = root ?? string.Empty;
            bool rootChanged = !_ready || !string.Equals(
                _root, nextRoot, StringComparison.OrdinalIgnoreCase);

            Details.Clear();
            if (details != null)
            {
                foreach (KeyValuePair<string, GrabDetail> entry in details)
                    Details[entry.Key] = entry.Value;
            }

            if (rootChanged)
            {
                ClearPeaks();
            }
            else
            {
                RemoveMissingPeaks(ColumnPeaks, Details);
                RemoveMissingPeaks(RowPeaks, Details);
            }

            _root = nextRoot;
            _settings = CurveVerdictSettingsSnapshot.Capture(threshold);
            _ready = true;
        }

        public void Reset()
        {
            Details.Clear();
            ColumnPeaks.Clear();
            RowPeaks.Clear();
            _root = string.Empty;
            _settings = null;
            _ready = false;
        }

        public void ClearPeaks()
        {
            ColumnPeaks.Clear();
            RowPeaks.Clear();
        }

        public bool HasBothPeaks(string grabId)
        {
            return ColumnPeaks.ContainsKey(grabId) && RowPeaks.ContainsKey(grabId);
        }

        public void Apply(ColumnCurvePeakIndexResult result)
        {
            if (result == null) return;
            foreach (KeyValuePair<string, ColumnCurvePeakRecord[]> entry in result.ByGrabId)
                ColumnPeaks[entry.Key] = entry.Value;
            foreach (KeyValuePair<string, RowCurvePeakRecord> entry in result.RowByGrabId)
                RowPeaks[entry.Key] = entry.Value;
        }

        public CurvePeakVerdictProjectionResult Project(ThresholdContext threshold)
        {
            CurvePeakVerdictProjectionResult result = CurvePeakVerdictProjector.Apply(
                Details, ColumnPeaks, RowPeaks, threshold);
            _settings = CurveVerdictSettingsSnapshot.Capture(threshold);
            return result;
        }

        private static void RemoveMissingPeaks<T>(
            Dictionary<string, T> peaks,
            Dictionary<string, GrabDetail> details)
        {
            var obsolete = new List<string>();
            foreach (string grabId in peaks.Keys)
            {
                if (!details.ContainsKey(grabId))
                    obsolete.Add(grabId);
            }

            foreach (string grabId in obsolete)
                peaks.Remove(grabId);
        }
    }
}
