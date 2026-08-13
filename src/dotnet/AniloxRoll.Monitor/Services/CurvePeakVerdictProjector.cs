using System;
using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class CurveVerdictSettingsSnapshot
    {
        private readonly float _columnNormalization;
        private readonly float _columnMeanThreshold;
        private readonly float _columnMaxThreshold;
        private readonly ColumnCurveDisplayMode _columnMode;
        private readonly float _rowNormalization;
        private readonly float _rowMeanThreshold;
        private readonly float _rowMaxThreshold;
        private readonly RidgeDirection _direction;

        private CurveVerdictSettingsSnapshot(ThresholdContext threshold)
        {
            _columnNormalization = threshold.CurrentHmV;
            _columnMeanThreshold = threshold.CurrentErrMean;
            _columnMaxThreshold = threshold.CurrentErrMax;
            _columnMode = threshold.ColumnCurveMode;
            _rowNormalization = threshold.CurrentHmH;
            _rowMeanThreshold = threshold.CurrentRowErrMean;
            _rowMaxThreshold = threshold.CurrentRowErrMax;
            _direction = threshold.RidgeDirection;
        }

        public static CurveVerdictSettingsSnapshot Capture(ThresholdContext threshold)
        {
            return threshold == null ? null : new CurveVerdictSettingsSnapshot(threshold);
        }

        public bool Matches(ThresholdContext threshold)
        {
            return threshold != null
                && _columnNormalization == threshold.CurrentHmV
                && _columnMeanThreshold == threshold.CurrentErrMean
                && _columnMaxThreshold == threshold.CurrentErrMax
                && _columnMode == threshold.ColumnCurveMode
                && _rowNormalization == threshold.CurrentHmH
                && _rowMeanThreshold == threshold.CurrentRowErrMean
                && _rowMaxThreshold == threshold.CurrentRowErrMax
                && _direction == threshold.RidgeDirection;
        }
    }

    internal sealed class CurvePeakVerdictProjectionResult
    {
        public int ColumnCount { get; set; }
        public int RowCount { get; set; }
    }

    /// <summary>
    /// Projects immutable curve peaks into the mutable report O/X view.
    /// This owner contains no UI or file access so threshold changes can be
    /// verified independently from report scheduling and rendering.
    /// </summary>
    internal static class CurvePeakVerdictProjector
    {
        public static CurvePeakVerdictProjectionResult Apply(
            IDictionary<string, GrabDetail> details,
            IDictionary<string, ColumnCurvePeakRecord[]> columns,
            IDictionary<string, RowCurvePeakRecord> rows,
            ThresholdContext threshold)
        {
            var result = new CurvePeakVerdictProjectionResult();
            if (details == null || details.Count == 0 || threshold == null)
                return result;

            foreach (GrabDetail detail in details.Values)
            {
                Array.Clear(detail.CamResult, 0, detail.CamResult.Length);
                detail.RowResult = null;
            }

            if (threshold.ColumnDetectionEnabled && columns != null)
            {
                foreach (KeyValuePair<string, ColumnCurvePeakRecord[]> entry in columns)
                {
                    GrabDetail detail;
                    if (!details.TryGetValue(entry.Key, out detail) || entry.Value == null)
                        continue;

                    int count = Math.Min(detail.CamResult.Length, entry.Value.Length);
                    for (int i = 0; i < count; i++)
                    {
                        ColumnCurvePeakRecord record = entry.Value[i];
                        if (record == null) continue;
                        ColumnVerdictEvaluation evaluation = threshold.EvaluateRawColumnCurve(
                            record.RawMeanPeak, record.RawMaxPeak, record.CaptureHmV);
                        if (!evaluation.HasData) continue;
                        detail.CamResult[i] = evaluation.IsFail;
                        result.ColumnCount++;
                    }
                }
            }

            if (threshold.RowDetectionEnabled && rows != null)
            {
                foreach (GrabDetail detail in details.Values)
                {
                    RowCurvePeakRecord record;
                    if (!rows.TryGetValue(detail.GrabId, out record) || record == null)
                        continue;

                    ColumnVerdictEvaluation evaluation = EvaluateRow(record, threshold);
                    if (!evaluation.HasData) continue;
                    detail.RowResult = evaluation.IsFail;
                    result.RowCount++;
                }
            }

            return result;
        }

        public static RowCurvePeakRecord CreateRowRecord(
            string grabId, float[] mean, float[] max, float captureHmV)
        {
            float meanPeak = ThresholdContext.FindPeakNormalized(mean);
            float maxPeak = ThresholdContext.FindPeakNormalized(max);
            if (float.IsNaN(meanPeak) && float.IsNaN(maxPeak)) return null;
            return new RowCurvePeakRecord
            {
                GrabId = grabId,
                CaptureHmV = captureHmV,
                RawMeanPeak = meanPeak,
                RawMaxPeak = maxPeak
            };
        }

        public static ColumnVerdictEvaluation EvaluateRow(
            RowCurvePeakRecord record, ThresholdContext threshold)
        {
            if (record == null || threshold == null || !threshold.RowDetectionEnabled)
                return ColumnVerdictEvaluation.NoData;
            return threshold.EvaluateRawRowCurve(
                record.RawMeanPeak, record.RawMaxPeak, record.CaptureHmV);
        }
    }
}
