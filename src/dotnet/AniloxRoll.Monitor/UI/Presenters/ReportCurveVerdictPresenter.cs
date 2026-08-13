using System;
using System.Globalization;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Services;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// Owns report O/X projection and its DVT evidence. The parent presenter
    /// decides when to refresh controls; this owner decides what verdict data
    /// and evidence those controls receive.
    /// </summary>
    internal sealed class ReportCurveVerdictPresenter
    {
        private readonly ReportCurveVerdictIndex _index;
        private readonly Func<ThresholdContext> _createThreshold;
        private readonly Func<CsvConfigSnapshot> _createFallbackConfig;
        private readonly Func<int> _cameraCount;
        private readonly Func<bool> _dvtEnabled;
        private readonly Action<string> _log;

        public ReportCurveVerdictPresenter(
            ReportCurveVerdictIndex index,
            Func<ThresholdContext> createThreshold,
            Func<CsvConfigSnapshot> createFallbackConfig,
            Func<int> cameraCount,
            Func<bool> dvtEnabled,
            Action<string> log)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));
            _createThreshold = createThreshold ?? throw new ArgumentNullException(nameof(createThreshold));
            _createFallbackConfig = createFallbackConfig ?? throw new ArgumentNullException(nameof(createFallbackConfig));
            _cameraCount = cameraCount ?? throw new ArgumentNullException(nameof(cameraCount));
            _dvtEnabled = dvtEnabled ?? throw new ArgumentNullException(nameof(dvtEnabled));
            _log = log ?? throw new ArgumentNullException(nameof(log));
        }

        public bool ApplyVisibleCurves(
            string grabId, SingleGrabCurveData data, GrabDetail detail)
        {
            if (string.IsNullOrWhiteSpace(grabId) || data == null || detail == null)
                return false;

            ThresholdContext threshold = _createThreshold();
            CsvConfigSnapshot config = data.Config ?? _createFallbackConfig();
            float captureHmV = data.Config?.HessianMaxFactorV ?? threshold.CurrentHmV;
            ColumnCurvePeakRecord[] records = ColumnCurvePeakIndex.ProjectVisibleRecords(
                grabId, data.ColumnMean, data.ColumnMax,
                config, captureHmV, _cameraCount());

            Array.Clear(detail.CamResult, 0, detail.CamResult.Length);
            if (threshold.ColumnDetectionEnabled && records != null)
            {
                _index.ColumnPeaks[grabId] = records;
                int count = Math.Min(detail.CamResult.Length, records.Length);
                for (int i = 0; i < count; i++)
                {
                    ColumnCurvePeakRecord record = records[i];
                    if (record == null) continue;
                    ColumnVerdictEvaluation evaluation = threshold.EvaluateRawColumnCurve(
                        record.RawMeanPeak, record.RawMaxPeak, record.CaptureHmV);
                    if (!evaluation.HasData) continue;

                    detail.CamResult[i] = evaluation.IsFail;
                    LogVisibleVerdict(
                        grabId, "column", i + 1, evaluation,
                        threshold.CurrentErrMean, threshold.CurrentErrMax, threshold);
                }
            }

            RowCurvePeakRecord rowRecord = CurvePeakVerdictProjector.CreateRowRecord(
                grabId, data.MergedRowMean, data.MergedRowMax, captureHmV);
            if (rowRecord != null)
                _index.RowPeaks[grabId] = rowRecord;
            else
                _index.RowPeaks.Remove(grabId);

            ColumnVerdictEvaluation rowEvaluation =
                CurvePeakVerdictProjector.EvaluateRow(rowRecord, threshold);
            detail.RowResult = rowEvaluation.HasData
                ? (bool?)rowEvaluation.IsFail
                : null;
            if (rowEvaluation.HasData)
            {
                LogVisibleVerdict(
                    grabId, "row", 0, rowEvaluation,
                    threshold.CurrentRowErrMean,
                    threshold.CurrentRowErrMax,
                    threshold);
            }
            return true;
        }

        public bool ApplyCurrentIfNeeded(string selectedGrabId)
        {
            ThresholdContext threshold = _createThreshold();
            if (_index.IsVerdictCurrent(threshold)) return false;

            CurvePeakVerdictProjectionResult result = _index.Project(threshold);
            _log(
                $"DT verdict refresh source=peak-index columns={result.ColumnCount} " +
                $"rows={result.RowCount}");
            AuditSelected(selectedGrabId, "settings");
            return true;
        }

        public CurvePeakVerdictProjectionResult Project(ThresholdContext threshold)
        {
            return _index.Project(threshold);
        }

        public void AuditSelected(string grabId, string trigger)
        {
            if (!_dvtEnabled()) return;
            Audit(grabId, trigger);
        }

        public void Audit(string grabId, string trigger)
        {
            if (string.IsNullOrWhiteSpace(grabId)) return;
            bool isClick = string.Equals(trigger, "click", StringComparison.Ordinal);
            string columnPrefix = isClick
                ? $"DT verdict click {grabId}"
                : $"DT verdict audit {grabId} trigger={trigger}";
            string rowPrefix = isClick
                ? $"DT row verdict click {grabId}"
                : $"DT row verdict audit {grabId} trigger={trigger}";
            ThresholdContext threshold = _createThreshold();
            _index.ColumnPeaks.TryGetValue(grabId, out ColumnCurvePeakRecord[] records);
            _index.Details.TryGetValue(grabId, out GrabDetail detail);
            string mode = threshold.ColumnCurveMode.ToString().ToLowerInvariant();
            int cameraCount = _cameraCount();

            for (int i = 0; i < cameraCount; i++)
            {
                ColumnCurvePeakRecord record = records != null && i < records.Length
                    ? records[i]
                    : null;
                ColumnVerdictEvaluation evaluation = record == null
                    ? null
                    : threshold.EvaluateRawColumnCurve(
                        record.RawMeanPeak, record.RawMaxPeak, record.CaptureHmV);
                bool? listResult = detail != null && i < detail.CamResult.Length
                    ? detail.CamResult[i]
                    : null;
                string source = evaluation == null || !evaluation.HasData
                    ? "missing"
                    : "visible-curve-index";
                _log(
                    $"{columnPrefix} cam={i + 1} mode={mode} " +
                    $"mean={FormatPeak(evaluation?.DisplayMeanPeak)}/{threshold.CurrentErrMean:F4} " +
                    $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Max ? 0 : 1)} " +
                    $"max={FormatPeak(evaluation?.DisplayMaxPeak)}/{threshold.CurrentErrMax:F4} " +
                    $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Mean ? 0 : 1)} " +
                    $"result={FormatResult(evaluation)} " +
                    $"cause={(evaluation == null ? "none" : evaluation.Cause.ToString().ToLowerInvariant())} " +
                    $"list={FormatListResult(listResult)} source={source}");
            }

            _index.RowPeaks.TryGetValue(grabId, out RowCurvePeakRecord rowRecord);
            ColumnVerdictEvaluation rowEvaluation = rowRecord == null
                ? null
                : threshold.EvaluateRawRowCurve(
                    rowRecord.RawMeanPeak, rowRecord.RawMaxPeak,
                    rowRecord.CaptureHmV);
            bool? rowListResult = detail?.RowResult;
            string rowSource = rowEvaluation == null || !rowEvaluation.HasData
                ? "missing"
                : "visible-curve-index";
            _log(
                $"{rowPrefix} mode={mode} " +
                $"mean={FormatPeak(rowEvaluation?.DisplayMeanPeak)}/{threshold.CurrentRowErrMean:F4} " +
                $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Max ? 0 : 1)} " +
                $"max={FormatPeak(rowEvaluation?.DisplayMaxPeak)}/{threshold.CurrentRowErrMax:F4} " +
                $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Mean ? 0 : 1)} " +
                $"result={FormatResult(rowEvaluation)} " +
                $"cause={(rowEvaluation == null ? "none" : rowEvaluation.Cause.ToString().ToLowerInvariant())} " +
                $"list={FormatListResult(rowListResult)} source={rowSource}");
            _log(isClick
                ? $"DT verdict click done {grabId} cams={cameraCount}"
                : $"DT verdict audit done {grabId} trigger={trigger} cams={cameraCount}");
        }

        private void LogVisibleVerdict(
            string grabId, string axis, int cameraId,
            ColumnVerdictEvaluation evaluation,
            float meanThreshold, float maxThreshold,
            ThresholdContext threshold)
        {
            string prefix = axis == "column"
                ? $"DT verdict {grabId} cam={cameraId}"
                : $"DT row verdict {grabId} merged=1";
            _log(
                prefix + " " +
                $"mode={threshold.ColumnCurveMode.ToString().ToLowerInvariant()} " +
                $"mean={evaluation.DisplayMeanPeak:F4}/{meanThreshold:F4} " +
                $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Max ? 0 : 1)} " +
                $"max={evaluation.DisplayMaxPeak:F4}/{maxThreshold:F4} " +
                $"enabled={(threshold.ColumnCurveMode == ColumnCurveDisplayMode.Mean ? 0 : 1)} " +
                $"result={(evaluation.IsFail ? "fail" : "pass")} " +
                $"cause={evaluation.Cause.ToString().ToLowerInvariant()} " +
                "source=visible-merged-curve");
        }

        private static string FormatPeak(float? value)
        {
            return value.HasValue
                ? value.Value.ToString("F4", CultureInfo.InvariantCulture)
                : "nan";
        }

        private static string FormatResult(ColumnVerdictEvaluation evaluation)
        {
            return evaluation == null || !evaluation.HasData
                ? "unknown"
                : evaluation.IsFail ? "fail" : "pass";
        }

        private static string FormatListResult(bool? result)
        {
            return result.HasValue
                ? result.Value ? "fail" : "pass"
                : "unknown";
        }
    }
}
