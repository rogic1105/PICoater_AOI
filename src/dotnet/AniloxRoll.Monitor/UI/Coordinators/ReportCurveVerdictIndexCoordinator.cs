using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal delegate ColumnCurvePeakIndexResult BuildCurvePeakSummaries(
        string root,
        IList<GrabIdInfo> grabInfos,
        IDictionary<string, float> captureHm,
        IDictionary<string, CsvConfigSnapshot> captureConfigs,
        int cameraCount,
        CancellationToken cancellationToken);

    internal delegate ColumnCurvePeakIndexResult BuildCurvePeakBinFallback(
        string root,
        IList<GrabIdInfo> grabInfos,
        IDictionary<string, float> captureHm,
        IDictionary<string, CsvConfigSnapshot> captureConfigs,
        int cameraCount,
        CancellationToken cancellationToken,
        Action<ColumnCurvePeakIndexResult> progress,
        int progressBatchSize);

    /// <summary>
    /// Owns the report peak-index build generation and summary-to-bin fallback.
    /// O/X projection and audit evidence remain in ReportCurveVerdictPresenter.
    /// </summary>
    internal sealed class ReportCurveVerdictIndexCoordinator : IDisposable
    {
        private readonly ReportCurveVerdictIndex _index;
        private readonly Func<string> _currentRoot;
        private readonly Func<ThresholdContext> _createThreshold;
        private readonly Func<Action, bool> _tryPost;
        private readonly Action _refreshAllViews;
        private readonly Action _refreshList;
        private readonly Action<string> _log;
        private readonly Action<string> _dvt;
        private readonly BuildCurvePeakSummaries _buildSummaries;
        private readonly BuildCurvePeakBinFallback _buildBinFallback;
        private CancellationTokenSource _cts;
        private int _generation;
        private bool _disposed;

        public ReportCurveVerdictIndexCoordinator(
            ReportCurveVerdictIndex index,
            Func<string> currentRoot,
            Func<ThresholdContext> createThreshold,
            Func<Action, bool> tryPost,
            Action refreshAllViews,
            Action refreshList,
            Action<string> log,
            Action<string> dvt)
            : this(
                index, currentRoot, createThreshold, tryPost,
                refreshAllViews, refreshList, log, dvt,
                ColumnCurvePeakIndex.BuildSummaries,
                ColumnCurvePeakIndex.BuildBinFallback)
        {
        }

        internal ReportCurveVerdictIndexCoordinator(
            ReportCurveVerdictIndex index,
            Func<string> currentRoot,
            Func<ThresholdContext> createThreshold,
            Func<Action, bool> tryPost,
            Action refreshAllViews,
            Action refreshList,
            Action<string> log,
            Action<string> dvt,
            BuildCurvePeakSummaries buildSummaries,
            BuildCurvePeakBinFallback buildBinFallback)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));
            _currentRoot = currentRoot ?? throw new ArgumentNullException(nameof(currentRoot));
            _createThreshold = createThreshold ?? throw new ArgumentNullException(nameof(createThreshold));
            _tryPost = tryPost ?? throw new ArgumentNullException(nameof(tryPost));
            _refreshAllViews = refreshAllViews ?? throw new ArgumentNullException(nameof(refreshAllViews));
            _refreshList = refreshList ?? throw new ArgumentNullException(nameof(refreshList));
            _log = log ?? throw new ArgumentNullException(nameof(log));
            _dvt = dvt ?? throw new ArgumentNullException(nameof(dvt));
            _buildSummaries = buildSummaries ??
                throw new ArgumentNullException(nameof(buildSummaries));
            _buildBinFallback = buildBinFallback ??
                throw new ArgumentNullException(nameof(buildBinFallback));
        }

        public void Start(
            bool resetExisting,
            string root,
            IList<GrabIdInfo> allInfos,
            IDictionary<string, float> captureHm,
            IDictionary<string, CsvConfigSnapshot> captureConfigs,
            int cameraCount)
        {
            if (_disposed) return;
            Cancel();
            if (resetExisting) _index.ClearPeaks();
            if (string.IsNullOrWhiteSpace(root) || allInfos == null || allInfos.Count == 0)
                return;

            int generation = _generation;
            List<GrabIdInfo> infos = allInfos
                .Where(info => !_index.HasBothPeaks(info.GrabId))
                .Select(CloneInfo)
                .ToList();
            if (infos.Count == 0) return;

            var hm = new Dictionary<string, float>(captureHm, StringComparer.Ordinal);
            var configs = new Dictionary<string, CsvConfigSnapshot>(captureConfigs, StringComparer.Ordinal);
            var cts = new CancellationTokenSource();
            _cts = cts;

            CancellationToken token = cts.Token;
            Task.Run(() => _buildSummaries(
                root, infos, hm, configs, cameraCount, token), token)
                .ContinueWith(task =>
                {
                    if (task.IsCanceled || token.IsCancellationRequested)
                    {
                        Release(cts);
                        return;
                    }
                    if (task.IsFaulted)
                    {
                        LogFailure("summaries", generation, task.Exception);
                        Release(cts);
                        return;
                    }
                    if (!_tryPost(() => CompleteSummaries(
                        generation, root, task.Result, hm, configs, cameraCount, cts)))
                        Release(cts);
                }, CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
        }

        public void Cancel()
        {
            if (_disposed) return;
            CancelCore();
        }

        public void Dispose()
        {
            if (_disposed) return;
            CancelCore();
            _disposed = true;
        }

        private void CancelCore()
        {
            _generation++;
            CancellationTokenSource cts = _cts;
            _cts = null;
            if (cts == null) return;
            try { cts.Cancel(); }
            finally { cts.Dispose(); }
        }

        private void CompleteSummaries(
            int generation,
            string root,
            ColumnCurvePeakIndexResult summaries,
            IDictionary<string, float> captureHm,
            IDictionary<string, CsvConfigSnapshot> captureConfigs,
            int cameraCount,
            CancellationTokenSource cts)
        {
            if (!IsCurrent(generation, root) || summaries == null)
            {
                Release(cts);
                return;
            }

            _index.Apply(summaries);
            CurvePeakVerdictProjectionResult applied = _index.Project(_createThreshold());
            _refreshAllViews();
            _log(
                $"DT verdict index apply=partial gen={generation} " +
                $"summaries={summaries.SummaryGrabCount} " +
                $"pending={summaries.PendingBinGrabInfos.Count}/{summaries.RequestedGrabCount} " +
                $"cams={applied.ColumnCount} verdicts={applied.ColumnCount} ms={summaries.ElapsedMilliseconds}");
            _log(
                $"DT verdict cache gen={generation} hits={summaries.CacheGrabCount}/" +
                $"{summaries.RequestedGrabCount} days={summaries.CacheDayCount} " +
                $"loadMs={summaries.CacheLoadMilliseconds}");
            LogRowResult("partial", generation, applied.RowCount);

            if (summaries.PendingBinGrabInfos.Count == 0)
            {
                Complete(generation, root, summaries, null, cts, false);
                return;
            }

            List<GrabIdInfo> pending = summaries.PendingBinGrabInfos
                .Select(CloneInfo)
                .ToList();
            CancellationToken token = cts.Token;
            Task.Factory.StartNew(() => _buildBinFallback(
                root, pending, captureHm, captureConfigs,
                cameraCount, token,
                progress => QueueProgress(generation, root, progress, token),
                8),
                token,
                TaskCreationOptions.LongRunning,
                TaskScheduler.Default)
                .ContinueWith(task =>
                {
                    if (task.IsCanceled || token.IsCancellationRequested)
                    {
                        Release(cts);
                        return;
                    }
                    if (task.IsFaulted)
                    {
                        LogFailure("bins", generation, task.Exception);
                        Release(cts);
                        return;
                    }
                    if (!_tryPost(() => Complete(
                        generation, root, summaries, task.Result, cts, true)))
                        Release(cts);
                }, CancellationToken.None, TaskContinuationOptions.None, TaskScheduler.Default);
        }

        private void QueueProgress(
            int generation,
            string root,
            ColumnCurvePeakIndexResult progress,
            CancellationToken token)
        {
            if (progress == null || token.IsCancellationRequested) return;
            _tryPost(() =>
            {
                if (!IsCurrent(generation, root) || token.IsCancellationRequested) return;
                _index.Apply(progress);
                _index.Project(_createThreshold());
                _refreshList();
                _dvt(
                    $"DT verdict index progress gen={generation} " +
                    $"batch={progress.BinFallbackGrabCount} " +
                    $"cams={progress.CameraCount} rows={progress.RowByGrabId.Count}");
            });
        }

        private void Complete(
            int generation,
            string root,
            ColumnCurvePeakIndexResult summaries,
            ColumnCurvePeakIndexResult bins,
            CancellationTokenSource cts,
            bool refreshViews)
        {
            if (!IsCurrent(generation, root) || summaries == null)
            {
                Release(cts);
                return;
            }
            if (bins != null) _index.Apply(bins);

            CurvePeakVerdictProjectionResult applied = _index.Project(_createThreshold());
            if (refreshViews) _refreshAllViews();
            int binCount = bins?.BinFallbackGrabCount ?? 0;
            int missingCount = bins?.MissingGrabCount ?? 0;
            long elapsed = summaries.ElapsedMilliseconds + (bins?.ElapsedMilliseconds ?? 0L);
            _log(
                $"DT verdict index apply=ok gen={generation} " +
                $"summaries={summaries.SummaryGrabCount} bins={binCount} " +
                $"missing={missingCount}/{summaries.RequestedGrabCount} " +
                $"cams={applied.ColumnCount} verdicts={applied.ColumnCount} ms={elapsed}");
            LogRowResult("ok", generation, applied.RowCount);
            Release(cts);
        }

        private void LogRowResult(string state, int generation, int rowCount)
        {
            _log(
                $"DT row verdict index apply={state} gen={generation} " +
                $"rows={rowCount} verdicts={rowCount} " +
                $"enabled={(_createThreshold().RowDetectionEnabled ? 1 : 0)}");
        }

        private bool IsCurrent(int generation, string root)
        {
            return generation == _generation && string.Equals(
                root, _currentRoot(), StringComparison.OrdinalIgnoreCase);
        }

        private void Release(CancellationTokenSource cts)
        {
            if (!ReferenceEquals(_cts, cts)) return;
            _cts = null;
            cts.Dispose();
        }

        private void LogFailure(string stage, int generation, AggregateException exception)
        {
            Exception error = exception?.Flatten().InnerException ?? exception;
            string errorName = error?.GetType().Name ?? "UnknownException";
            _log(
                $"DT verdict index apply=failed gen={generation} " +
                $"stage={stage} error={errorName}");
        }

        private static GrabIdInfo CloneInfo(GrabIdInfo info)
        {
            return new GrabIdInfo
            {
                GrabId = info.GrabId,
                Earliest = info.Earliest,
                Latest = info.Latest
            };
        }
    }
}
