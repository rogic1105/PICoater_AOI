using System;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Owns the Data range-selection timing policy. Fast list and curve previews are sampled
    /// independently, while the settled selection performs one authoritative refresh.
    /// </summary>
    internal sealed class DataRangePreviewCoordinator : IDisposable
    {
        internal const int ListPreviewIntervalMs = 33;
        internal const int CurvePreviewIntervalMs = 80;
        internal const int SettleIntervalMs = 150;
        internal const int CurveSampleLimit = 50;

        private readonly Action _clearTransientPresentation;
        private readonly Action _applySettledSelection;
        private readonly Func<int, bool> _applyListPreview;
        private readonly Func<int, Func<int>, CancellationToken, Task> _applyCurvePreviewAsync;
        private readonly Action<string> _flow;
        private readonly System.Windows.Forms.Timer _settleTimer;
        private readonly System.Windows.Forms.Timer _listTimer;
        private readonly System.Windows.Forms.Timer _curveTimer;

        private CancellationTokenSource _curveCancellation;
        private int _generation;
        private int _listAppliedGeneration;
        private int _curveAppliedGeneration;
        private bool _curveRunning;
        private bool _disposed;

        public DataRangePreviewCoordinator(
            Action clearTransientPresentation,
            Action applySettledSelection,
            Func<int, bool> applyListPreview,
            Func<int, Func<int>, CancellationToken, Task> applyCurvePreviewAsync,
            Action<string> flow)
        {
            _clearTransientPresentation = clearTransientPresentation ??
                throw new ArgumentNullException(nameof(clearTransientPresentation));
            _applySettledSelection = applySettledSelection ??
                throw new ArgumentNullException(nameof(applySettledSelection));
            _applyListPreview = applyListPreview ??
                throw new ArgumentNullException(nameof(applyListPreview));
            _applyCurvePreviewAsync = applyCurvePreviewAsync ??
                throw new ArgumentNullException(nameof(applyCurvePreviewAsync));
            _flow = flow;

            _settleTimer = new System.Windows.Forms.Timer { Interval = SettleIntervalMs };
            _settleTimer.Tick += SettleTimer_Tick;
            _listTimer = new System.Windows.Forms.Timer { Interval = ListPreviewIntervalMs };
            _listTimer.Tick += ListTimer_Tick;
            _curveTimer = new System.Windows.Forms.Timer { Interval = CurvePreviewIntervalMs };
            _curveTimer.Tick += CurveTimer_Tick;
        }

        public void Start()
        {
            if (_disposed) return;

            _clearTransientPresentation();
            _settleTimer.Stop();
            _settleTimer.Start();

            _generation++;
            if (!_listTimer.Enabled) _listTimer.Start();
            if (!_curveTimer.Enabled) _curveTimer.Start();
        }

        public void Cancel()
        {
            if (_disposed) return;

            _generation++;
            _curveCancellation?.Cancel();
            _settleTimer.Stop();
            _listTimer.Stop();
            _curveTimer.Stop();
        }

        private void SettleTimer_Tick(object sender, EventArgs e)
        {
            _settleTimer.Stop();
            _flow?.Invoke("DT range settle → refresh");
            _applySettledSelection();
        }

        private void ListTimer_Tick(object sender, EventArgs e)
        {
            if (_listAppliedGeneration == _generation)
            {
                _listTimer.Stop();
                return;
            }

            int generation = _generation;
            if (!_applyListPreview(generation))
            {
                _listTimer.Stop();
                return;
            }

            if (generation == _generation)
                _listAppliedGeneration = generation;
        }

        private async void CurveTimer_Tick(object sender, EventArgs e)
        {
            if (_curveRunning || _curveAppliedGeneration == _generation)
            {
                if (!_curveRunning) _curveTimer.Stop();
                return;
            }

            int generation = _generation;
            var cancellation = new CancellationTokenSource();
            _curveCancellation = cancellation;
            _curveRunning = true;
            try
            {
                await _applyCurvePreviewAsync(
                    generation,
                    () => _generation,
                    cancellation.Token);
                if (!cancellation.IsCancellationRequested)
                    _curveAppliedGeneration = generation;
            }
            catch (OperationCanceledException)
            {
                // A newer range selection or teardown owns cancellation.
            }
            finally
            {
                if (ReferenceEquals(_curveCancellation, cancellation))
                    _curveCancellation = null;
                cancellation.Dispose();
                _curveRunning = false;
                if (_curveAppliedGeneration == _generation)
                    _curveTimer.Stop();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            Cancel();
            _disposed = true;
            _settleTimer.Dispose();
            _listTimer.Dispose();
            _curveTimer.Dispose();
            _curveCancellation = null;
        }
    }
}
