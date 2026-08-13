using System;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Coalesces rapid report single-grab selections and applies only the
    /// latest selection once per UI frame interval.
    /// </summary>
    internal sealed class ReportSingleGrabSelectionCoordinator : IDisposable
    {
        internal const int IntervalMs = 33;

        private readonly Func<string> _selectedGrabId;
        private readonly Action _applyLatestSelection;
        private readonly Action<string> _flow;
        private readonly System.Windows.Forms.Timer _timer;
        private int _pendingRequests;
        private bool _disposed;

        public ReportSingleGrabSelectionCoordinator(
            Func<string> selectedGrabId,
            Action applyLatestSelection,
            Action<string> flow)
        {
            _selectedGrabId = selectedGrabId ??
                throw new ArgumentNullException(nameof(selectedGrabId));
            _applyLatestSelection = applyLatestSelection ??
                throw new ArgumentNullException(nameof(applyLatestSelection));
            _flow = flow ?? throw new ArgumentNullException(nameof(flow));
            _timer = new System.Windows.Forms.Timer { Interval = IntervalMs };
            _timer.Tick += Timer_Tick;
        }

        public void Schedule()
        {
            if (_disposed) return;
            _pendingRequests++;
            if (!_timer.Enabled) _timer.Start();
        }

        public void Cancel()
        {
            if (_disposed) return;
            _timer.Stop();
            _pendingRequests = 0;
        }

        internal void ApplyPendingNow()
        {
            if (_disposed) return;
            _timer.Stop();
            int requestCount = _pendingRequests;
            _pendingRequests = 0;
            if (requestCount == 0) return;

            string grabId = _selectedGrabId();
            if (string.IsNullOrWhiteSpace(grabId)) return;

            _flow($"ui:【報表序號】→ {grabId}");
            if (requestCount > 1)
            {
                _flow(
                    $"DT selected coalesced {grabId} " +
                    $"skipped={requestCount - 1} intervalMs={IntervalMs}");
            }
            _applyLatestSelection();
        }

        public void Dispose()
        {
            if (_disposed) return;
            Cancel();
            _disposed = true;
            _timer.Tick -= Timer_Tick;
            _timer.Dispose();
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            ApplyPendingNow();
        }
    }
}
