using System;
using System.Threading;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// 單次抓取倒數的唯一 owner；到時只發布 intent，實際停止仍由 Form 的共用抓取流程處理。
    ///
    /// State + Event -> Next + Action:
    /// Idle + Arm -> Armed + 建立 one-shot timer
    /// Armed + Arm -> Armed + 作廢舊 generation、重設 timer
    /// Armed + Disarm -> Idle + 作廢舊 generation、釋放 timer
    /// Armed + Elapsed -> Idle + 發布一次 Elapsed
    /// Any + Dispose -> Disposed + 作廢 callback、釋放 timer
    /// </summary>
    internal sealed class GrabDurationCoordinator : IDisposable
    {
        private readonly object _gate = new object();
        private readonly Action<int> _elapsed;
        private Timer _timer;
        private int _generation;
        private bool _disposed;

        public GrabDurationCoordinator(Action<int> elapsed)
        {
            _elapsed = elapsed ?? throw new ArgumentNullException(nameof(elapsed));
        }

        public void Arm(int seconds)
        {
            if (seconds < 1) throw new ArgumentOutOfRangeException(nameof(seconds));

            lock (_gate)
            {
                ThrowIfDisposed();
                _generation++;
                int generation = _generation;
                DisposeTimerLocked();

                long requestedMs = (long)seconds * 1000L;
                int dueMs = requestedMs > int.MaxValue ? int.MaxValue : (int)requestedMs;
                _timer = new Timer(_ => OnElapsed(generation, seconds), null, dueMs, Timeout.Infinite);
            }
        }

        public void Disarm()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _generation++;
                DisposeTimerLocked();
            }
        }

        private void OnElapsed(int generation, int seconds)
        {
            lock (_gate)
            {
                if (_disposed || generation != _generation) return;
                _generation++;
                DisposeTimerLocked();
            }

            _elapsed(seconds);
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _generation++;
                DisposeTimerLocked();
            }
        }

        private void DisposeTimerLocked()
        {
            Timer timer = _timer;
            _timer = null;
            timer?.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(GrabDurationCoordinator));
        }
    }
}
