using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Serializes Review period loads while preserving every distinct user-selected period.
    /// Duplicate requests already running or queued are ignored; Invalidate prevents an old
    /// period result from applying after the user switches to grab-id mode.
    /// </summary>
    internal sealed class ReviewPeriodLoadCoordinator
    {
        internal sealed class Request
        {
            public DateTime Period { get; set; }
            public bool ProcessedMode { get; set; }
            internal int Generation { get; set; }
        }

        private readonly object _gate = new object();
        private readonly Queue<Request> _pending = new Queue<Request>();
        private readonly Func<Request, Func<bool>, Task> _loadAsync;
        private readonly Action<Exception> _onError;
        private Request _runningRequest;
        private bool _running;
        private int _generation;
        private Task _drainTask = Task.CompletedTask;

        public ReviewPeriodLoadCoordinator(
            Func<Request, Func<bool>, Task> loadAsync,
            Action<Exception> onError = null)
        {
            _loadAsync = loadAsync ?? throw new ArgumentNullException(nameof(loadAsync));
            _onError = onError;
        }

        public Task Enqueue(DateTime period, bool processedMode)
        {
            bool startDrain = false;
            lock (_gate)
            {
                if (Matches(_runningRequest, period, processedMode) ||
                    _pending.Any(x => Matches(x, period, processedMode)))
                    return _drainTask;

                _pending.Enqueue(new Request
                {
                    Period = period,
                    ProcessedMode = processedMode,
                    Generation = _generation
                });
                if (!_running)
                {
                    _running = true;
                    startDrain = true;
                }
            }

            if (startDrain)
            {
                Task drain = DrainAsync();
                lock (_gate) _drainTask = drain;
            }
            lock (_gate) return _drainTask;
        }

        public void Invalidate()
        {
            lock (_gate)
            {
                _generation++;
                _pending.Clear();
            }
        }

        private async Task DrainAsync()
        {
            while (true)
            {
                Request request;
                lock (_gate)
                {
                    if (_pending.Count == 0)
                    {
                        _runningRequest = null;
                        _running = false;
                        return;
                    }
                    request = _pending.Dequeue();
                    _runningRequest = request;
                }

                try
                {
                    await _loadAsync(request, () => IsCurrent(request));
                }
                catch (Exception ex)
                {
                    _onError?.Invoke(ex);
                }
                finally
                {
                    lock (_gate)
                    {
                        if (ReferenceEquals(_runningRequest, request))
                            _runningRequest = null;
                    }
                }
            }
        }

        private bool IsCurrent(Request request)
        {
            lock (_gate) return request.Generation == _generation;
        }

        private static bool Matches(Request request, DateTime period, bool processedMode)
            => request != null && request.Period == period && request.ProcessedMode == processedMode;
    }
}
