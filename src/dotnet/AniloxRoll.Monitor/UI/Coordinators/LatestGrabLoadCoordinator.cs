using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal sealed class SingleGrabLoadRequest
    {
        public string GrabId { get; set; }
        public DateTime HintFrom { get; set; }
        public DateTime HintTo { get; set; }
        internal int Sequence { get; set; }
        internal int CoalescedCount { get; set; }
        internal int InvalidationVersion { get; set; }
    }

    /// <summary>
    /// Serializes single-grab loads and retains only the latest request that has not started.
    /// The running request may finish its IO, but IsCurrent prevents stale results from applying.
    /// </summary>
    internal sealed class LatestGrabLoadCoordinator
    {
        private readonly object _gate = new object();
        private readonly Func<SingleGrabLoadRequest, Task> _loadAsync;
        private readonly Func<int, Task> _delayAsync;
        private readonly int _minimumCycleMs;
        private SingleGrabLoadRequest _pending;
        private bool _running;
        private int _sequence;
        private int _invalidationVersion;

        public LatestGrabLoadCoordinator(
            Func<SingleGrabLoadRequest, Task> loadAsync,
            int minimumCycleMs = 0,
            Func<int, Task> delayAsync = null)
        {
            _loadAsync = loadAsync ?? throw new ArgumentNullException(nameof(loadAsync));
            if (minimumCycleMs < 0)
                throw new ArgumentOutOfRangeException(nameof(minimumCycleMs));
            _minimumCycleMs = minimumCycleMs;
            _delayAsync = delayAsync ?? Task.Delay;
        }

        public Task Enqueue(string grabId, DateTime hintFrom, DateTime hintTo)
        {
            lock (_gate)
            {
                var request = new SingleGrabLoadRequest
                {
                    GrabId = grabId,
                    HintFrom = hintFrom,
                    HintTo = hintTo,
                    Sequence = Interlocked.Increment(ref _sequence),
                    InvalidationVersion = Volatile.Read(ref _invalidationVersion),
                    CoalescedCount = _pending == null
                        ? 0
                        : _pending.CoalescedCount + 1
                };
                _pending = request;
                if (_running)
                    return Task.CompletedTask;
                _running = true;
            }

            return DrainAsync();
        }

        public bool IsCurrent(SingleGrabLoadRequest request)
            => request != null && request.Sequence == Volatile.Read(ref _sequence);

        /// <summary>
        /// Returns true while a serialized running result may still be presented.
        /// A newer pending selection does not invalidate it because the pending result can only
        /// apply later. Explicit Invalidate calls still reject it when the owning mode changes.
        /// </summary>
        public bool CanApplyStarted(SingleGrabLoadRequest request)
            => request != null &&
               request.InvalidationVersion == Volatile.Read(ref _invalidationVersion);

        /// <summary>
        /// Invalidates the running request and drops the not-yet-started request.
        /// The running IO may finish, but its result can no longer be applied.
        /// </summary>
        public void Invalidate()
        {
            lock (_gate)
            {
                _pending = null;
                Interlocked.Increment(ref _sequence);
                Interlocked.Increment(ref _invalidationVersion);
            }
        }

        private async Task DrainAsync()
        {
            while (true)
            {
                SingleGrabLoadRequest request;
                lock (_gate)
                {
                    request = _pending;
                    _pending = null;
                    if (request == null)
                    {
                        _running = false;
                        return;
                    }
                }

                var cycle = Stopwatch.StartNew();
                await _loadAsync(request);

                int remainingMs = _minimumCycleMs - (int)cycle.ElapsedMilliseconds;
                if (remainingMs > 0)
                    await _delayAsync(remainingMs);
            }
        }
    }
}
