using System;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal sealed class ReviewCurveLoadRequest
    {
        public string GrabId { get; set; }
        public DateTime HintFrom { get; set; }
        public DateTime HintTo { get; set; }
        internal int Sequence { get; set; }
    }

    /// <summary>
    /// Serializes review curve loads and retains only the latest request that has not started.
    /// The running request may finish its IO, but IsCurrent prevents stale results from applying.
    /// </summary>
    internal sealed class ReviewCurveLoadCoordinator
    {
        private readonly object _gate = new object();
        private readonly Func<ReviewCurveLoadRequest, Task> _loadAsync;
        private ReviewCurveLoadRequest _pending;
        private bool _running;
        private int _sequence;

        public ReviewCurveLoadCoordinator(Func<ReviewCurveLoadRequest, Task> loadAsync)
        {
            _loadAsync = loadAsync ?? throw new ArgumentNullException(nameof(loadAsync));
        }

        public Task Enqueue(string grabId, DateTime hintFrom, DateTime hintTo)
        {
            lock (_gate)
            {
                _pending = new ReviewCurveLoadRequest
                {
                    GrabId = grabId,
                    HintFrom = hintFrom,
                    HintTo = hintTo,
                    Sequence = Interlocked.Increment(ref _sequence)
                };
                if (_running)
                    return Task.CompletedTask;
                _running = true;
            }

            return DrainAsync();
        }

        public bool IsCurrent(ReviewCurveLoadRequest request)
            => request != null && request.Sequence == Volatile.Read(ref _sequence);

        private async Task DrainAsync()
        {
            while (true)
            {
                ReviewCurveLoadRequest request;
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

                await _loadAsync(request);
            }
        }
    }
}
