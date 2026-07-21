using System.Threading;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Owns the latest-image token and the matching busy lease for Review image loads.
    /// Invalidating a pending load releases its lease before the debounce starts a new load.
    /// </summary>
    internal sealed class ReviewImageLoadGate
    {
        private int _sequence;
        private int _busySequence;

        public int Begin()
        {
            int sequence = Interlocked.Increment(ref _sequence);
            Volatile.Write(ref _busySequence, sequence);
            return sequence;
        }

        public bool Invalidate()
        {
            Interlocked.Increment(ref _sequence);
            return Interlocked.Exchange(ref _busySequence, 0) != 0;
        }

        public bool IsCurrent(int sequence)
            => sequence == Volatile.Read(ref _sequence);

        public bool Complete(int sequence)
            => Interlocked.CompareExchange(ref _busySequence, 0, sequence) == sequence;
    }
}
