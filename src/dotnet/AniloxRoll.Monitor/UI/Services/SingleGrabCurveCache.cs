using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Services
{
    /// <summary>Immutable raw curves for one grab before view-time rescaling.</summary>
    internal sealed class SingleGrabCurveProfile
    {
        public SingleGrabCurveProfile(
            float[][] mean, float[][] max, int captureCount,
            string storageSource, long lookupMs, long mergeMs, long summaryMs)
            : this(mean, max, null, null, captureCount,
                storageSource, lookupMs, mergeMs, summaryMs)
        {
        }

        public SingleGrabCurveProfile(
            float[][] mean, float[][] max, float[] rowMean, float[] rowMax,
            int captureCount, string storageSource,
            long lookupMs, long mergeMs, long summaryMs,
            int matchedCameraCount = 0, string alignmentMode = null)
        {
            Mean = mean ?? new float[0][];
            Max = max ?? new float[0][];
            RowMean = rowMean;
            RowMax = rowMax;
            CaptureCount = captureCount;
            StorageSource = storageSource ?? "bins";
            LookupMs = lookupMs;
            MergeMs = mergeMs;
            SummaryMs = summaryMs;
            MatchedCameraCount = matchedCameraCount;
            AlignmentMode = alignmentMode ?? "unknown";
            EstimatedBytes = EstimateBytes(Mean) + EstimateBytes(Max) +
                EstimateBytes(RowMean) + EstimateBytes(RowMax);
        }

        public float[][] Mean { get; }
        public float[][] Max { get; }
        public float[] RowMean { get; }
        public float[] RowMax { get; }
        public int CaptureCount { get; }
        public string StorageSource { get; }
        public long LookupMs { get; }
        public long MergeMs { get; }
        public long SummaryMs { get; }
        public int MatchedCameraCount { get; }
        public string AlignmentMode { get; }
        public long EstimatedBytes { get; }

        private static long EstimateBytes(float[][] arrays)
        {
            long bytes = 0;
            for (int i = 0; i < arrays.Length; i++)
                if (arrays[i] != null) bytes += (long)arrays[i].Length * sizeof(float);
            return bytes;
        }

        private static long EstimateBytes(float[] array) =>
            array == null ? 0 : (long)array.Length * sizeof(float);
    }

    /// <summary>
    /// Bounded LRU for merged single-grab curves. Concurrent requests for the same
    /// key share one load so foreground selection can join an adjacent prefetch.
    /// </summary>
    internal sealed class SingleGrabCurveCache : IDisposable
    {
        private sealed class CacheEntry
        {
            public SingleGrabCurveProfile Profile;
            public LinkedListNode<string> Node;
        }

        private readonly object _sync = new object();
        private readonly int _maxEntries;
        private readonly long _maxBytes;
        private readonly Dictionary<string, CacheEntry> _entries =
            new Dictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, Task<SingleGrabCurveProfile>> _inflight =
            new Dictionary<string, Task<SingleGrabCurveProfile>>(StringComparer.OrdinalIgnoreCase);
        private readonly LinkedList<string> _lru = new LinkedList<string>();

        private long _cachedBytes;
        private int _generation;
        private bool _disposed;

        public SingleGrabCurveCache(int maxEntries, long maxBytes)
        {
            if (maxEntries <= 0) throw new ArgumentOutOfRangeException(nameof(maxEntries));
            if (maxBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxBytes));
            _maxEntries = maxEntries;
            _maxBytes = maxBytes;
        }

        public static string BuildKey(
            string root, GrabIdInfo info, int cameraCount)
        {
            if (string.IsNullOrWhiteSpace(root))
                throw new ArgumentException("Root path is required.", nameof(root));
            if (info == null) throw new ArgumentNullException(nameof(info));
            string normalizedRoot = Path.GetFullPath(root).TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return normalizedRoot + "|" + info.GrabId + "|" + info.Earliest.Ticks + "|" +
                info.Latest.Ticks + "|" + cameraCount;
        }

        public long CachedBytes
        {
            get { lock (_sync) return _cachedBytes; }
        }

        public int Count
        {
            get { lock (_sync) return _entries.Count; }
        }

        public bool TryGet(string key, out SingleGrabCurveProfile profile)
        {
            lock (_sync)
            {
                if (!_disposed && _entries.TryGetValue(key, out CacheEntry entry))
                {
                    _lru.Remove(entry.Node);
                    _lru.AddFirst(entry.Node);
                    profile = entry.Profile;
                    return true;
                }
            }

            profile = null;
            return false;
        }

        public Task<SingleGrabCurveProfile> GetOrLoadAsync(
            string key, Func<SingleGrabCurveProfile> loader)
        {
            if (string.IsNullOrEmpty(key)) throw new ArgumentException("Cache key is required.", nameof(key));
            if (loader == null) throw new ArgumentNullException(nameof(loader));

            lock (_sync)
            {
                ThrowIfDisposed();
                if (_entries.TryGetValue(key, out CacheEntry entry))
                {
                    _lru.Remove(entry.Node);
                    _lru.AddFirst(entry.Node);
                    return Task.FromResult(entry.Profile);
                }

                if (_inflight.TryGetValue(key, out Task<SingleGrabCurveProfile> existing))
                    return existing;

                int generation = _generation;
                var completion = new TaskCompletionSource<SingleGrabCurveProfile>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                _inflight[key] = completion.Task;
                Task.Run(() => CompleteLoad(key, loader, generation, completion));
                return completion.Task;
            }
        }

        public void Clear()
        {
            lock (_sync)
            {
                _generation++;
                _entries.Clear();
                _inflight.Clear();
                _lru.Clear();
                _cachedBytes = 0;
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                _generation++;
                _entries.Clear();
                _inflight.Clear();
                _lru.Clear();
                _cachedBytes = 0;
            }
        }

        private void CompleteLoad(
            string key,
            Func<SingleGrabCurveProfile> loader,
            int generation,
            TaskCompletionSource<SingleGrabCurveProfile> completion)
        {
            try
            {
                SingleGrabCurveProfile profile = loader();
                lock (_sync)
                {
                    RemoveInflight(key, completion.Task);
                    if (!_disposed && generation == _generation && profile != null)
                        AddOrTouch(key, profile);
                }
                completion.TrySetResult(profile);
            }
            catch (OperationCanceledException)
            {
                lock (_sync) RemoveInflight(key, completion.Task);
                completion.TrySetCanceled();
            }
            catch (Exception ex)
            {
                lock (_sync) RemoveInflight(key, completion.Task);
                completion.TrySetException(ex);
            }
        }

        private void RemoveInflight(string key, Task<SingleGrabCurveProfile> task)
        {
            if (_inflight.TryGetValue(key, out Task<SingleGrabCurveProfile> current) &&
                ReferenceEquals(current, task))
                _inflight.Remove(key);
        }

        private void AddOrTouch(string key, SingleGrabCurveProfile profile)
        {
            if (profile.EstimatedBytes > _maxBytes) return;

            if (_entries.TryGetValue(key, out CacheEntry existing))
            {
                _cachedBytes -= existing.Profile.EstimatedBytes;
                existing.Profile = profile;
                _cachedBytes += profile.EstimatedBytes;
                _lru.Remove(existing.Node);
                _lru.AddFirst(existing.Node);
            }
            else
            {
                var node = new LinkedListNode<string>(key);
                _lru.AddFirst(node);
                _entries[key] = new CacheEntry { Profile = profile, Node = node };
                _cachedBytes += profile.EstimatedBytes;
            }

            while (_entries.Count > _maxEntries || _cachedBytes > _maxBytes)
            {
                LinkedListNode<string> node = _lru.Last;
                if (node == null) break;
                CacheEntry removed = _entries[node.Value];
                _cachedBytes -= removed.Profile.EstimatedBytes;
                _entries.Remove(node.Value);
                _lru.RemoveLast();
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(SingleGrabCurveCache));
        }
    }
}
