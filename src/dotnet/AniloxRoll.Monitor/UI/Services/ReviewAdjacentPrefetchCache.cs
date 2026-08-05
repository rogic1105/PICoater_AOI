using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Services
{
    internal enum ReviewCacheAccess
    {
        Cold,
        Joined,
        Hit
    }

    internal sealed class ReviewThumbnailSnapshot
    {
        public byte[][] GrayFrames { get; set; }
        public int[] GrayWidths { get; set; }
        public int[] GrayHeights { get; set; }
        public long DecodeMs { get; set; }
        public double PixelScaleRatio { get; set; }
        public string PreviewSource { get; set; }
        public int PreviewWidth { get; set; }
        public int PreviewHeight { get; set; }

        public int ImageCount
        {
            get
            {
                int count = 0;
                if (GrayFrames != null)
                {
                    for (int i = 0; i < GrayFrames.Length; i++)
                        if (GrayFrames[i] != null) count++;
                }
                return count;
            }
        }

        public bool IsUsable => ImageCount > 0 && PixelScaleRatio > 1.0;

        public long EstimatedBytes
        {
            get
            {
                long bytes = 0;
                if (GrayFrames != null)
                {
                    for (int i = 0; i < GrayFrames.Length; i++)
                        if (GrayFrames[i] != null) bytes += GrayFrames[i].LongLength;
                }
                return bytes;
            }
        }
    }

    /// <summary>
    /// Bounded single-flight LRU used by adjacent Review prefetch. A foreground request joins
    /// an in-flight prefetch instead of reading the same files a second time.
    /// </summary>
    internal sealed class ReviewAsyncLruCache<T> : IDisposable where T : class
    {
        private sealed class CacheEntry
        {
            public T Value;
            public long Size;
            public LinkedListNode<string> Node;
        }

        private readonly object _sync = new object();
        private readonly int _maxEntries;
        private readonly long _maxSize;
        private readonly Func<T, long> _sizeOf;
        private readonly Dictionary<string, CacheEntry> _entries =
            new Dictionary<string, CacheEntry>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, Task<T>> _inflight =
            new Dictionary<string, Task<T>>(StringComparer.OrdinalIgnoreCase);
        private readonly LinkedList<string> _lru = new LinkedList<string>();
        private long _cachedSize;
        private int _generation;
        private bool _disposed;

        public ReviewAsyncLruCache(int maxEntries, long maxSize, Func<T, long> sizeOf)
        {
            if (maxEntries <= 0) throw new ArgumentOutOfRangeException(nameof(maxEntries));
            if (maxSize <= 0) throw new ArgumentOutOfRangeException(nameof(maxSize));
            _maxEntries = maxEntries;
            _maxSize = maxSize;
            _sizeOf = sizeOf ?? throw new ArgumentNullException(nameof(sizeOf));
        }

        public int Count
        {
            get { lock (_sync) return _entries.Count; }
        }

        public long CachedSize
        {
            get { lock (_sync) return _cachedSize; }
        }

        public bool TryGet(string key, out T value)
        {
            lock (_sync)
            {
                if (!_disposed && _entries.TryGetValue(key, out CacheEntry entry))
                {
                    Touch(entry);
                    value = entry.Value;
                    return true;
                }
            }
            value = null;
            return false;
        }

        public Task<T> GetOrLoadAsync(
            string key, Func<T> loader, out ReviewCacheAccess access)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException("Cache key is required.", nameof(key));
            if (loader == null) throw new ArgumentNullException(nameof(loader));

            lock (_sync)
            {
                ThrowIfDisposed();
                if (_entries.TryGetValue(key, out CacheEntry entry))
                {
                    Touch(entry);
                    access = ReviewCacheAccess.Hit;
                    return Task.FromResult(entry.Value);
                }
                if (_inflight.TryGetValue(key, out Task<T> existing))
                {
                    access = ReviewCacheAccess.Joined;
                    return existing;
                }

                int generation = _generation;
                var completion = new TaskCompletionSource<T>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                _inflight[key] = completion.Task;
                access = ReviewCacheAccess.Cold;
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
                _cachedSize = 0;
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                ClearCore();
            }
        }

        private void CompleteLoad(
            string key, Func<T> loader, int generation,
            TaskCompletionSource<T> completion)
        {
            try
            {
                T value = loader();
                lock (_sync)
                {
                    RemoveInflight(key, completion.Task);
                    if (!_disposed && generation == _generation && value != null)
                        AddOrTouch(key, value);
                }
                completion.TrySetResult(value);
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

        private void AddOrTouch(string key, T value)
        {
            long size = Math.Max(0, _sizeOf(value));
            if (size > _maxSize) return;

            if (_entries.TryGetValue(key, out CacheEntry existing))
            {
                _cachedSize -= existing.Size;
                existing.Value = value;
                existing.Size = size;
                _cachedSize += size;
                Touch(existing);
            }
            else
            {
                var node = new LinkedListNode<string>(key);
                _lru.AddFirst(node);
                _entries[key] = new CacheEntry
                {
                    Value = value,
                    Size = size,
                    Node = node
                };
                _cachedSize += size;
            }

            while (_entries.Count > _maxEntries || _cachedSize > _maxSize)
            {
                LinkedListNode<string> node = _lru.Last;
                if (node == null) break;
                CacheEntry removed = _entries[node.Value];
                _cachedSize -= removed.Size;
                _entries.Remove(node.Value);
                _lru.RemoveLast();
            }
        }

        private void Touch(CacheEntry entry)
        {
            _lru.Remove(entry.Node);
            _lru.AddFirst(entry.Node);
        }

        private void RemoveInflight(string key, Task<T> task)
        {
            if (_inflight.TryGetValue(key, out Task<T> current) &&
                ReferenceEquals(current, task))
                _inflight.Remove(key);
        }

        private void ClearCore()
        {
            _generation++;
            _entries.Clear();
            _inflight.Clear();
            _lru.Clear();
            _cachedSize = 0;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(GetType().Name);
        }
    }

    internal static class ReviewAdjacentPrefetchPolicy
    {
        public static GrabIdInfo[] Select(
            IList<GrabIdInfo> items, int currentIndex, int direction)
        {
            if (items == null || currentIndex < 0 || currentIndex >= items.Count)
                return new GrabIdInfo[0];

            int first = direction < 0 ? currentIndex - 1 : currentIndex + 1;
            int second = direction < 0 ? currentIndex + 1 : currentIndex - 1;
            var selected = new List<GrabIdInfo>(2);
            AddCopy(items, first, selected);
            AddCopy(items, second, selected);
            return selected.ToArray();
        }

        private static void AddCopy(
            IList<GrabIdInfo> items, int index, List<GrabIdInfo> selected)
        {
            if (index < 0 || index >= items.Count) return;
            GrabIdInfo source = items[index];
            if (source == null || string.IsNullOrWhiteSpace(source.GrabId)) return;
            selected.Add(new GrabIdInfo
            {
                GrabId = source.GrabId,
                Earliest = source.Earliest,
                Latest = source.Latest
            });
        }
    }
}
