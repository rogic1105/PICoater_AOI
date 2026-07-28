using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.DvtRunner
{
    internal sealed class FlowLogMonitor
    {
        private readonly string _logDirectory;
        private readonly Dictionary<string, long> _initialLengths =
            new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, long> _offsets =
            new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        private readonly Queue<string> _pendingLines = new Queue<string>();

        public FlowLogMonitor(string logDirectory)
        {
            _logDirectory = logDirectory;
        }

        public event Action<string> LineObserved;

        public void BeginSession()
        {
            _initialLengths.Clear();
            _offsets.Clear();
            _pendingLines.Clear();
            if (!Directory.Exists(_logDirectory)) return;

            foreach (string path in Directory.GetFiles(_logDirectory, "trace-*.log"))
            {
                try { _initialLengths[path] = new FileInfo(path).Length; }
                catch (IOException) { }
            }
        }

        public async Task<string> WaitForAsync(
            string pattern,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            var regex = new Regex(pattern, RegexOptions.CultureInvariant);
            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ReadAvailableLines();
                while (_pendingLines.Count > 0)
                {
                    string line = _pendingLines.Dequeue();
                    LineObserved?.Invoke(line);
                    if (regex.IsMatch(line)) return line;
                }
                await Task.Delay(120, cancellationToken);
            }
            throw new TimeoutException("Flow evidence timed out: " + pattern);
        }

        private void ReadAvailableLines()
        {
            if (!Directory.Exists(_logDirectory)) return;
            string path = Directory.GetFiles(_logDirectory, "trace-*.log")
                .Select(p => new FileInfo(p))
                .OrderByDescending(f => f.LastWriteTimeUtc)
                .Select(f => f.FullName)
                .FirstOrDefault();
            if (path == null) return;

            long offset;
            if (!_offsets.TryGetValue(path, out offset))
            {
                long initial;
                offset = _initialLengths.TryGetValue(path, out initial) ? initial : 0L;
                _offsets[path] = offset;
            }

            FileInfo info;
            try { info = new FileInfo(path); }
            catch (IOException) { return; }
            if (!info.Exists || info.Length <= offset) return;
            if (info.Length < offset) offset = 0;

            try
            {
                using (var stream = new FileStream(
                    path, FileMode.Open, FileAccess.Read,
                    FileShare.ReadWrite | FileShare.Delete))
                {
                    stream.Seek(offset, SeekOrigin.Begin);
                    using (var reader = new StreamReader(
                        stream, Encoding.UTF8, true, 4096, leaveOpen: true))
                    {
                        string line;
                        while ((line = reader.ReadLine()) != null)
                            _pendingLines.Enqueue(line);
                    }
                    _offsets[path] = stream.Length;
                }
            }
            catch (IOException)
            {
                // The writer may be rotating the log; retry on the next poll.
            }
        }
    }
}
