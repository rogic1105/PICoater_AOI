using System;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.Core.Camera
{
    internal sealed class CaptureSaveSessionGate
    {
        private readonly object _sync = new object();
        private string _sessionId = string.Empty;
        private bool _accepting;
        private int _inFlight;
        private TaskCompletionSource<bool> _idleSource;

        public void Begin(string sessionId)
        {
            lock (_sync)
            {
                if (_inFlight != 0)
                    throw new InvalidOperationException(
                        "Previous capture saves are still draining.");
                _sessionId = sessionId ?? string.Empty;
                _accepting = !string.IsNullOrWhiteSpace(_sessionId);
            }
        }

        public bool TryEnter(string sessionId)
        {
            lock (_sync)
            {
                if (!_accepting ||
                    !string.Equals(_sessionId, sessionId, StringComparison.Ordinal))
                    return false;
                if (_inFlight == 0)
                {
                    _idleSource = new TaskCompletionSource<bool>(
                        TaskCreationOptions.RunContinuationsAsynchronously);
                }
                _inFlight++;
                return true;
            }
        }

        public Task Close()
        {
            lock (_sync)
            {
                _accepting = false;
                return _inFlight == 0 || _idleSource == null
                    ? Task.CompletedTask
                    : _idleSource.Task;
            }
        }

        public void Complete()
        {
            TaskCompletionSource<bool> completed = null;
            lock (_sync)
            {
                if (_inFlight <= 0)
                    throw new InvalidOperationException(
                        "Capture save completion has no matching entry.");
                _inFlight--;
                if (_inFlight == 0) completed = _idleSource;
            }
            completed?.TrySetResult(true);
        }
    }
}
