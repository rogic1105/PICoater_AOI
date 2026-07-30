using System;
using System.Diagnostics;
using System.Threading;
using LightBridge.Core;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal sealed class LightConnectionSnapshot
    {
        public LightConnectionSnapshot(
            bool enabled,
            bool connected,
            bool hasProbed,
            bool probeInFlight,
            int reconnectSeconds)
        {
            Enabled = enabled;
            Connected = connected;
            HasProbed = hasProbed;
            ProbeInFlight = probeInFlight;
            ReconnectSeconds = reconnectSeconds;
        }

        public bool Enabled { get; }
        public bool Connected { get; }
        public bool HasProbed { get; }
        public bool ProbeInFlight { get; }
        public int ReconnectSeconds { get; }
    }

    /// <summary>
    /// Owns the light controller lifecycle, background probing, and reconnect policy.
    /// It deliberately has no WinForms dependency; the form renders snapshots.
    /// </summary>
    internal sealed class LightConnectionCoordinator : IDisposable
    {
        private const int ProbeIntervalTicks = 4;
        private const int FullScanEveryAttempts = 5;

        private readonly object _sync = new object();
        private readonly AutoResetEvent _probeSignal =
            new AutoResetEvent(false);
        private readonly Thread _probeThread;
        private readonly int _telemetryTickMs;

        private LightController _controller;
        private Action _pendingProbe;
        private bool _enabled;
        private bool _hasProbed;
        private bool _probeInFlight;
        private bool _disposed;
        private string _preferredPort;
        private int _channel;
        private int _probeTickCounter;
        private int _reconnectAttemptCount;
        private int _generation;

        public LightConnectionCoordinator(int telemetryTickMs)
        {
            if (telemetryTickMs <= 0)
                throw new ArgumentOutOfRangeException(nameof(telemetryTickMs));

            _telemetryTickMs = telemetryTickMs;
            _probeThread = new Thread(ProbeWorkerLoop)
            {
                IsBackground = true,
                Name = "LightProbe"
            };
            _probeThread.Start();
        }

        public event Action StateChanged;
        public event Action<string> ActivePortChanged;

        public LightConnectionSnapshot Snapshot
        {
            get
            {
                lock (_sync)
                {
                    return new LightConnectionSnapshot(
                        _enabled,
                        _controller != null && _controller.IsConnected,
                        _hasProbed,
                        _probeInFlight,
                        CountdownSeconds(_probeTickCounter));
                }
            }
        }

        public void Start(string preferredPort, int channel)
        {
            LightController staleController;
            int generation;

            lock (_sync)
            {
                ThrowIfDisposed();
                generation = ++_generation;
                _enabled = true;
                _preferredPort = preferredPort;
                _channel = channel;
                _hasProbed = false;
                _probeInFlight = true;
                _probeTickCounter = 0;
                _reconnectAttemptCount = 0;
                staleController = _controller;
                _controller = null;
            }

            DisposeController(staleController);
            RaiseStateChanged();
            QueueProbe(() =>
                RunInitialProbe(generation, preferredPort, channel));
        }

        public void Disable()
        {
            LightController staleController;
            lock (_sync)
            {
                if (_disposed) return;
                ++_generation;
                _enabled = false;
                _hasProbed = false;
                _probeInFlight = false;
                _probeTickCounter = 0;
                _reconnectAttemptCount = 0;
                staleController = _controller;
                _controller = null;
            }

            DisposeController(staleController);
            RaiseStateChanged();
        }

        public void Tick()
        {
            int generation = 0;
            LightController controller = null;
            string preferredPort = null;
            int channel = 0;
            bool startProbe = false;

            lock (_sync)
            {
                if (_disposed || !_enabled) return;
                if (++_probeTickCounter >= ProbeIntervalTicks)
                {
                    _probeTickCounter = 0;
                    if (!_probeInFlight)
                    {
                        _probeInFlight = true;
                        generation = _generation;
                        controller = _controller;
                        preferredPort = _preferredPort;
                        channel = _channel;
                        startProbe = true;
                    }
                }
            }

            if (startProbe)
            {
                QueueProbe(() => RunPeriodicProbe(
                    generation, controller, preferredPort, channel));
            }
        }

        public void TurnOn(int channel, int brightness)
        {
            LightController controller = GetConnectedController();
            controller?.TurnOn(channel, brightness);
        }

        public void TurnOff(int channel)
        {
            LightController controller = GetConnectedController();
            controller?.TurnOff(channel);
        }

        public void SetBrightness(int channel, int brightness)
        {
            LightController controller = GetConnectedController();
            controller?.SetBrightness(channel, brightness);
        }

        internal static bool ShouldRunFullPortScan(int reconnectAttempt)
        {
            return reconnectAttempt > 0 &&
                   reconnectAttempt % FullScanEveryAttempts == 0;
        }

        private void RunInitialProbe(
            int generation,
            string preferredPort,
            int channel)
        {
            var candidate = new LightController();
            string found = null;
            try
            {
                found = candidate.AutoDetect(preferredPort, channel);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[Light.InitialProbe] {ex.GetType().Name}: {ex.Message}");
            }

            bool accepted = false;
            lock (_sync)
            {
                if (!_disposed && _enabled && generation == _generation)
                {
                    _hasProbed = true;
                    _probeInFlight = false;
                    if (found != null)
                    {
                        _controller = candidate;
                        _reconnectAttemptCount = 0;
                        accepted = true;
                    }
                }
            }

            if (!accepted)
                DisposeController(candidate);

            if (accepted &&
                !string.Equals(
                    found,
                    preferredPort,
                    StringComparison.OrdinalIgnoreCase))
            {
                ActivePortChanged?.Invoke(found);
            }

            RaiseStateChanged();
        }

        private void RunPeriodicProbe(
            int generation,
            LightController controller,
            string preferredPort,
            int channel)
        {
            LightController replacement = null;
            string found = null;
            bool replace = false;

            try
            {
                if (controller != null && controller.IsConnected)
                {
                    if (controller.Probe(channel))
                    {
                        lock (_sync)
                        {
                            if (generation == _generation)
                                _reconnectAttemptCount = 0;
                        }
                    }
                }
                else
                {
                    int attempt;
                    lock (_sync)
                    {
                        if (_disposed || !_enabled || generation != _generation)
                            return;
                        attempt = ++_reconnectAttemptCount;
                    }

                    bool fullScan = ShouldRunFullPortScan(attempt);
                    replacement = new LightController();
                    found = fullScan
                        ? replacement.AutoDetect(preferredPort, channel)
                        : (replacement.TryConnect(preferredPort, channel)
                            ? preferredPort
                            : null);
                    if (found != null)
                    {
                        lock (_sync)
                        {
                            if (!_disposed && _enabled && generation == _generation)
                            {
                                _controller = replacement;
                                _reconnectAttemptCount = 0;
                                replace = true;
                            }
                        }
                    }
                }
            }
            catch
            {
                // Probe failures are represented by the disconnected snapshot.
            }
            finally
            {
                lock (_sync)
                {
                    if (generation == _generation)
                        _probeInFlight = false;
                }
            }

            if (replace)
            {
                DisposeController(controller);
                replacement = null;
                if (!string.Equals(
                    found,
                    preferredPort,
                    StringComparison.OrdinalIgnoreCase))
                {
                    ActivePortChanged?.Invoke(found);
                }
            }

            DisposeController(replacement);
            RaiseStateChanged();
        }

        private LightController GetConnectedController()
        {
            lock (_sync)
            {
                if (_disposed ||
                    _controller == null ||
                    !_controller.IsConnected)
                {
                    return null;
                }

                return _controller;
            }
        }

        private int CountdownSeconds(int elapsedTicks)
        {
            return Math.Max(
                1,
                (int)Math.Ceiling(
                    (ProbeIntervalTicks - elapsedTicks) *
                    _telemetryTickMs /
                    1000.0));
        }

        private void RaiseStateChanged()
        {
            Action handler;
            lock (_sync)
            {
                if (_disposed) return;
                handler = StateChanged;
            }
            handler?.Invoke();
        }

        private void QueueProbe(Action action)
        {
            lock (_sync)
            {
                if (_disposed)
                    return;
                _pendingProbe = action;
            }
            _probeSignal.Set();
        }

        private void ProbeWorkerLoop()
        {
            while (true)
            {
                _probeSignal.WaitOne();

                Action action;
                lock (_sync)
                {
                    if (_disposed)
                        return;
                    action = _pendingProbe;
                    _pendingProbe = null;
                }

                try
                {
                    action?.Invoke();
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning(
                        $"[Light.ProbeWorker] {ex.GetType().Name}: {ex.Message}");
                }
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(LightConnectionCoordinator));
        }

        public void Dispose()
        {
            LightController staleController;
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                ++_generation;
                _enabled = false;
                _probeInFlight = false;
                staleController = _controller;
                _controller = null;
                _pendingProbe = null;
            }

            _probeSignal.Set();
            bool workerStopped =
                Thread.CurrentThread == _probeThread ||
                _probeThread.Join(15000);
            if (!workerStopped)
            {
                Trace.TraceWarning(
                    "[Light.Dispose] timed out waiting for probe worker");
            }

            DisposeController(staleController);
            if (workerStopped)
                _probeSignal.Dispose();
        }

        private static void DisposeController(LightController controller)
        {
            if (controller == null) return;
            try { controller.Dispose(); }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[Light.Dispose] {ex.GetType().Name}: {ex.Message}");
            }
        }
    }
}
