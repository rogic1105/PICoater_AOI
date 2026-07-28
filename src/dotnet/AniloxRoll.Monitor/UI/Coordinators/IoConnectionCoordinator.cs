using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal sealed class IoConnectionOptions
    {
        public IoConnectionOptions(
            bool enabled,
            string model,
            string ip,
            int port,
            bool stopCaptureOnStartLow)
        {
            Enabled = enabled;
            Model = model;
            Ip = ip;
            Port = port;
            StopCaptureOnStartLow = stopCaptureOnStartLow;
        }

        public bool Enabled { get; }
        public string Model { get; }
        public string Ip { get; }
        public int Port { get; }
        public bool StopCaptureOnStartLow { get; }
    }

    /// <summary>
    /// Owns one active IO controller generation and serializes replacement or shutdown.
    /// Product capture policy remains in the Form and IoGrabController FSM.
    /// </summary>
    internal sealed class IoConnectionCoordinator
    {
        private const int ReconnectIntervalMs = 3000;
        private const int ReadWriteTimeoutMs = 500;

        private readonly SemaphoreSlim _lifecycleGate =
            new SemaphoreSlim(1, 1);

        private IoGrabController _controller;
        private Task _startTask = Task.CompletedTask;
        private int _requestedGeneration;
        private int _activeGeneration;
        private int _shutdown;

        public event Action<IoGrabController, int> StartRequested;
        public event Action<IoGrabController, int, IoStopRequestReason> StopRequested;
        public event Action<IoGrabController, int, IoState> StateChanged;
        public event Action<IoGrabController, int, bool> ConnectionChanged;
        public event Action<IoGrabController, int, IoSnapshot> IoUpdated;

        public IoGrabController CurrentController =>
            Volatile.Read(ref _controller);

        public int CurrentGeneration =>
            Volatile.Read(ref _activeGeneration);

        public Task StartAsync(IoConnectionOptions options)
        {
            return ReplaceAsync(options);
        }

        public Task RestartAsync(IoConnectionOptions options)
        {
            return ReplaceAsync(options);
        }

        public bool IsCurrent(
            IoGrabController controller,
            int generation)
        {
            return Volatile.Read(ref _shutdown) == 0 &&
                   generation == Volatile.Read(ref _requestedGeneration) &&
                   generation == Volatile.Read(ref _activeGeneration) &&
                   ReferenceEquals(
                       Volatile.Read(ref _controller),
                       controller);
        }

        private async Task ReplaceAsync(IoConnectionOptions options)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));
            if (Volatile.Read(ref _shutdown) != 0)
                return;

            int requestedGeneration =
                Interlocked.Increment(ref _requestedGeneration);
            await _lifecycleGate.WaitAsync();
            try
            {
                if (requestedGeneration !=
                    Volatile.Read(ref _requestedGeneration))
                {
                    FlowTrace.Log(
                        $"IO controller restart coalesced generation={requestedGeneration}");
                    return;
                }

                IoGrabController oldController =
                    Volatile.Read(ref _controller);
                Task oldStartTask = _startTask;
                int oldGeneration =
                    Volatile.Read(ref _activeGeneration);
                Volatile.Write(ref _controller, null);
                Volatile.Write(ref _activeGeneration, 0);
                _startTask = Task.CompletedTask;

                if (oldController != null)
                {
                    FlowTrace.Log(
                        $"IO controller stop generation={oldGeneration} reason=settings");
                    await StopAndDisposeAsync(
                        oldController,
                        oldStartTask,
                        "Restart.IO");
                }

                if (requestedGeneration !=
                        Volatile.Read(ref _requestedGeneration) ||
                    Volatile.Read(ref _shutdown) != 0)
                {
                    FlowTrace.Log(
                        $"IO controller restart coalesced generation={requestedGeneration}");
                    return;
                }

                if (!options.Enabled)
                    return;

                var controller = new IoGrabController(options.Model)
                {
                    ReconnectIntervalMs = ReconnectIntervalMs,
                    ReadWriteTimeoutMs = ReadWriteTimeoutMs,
                    StopCaptureOnStartLow =
                        options.StopCaptureOnStartLow
                };
                Volatile.Write(ref _controller, controller);
                Volatile.Write(
                    ref _activeGeneration,
                    requestedGeneration);
                AttachControllerEvents(
                    controller,
                    requestedGeneration);

                FlowTrace.Log(
                    $"IO controller start generation={requestedGeneration} " +
                    $"endpoint={options.Ip}:{options.Port}");
                _startTask = StartControllerAsync(
                    controller,
                    requestedGeneration,
                    options.Ip,
                    options.Port);
            }
            finally
            {
                _lifecycleGate.Release();
            }
        }

        private void AttachControllerEvents(
            IoGrabController controller,
            int generation)
        {
            controller.OnStartRequested += () =>
            {
                if (IsCurrent(controller, generation))
                    StartRequested?.Invoke(controller, generation);
            };
            controller.OnStopRequested += reason =>
            {
                if (IsCurrent(controller, generation))
                    StopRequested?.Invoke(
                        controller,
                        generation,
                        reason);
            };
            controller.OnStateChanged += state =>
            {
                if (IsCurrent(controller, generation))
                    StateChanged?.Invoke(
                        controller,
                        generation,
                        state);
            };
            controller.OnConnectionChanged += connected =>
            {
                if (IsCurrent(controller, generation))
                    ConnectionChanged?.Invoke(
                        controller,
                        generation,
                        connected);
            };
            controller.OnIoUpdated += snapshot =>
            {
                if (IsCurrent(controller, generation))
                    IoUpdated?.Invoke(
                        controller,
                        generation,
                        snapshot);
            };
        }

        private async Task StartControllerAsync(
            IoGrabController controller,
            int generation,
            string ip,
            int port)
        {
            try
            {
                await controller.StartAsync(ip, port);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[IO.Start generation={generation}] " +
                    $"{ex.GetType().Name}: {ex.Message}");
                if (IsCurrent(controller, generation))
                    ConnectionChanged?.Invoke(
                        controller,
                        generation,
                        false);
            }
        }

        public async Task ShutdownAsync()
        {
            Interlocked.Exchange(ref _shutdown, 1);
            Interlocked.Increment(ref _requestedGeneration);
            await _lifecycleGate.WaitAsync();
            try
            {
                IoGrabController controller =
                    Volatile.Read(ref _controller);
                Task startTask = _startTask;
                int generation =
                    Volatile.Read(ref _activeGeneration);
                Volatile.Write(ref _controller, null);
                Volatile.Write(ref _activeGeneration, 0);
                _startTask = Task.CompletedTask;
                if (controller == null)
                    return;

                FlowTrace.Log(
                    $"IO controller stop generation={generation} reason=shutdown");
                await StopAndDisposeAsync(
                    controller,
                    startTask,
                    "Shutdown.IO");
            }
            finally
            {
                _lifecycleGate.Release();
            }
        }

        private static async Task StopAndDisposeAsync(
            IoGrabController controller,
            Task startTask,
            string traceOwner)
        {
            try { await startTask; }
            catch { }

            try { await controller.StopAsync(); }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[{traceOwner}] {ex.GetType().Name}: {ex.Message}");
            }

            try { controller.Dispose(); }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[{traceOwner}.Dispose] " +
                    $"{ex.GetType().Name}: {ex.Message}");
            }
        }
    }
}
