using System;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    internal enum CaptureStopState
    {
        Idle,
        WaitingForFirstSet,
        ArmedIo,
        ArmedTime,
        ArmedHeight,
        StopPending,
        Disposed
    }

    internal enum CaptureStopTrigger
    {
        IoRequest,
        TimerElapsed,
        HeightReached
    }

    internal sealed class CaptureStopRequest
    {
        public CaptureStopRequest(
            CaptureStopCondition condition,
            CaptureStopTrigger trigger,
            bool ioControlled,
            string grabId,
            int limit,
            int observed,
            IoStopRequestReason? ioReason)
        {
            Condition = condition;
            Trigger = trigger;
            IsIoControlled = ioControlled;
            GrabId = grabId ?? string.Empty;
            Limit = limit;
            Observed = observed;
            IoReason = ioReason;
        }

        public CaptureStopCondition Condition { get; }
        public CaptureStopTrigger Trigger { get; }
        public bool IsIoControlled { get; }
        public string GrabId { get; }
        public int Limit { get; }
        public int Observed { get; }
        public IoStopRequestReason? IoReason { get; }

        public bool DrainIoTail =>
            IoReason == IoStopRequestReason.StartLow;

        public bool NotifyFixedGrabCompleted =>
            IsIoControlled &&
            (Condition == CaptureStopCondition.Time ||
             Condition == CaptureStopCondition.Height);

        public string CreateIntentLine()
        {
            if (Trigger == CaptureStopTrigger.HeightReached)
            {
                return
                    $"auto:抓取停止 condition=Height rows={Observed} " +
                    $"limit={Limit}px grab={GrabId}";
            }

            if (Trigger == CaptureStopTrigger.IoRequest)
            {
                return
                    $"io:stop reason={IoReason} condition={Condition}";
            }

            return
                $"auto:抓取停止 condition={Condition} " +
                $"limit={Limit}s grab={GrabId}";
        }
    }

    /// <summary>
    /// Owns the snapshotted stop condition and accepts exactly one terminal
    /// trigger for a capture. Actual grab shutdown remains in the Form.
    ///
    /// State + Event -> Next State + Action:
    /// Idle + Arm(IO) -> ArmedIo + wait for IO stop request.
    /// Idle + Arm(Time) -> WaitingForFirstSet + wait without a timer.
    /// Idle + Arm(Height) -> ArmedHeight + watch common rows.
    /// WaitingForFirstSet + FirstSetReady -> ArmedTime + arm fixed timer.
    /// WaitingForFirstSet + FirstSetFailed -> StopPending + cancel timer.
    /// ArmedIo + IO request -> StopPending + request stop; drain only StartLow.
    /// WaitingForFirstSet/ArmedTime/ArmedHeight + IO request -> unchanged + ignore.
    /// ArmedTime + TimerElapsed -> StopPending + request stop.
    /// ArmedHeight + CommonRowsReached -> StopPending + request stop.
    /// StopPending + any terminal trigger -> StopPending + ignore duplicate.
    /// Any active state + Complete/Cancel -> Idle + disarm timer.
    /// Any state + Dispose -> Disposed + disarm timer.
    /// </summary>
    internal sealed class CaptureStopCoordinator : IDisposable
    {
        private readonly object _gate = new object();
        private readonly Action<CaptureStopRequest> _stopRequested;
        private readonly GrabDurationCoordinator _duration;

        private CaptureStopState _state = CaptureStopState.Idle;
        private CaptureStopCondition _condition =
            CaptureStopCondition.IoSignal;
        private bool _ioControlled;
        private int _configuredSeconds;
        private int _heightLimitRows;
        private string _grabId = string.Empty;
        private bool _disposed;

        public CaptureStopCoordinator(
            Action<CaptureStopRequest> stopRequested)
        {
            _stopRequested = stopRequested ??
                throw new ArgumentNullException(nameof(stopRequested));
            _duration = new GrabDurationCoordinator(HandleTimerElapsed);
        }

        public CaptureStopState State
        {
            get { lock (_gate) return _state; }
        }

        public CaptureStopCondition Condition
        {
            get { lock (_gate) return _condition; }
        }

        public bool Arm(
            CaptureStopCondition condition,
            bool ioControlled,
            int configuredSeconds,
            int boundaryGraceSeconds,
            int heightLimitRows,
            string grabId)
        {
            if (configuredSeconds < 1)
                throw new ArgumentOutOfRangeException(nameof(configuredSeconds));
            if (boundaryGraceSeconds < 0)
                throw new ArgumentOutOfRangeException(nameof(boundaryGraceSeconds));
            if (heightLimitRows < 1)
                throw new ArgumentOutOfRangeException(nameof(heightLimitRows));

            string logLine;
            bool waitsForFirstSet;
            lock (_gate)
            {
                ThrowIfDisposed();
                _duration.Disarm();
                _condition = condition;
                _ioControlled = ioControlled;
                _configuredSeconds = configuredSeconds;
                _heightLimitRows = heightLimitRows;
                _grabId = grabId ?? string.Empty;

                string source = ioControlled ? "io" : "manual";
                switch (condition)
                {
                    case CaptureStopCondition.Height:
                        _state = CaptureStopState.ArmedHeight;
                        waitsForFirstSet = false;
                        logLine =
                            $"grab stop armed condition=height " +
                            $"limit={heightLimitRows}px source={source} " +
                            $"grab={_grabId}";
                        break;

                    case CaptureStopCondition.IoSignal:
                        _state = CaptureStopState.ArmedIo;
                        waitsForFirstSet = false;
                        logLine =
                            $"grab stop armed condition={condition} " +
                            $"limit=io-low " +
                            $"configured={configuredSeconds}s " +
                            $"grace=unused " +
                            $"source={source} grab={_grabId}";
                        break;

                    default:
                        _state = CaptureStopState.WaitingForFirstSet;
                        waitsForFirstSet = true;
                        logLine =
                            $"grab stop waiting condition=Time " +
                            $"configured={configuredSeconds}s " +
                            $"source={source} grab={_grabId}";
                        break;
                }
            }

            FlowTrace.Log(logLine);
            return waitsForFirstSet;
        }

        public bool ActivateTimeAfterFirstSet()
        {
            string logLine;
            lock (_gate)
            {
                if (_disposed ||
                    _state != CaptureStopState.WaitingForFirstSet)
                    return false;

                _state = CaptureStopState.ArmedTime;
                _duration.Arm(_configuredSeconds);
                string source = _ioControlled ? "io" : "manual";
                logLine =
                    $"grab stop armed condition=Time " +
                    $"limit={_configuredSeconds}s " +
                    $"configured={_configuredSeconds}s grace=0s " +
                    $"source={source} start=first-set grab={_grabId}";
            }

            FlowTrace.Log(logLine);
            return true;
        }

        public bool FailFirstSet()
        {
            lock (_gate)
            {
                if (_disposed ||
                    _state != CaptureStopState.WaitingForFirstSet)
                    return false;

                _state = CaptureStopState.StopPending;
                _duration.Disarm();
                return true;
            }
        }

        public bool TryRequestIoStop(
            IoStopRequestReason reason,
            out CaptureStopRequest request)
        {
            lock (_gate)
            {
                request = null;
                if (_disposed ||
                    _state != CaptureStopState.ArmedIo ||
                    _condition != CaptureStopCondition.IoSignal)
                    return false;

                _state = CaptureStopState.StopPending;
                _duration.Disarm();
                request = CreateRequestLocked(
                    CaptureStopTrigger.IoRequest,
                    0,
                    0,
                    reason);
                return true;
            }
        }

        public void ObserveCommonRows(int commonRows)
        {
            CaptureStopRequest request = null;
            lock (_gate)
            {
                if (_disposed ||
                    _state != CaptureStopState.ArmedHeight ||
                    commonRows < _heightLimitRows)
                    return;

                _state = CaptureStopState.StopPending;
                request = CreateRequestLocked(
                    CaptureStopTrigger.HeightReached,
                    _heightLimitRows,
                    commonRows,
                    null);
            }

            _stopRequested(request);
        }

        internal void HandleTimerElapsed(int elapsedSeconds)
        {
            CaptureStopRequest request = null;
            lock (_gate)
            {
                if (_disposed ||
                    _state != CaptureStopState.ArmedTime)
                    return;

                _state = CaptureStopState.StopPending;
                request = CreateRequestLocked(
                    CaptureStopTrigger.TimerElapsed,
                    elapsedSeconds,
                    elapsedSeconds,
                    null);
            }

            _stopRequested(request);
        }

        public void CompleteStop()
        {
            ResetToIdle();
        }

        public void Cancel()
        {
            ResetToIdle();
        }

        private CaptureStopRequest CreateRequestLocked(
            CaptureStopTrigger trigger,
            int limit,
            int observed,
            IoStopRequestReason? ioReason)
        {
            return new CaptureStopRequest(
                _condition,
                trigger,
                _ioControlled,
                _grabId,
                limit,
                observed,
                ioReason);
        }

        private void ResetToIdle()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _duration.Disarm();
                _state = CaptureStopState.Idle;
                _configuredSeconds = 0;
                _heightLimitRows = 0;
                _grabId = string.Empty;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _state = CaptureStopState.Disposed;
                _duration.Dispose();
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(CaptureStopCoordinator));
        }
    }
}
