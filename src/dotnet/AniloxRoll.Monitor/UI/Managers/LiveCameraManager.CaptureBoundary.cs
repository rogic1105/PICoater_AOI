using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        private enum CapturePhaseEligibility
        {
            Invalid,
            Synchronizing,
            Verified
        }

        // State + Event -> Next State + Action:
        // Closed + Arm -> AwaitingHeadProbe + discard one boundary frame per camera.
        // AwaitingHeadProbe + aligned complete probe -> AwaitingFirstSet + allow product frames.
        // AwaitingHeadProbe + invalid complete probe -> Rejected + close gate and fail waiter.
        // AwaitingFirstSet + aligned complete set -> Active + complete waiter.
        // AwaitingFirstSet + invalid complete set -> Rejected + close gate and fail waiter.
        // Any + Stop/rearm -> Closed/new AwaitingHeadProbe + invalidate the previous waiter.
        private enum CaptureBoundaryAdmissionState
        {
            Closed,
            AwaitingHeadProbe,
            AwaitingFirstSet,
            Active,
            Rejected
        }

        private const int IdleCapturePreparationRetryMs = 3000;
        private readonly object _captureBoundaryLock = new object();
        private CapturePhaseEligibility _capturePhaseEligibility =
            CapturePhaseEligibility.Invalid;
        private CaptureBoundaryAdmissionState _captureBoundaryAdmissionState =
            CaptureBoundaryAdmissionState.Closed;
        private readonly Dictionary<int, long> _headBoundaryTicks =
            new Dictionary<int, long>();
        private readonly Dictionary<int, long> _firstAcceptedTicks =
            new Dictionary<int, long>();
        private TaskCompletionSource<bool> _firstSetReadyCompletion;
        private readonly HashSet<int> _firstPendingCameraIds =
            new HashSet<int>();
        private readonly HashSet<int> _headBoundaryPendingCameraIds =
            new HashSet<int>();
        private readonly Dictionary<int, long> _lastAcceptedTicks =
            new Dictionary<int, long>();
        private readonly Dictionary<int, int> _completedRowsByCamera =
            new Dictionary<int, int>();
        private int _lastPublishedCommonRows;
        private readonly Dictionary<int, long> _tailBaselineTicks =
            new Dictionary<int, long>();
        private readonly Dictionary<int, long> _tailAcceptedTicks =
            new Dictionary<int, long>();
        private readonly HashSet<int> _tailPendingCameraIds =
            new HashSet<int>();
        private TaskCompletionSource<bool> _tailDrainCompletion;
        private bool _tailDrainActive;
        private string _captureStartPath = "none";
        private int _idleCapturePreparationInFlight;
        private long _idleCapturePreparationNotBeforeTimestamp;

        public bool IsCaptureTailDrainActive
        {
            get
            {
                lock (_captureBoundaryLock)
                    return _tailDrainActive;
            }
        }

        public bool TryGetCaptureStandbyReady(out string reason)
        {
            if (!IsAllocated || IsReleasing || _isParameterReconfiguring ||
                _isCaptureSynchronizing || IsLiveGrabbing)
            {
                reason = "manager-busy";
                return false;
            }

            AniloxCamera[] targets;
            try
            {
                targets = _cameras
                    .Where(cam => cam != null && cam.IsConnected)
                    .ToArray();
            }
            catch (Exception)
            {
                reason = "camera-snapshot-failed";
                return false;
            }

            if (targets.Length == 0)
            {
                reason = "no-targets";
                return false;
            }

            return CanUseVerifiedStandby(
                targets,
                out reason,
                false,
                "acquisition io-ready phase");
        }

        public int GetCaptureBoundaryGraceSeconds()
        {
            int framePeriodMs = Math.Max(1, GetMaxFramePeriodMs());
            return Math.Max(2, (int)Math.Ceiling(framePeriodMs / 1000.0) + 1);
        }

        public int GetCaptureFirstSetTimeoutMs()
        {
            return Math.Max(3000, Math.Min(30000, GetMaxFramePeriodMs() * 4 + 500));
        }

        /// <summary>
        /// Waits for the first complete, phase-validated frame set after the current gate opens.
        /// Stop/rearm completes the previous waiter with false.
        /// </summary>
        public async Task<bool> WaitForCaptureFirstSetReadyAsync(int timeoutMs)
        {
            TaskCompletionSource<bool> completion;
            lock (_captureBoundaryLock)
                completion = _firstSetReadyCompletion;
            if (completion == null) return false;

            int boundedTimeout = Math.Max(1, timeoutMs);
            Task winner = await Task.WhenAny(
                completion.Task,
                Task.Delay(boundedTimeout)).ConfigureAwait(false);
            if (winner != completion.Task)
            {
                FlowTrace.Log(
                    $"capture first-set timeout limitMs={boundedTimeout}");
                return false;
            }
            return await completion.Task.ConfigureAwait(false);
        }

        public async Task<bool> DrainIoTailAsync()
        {
            TaskCompletionSource<bool> completion;
            int timeoutMs;
            lock (_captureBoundaryLock)
            {
                if (!_captureGateOpen || !IsLiveGrabbing)
                    return false;
                if (_tailDrainActive)
                {
                    completion = _tailDrainCompletion;
                    timeoutMs = GetTailDrainTimeoutMs();
                }
                else
                {
                    _tailDrainActive = true;
                    _tailPendingCameraIds.Clear();
                    _tailBaselineTicks.Clear();
                    _tailAcceptedTicks.Clear();
                    foreach (AniloxCamera cam in _cameras)
                    {
                        if (cam == null || !cam.IsConnected) continue;
                        _tailPendingCameraIds.Add(cam.CameraId);
                        long baseline;
                        _tailBaselineTicks[cam.CameraId] =
                            _lastAcceptedTicks.TryGetValue(cam.CameraId, out baseline)
                                ? baseline
                                : cam.LastFrameStartTicks;
                    }
                    completion = new TaskCompletionSource<bool>(
                        TaskCreationOptions.RunContinuationsAsynchronously);
                    _tailDrainCompletion = completion;
                    timeoutMs = GetTailDrainTimeoutMs();
                    FlowTrace.Log(
                        $"capture tail begin cams={string.Join(",", _tailPendingCameraIds)} " +
                        $"timeoutMs={timeoutMs}");
                    if (_tailPendingCameraIds.Count == 0)
                        completion.TrySetResult(true);
                }
            }

            Task winner = await Task.WhenAny(
                completion.Task,
                Task.Delay(timeoutMs));
            bool completed = winner == completion.Task && completion.Task.Result;
            string pending;
            lock (_captureBoundaryLock)
                pending = string.Join(",", _tailPendingCameraIds);
            FlowTrace.Log(
                $"capture tail {(completed ? "complete" : "timeout")} pending={pending}");
            if (!completed)
                InvalidateCapturePhase("tail-timeout:" + pending);
            return completed;
        }

        private int GetTailDrainTimeoutMs()
        {
            return Math.Max(1500, Math.Min(10000, GetMaxFramePeriodMs() * 3 + 500));
        }

        private void NotifyCaptureFrameCompleted(int cameraId, long frameStartTicks)
        {
            TaskCompletionSource<bool> completion = null;
            bool accepted = false;
            int commonRows = 0;
            lock (_captureBoundaryLock)
            {
                if (_completedRowsByCamera.ContainsKey(cameraId))
                {
                    AniloxCamera camera = _cameras.FirstOrDefault(
                        item => item != null && item.CameraId == cameraId);
                    int frameHeight = Math.Max(0, camera?.FrameHeight ?? 0);
                    _completedRowsByCamera[cameraId] += frameHeight;
                    int completedByAll = _completedRowsByCamera.Values.Min();
                    if (completedByAll > _lastPublishedCommonRows)
                    {
                        _lastPublishedCommonRows = completedByAll;
                        commonRows = completedByAll;
                    }
                }

                if (_tailDrainActive)
                {
                    long tailTick;
                    if (_tailAcceptedTicks.TryGetValue(cameraId, out tailTick) &&
                        tailTick == frameStartTicks)
                    {
                        accepted = _tailPendingCameraIds.Remove(cameraId);
                        if (accepted && _tailPendingCameraIds.Count == 0)
                            completion = _tailDrainCompletion;
                    }
                }
            }

            if (commonRows > 0)
                OnCaptureCommonRowsCompleted?.Invoke(commonRows);

            if (accepted)
            {
                FlowTrace.Log(
                    $"capture tail frame complete cam{cameraId} tick={frameStartTicks}");
                completion?.TrySetResult(true);
            }
        }

        private bool IsCaptureFrameAccepted(int cameraId, long frameStartTicks)
        {
            if (!_captureGateOpen || !IsLiveGrabbing || frameStartTicks <= 0)
                return false;

            bool headBoundaryDropped = false;
            bool headProbeReady = false;
            bool firstSetReady = false;
            bool tailAccepted = false;
            lock (_captureBoundaryLock)
            {
                if (_captureBoundaryAdmissionState ==
                        CaptureBoundaryAdmissionState.Closed ||
                    _captureBoundaryAdmissionState ==
                        CaptureBoundaryAdmissionState.Rejected)
                    return false;

                // Hot standby keeps filling a line-scan frame while the light is off between
                // captures. Treat the first callback from every camera as a phase probe: it is
                // always discarded, and no later callback is admitted until the complete probe
                // set proves that the cameras are still aligned at the IO boundary.
                if (_captureBoundaryAdmissionState ==
                    CaptureBoundaryAdmissionState.AwaitingHeadProbe)
                {
                    if (_headBoundaryPendingCameraIds.Remove(cameraId))
                    {
                        _headBoundaryTicks[cameraId] = frameStartTicks;
                        headBoundaryDropped = true;
                        headProbeReady = _headBoundaryPendingCameraIds.Count == 0;
                    }
                    else
                    {
                        return false;
                    }
                }
                else if (_captureBoundaryAdmissionState ==
                             CaptureBoundaryAdmissionState.AwaitingFirstSet ||
                         _captureBoundaryAdmissionState ==
                             CaptureBoundaryAdmissionState.Active)
                {
                    if (_tailDrainActive)
                    {
                        if (!_tailPendingCameraIds.Contains(cameraId) ||
                            _tailAcceptedTicks.ContainsKey(cameraId))
                            return false;

                        long baseline;
                        _tailBaselineTicks.TryGetValue(cameraId, out baseline);
                        if (frameStartTicks == baseline)
                            return false;
                        _tailAcceptedTicks[cameraId] = frameStartTicks;
                        tailAccepted = true;
                    }

                    _lastAcceptedTicks[cameraId] = frameStartTicks;
                    if (_captureBoundaryAdmissionState ==
                            CaptureBoundaryAdmissionState.AwaitingFirstSet &&
                        _firstPendingCameraIds.Remove(cameraId))
                    {
                        _firstAcceptedTicks[cameraId] = frameStartTicks;
                        firstSetReady = _firstPendingCameraIds.Count == 0;
                    }
                }
                else
                {
                    return false;
                }
            }

            if (headBoundaryDropped)
            {
                FlowTrace.Log(
                    $"capture head frame dropped cam{cameraId} tick={frameStartTicks} " +
                    "reason=cross-boundary");
                if (headProbeReady)
                    ValidateHeadBoundaryProbeSet();
                return false;
            }
            if (tailAccepted)
            {
                FlowTrace.Log(
                    $"capture tail frame accepted cam{cameraId} tick={frameStartTicks}");
            }
            if (firstSetReady)
                return ValidateFirstAcceptedFrameSet();
            return true;
        }

        private void ValidateHeadBoundaryProbeSet()
        {
            Dictionary<int, long> ticks;
            AniloxCamera[] targets;
            TaskCompletionSource<bool> completion;
            lock (_captureBoundaryLock)
            {
                if (_captureBoundaryAdmissionState !=
                    CaptureBoundaryAdmissionState.AwaitingHeadProbe)
                    return;
                ticks = new Dictionary<int, long>(_headBoundaryTicks);
                targets = _cameras
                    .Where(cam => cam != null && ticks.ContainsKey(cam.CameraId))
                    .ToArray();
                completion = _firstSetReadyCompletion;
            }

            string reason;
            bool aligned = TryValidateCapturePhase(
                targets,
                cam => ticks[cam.CameraId],
                out reason,
                true,
                "capture head phase");
            bool transitionApplied;
            lock (_captureBoundaryLock)
            {
                transitionApplied = _captureBoundaryAdmissionState ==
                    CaptureBoundaryAdmissionState.AwaitingHeadProbe;
                if (transitionApplied)
                {
                    _captureBoundaryAdmissionState = aligned
                        ? CaptureBoundaryAdmissionState.AwaitingFirstSet
                        : CaptureBoundaryAdmissionState.Rejected;
                }
            }
            if (!transitionApplied)
                return;

            FlowTrace.Log(
                $"capture head guard path={_captureStartPath} " +
                $"cams={string.Join(",", targets.Select(cam => cam.CameraId))} " +
                $"aligned={aligned}");
            if (aligned)
                return;

            _captureGateOpen = false;
            completion?.TrySetResult(false);
            InvalidateCapturePhase("head-probe-" + reason);
        }

        private bool ValidateFirstAcceptedFrameSet()
        {
            Dictionary<int, long> ticks;
            AniloxCamera[] targets;
            TaskCompletionSource<bool> completion;
            lock (_captureBoundaryLock)
            {
                ticks = new Dictionary<int, long>(_firstAcceptedTicks);
                targets = _cameras
                    .Where(cam => cam != null && ticks.ContainsKey(cam.CameraId))
                    .ToArray();
                completion = _firstSetReadyCompletion;
            }

            string reason;
            bool aligned = TryValidateCapturePhase(
                targets,
                cam => ticks[cam.CameraId],
                out reason,
                true,
                "capture first-set phase");
            bool transitionApplied;
            lock (_captureBoundaryLock)
            {
                transitionApplied = _captureBoundaryAdmissionState ==
                    CaptureBoundaryAdmissionState.AwaitingFirstSet;
                if (transitionApplied)
                {
                    _captureBoundaryAdmissionState = aligned
                        ? CaptureBoundaryAdmissionState.Active
                        : CaptureBoundaryAdmissionState.Rejected;
                }
            }
            if (!transitionApplied)
                return false;

            FlowTrace.Log(
                $"capture first-set ready path={_captureStartPath} " +
                $"cams={string.Join(",", targets.Select(cam => cam.CameraId))} " +
                $"aligned={aligned}");
            completion?.TrySetResult(aligned);
            if (!aligned)
            {
                _captureGateOpen = false;
                InvalidateCapturePhase("first-set-" + reason);
            }
            return aligned;
        }

        private bool ArmCaptureBoundary(IList<AniloxCamera> targets)
        {
            if (targets == null || targets.Count == 0)
                return false;

            TaskCompletionSource<bool> previousFirstSet;
            lock (_captureBoundaryLock)
            {
                previousFirstSet = _firstSetReadyCompletion;
                _firstSetReadyCompletion = new TaskCompletionSource<bool>(
                    TaskCreationOptions.RunContinuationsAsynchronously);
                _captureBoundaryAdmissionState =
                    CaptureBoundaryAdmissionState.AwaitingHeadProbe;
                _headBoundaryTicks.Clear();
                _firstAcceptedTicks.Clear();
                _firstPendingCameraIds.Clear();
                _headBoundaryPendingCameraIds.Clear();
                _lastAcceptedTicks.Clear();
                _completedRowsByCamera.Clear();
                _lastPublishedCommonRows = 0;
                _tailBaselineTicks.Clear();
                _tailAcceptedTicks.Clear();
                _tailPendingCameraIds.Clear();
                _tailDrainActive = false;
                _tailDrainCompletion = null;
                foreach (AniloxCamera cam in targets)
                {
                    _firstPendingCameraIds.Add(cam.CameraId);
                    _headBoundaryPendingCameraIds.Add(cam.CameraId);
                    _completedRowsByCamera[cam.CameraId] = 0;
                }
            }
            previousFirstSet?.TrySetResult(false);
            return true;
        }

        private void ClearCaptureBoundary()
        {
            TaskCompletionSource<bool> completion;
            TaskCompletionSource<bool> firstSetCompletion;
            lock (_captureBoundaryLock)
            {
                completion = _tailDrainCompletion;
                firstSetCompletion = _firstSetReadyCompletion;
                _firstSetReadyCompletion = null;
                _captureBoundaryAdmissionState =
                    CaptureBoundaryAdmissionState.Closed;
                _headBoundaryTicks.Clear();
                _firstAcceptedTicks.Clear();
                _firstPendingCameraIds.Clear();
                _headBoundaryPendingCameraIds.Clear();
                _lastAcceptedTicks.Clear();
                _completedRowsByCamera.Clear();
                _lastPublishedCommonRows = 0;
                _tailBaselineTicks.Clear();
                _tailAcceptedTicks.Clear();
                _tailPendingCameraIds.Clear();
                _tailDrainActive = false;
                _tailDrainCompletion = null;
                _captureStartPath = "none";
            }
            completion?.TrySetResult(false);
            firstSetCompletion?.TrySetResult(false);
        }

        private void SetCaptureStartPath(string path)
        {
            lock (_captureBoundaryLock)
                _captureStartPath = path ?? "none";
        }

        private void SetCapturePhaseSynchronizing(string reason)
        {
            CapturePhaseEligibility previous;
            lock (_captureBoundaryLock)
            {
                previous = _capturePhaseEligibility;
                _capturePhaseEligibility = CapturePhaseEligibility.Synchronizing;
            }
            FlowTrace.Log(
                $"acquisition phase synchronizing reason={reason} previous={previous}");
        }

        private void MarkCapturePhaseVerified(string reason)
        {
            lock (_captureBoundaryLock)
                _capturePhaseEligibility = CapturePhaseEligibility.Verified;
            FlowTrace.Log($"acquisition phase verified reason={reason}");
        }

        private void InvalidateCapturePhase(string reason)
        {
            CapturePhaseEligibility previous;
            lock (_captureBoundaryLock)
            {
                previous = _capturePhaseEligibility;
                _capturePhaseEligibility = CapturePhaseEligibility.Invalid;
            }
            if (previous != CapturePhaseEligibility.Invalid)
            {
                FlowTrace.Log(
                    $"acquisition phase invalidated reason={reason} previous={previous}");
            }
        }

        private bool CanUseVerifiedStandby(
            IList<AniloxCamera> targets,
            out string reason,
            bool logAligned = true,
            string logPrefix = "acquisition standby phase")
        {
            lock (_captureBoundaryLock)
            {
                if (_capturePhaseEligibility != CapturePhaseEligibility.Verified)
                {
                    reason = "phase-" + _capturePhaseEligibility;
                    return false;
                }
            }

            long now = Stopwatch.GetTimestamp();
            long frequency = Stopwatch.Frequency;
            int maxAgeMs = Math.Max(5000, GetMaxFramePeriodMs(targets) * 3 + 1000);
            foreach (AniloxCamera cam in targets)
            {
                if (cam == null || !cam.IsConnected || !cam.IsLive ||
                    !cam.IsAcquisitionWarm)
                {
                    reason = "cam" + (cam?.CameraId ?? 0) + "-not-warm";
                    return false;
                }

                long observed = cam.LastFrameObservedTimestamp;
                double ageMs = observed > 0 && frequency > 0
                    ? (now - observed) * 1000.0 / frequency
                    : double.MaxValue;
                if (ageMs < 0 || ageMs > maxAgeMs)
                {
                    reason = "cam" + cam.CameraId + "-stale";
                    return false;
                }
            }

            return TryValidateCapturePhase(
                targets,
                cam => cam.LastFrameStartTicks,
                out reason,
                logAligned,
                logPrefix);
        }

        private bool TryValidateCapturePhase(
            IList<AniloxCamera> targets,
            Func<AniloxCamera, long> tickSelector,
            out string reason,
            bool logAligned,
            string logPrefix)
        {
            if (targets == null || targets.Count == 0)
            {
                reason = "no-targets";
                return false;
            }

            CapturePhaseSample[] samples = targets
                .Select(cam => new CapturePhaseSample
                {
                    CameraId = cam.CameraId,
                    SystemNum = GetCameraSystemNum(cam.CameraId),
                    FrameStartTicks = tickSelector(cam),
                    ClockFrequencyHz = cam.DataLatchClockFreqHz,
                    FrameHeight = cam.FrameHeight,
                    AppliedLineRateHz = cam.AppliedLineRateHz
                })
                .ToArray();
            return TryValidateCapturePhaseSamples(
                samples, out reason, logAligned, logPrefix, null);
        }

        private static bool TryValidateCapturePhaseSamples(
            IList<CapturePhaseSample> samples,
            out string reason,
            bool logAligned,
            string logPrefix,
            string sampleSource)
        {
            if (samples == null || samples.Count == 0)
            {
                reason = "no-samples";
                return false;
            }

            foreach (IGrouping<int, CapturePhaseSample> group in
                samples.GroupBy(sample => sample.SystemNum))
            {
                CapturePhaseSample[] cameras = group
                    .OrderBy(sample => sample.CameraId)
                    .ToArray();
                long frequency = cameras[0].ClockFrequencyHz;
                bool measurable = group.Key >= 0 && frequency > 0 && cameras.All(cam =>
                    cam.FrameStartTicks > 0 &&
                    cam.ClockFrequencyHz == frequency &&
                    cam.FrameHeight > 0 &&
                    cam.AppliedLineRateHz > 0);
                long[] periods = measurable
                    ? cameras.Select(cam => (long)Math.Round(
                        cam.FrameHeight * (double)frequency / cam.AppliedLineRateHz))
                        .ToArray()
                    : new long[0];
                long periodTicks = periods.Length > 0
                    ? (long)Math.Round(periods.Average(value => (double)value))
                    : 0;
                double periodMismatchMs = periods.Length > 1
                    ? (periods.Max() - periods.Min()) * 1000.0 / frequency
                    : 0.0;
                long spreadTicks = 0;
                measurable = measurable &&
                    periodMismatchMs <= CapturePhaseToleranceMs &&
                    CapturePhaseMath.TryGetCircularSpreadTicks(
                        cameras.Select(cam => cam.FrameStartTicks),
                        periodTicks,
                        out spreadTicks);
                double spreadMs = measurable
                    ? spreadTicks * 1000.0 / frequency
                    : 0.0;
                bool aligned = measurable &&
                    (cameras.Length == 1 || spreadMs <= CapturePhaseToleranceMs);

                if (logAligned || !aligned)
                {
                    FlowTrace.Log(
                        $"{logPrefix} system={group.Key} " +
                        $"cams={string.Join(",", cameras.Select(cam => cam.CameraId))} " +
                        $"periodMs={(periodTicks > 0 && frequency > 0 ? periodTicks * 1000.0 / frequency : 0.0):F3} " +
                        $"periodMismatchMs={periodMismatchMs:F3} spreadTicks={spreadTicks} " +
                        $"spreadMs={spreadMs:F3} limitMs={CapturePhaseToleranceMs:F3} " +
                        $"measurable={measurable} aligned={aligned}" +
                        (string.IsNullOrEmpty(sampleSource)
                            ? string.Empty
                            : $" sampleSource={sampleSource}"));
                }

                if (!aligned)
                {
                    reason = measurable
                        ? "phase-drift-system" + group.Key
                        : "phase-unmeasurable-system" + group.Key;
                    return false;
                }
            }

            reason = "verified";
            return true;
        }

        private async Task PrepareIdleCaptureStandbyAsync(AniloxCamera[] snapshot)
        {
            if (snapshot == null || snapshot.Length == 0 || IsReleasing ||
                IsLiveGrabbing || !AreCamerasHwReady || _isParameterReconfiguring)
                return;

            long now = Stopwatch.GetTimestamp();
            long notBefore = Interlocked.Read(
                ref _idleCapturePreparationNotBeforeTimestamp);
            if (notBefore > 0 && now < notBefore)
                return;

            AniloxCamera[] targets = snapshot
                .Where(cam => cam != null && cam.IsConnected)
                .ToArray();
            if (targets.Length == 0)
                return;

            string reason;
            if (CanUseVerifiedStandby(
                targets, out reason, false, "acquisition idle phase"))
                return;
            if (Interlocked.CompareExchange(
                ref _idleCapturePreparationInFlight, 1, 0) != 0)
                return;

            await _allocationGate.WaitAsync();
            try
            {
                if (!IsAllocated || IsReleasing || IsLiveGrabbing ||
                    !AreCamerasHwReady || _isParameterReconfiguring)
                    return;

                targets = _cameras
                    .Where(cam => cam != null && cam.IsConnected)
                    .ToArray();
                if (targets.Length == 0)
                    return;
                if (CanUseVerifiedStandby(
                    targets, out reason, false, "acquisition idle phase"))
                    return;

                SetCapturePhaseSynchronizing("idle");
                _isCaptureSynchronizing = true;
                FlowTrace.Log(
                    $"acquisition idle prepare begin reason={reason} cams={targets.Length}");

                AcquisitionSyncResult sync;
                try
                {
                    sync = await SynchronizeAcquisitionAsync(
                        "idle",
                        targets,
                        null,
                        () => ReapplyLineRatesForSynchronization("idle", targets),
                        () => IsReleasing || IsLiveGrabbing);
                }
                finally
                {
                    _isCaptureSynchronizing = false;
                }

                if (!sync.Succeeded)
                {
                    InvalidateCapturePhase(
                        "idle-sync-" + (sync.Error ?? "failed"));
                    long retryTicks = (long)Math.Ceiling(
                        Stopwatch.Frequency * IdleCapturePreparationRetryMs / 1000.0);
                    Interlocked.Exchange(
                        ref _idleCapturePreparationNotBeforeTimestamp,
                        Stopwatch.GetTimestamp() + Math.Max(1, retryTicks));
                    FlowTrace.Log(
                        $"acquisition idle prepare failed error={sync.Error} " +
                        $"retryMs={IdleCapturePreparationRetryMs}");
                    return;
                }

                Interlocked.Exchange(ref _idleCapturePreparationNotBeforeTimestamp, 0);
                MarkCapturePhaseVerified("idle-sync");
                FlowTrace.Log(
                    $"acquisition idle prepare ready cams={targets.Length}");
            }
            finally
            {
                _allocationGate.Release();
                Interlocked.Exchange(ref _idleCapturePreparationInFlight, 0);
            }
        }
    }
}
