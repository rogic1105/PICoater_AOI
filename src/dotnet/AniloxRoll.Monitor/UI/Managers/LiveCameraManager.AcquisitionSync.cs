using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        private volatile bool _isCaptureSynchronizing;

        private const int CaptureSynchronizationMaxAttempts = 3;
        private const double CapturePhaseToleranceMs = 5.0;

        private sealed class AcquisitionWarmSample
        {
            public int CameraId;
            public int SystemNum;
            public long FrameStartTicks;
            public long FrameStartSequence;
            public long ClockFrequencyHz;
        }

        private sealed class AcquisitionSyncResult
        {
            public bool Succeeded;
            public bool Canceled;
            public string Error;
            public List<AcquisitionWarmSample> Samples =
                new List<AcquisitionWarmSample>();
        }

        private async Task<AcquisitionSyncResult> SynchronizeAcquisitionAsync(
            string reason,
            AniloxCamera[] targets,
            Action applyWhilePaused,
            Action resetTimingWhilePaused,
            Func<bool> cancellationRequested,
            bool validateFramePeriod)
        {
            var result = new AcquisitionSyncResult();
            bool actionApplied = applyWhilePaused == null;

            if (validateFramePeriod)
            {
                foreach (AniloxCamera cam in targets)
                    cam.SetFramePeriodObservationEnabled(true);
            }

            try
            {
                for (int attempt = 1; attempt <= CaptureSynchronizationMaxAttempts; attempt++)
                {
                    if (cancellationRequested != null && cancellationRequested())
                    {
                        result.Canceled = true;
                        result.Error = "Canceled";
                        return result;
                    }

                    FlowTrace.Log(
                        $"acquisition sync begin reason={reason} attempt={attempt} " +
                        $"gate=closed cams={targets.Length}");

                    Exception failure = null;
                    try
                    {
                        await Task.Run(() =>
                            System.Threading.Tasks.Parallel.ForEach(
                                targets, cam => cam.PauseAcquisition()));
                        FlowTrace.Log(
                            $"acquisition sync paused reason={reason} attempt={attempt} " +
                            $"cams={targets.Length}");

                        if (!actionApplied)
                        {
                            await Task.Run(applyWhilePaused);
                            actionApplied = true;
                        }
                        if (resetTimingWhilePaused != null)
                            await Task.Run(resetTimingWhilePaused);
                    }
                    catch (Exception ex)
                    {
                        failure = ex;
                    }
                    finally
                    {
                        try
                        {
                            // M_START is intentionally issued back-to-back from one worker. The
                            // measured first hardware ticks, not a fixed delay, decide readiness.
                            await Task.Run(() =>
                            {
                                foreach (var cam in targets)
                                    cam.ResumeAcquisition();
                            });
                        }
                        catch (Exception ex)
                        {
                            if (failure == null) failure = ex;
                        }
                    }

                    if (failure != null)
                    {
                        result.Error = failure.GetType().Name;
                        FlowTrace.Log(
                            $"acquisition sync failed reason={reason} attempt={attempt} " +
                            $"gate=closed error={result.Error}");
                        return result;
                    }

                    FlowTrace.Log(
                        $"acquisition sync resumed reason={reason} attempt={attempt} " +
                        $"cams={targets.Length}");

                    AcquisitionSyncResult warm = await WaitForAcquisitionWarmAsync(
                        reason, attempt, targets, cancellationRequested);
                    if (!warm.Succeeded)
                        return warm;

                    result.Samples = warm.Samples;
                    bool framePeriodAligned = !validateFramePeriod ||
                        await WaitAndValidateFramePeriodsAsync(
                            reason, attempt, targets, warm.Samples, cancellationRequested);
                    bool phaseAligned = LogAndValidateCapturePhase(
                        reason, attempt, warm.Samples);
                    if (framePeriodAligned && phaseAligned)
                    {
                        result.Succeeded = true;
                        result.Error = null;
                        FlowTrace.Log(
                            $"acquisition sync complete reason={reason} attempts={attempt} " +
                            $"cams={targets.Length} phase=True");
                        return result;
                    }

                    FlowTrace.Log(
                        $"acquisition sync retry reason={reason} attempt={attempt} " +
                        $"error={(framePeriodAligned ? "PhaseOutOfRange" : "FramePeriodOutOfRange")}");
                }

                result.Error = "SynchronizationOutOfRange";
                return result;
            }
            finally
            {
                if (validateFramePeriod)
                {
                    foreach (AniloxCamera cam in targets)
                        cam.SetFramePeriodObservationEnabled(false);
                }
            }
        }

        private async Task<AcquisitionSyncResult> WaitForAcquisitionWarmAsync(
            string reason,
            int attempt,
            IList<AniloxCamera> targets,
            Func<bool> cancellationRequested)
        {
            var result = new AcquisitionSyncResult();
            int framePeriodMs = GetMaxFramePeriodMs(targets);
            int timeoutMs = Math.Max(5000, Math.Min(60000, framePeriodMs * 5 + 2000));
            var pending = new HashSet<int>(targets.Select(cam => cam.CameraId));
            var stopwatch = Stopwatch.StartNew();

            while (pending.Count > 0 && stopwatch.ElapsedMilliseconds <= timeoutMs)
            {
                if (IsReleasing ||
                    (cancellationRequested != null && cancellationRequested()))
                {
                    result.Canceled = true;
                    result.Error = "Canceled";
                    FlowTrace.Log(
                        $"acquisition sync canceled reason={reason} attempt={attempt} " +
                        $"gate=closed");
                    return result;
                }

                foreach (var cam in targets)
                {
                    if (!pending.Contains(cam.CameraId) || !cam.IsAcquisitionWarm)
                        continue;

                    var sample = new AcquisitionWarmSample
                    {
                        CameraId = cam.CameraId,
                        SystemNum = GetCameraSystemNum(cam.CameraId),
                        FrameStartTicks = cam.LastFrameStartTicks,
                        FrameStartSequence = cam.FrameStartSequence,
                        ClockFrequencyHz = cam.DataLatchClockFreqHz
                    };
                    result.Samples.Add(sample);
                    pending.Remove(cam.CameraId);
                    FlowTrace.Log(
                        $"acquisition sync ready reason={reason} attempt={attempt} " +
                        $"cam{sample.CameraId} system={sample.SystemNum} " +
                        $"tick={sample.FrameStartTicks} freq={sample.ClockFrequencyHz}");
                }

                if (pending.Count == 0)
                {
                    result.Succeeded = true;
                    return result;
                }
                await Task.Delay(20);
            }

            result.Error = "WarmTimeout";
            FlowTrace.Log(
                $"acquisition sync timeout reason={reason} attempt={attempt} " +
                $"pending={string.Join(",", pending)} limitMs={timeoutMs}");
            return result;
        }

        private async Task<bool> WaitAndValidateFramePeriodsAsync(
            string reason,
            int attempt,
            IList<AniloxCamera> targets,
            IList<AcquisitionWarmSample> samples,
            Func<bool> cancellationRequested)
        {
            var sampleByCamera = samples.ToDictionary(sample => sample.CameraId);
            var pending = new HashSet<int>(sampleByCamera.Keys);
            int maxExpectedMs = targets
                .Select(cam => (int)Math.Ceiling(
                    AcquisitionFramePeriodPolicy.ExpectedMs(
                        cam.FrameHeight, cam.AppliedLineRateHz)))
                .DefaultIfEmpty(0)
                .Max();
            int timeoutMs = Math.Max(5000, Math.Min(60000, maxExpectedMs * 4 + 2000));
            bool allAligned = true;
            var stopwatch = Stopwatch.StartNew();

            while (pending.Count > 0 && stopwatch.ElapsedMilliseconds <= timeoutMs)
            {
                if (IsReleasing ||
                    (cancellationRequested != null && cancellationRequested()))
                {
                    return false;
                }

                foreach (AniloxCamera cam in targets)
                {
                    if (!pending.Contains(cam.CameraId)) continue;

                    AcquisitionWarmSample first = sampleByCamera[cam.CameraId];
                    long sequence = cam.FrameStartSequence;
                    if (sequence <= first.FrameStartSequence) continue;

                    double expectedMs;
                    double actualMs;
                    double toleranceMs;
                    bool aligned = AcquisitionFramePeriodPolicy.IsWithinTolerance(
                        cam.FrameHeight,
                        cam.AppliedLineRateHz,
                        first.FrameStartTicks,
                        cam.LastFrameStartTicks,
                        sequence - first.FrameStartSequence,
                        first.ClockFrequencyHz,
                        out expectedMs,
                        out actualMs,
                        out toleranceMs);
                    allAligned &= aligned;
                    pending.Remove(cam.CameraId);

                    FlowTrace.Log(
                        $"acquisition sync rate reason={reason} attempt={attempt} " +
                        $"cam{cam.CameraId} expectedMs={expectedMs:F3} " +
                        $"actualMs={actualMs:F3} toleranceMs={toleranceMs:F3} " +
                        $"aligned={aligned}");
                }

                if (pending.Count > 0)
                    await Task.Delay(20);
            }

            if (pending.Count > 0)
            {
                FlowTrace.Log(
                    $"acquisition sync rate timeout reason={reason} attempt={attempt} " +
                    $"pending={string.Join(",", pending)} limitMs={timeoutMs}");
                return false;
            }
            return allAligned;
        }

        private bool LogAndValidateCapturePhase(
            string reason,
            int attempt,
            IList<AcquisitionWarmSample> samples)
        {
            bool allAligned = samples != null && samples.Count > 0;
            foreach (var group in (samples ?? new List<AcquisitionWarmSample>())
                .GroupBy(sample => sample.SystemNum))
            {
                var ordered = group.OrderBy(sample => sample.CameraId).ToArray();
                bool measurable = (ordered.Length == 1 || group.Key >= 0) &&
                    ordered.All(sample =>
                        sample.FrameStartTicks > 0 &&
                        sample.ClockFrequencyHz > 0 &&
                        sample.ClockFrequencyHz == ordered[0].ClockFrequencyHz);
                long spreadTicks = measurable && ordered.Length > 1
                    ? ordered.Max(sample => sample.FrameStartTicks) -
                      ordered.Min(sample => sample.FrameStartTicks)
                    : 0;
                double spreadMs = measurable && ordered.Length > 1
                    ? spreadTicks * 1000.0 / ordered[0].ClockFrequencyHz
                    : 0.0;
                bool aligned = measurable &&
                    (ordered.Length == 1 || spreadMs <= CapturePhaseToleranceMs);
                allAligned &= aligned;

                string spreadText = spreadMs.ToString(
                    "F3", System.Globalization.CultureInfo.InvariantCulture);
                string limitText = CapturePhaseToleranceMs.ToString(
                    "F3", System.Globalization.CultureInfo.InvariantCulture);
                FlowTrace.Log(
                    $"acquisition sync phase reason={reason} attempt={attempt} " +
                    $"system={group.Key} cams={string.Join(",", ordered.Select(s => s.CameraId))} " +
                    $"spreadTicks={spreadTicks} spreadMs={spreadText} " +
                    $"limitMs={limitText} measurable={measurable} aligned={aligned}");
            }
            return allAligned;
        }

        private int GetCameraSystemNum(int cameraId)
        {
            CameraHardwareConfig config = _cameraHardwareConfigs?
                .FirstOrDefault(item => item.Id == cameraId);
            return config?.SystemNum ?? -1;
        }

        private static void ReapplyLineRatesForSynchronization(
            string reason, IEnumerable<AniloxCamera> targets)
        {
            var applied = new List<string>();
            foreach (var cam in targets)
            {
                double lineRateHz = cam.AppliedLineRateHz;
                if (lineRateHz <= 0) continue;
                cam.SetLineRateHz(lineRateHz);
                applied.Add(
                    "cam" + cam.CameraId + "=" +
                    lineRateHz.ToString(
                        "0.###", System.Globalization.CultureInfo.InvariantCulture));
            }
            FlowTrace.Log(
                $"acquisition sync timing-reset reason={reason} " +
                $"lineRates={string.Join(",", applied)}");
        }
    }

    internal static class AcquisitionFramePeriodPolicy
    {
        private const double RelativeTolerance = 0.20;
        private const double MinimumToleranceMs = 100.0;

        public static double ExpectedMs(int frameHeight, double lineRateHz)
        {
            return frameHeight > 0 && lineRateHz > 0
                ? frameHeight * 1000.0 / lineRateHz
                : 0.0;
        }

        public static bool IsWithinTolerance(
            int frameHeight,
            double lineRateHz,
            long firstTicks,
            long lastTicks,
            long frameCount,
            long clockFrequencyHz,
            out double expectedMs,
            out double actualMs,
            out double toleranceMs)
        {
            expectedMs = ExpectedMs(frameHeight, lineRateHz);
            actualMs = frameCount > 0 && clockFrequencyHz > 0 && lastTicks > firstTicks
                ? (lastTicks - firstTicks) * 1000.0 / clockFrequencyHz / frameCount
                : 0.0;
            toleranceMs = Math.Max(MinimumToleranceMs, expectedMs * RelativeTolerance);
            return expectedMs > 0 && actualMs > 0 &&
                Math.Abs(actualMs - expectedMs) <= toleranceMs;
        }
    }
}
