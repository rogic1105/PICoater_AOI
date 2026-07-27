using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Camera
{
    internal sealed class RowPhaseFrameData
    {
        public int CameraId;
        public byte[] Pixels;
        public int Width;
        public int Height;
        public long FrameStartTicks;
    }

    internal sealed class RowPhaseFramePlan
    {
        public static RowPhaseFramePlan PassThrough(int height)
        {
            return new RowPhaseFramePlan
            {
                Accepted = height > 0,
                CommonHeight = Math.Max(0, height)
            };
        }

        public bool Accepted;
        public long BatchId;
        public int CameraId;
        public int SourceTop;
        public int CommonHeight;
        public int FixedOffsetRows;
        public int DynamicOffsetRows;
        public int TotalOffsetRows;
        public bool AutoAlignmentTrusted;
        public double Confidence;
        public string Reason;
    }

    internal sealed class RowPhasePairResult
    {
        public int LeftCameraId;
        public int RightCameraId;
        public int ShiftRows;
        public double Correlation;
        public double PeakMargin;
        public bool Trusted;
    }

    internal static class RowPhaseAlignmentMath
    {
        private const int ColumnSampleStep = 16;
        private const int CorrelationRowStep = 4;
        private const double MinimumCorrelation = 0.45;
        private const double MinimumPeakMargin = 0.015;

        public static int MillimetersToRows(
            double offsetMm,
            double speedMPerMin,
            double lineRateHz)
        {
            if (Math.Abs(offsetMm) < 1e-12) return 0;
            if (speedMPerMin <= 0 || lineRateHz <= 0) return 0;
            double mmPerRow = speedMPerMin * 1000.0 / 60.0 / lineRateHz;
            return mmPerRow > 0
                ? (int)Math.Round(offsetMm / mmPerRow)
                : 0;
        }

        public static bool TryEstimateDynamicOffsets(
            IList<RowPhaseFrameData> frames,
            double[] startMm,
            double[] opsUm,
            int[] fixedOffsetRows,
            int searchRadiusRows,
            out int[] dynamicOffsetRows,
            out RowPhasePairResult[] pairResults,
            out double confidence,
            out string reason)
        {
            dynamicOffsetRows = new int[Math.Max(7, MaxCameraId(frames))];
            pairResults = new RowPhasePairResult[0];
            confidence = 0;
            reason = "not-enough-cameras";
            if (frames == null || frames.Count < 2)
                return false;

            RowPhaseFrameData[] ordered = frames
                .Where(IsValidFrame)
                .OrderBy(frame => GetArrayValue(startMm, frame.CameraId - 1, frame.CameraId))
                .ToArray();
            if (ordered.Length < 2)
            {
                reason = "invalid-frame";
                return false;
            }

            var pairs = new List<RowPhasePairResult>(ordered.Length - 1);
            int[] totalOffsets = new int[dynamicOffsetRows.Length];
            RowPhaseFrameData anchor = ordered[0];
            totalOffsets[anchor.CameraId - 1] =
                GetArrayValue(fixedOffsetRows, anchor.CameraId - 1, 0);

            for (int i = 0; i < ordered.Length - 1; i++)
            {
                RowPhaseFrameData left = ordered[i];
                RowPhaseFrameData right = ordered[i + 1];
                int leftFixed = GetArrayValue(fixedOffsetRows, left.CameraId - 1, 0);
                int rightFixed = GetArrayValue(fixedOffsetRows, right.CameraId - 1, 0);
                int expectedShift = leftFixed - rightFixed;

                RowPhasePairResult pair;
                if (!TryEstimatePairShift(
                    left, right, startMm, opsUm, expectedShift,
                    Math.Max(0, searchRadiusRows), out pair))
                {
                    pairs.Add(pair);
                    pairResults = pairs.ToArray();
                    confidence = pairs.Count == 0
                        ? 0
                        : pairs.Min(item => item.Correlation);
                    reason = "low-confidence-cam" + left.CameraId + "-cam" + right.CameraId;
                    return false;
                }

                pairs.Add(pair);
                int leftTotal = totalOffsets[left.CameraId - 1];
                int rightTotal = leftTotal - pair.ShiftRows;
                totalOffsets[right.CameraId - 1] = rightTotal;
                dynamicOffsetRows[right.CameraId - 1] = rightTotal - rightFixed;
            }

            pairResults = pairs.ToArray();
            confidence = pairs.Min(pair => pair.Correlation);
            reason = "trusted";
            return true;
        }

        public static bool TryBuildCropPlans(
            IList<RowPhaseFrameData> frames,
            int[] fixedOffsetRows,
            int[] dynamicOffsetRows,
            long batchId,
            bool autoTrusted,
            double confidence,
            string reason,
            out Dictionary<int, RowPhaseFramePlan> plans)
        {
            plans = new Dictionary<int, RowPhaseFramePlan>();
            if (frames == null || frames.Count == 0)
                return false;

            int commonTop = int.MinValue;
            int commonBottom = int.MaxValue;
            foreach (RowPhaseFrameData frame in frames)
            {
                if (!IsValidFrame(frame)) return false;
                int index = frame.CameraId - 1;
                int fixedRows = GetArrayValue(fixedOffsetRows, index, 0);
                int dynamicRows = GetArrayValue(dynamicOffsetRows, index, 0);
                int totalRows = fixedRows + dynamicRows;
                commonTop = Math.Max(commonTop, totalRows);
                commonBottom = Math.Min(commonBottom, totalRows + frame.Height);
            }

            int commonHeight = commonBottom - commonTop;
            if (commonHeight <= 0)
                return false;

            foreach (RowPhaseFrameData frame in frames)
            {
                int index = frame.CameraId - 1;
                int fixedRows = GetArrayValue(fixedOffsetRows, index, 0);
                int dynamicRows = GetArrayValue(dynamicOffsetRows, index, 0);
                int totalRows = fixedRows + dynamicRows;
                int sourceTop = commonTop - totalRows;
                if (sourceTop < 0 || sourceTop + commonHeight > frame.Height)
                    return false;

                plans[frame.CameraId] = new RowPhaseFramePlan
                {
                    Accepted = true,
                    BatchId = batchId,
                    CameraId = frame.CameraId,
                    SourceTop = sourceTop,
                    CommonHeight = commonHeight,
                    FixedOffsetRows = fixedRows,
                    DynamicOffsetRows = dynamicRows,
                    TotalOffsetRows = totalRows,
                    AutoAlignmentTrusted = autoTrusted,
                    Confidence = confidence,
                    Reason = reason
                };
            }
            return true;
        }

        private static bool TryEstimatePairShift(
            RowPhaseFrameData left,
            RowPhaseFrameData right,
            double[] startMm,
            double[] opsUm,
            int expectedShift,
            int searchRadius,
            out RowPhasePairResult result)
        {
            result = new RowPhasePairResult
            {
                LeftCameraId = left?.CameraId ?? 0,
                RightCameraId = right?.CameraId ?? 0,
                ShiftRows = expectedShift
            };
            if (!IsValidFrame(left) || !IsValidFrame(right))
                return false;

            double leftOps = GetArrayValue(opsUm, left.CameraId - 1, 0.0);
            double rightOps = GetArrayValue(opsUm, right.CameraId - 1, 0.0);
            double leftStart = GetArrayValue(startMm, left.CameraId - 1, 0.0);
            double rightStart = GetArrayValue(startMm, right.CameraId - 1, 0.0);
            if (leftOps <= 0 || rightOps <= 0)
                return false;

            double overlapStart = Math.Max(leftStart, rightStart);
            double overlapEnd = Math.Min(
                leftStart + left.Width * leftOps / 1000.0,
                rightStart + right.Width * rightOps / 1000.0);
            if (overlapEnd <= overlapStart)
                return false;

            int leftX0 = Clamp(
                (int)Math.Ceiling((overlapStart - leftStart) * 1000.0 / leftOps),
                0, left.Width - 1);
            int leftX1 = Clamp(
                (int)Math.Floor((overlapEnd - leftStart) * 1000.0 / leftOps),
                leftX0 + 1, left.Width);
            int rightX0 = Clamp(
                (int)Math.Ceiling((overlapStart - rightStart) * 1000.0 / rightOps),
                0, right.Width - 1);
            int rightX1 = Clamp(
                (int)Math.Floor((overlapEnd - rightStart) * 1000.0 / rightOps),
                rightX0 + 1, right.Width);
            if (leftX1 - leftX0 < 16 || rightX1 - rightX0 < 16)
                return false;

            double[] leftSignature = BuildRowSignature(left, leftX0, leftX1);
            double[] rightSignature = BuildRowSignature(right, rightX0, rightX1);
            int minimumShift = expectedShift - searchRadius;
            int maximumShift = expectedShift + searchRadius;

            var candidates = new List<KeyValuePair<int, double>>();
            for (int shift = minimumShift; shift <= maximumShift; shift++)
            {
                double score;
                if (TryCorrelation(leftSignature, rightSignature, shift, out score))
                    candidates.Add(new KeyValuePair<int, double>(shift, score));
            }
            if (candidates.Count == 0)
                return false;

            KeyValuePair<int, double> best = candidates
                .OrderByDescending(item => item.Value)
                .First();
            int exclusion = Math.Max(4, Math.Min(20, searchRadius / 10));
            double second = candidates
                .Where(item => Math.Abs(item.Key - best.Key) > exclusion)
                .Select(item => item.Value)
                .DefaultIfEmpty(-1.0)
                .Max();

            result.ShiftRows = best.Key;
            result.Correlation = best.Value;
            result.PeakMargin = best.Value - second;
            result.Trusted =
                best.Value >= MinimumCorrelation &&
                result.PeakMargin >= MinimumPeakMargin;
            return result.Trusted;
        }

        private static double[] BuildRowSignature(
            RowPhaseFrameData frame,
            int x0,
            int x1)
        {
            var result = new double[frame.Height];
            for (int y = 0; y < frame.Height; y++)
            {
                int row = y * frame.Width;
                double sum = 0;
                int count = 0;
                for (int x = x0; x < x1; x += ColumnSampleStep)
                {
                    sum += frame.Pixels[row + x];
                    count++;
                }
                result[y] = count > 0 ? sum / count : 0;
            }

            // Vertical derivative removes constant brightness differences between cameras.
            for (int y = frame.Height - 1; y >= 1; y--)
                result[y] -= result[y - 1];
            result[0] = 0;
            return result;
        }

        private static bool TryCorrelation(
            double[] left,
            double[] right,
            int shift,
            out double score)
        {
            score = 0;
            int start = Math.Max(1, -shift + 1);
            int end = Math.Min(left.Length, right.Length - shift);
            if (end - start < 64)
                return false;

            double sumLeft = 0;
            double sumRight = 0;
            int count = 0;
            for (int y = start; y < end; y += CorrelationRowStep)
            {
                sumLeft += left[y];
                sumRight += right[y + shift];
                count++;
            }
            if (count < 16)
                return false;

            double meanLeft = sumLeft / count;
            double meanRight = sumRight / count;
            double covariance = 0;
            double energyLeft = 0;
            double energyRight = 0;
            for (int y = start; y < end; y += CorrelationRowStep)
            {
                double a = left[y] - meanLeft;
                double b = right[y + shift] - meanRight;
                covariance += a * b;
                energyLeft += a * a;
                energyRight += b * b;
            }
            if (energyLeft < 1e-6 || energyRight < 1e-6)
                return false;

            score = covariance / Math.Sqrt(energyLeft * energyRight);
            return !double.IsNaN(score) && !double.IsInfinity(score);
        }

        private static bool IsValidFrame(RowPhaseFrameData frame)
        {
            if (frame == null || frame.CameraId <= 0 ||
                frame.Pixels == null || frame.Width <= 0 || frame.Height <= 0)
                return false;
            long required = (long)frame.Width * frame.Height;
            return required <= frame.Pixels.Length;
        }

        private static int MaxCameraId(IList<RowPhaseFrameData> frames)
        {
            return frames == null || frames.Count == 0
                ? 0
                : frames.Where(frame => frame != null)
                    .Select(frame => frame.CameraId)
                    .DefaultIfEmpty(0)
                    .Max();
        }

        private static T GetArrayValue<T>(T[] values, int index, T fallback)
        {
            return values != null && index >= 0 && index < values.Length
                ? values[index]
                : fallback;
        }

        private static int Clamp(int value, int minimum, int maximum)
        {
            if (value < minimum) return minimum;
            return value > maximum ? maximum : value;
        }
    }

    /// <summary>
    /// State table:
    /// Idle + Arm -> Collecting; Collecting + unique frame -> Collecting;
    /// Collecting + all cameras -> release one aligned batch; Collecting + timeout/cancel -> drop batch.
    /// The first complete batch calibrates dynamic offsets. Later batches reuse them but still wait
    /// for every active camera, preventing one camera from running a whole frame generation ahead.
    /// </summary>
    internal sealed class RowPhaseAlignmentCoordinator
    {
        private sealed class Batch
        {
            public long Id;
            public readonly Dictionary<int, RowPhaseFrameData> Frames =
                new Dictionary<int, RowPhaseFrameData>();
            public Dictionary<int, RowPhaseFramePlan> Plans;
            public bool Completed;
        }

        private readonly object _sync = new object();
        private HashSet<int> _activeCameraIds = new HashSet<int>();
        private HashSet<int> _resyncPendingCameraIds = new HashSet<int>();
        private double[] _startMm = new double[0];
        private double[] _opsUm = new double[0];
        private int[] _fixedOffsetRows = new int[0];
        private int[] _dynamicOffsetRows = new int[0];
        private int _searchRadiusRows;
        private int _timeoutMs = 3000;
        private bool _autoEnabled;
        private bool _armed;
        private bool _calibrated;
        private bool _autoTrusted;
        private double _confidence;
        private string _calibrationReason = "not-calibrated";
        private long _nextBatchId;
        private Batch _current;

        public void Configure(
            bool autoEnabled,
            int searchRadiusRows,
            int timeoutMs,
            double[] startMm,
            double[] opsUm,
            int[] fixedOffsetRows)
        {
            lock (_sync)
            {
                _autoEnabled = autoEnabled;
                _searchRadiusRows = Math.Max(0, searchRadiusRows);
                _timeoutMs = Math.Max(500, timeoutMs);
                _startMm = (double[])(startMm?.Clone() ?? new double[0]);
                _opsUm = (double[])(opsUm?.Clone() ?? new double[0]);
                _fixedOffsetRows = (int[])(fixedOffsetRows?.Clone() ?? new int[0]);
                _dynamicOffsetRows = new int[Math.Max(7, _fixedOffsetRows.Length)];
                _calibrated = false;
                _autoTrusted = false;
                _confidence = 0;
                _calibrationReason = "not-calibrated";
            }
        }

        public void Arm(IEnumerable<int> cameraIds)
        {
            lock (_sync)
            {
                CancelCurrentLocked("rearm");
                _activeCameraIds = new HashSet<int>(
                    (cameraIds ?? Enumerable.Empty<int>()).Where(id => id > 0));
                _resyncPendingCameraIds.Clear();
                _dynamicOffsetRows = new int[Math.Max(
                    7,
                    Math.Max(_fixedOffsetRows.Length,
                        _activeCameraIds.DefaultIfEmpty(0).Max()))];
                _calibrated = false;
                _autoTrusted = false;
                _confidence = 0;
                _calibrationReason = "not-calibrated";
                _armed = _activeCameraIds.Count > 0;
                _nextBatchId = 0;
            }
        }

        public void Cancel(string reason)
        {
            lock (_sync)
            {
                _armed = false;
                _resyncPendingCameraIds.Clear();
                CancelCurrentLocked(reason ?? "cancel");
            }
        }

        public RowPhaseFramePlan Align(RowPhaseFrameData frame)
        {
            if (frame == null)
                return RowPhaseFramePlan.PassThrough(0);

            Batch batch;
            lock (_sync)
            {
                if (!_armed || !_activeCameraIds.Contains(frame.CameraId))
                    return RowPhaseFramePlan.PassThrough(frame.Height);

                if (_resyncPendingCameraIds.Remove(frame.CameraId))
                {
                    FlowTrace.Log(
                        $"row phase resync drop cam={frame.CameraId} " +
                        $"pending={string.Join(",", _resyncPendingCameraIds.OrderBy(id => id))}");
                    return Rejected(frame, 0, "resync-drop");
                }

                bool hasFixedOffset = _fixedOffsetRows.Any(value => value != 0);
                if (!_autoEnabled && !hasFixedOffset)
                    return RowPhaseFramePlan.PassThrough(frame.Height);

                if (_current == null)
                    _current = new Batch { Id = ++_nextBatchId };
                batch = _current;

                if (batch.Frames.ContainsKey(frame.CameraId))
                {
                    FlowTrace.Log(
                        $"row phase duplicate batch={batch.Id} cam={frame.CameraId} drop=True");
                    return Rejected(frame, batch.Id, "duplicate-camera");
                }

                batch.Frames.Add(frame.CameraId, frame);
                if (batch.Frames.Count == _activeCameraIds.Count &&
                    _activeCameraIds.All(batch.Frames.ContainsKey))
                {
                    CompleteBatchLocked(batch);
                    _current = null;
                    global::System.Threading.Monitor.PulseAll(_sync);
                }
                else
                {
                    Stopwatch wait = Stopwatch.StartNew();
                    while (!batch.Completed && _armed && wait.ElapsedMilliseconds < _timeoutMs)
                    {
                        int remaining = _timeoutMs - (int)wait.ElapsedMilliseconds;
                        if (remaining <= 0) break;
                        global::System.Threading.Monitor.Wait(_sync, remaining);
                    }

                    if (!batch.Completed)
                    {
                        string pending = string.Join(
                            ",",
                            _activeCameraIds
                                .Where(id => !batch.Frames.ContainsKey(id))
                                .OrderBy(id => id));
                        RejectBatchLocked(batch, "timeout-pending-" + pending);
                        _resyncPendingCameraIds = new HashSet<int>(_activeCameraIds);
                        if (ReferenceEquals(_current, batch))
                            _current = null;
                        FlowTrace.Log(
                            $"row phase timeout batch={batch.Id} pending={pending} " +
                            $"limitMs={_timeoutMs}");
                        global::System.Threading.Monitor.PulseAll(_sync);
                    }
                }

                RowPhaseFramePlan plan;
                if (batch.Plans != null &&
                    batch.Plans.TryGetValue(frame.CameraId, out plan))
                    return plan;
                return Rejected(frame, batch.Id, "missing-plan");
            }
        }

        private void CompleteBatchLocked(Batch batch)
        {
            var frames = batch.Frames.Values.OrderBy(item => item.CameraId).ToArray();
            if (!_calibrated)
            {
                RowPhasePairResult[] pairs = Array.Empty<RowPhasePairResult>();
                string reason = _autoEnabled ? "not-evaluated" : "auto-disabled";
                double confidence = 0;
                int[] dynamicRows = new int[Math.Max(7, _fixedOffsetRows.Length)];
                bool trusted = _autoEnabled && _searchRadiusRows > 0 &&
                    RowPhaseAlignmentMath.TryEstimateDynamicOffsets(
                        frames,
                        _startMm,
                        _opsUm,
                        _fixedOffsetRows,
                        _searchRadiusRows,
                        out dynamicRows,
                        out pairs,
                        out confidence,
                        out reason);

                _dynamicOffsetRows = trusted
                    ? dynamicRows
                    : new int[Math.Max(7, _fixedOffsetRows.Length)];
                _autoTrusted = trusted;
                _confidence = confidence;
                _calibrationReason = _autoEnabled
                    ? reason
                    : "auto-disabled";
                _calibrated = true;

                FlowTrace.Log(
                    $"row phase calibrated batch={batch.Id} auto={_autoEnabled} " +
                    $"trusted={trusted} confidence={confidence:F3} reason={_calibrationReason} " +
                    $"fixed={FormatOffsets(_fixedOffsetRows, _activeCameraIds)} " +
                    $"dynamic={FormatOffsets(_dynamicOffsetRows, _activeCameraIds)}");
                if (pairs != null)
                {
                    foreach (RowPhasePairResult pair in pairs)
                    {
                        FlowTrace.Dvt(
                            $"row phase pair batch={batch.Id} cams={pair.LeftCameraId}-{pair.RightCameraId} " +
                            $"shift={pair.ShiftRows} corr={pair.Correlation:F3} " +
                            $"margin={pair.PeakMargin:F3} trusted={pair.Trusted}");
                    }
                }
            }

            Dictionary<int, RowPhaseFramePlan> plans;
            if (!RowPhaseAlignmentMath.TryBuildCropPlans(
                frames,
                _fixedOffsetRows,
                _dynamicOffsetRows,
                batch.Id,
                _autoTrusted,
                _confidence,
                _calibrationReason,
                out plans))
            {
                RejectBatchLocked(batch, "no-common-range");
                FlowTrace.Log($"row phase rejected batch={batch.Id} reason=no-common-range");
                return;
            }

            batch.Plans = plans;
            batch.Completed = true;
            RowPhaseFramePlan first = plans.Values.First();
            FlowTrace.Dvt(
                $"row phase batch ready batch={batch.Id} cams={plans.Count} " +
                $"height={first.CommonHeight} trusted={_autoTrusted} " +
                $"total={FormatPlanOffsets(plans)}");
        }

        private void RejectBatchLocked(Batch batch, string reason)
        {
            if (batch.Completed) return;
            batch.Plans = new Dictionary<int, RowPhaseFramePlan>();
            foreach (RowPhaseFrameData frame in batch.Frames.Values)
                batch.Plans[frame.CameraId] = Rejected(frame, batch.Id, reason);
            batch.Completed = true;
        }

        private void CancelCurrentLocked(string reason)
        {
            if (_current == null) return;
            RejectBatchLocked(_current, "canceled-" + reason);
            _current = null;
            global::System.Threading.Monitor.PulseAll(_sync);
        }

        private static RowPhaseFramePlan Rejected(
            RowPhaseFrameData frame,
            long batchId,
            string reason)
        {
            return new RowPhaseFramePlan
            {
                Accepted = false,
                BatchId = batchId,
                CameraId = frame?.CameraId ?? 0,
                Reason = reason
            };
        }

        private static string FormatOffsets(int[] values, IEnumerable<int> cameraIds)
        {
            return string.Join(
                ",",
                cameraIds.OrderBy(id => id).Select(id =>
                    "cam" + id + "=" +
                    (values != null && id - 1 < values.Length ? values[id - 1] : 0)));
        }

        private static string FormatPlanOffsets(
            IDictionary<int, RowPhaseFramePlan> plans)
        {
            return string.Join(
                ",",
                plans.OrderBy(item => item.Key)
                    .Select(item => "cam" + item.Key + "=" + item.Value.TotalOffsetRows));
        }
    }
}
