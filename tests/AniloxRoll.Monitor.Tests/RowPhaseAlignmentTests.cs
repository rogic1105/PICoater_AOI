using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Camera;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class RowPhaseAlignmentTests
    {
        [Test]
        public void MillimetersToRows_UsesWebSpeedAndLineRate()
        {
            int rows = RowPhaseAlignmentMath.MillimetersToRows(10.0, 40.0, 3000.0);

            Assert.That(rows, Is.EqualTo(45));
        }

        [Test]
        public void EstimateDynamicOffsets_FindsKnownShiftInOverlap()
        {
            const int width = 128;
            const int height = 240;
            const int shift = 12;
            byte[] left = BuildFrame(width, height, 0, 0);
            byte[] right = BuildFrame(width, height, 64, -shift);
            var frames = new[]
            {
                Frame(1, left, width, height),
                Frame(2, right, width, height)
            };

            int[] dynamicRows;
            RowPhasePairResult[] pairs;
            double confidence;
            string reason;
            bool ok = RowPhaseAlignmentMath.TryEstimateDynamicOffsets(
                frames,
                new[] { 0.0, 64.0 },
                new[] { 1000.0, 1000.0 },
                new[] { 0, 0 },
                30,
                out dynamicRows,
                out pairs,
                out confidence,
                out reason);

            Assert.That(ok, Is.True, reason);
            Assert.That(pairs, Has.Length.EqualTo(1));
            Assert.That(pairs[0].ShiftRows, Is.EqualTo(shift));
            Assert.That(dynamicRows[0], Is.EqualTo(0));
            Assert.That(dynamicRows[1], Is.EqualTo(-shift));
            Assert.That(confidence, Is.GreaterThan(0.9));
        }

        [Test]
        public void EstimateDynamicOffsets_FlatOverlapIsRejected()
        {
            const int width = 128;
            const int height = 240;
            var frames = new[]
            {
                Frame(1, Filled(width * height, 80), width, height),
                Frame(2, Filled(width * height, 80), width, height)
            };

            int[] dynamicRows;
            RowPhasePairResult[] pairs;
            double confidence;
            string reason;
            bool ok = RowPhaseAlignmentMath.TryEstimateDynamicOffsets(
                frames,
                new[] { 0.0, 64.0 },
                new[] { 1000.0, 1000.0 },
                new[] { 0, 0 },
                30,
                out dynamicRows,
                out pairs,
                out confidence,
                out reason);

            Assert.That(ok, Is.False);
            Assert.That(reason, Does.StartWith("low-confidence"));
        }

        [Test]
        public void BuildCropPlans_UsesOnlyCommonValidRows()
        {
            const int width = 16;
            const int height = 100;
            var frames = new[]
            {
                Frame(1, new byte[width * height], width, height),
                Frame(2, new byte[width * height], width, height),
                Frame(3, new byte[width * height], width, height)
            };

            Dictionary<int, RowPhaseFramePlan> plans;
            bool ok = RowPhaseAlignmentMath.TryBuildCropPlans(
                frames,
                new[] { 5, -3, 0 },
                new[] { 0, -7, 2 },
                9,
                true,
                0.8,
                "trusted",
                out plans);

            Assert.That(ok, Is.True);
            Assert.That(plans[1].SourceTop, Is.EqualTo(0));
            Assert.That(plans[2].SourceTop, Is.EqualTo(15));
            Assert.That(plans[3].SourceTop, Is.EqualTo(3));
            Assert.That(plans[1].CommonHeight, Is.EqualTo(85));
            Assert.That(plans[2].CommonHeight, Is.EqualTo(85));
            Assert.That(plans[3].CommonHeight, Is.EqualTo(85));
        }

        [Test]
        public void Coordinator_ConcurrentCameraFrames_ReleaseOneCommonBatch()
        {
            const int width = 16;
            const int height = 100;
            var coordinator = new RowPhaseAlignmentCoordinator();
            coordinator.Configure(
                false,
                0,
                1000,
                new[] { 0.0, 1.0, 2.0 },
                new[] { 1000.0, 1000.0, 1000.0 },
                new[] { 5, -3, 0 });
            coordinator.Arm(new[] { 1, 2, 3 });

            Task<RowPhaseFramePlan>[] tasks =
            {
                Task.Run(() => coordinator.Align(
                    Frame(1, new byte[width * height], width, height))),
                Task.Run(() => coordinator.Align(
                    Frame(2, new byte[width * height], width, height))),
                Task.Run(() => coordinator.Align(
                    Frame(3, new byte[width * height], width, height)))
            };

            Assert.That(Task.WaitAll(tasks, 2000), Is.True);
            Assert.That(tasks[0].Result.Accepted, Is.True);
            Assert.That(tasks[1].Result.Accepted, Is.True);
            Assert.That(tasks[2].Result.Accepted, Is.True);
            Assert.That(tasks[0].Result.BatchId, Is.EqualTo(tasks[1].Result.BatchId));
            Assert.That(tasks[1].Result.BatchId, Is.EqualTo(tasks[2].Result.BatchId));
            Assert.That(tasks[0].Result.CommonHeight, Is.EqualTo(92));
            Assert.That(tasks[1].Result.CommonHeight, Is.EqualTo(92));
            Assert.That(tasks[2].Result.CommonHeight, Is.EqualTo(92));
        }

        private static RowPhaseFrameData Frame(
            int cameraId,
            byte[] pixels,
            int width,
            int height)
        {
            return new RowPhaseFrameData
            {
                CameraId = cameraId,
                Pixels = pixels,
                Width = width,
                Height = height
            };
        }

        private static byte[] BuildFrame(
            int width,
            int height,
            int physicalXStart,
            int globalRowOffset)
        {
            var pixels = new byte[width * height];
            for (int y = 0; y < height; y++)
            {
                int globalY = y + globalRowOffset;
                for (int x = 0; x < width; x++)
                {
                    int physicalX = physicalXStart + x;
                    int value =
                        100 +
                        Hash(globalY) % 70 +
                        Hash(physicalX / 4) % 20;
                    pixels[y * width + x] = (byte)Math.Max(0, Math.Min(255, value));
                }
            }
            return pixels;
        }

        private static int Hash(int value)
        {
            unchecked
            {
                uint x = (uint)(value + 10000);
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                return (int)(x & 0x7fffffff);
            }
        }

        private static byte[] Filled(int count, byte value)
        {
            var result = new byte[count];
            for (int i = 0; i < result.Length; i++)
                result[i] = value;
            return result;
        }
    }
}
