using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionMuraProfileRepositoryTests
    {
        private string _tempRoot;

        [SetUp]
        public void SetUp()
        {
            _tempRoot = Path.Combine(Path.GetTempPath(),
                "InspectionMuraProfileRepositoryTests_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(_tempRoot);
        }

        [TearDown]
        public void TearDown()
        {
            try { Directory.Delete(_tempRoot, true); } catch { }
        }

        [Test]
        public void LoadAverage_MultipleGrabs_AveragesMeanAndTakesPointwiseMax()
        {
            DateTime first = new DateTime(2026, 7, 13, 10, 0, 0);
            DateTime second = first.AddMinutes(1);
            WriteCurves(first, new[] { 10f, 30f }, new[] { 40f, 20f });
            WriteCurves(second, new[] { 30f, 50f }, new[] { 20f, 60f });
            var infos = new List<GrabIdInfo>
            {
                new GrabIdInfo { GrabId = "260713-100000", Earliest = first },
                new GrabIdInfo { GrabId = "260713-100100", Earliest = second }
            };

            var result = InspectionMuraProfileRepository.LoadAverage(_tempRoot, infos);

            Assert.That(result.Mean[1], Is.EqualTo(new[] { 20f, 40f }));
            Assert.That(result.Max[1], Is.EqualTo(new[] { 40f, 60f }));
        }

        [Test]
        public void CurveBinFile_LoadVersion2_ReturnsExactPayload()
        {
            string path = Path.Combine(_tempRoot, "curve-v2.bin");
            WriteCurveBin(path, new[] { -1.5f, 0f, 300.25f }, 2);

            var result = CurveBinFile.Load(path);

            Assert.That(result, Is.EqualTo(new[] { -1.5f, 0f, 300.25f }));
        }

        [Test]
        public void MergeCurves_DifferentLengths_PreservesPointwiseMeanAndMax()
        {
            string[] imagePaths =
            {
                WriteCurvePair("capture-1", new[] { 1f, 2f }, new[] { 5f, 4f }),
                WriteCurvePair("capture-2", new[] { 3f, 4f }, new[] { 4f, 6f }),
                WriteCurvePair("capture-3", new[] { 5f }, new[] { 7f })
            };

            CurveMergeHelper.MergeCurves(imagePaths, out var mean, out var max);

            Assert.That(mean, Is.EqualTo(new[] { 3f, 3f }));
            Assert.That(max, Is.EqualTo(new[] { 7f, 6f }));
        }

        [Test]
        public void MergeCurves_MissingPair_ReportsOnlyCompleteCaptures()
        {
            string complete = WriteCurvePair(
                "capture-complete", new[] { 1f }, new[] { 2f });
            string incomplete = Path.Combine(
                _tempRoot, "capture-incomplete" + CaptureFileNaming.RawJpg);
            WriteCurveBin(
                CaptureFileNaming.StripRawJpg(incomplete) + CaptureFileNaming.MeanC,
                new[] { 9f });

            CurveMergeHelper.MergeCurves(
                new[] { complete, incomplete }, out var mean, out var max,
                out int mergedCount, System.Threading.CancellationToken.None);

            Assert.That(mergedCount, Is.EqualTo(1));
            Assert.That(mean, Is.EqualTo(new[] { 1f }));
            Assert.That(max, Is.EqualTo(new[] { 2f }));
        }

        private void WriteCurves(DateTime timestamp, float[] mean, float[] max)
        {
            string directory = CaptureStoragePaths.DateImageDir(_tempRoot, timestamp);
            Directory.CreateDirectory(directory);
            string baseName = timestamp.ToString("yyyyMMdd_HHmmss.fff") + "-1";
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MeanC), mean);
            WriteCurveBin(Path.Combine(directory, baseName + CaptureFileNaming.MaxC), max);
        }

        private string WriteCurvePair(string fileName, float[] mean, float[] max)
        {
            string imagePath = Path.Combine(_tempRoot, fileName + CaptureFileNaming.RawJpg);
            string basePath = CaptureFileNaming.StripRawJpg(imagePath);
            WriteCurveBin(basePath + CaptureFileNaming.MeanC, mean);
            WriteCurveBin(basePath + CaptureFileNaming.MaxC, max);
            return imagePath;
        }

        private static void WriteCurveBin(string path, float[] values, int version = 1)
        {
            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                writer.Write(version);
                writer.Write(1.0f);
                if (version >= 2)
                {
                    writer.Write(128);
                    writer.Write(50.0f);
                }
                writer.Write(values.Length);
                foreach (float value in values) writer.Write(value);
            }
        }
    }
}
