using System;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class InspectionCsvReaderTests
    {
        [Test]
        public void TryParseRecord_LegacyFourColumns_ReturnsRequiredFields()
        {
            bool parsed = InspectionCsvReader.TryParseRecord(
                "260713-120000,20260713_120000.123-3,1,0", out var record);

            Assert.That(parsed, Is.True);
            Assert.That(record.GrabId, Is.EqualTo("260713-120000"));
            Assert.That(record.FileName, Is.EqualTo("20260713_120000.123-3"));
            Assert.That(record.MaxExceed, Is.EqualTo(1));
            Assert.That(record.MeanExceed, Is.EqualTo(0));
            Assert.That(float.IsNaN(record.MaxCMean), Is.True);
        }

        [Test]
        public void TryParseRecord_CurrentTenColumns_ReturnsAllFields()
        {
            bool parsed = InspectionCsvReader.TryParseRecord(
                "260713-120000,20260713_120000.123-3,1,0,0.25,0.5,3001,6000.5,50.25,0.75",
                out var record);

            Assert.That(parsed, Is.True);
            Assert.That(record.MeanPeak, Is.EqualTo(0.25f));
            Assert.That(record.MaxPeak, Is.EqualTo(0.5f));
            Assert.That(record.GrabHeight, Is.EqualTo(3001));
            Assert.That(record.LineRateHz, Is.EqualTo(6000.5d));
            Assert.That(record.ExposureUs, Is.EqualTo(50.25d));
            Assert.That(record.MaxCMean, Is.EqualTo(0.75f));
            Assert.That(float.IsNaN(record.MeanRPeak), Is.True);
            Assert.That(float.IsNaN(record.MaxRPeak), Is.True);
        }

        [Test]
        public void TryParseRecord_CurrentTwelveColumns_ReturnsRowPeaks()
        {
            bool parsed = InspectionCsvReader.TryParseRecord(
                "260713-120000,20260713_120000.123-3,1,0,0.25,0.5,3001,6000.5,50.25,0.75,0.125,0.875",
                out var record);

            Assert.That(parsed, Is.True);
            Assert.That(record.MeanRPeak, Is.EqualTo(0.125f));
            Assert.That(record.MaxRPeak, Is.EqualTo(0.875f));
        }

        [TestCase("")]
        [TestCase("#CFG,ignored")]
        [TestCase("Id,FileName,MaxExceed,MeanExceed")]
        public void TryParseRecord_NonDataLine_ReturnsFalse(string line)
        {
            Assert.That(InspectionCsvReader.TryParseRecord(line, out _), Is.False);
        }

        [Test]
        public void TimestampAndCameraId_ValidFileName_ReturnExpectedValues()
        {
            const string fileName = "20260713_120000.123-7";

            Assert.That(InspectionCsvReader.TryParseTimestamp(fileName, out var timestamp), Is.True);
            Assert.That(timestamp, Is.EqualTo(new DateTime(2026, 7, 13, 12, 0, 0, 123)));
            Assert.That(InspectionCsvReader.TryExtractCameraId(fileName, out int cameraId), Is.True);
            Assert.That(cameraId, Is.EqualTo(7));
        }

        [Test]
        public void TryUpdateHmFromConfig_ConfigLine_UpdatesCaptureValue()
        {
            var snapshot = new CsvConfigSnapshot(
                new double[7], new double[7], new int[7], new double[7], new double[7],
                3.25f, 1f, 0f, 0f, 0f, 0f, 0d, 0d,
                new DateTime(2026, 7, 13, 12, 0, 0));
            float captureHmV = 1f;

            bool handled = InspectionCsvReader.TryUpdateHmFromConfig(
                snapshot.ToCsvLine(), ref captureHmV);

            Assert.That(handled, Is.True);
            Assert.That(captureHmV, Is.EqualTo(3.25f));
        }
    }
}
