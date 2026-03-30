using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Moq;
using NUnit.Framework;
using PlcBridge.Core;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    /// <summary>
    /// 長時間壓力測試（跑一整晚用）。
    /// 使用 nunit3-console --where "cat == Stress" 單獨執行。
    /// </summary>
    [TestFixture, Category("Stress")]
    public class StressTests
    {
        // ── PLC FSM 循環壓力測試 ──

        [Test]
        public async Task PlcFsm_1000000Cycles_NoStateCorruption()
        {
            const int Cycles = 1000000;
            var mockPlc = new Mock<IModbusTcpClient>();
            mockPlc.SetupProperty(p => p.ReadWriteTimeoutMs, 2000);
            mockPlc.Setup(p => p.WriteDo(It.IsAny<int>(), It.IsAny<bool>())).Returns(Task.CompletedTask);
            mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);
            mockPlc.Setup(p => p.IsConnected).Returns(true);

            using (var ctrl = new PlcGrabController(mockPlc.Object))
            {
                int startCount = 0, stopCount = 0;
                ctrl.OnStartRequested += () => startCount++;
                ctrl.OnStopRequested  += () => stopCount++;

                await ctrl.StartAsync("192.168.255.1");
                Assert.That(ctrl.CurrentState, Is.EqualTo(PlcState.Idle));

                for (int i = 0; i < Cycles; i++)
                {
                    // Rising edge → Running
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, true, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(PlcState.Running),
                        $"Cycle {i}: expected Running after rising edge");

                    // Falling edge → Idle
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(PlcState.Idle),
                        $"Cycle {i}: expected Idle after falling edge");
                }

                Assert.That(startCount, Is.EqualTo(Cycles), $"Expected {Cycles} start events");
                Assert.That(stopCount, Is.EqualTo(Cycles), $"Expected {Cycles} stop events");
            }
        }

        [Test]
        public async Task PlcFsm_FaultRecoveryCycles_500000_NoLeak()
        {
            const int Cycles = 500000;
            var mockPlc = new Mock<IModbusTcpClient>();
            mockPlc.SetupProperty(p => p.ReadWriteTimeoutMs, 2000);
            mockPlc.Setup(p => p.WriteDo(It.IsAny<int>(), It.IsAny<bool>())).Returns(Task.CompletedTask);
            mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);
            mockPlc.Setup(p => p.IsConnected).Returns(true);

            using (var ctrl = new PlcGrabController(mockPlc.Object))
            {
                await ctrl.StartAsync("192.168.255.1");

                for (int i = 0; i < Cycles; i++)
                {
                    // PLC ALIVE lost → Faulted
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { false, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(PlcState.Faulted),
                        $"Cycle {i}: expected Faulted");

                    // PLC ALIVE restored → Idle
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(PlcState.Idle),
                        $"Cycle {i}: expected Idle after recovery");
                }
            }
        }

        // ── CSV 寫入耐久測試 ──

        [Test]
        public void CsvWrite_500000Records_NoDataloss()
        {
            const int RecordCount = 500000;
            string tempRoot = Path.Combine(Path.GetTempPath(), "StressCsv_" + Guid.NewGuid().ToString("N"));

            try
            {
                Directory.CreateDirectory(tempRoot);
                var svc = new InspectionLogService(() => tempRoot, startIdNum: 0);
                var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
                var config = new CsvConfigSnapshot(
                    new double[7], new double[7], 1.0f, 0.5f, 0.8f, ts);

                for (int i = 0; i < RecordCount; i++)
                {
                    string grabId = $"A{(i + 1):D5}";
                    int camId = (i % 7) + 1;
                    string fileName = $"{ts:yyyyMMdd_HHmmss}.{i:D3}-{camId}";
                    float meanPeak = (i % 2 == 0) ? 0.3f : 0.7f;
                    float maxPeak  = (i % 3 == 0) ? 1.0f : 0.5f;

                    svc.AppendRecord(grabId, fileName, meanPeak, maxPeak,
                        0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);
                }

                // Verify: read back CSV and count data lines
                string csvPath = Path.Combine(tempRoot, "2026", "202603", "20260330.csv");
                Assert.That(File.Exists(csvPath), Is.True);

                int dataLines = 0;
                foreach (string line in File.ReadAllLines(csvPath))
                {
                    if (!string.IsNullOrWhiteSpace(line) &&
                        !line.StartsWith("#") &&
                        !line.StartsWith("Id,"))
                        dataLines++;
                }
                Assert.That(dataLines, Is.EqualTo(RecordCount),
                    $"Expected {RecordCount} data lines, got {dataLines}");
            }
            finally
            {
                try { Directory.Delete(tempRoot, true); } catch { }
            }
        }

        // ── AcquisitionSettings Save/Load 循環壓力測試 ──

        [Test]
        public void AcquisitionSettings_145000SaveLoadCycles_NoDataloss()
        {
            const int Cycles = 145000;
            string tempDir = Path.Combine(Path.GetTempPath(), "StressAcq_" + Guid.NewGuid().ToString("N"));
            string configDir = Path.Combine(tempDir, "Config");

            try
            {
                Directory.CreateDirectory(configDir);
                string jsonPath = Path.Combine(configDir, "acquisition-settings.json");
                var rng = new Random(42);

                for (int i = 0; i < Cycles; i++)
                {
                    var settings = new AcquisitionSettings();
                    for (int c = 0; c < 7; c++)
                    {
                        settings.CameraGrabHeight[c] = rng.Next(1000, 10000);
                        settings.CameraExposureTimeUs[c] = rng.Next(10, 500);
                        settings.CameraLineRateHz[c] = rng.Next(1000, 20000);
                    }
                    settings.Validate();

                    // Save
                    string json = SerializeJson(settings);
                    File.WriteAllText(jsonPath, json);

                    // Load
                    string loaded = File.ReadAllText(jsonPath);
                    var parsed = ParseJson(loaded);
                    parsed.Validate();

                    // Verify
                    for (int c = 0; c < 7; c++)
                    {
                        Assert.That(parsed.CameraGrabHeight[c], Is.EqualTo(settings.CameraGrabHeight[c]),
                            $"Cycle {i}, cam {c}: GrabHeight mismatch");
                        Assert.That(parsed.CameraExposureTimeUs[c],
                            Is.EqualTo(settings.CameraExposureTimeUs[c]).Within(0.01),
                            $"Cycle {i}, cam {c}: ExposureTimeUs mismatch");
                        Assert.That(parsed.CameraLineRateHz[c],
                            Is.EqualTo(settings.CameraLineRateHz[c]).Within(0.01),
                            $"Cycle {i}, cam {c}: LineRateHz mismatch");
                    }
                }
            }
            finally
            {
                try { Directory.Delete(tempDir, true); } catch { }
            }
        }

        // ── CsvConfigSnapshot ToCsvLine/TryParse 大量循環 ──

        [Test]
        public void CsvConfigSnapshot_1000000RoundTrips_NoCorruption()
        {
            const int Cycles = 1000000;
            var rng = new Random(42);

            for (int i = 0; i < Cycles; i++)
            {
                var ops = new double[7];
                var pos = new double[7];
                for (int c = 0; c < 7; c++)
                {
                    ops[c] = Math.Round(rng.NextDouble() * 100, 2);
                    pos[c] = Math.Round(rng.NextDouble() * 200, 2);
                }
                float hessian = (float)Math.Round(rng.NextDouble() * 5, 4);
                float errMean = (float)Math.Round(rng.NextDouble() * 2, 4);
                float errMax  = (float)Math.Round(rng.NextDouble() * 3, 4);
                var ts = new DateTime(2026, 1, 1).AddSeconds(rng.Next(0, 31536000));
                // Truncate to millisecond precision
                ts = new DateTime(ts.Year, ts.Month, ts.Day, ts.Hour, ts.Minute, ts.Second, ts.Millisecond);

                var snap = new CsvConfigSnapshot(ops, pos, hessian, errMean, errMax, ts);
                string csv = snap.ToCsvLine();
                bool ok = CsvConfigSnapshot.TryParse(csv, out var parsed);

                Assert.That(ok, Is.True, $"Cycle {i}: TryParse failed");
                Assert.That(parsed.Timestamp, Is.EqualTo(ts), $"Cycle {i}: Timestamp mismatch");

                for (int c = 0; c < 7; c++)
                {
                    Assert.That(parsed.CamOps[c], Is.EqualTo(ops[c]).Within(0.01),
                        $"Cycle {i}, cam {c}: CamOps mismatch");
                    Assert.That(parsed.CamPos[c], Is.EqualTo(pos[c]).Within(0.01),
                        $"Cycle {i}, cam {c}: CamPos mismatch");
                }
            }
        }

        // ── InspectionStatisticsService 大量 CSV 統計壓力測試 ──

        [Test]
        public void Statistics_LargeCsv_200000Records_CorrectCounts()
        {
            const int RecordCount = 200000;
            string tempRoot = Path.Combine(Path.GetTempPath(), "StressStat_" + Guid.NewGuid().ToString("N"));

            try
            {
                string dir = Path.Combine(tempRoot, "2026", "202603");
                Directory.CreateDirectory(dir);
                string csvPath = Path.Combine(dir, "20260330.csv");

                var sb = new System.Text.StringBuilder();
                sb.AppendLine("Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs");

                int expectedPass = 0, expectedFail = 0;
                for (int i = 0; i < RecordCount; i++)
                {
                    string grabId = $"A{(i + 1):D5}";
                    int camId = 1; // all CAM1 for simplicity
                    int maxExceed  = (i % 5 == 0) ? 1 : 0;
                    int meanExceed = (i % 7 == 0) ? 1 : 0;
                    if (maxExceed == 0 && meanExceed == 0) expectedPass++;
                    else expectedFail++;

                    sb.AppendLine($"{grabId},20260330_100000.{i:D3}-{camId},{maxExceed},{meanExceed},0.3,0.6,3001,3001.0,149.0");
                }

                File.WriteAllText(csvPath, sb.ToString());

                var stats = InspectionStatisticsService.Compute(
                    tempRoot, new DateTime(2026, 3, 30), new DateTime(2026, 3, 31));

                Assert.That(stats[1].Pass, Is.EqualTo(expectedPass), "CAM1 pass count mismatch");
                Assert.That(stats[1].Fail, Is.EqualTo(expectedFail), "CAM1 fail count mismatch");
                Assert.That(stats[1].Total, Is.EqualTo(RecordCount));
            }
            finally
            {
                try { Directory.Delete(tempRoot, true); } catch { }
            }
        }

        // ── Mirror helpers for AcquisitionSettings JSON ──

        private static string SerializeJson(AcquisitionSettings s)
        {
            return "{\n" +
                $"  \"CameraGrabHeight\": [{string.Join(", ", s.CameraGrabHeight)}],\n" +
                $"  \"CameraExposureTimeUs\": [{string.Join(", ", s.CameraExposureTimeUs)}],\n" +
                $"  \"CameraLineRateHz\": [{string.Join(", ", s.CameraLineRateHz)}]\n" +
                "}";
        }

        private static AcquisitionSettings ParseJson(string json)
        {
            var result = new AcquisitionSettings();
            var ht = ParseArray(json, "CameraGrabHeight", s => int.Parse(s.Trim()));
            if (ht != null) result.CameraGrabHeight = ht;
            var exp = ParseDoubleArray(json, "CameraExposureTimeUs");
            if (exp != null) result.CameraExposureTimeUs = exp;
            var lr = ParseDoubleArray(json, "CameraLineRateHz");
            if (lr != null) result.CameraLineRateHz = lr;
            return result;
        }

        private static int[] ParseArray(string json, string key, Func<string, int> converter)
        {
            var pattern = new System.Text.RegularExpressions.Regex(
                "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
            var m = pattern.Match(json);
            if (!m.Success) return null;
            string body = m.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(body)) return new int[0];
            var parts = body.Split(',');
            var arr = new int[parts.Length];
            for (int i = 0; i < parts.Length; i++) arr[i] = converter(parts[i]);
            return arr;
        }

        private static double[] ParseDoubleArray(string json, string key)
        {
            var pattern = new System.Text.RegularExpressions.Regex(
                "\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
            var m = pattern.Match(json);
            if (!m.Success) return null;
            string body = m.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(body)) return new double[0];
            var parts = body.Split(',');
            var arr = new double[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                arr[i] = double.Parse(parts[i].Trim(), System.Globalization.CultureInfo.InvariantCulture);
            return arr;
        }
    }
}
