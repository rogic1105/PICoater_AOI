using System;
using System.Collections.Generic;
using System.Diagnostics;
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
    /// 長時間壓力測試。時間可透過環境變數 STRESS_MINUTES 調整（預設 60 分鐘）。
    /// 使用 run_tests.bat 選擇壓力測試，會自動詢問分鐘數。
    /// </summary>
    [TestFixture, Category("Stress")]
    public class StressTests
    {
        private const int NumStressTests = 6;

        // ── 讀取 STRESS_MINUTES 環境變數，預設 60 分鐘 ──
        private static double StressMinutes
        {
            get
            {
                var env = Environment.GetEnvironmentVariable("STRESS_MINUTES");
                if (!string.IsNullOrEmpty(env) &&
                    double.TryParse(env, System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out var m) && m > 0)
                    return m;
                return 60.0;
            }
        }

        /// <summary>
        /// 根據每分鐘基準 cycle 數，按總時間等分給每個測試。
        /// cyclesPerMinute: 該測試每分鐘可跑的 cycle 數（實測校準）。
        /// </summary>
        private static int Scale(int cyclesPerMinute) =>
            Math.Max(100, (int)(cyclesPerMinute * StressMinutes / NumStressTests));

        /// <summary>
        /// 印出測試開始資訊：名稱、cycle 數、預估時間、開始時間。
        /// </summary>
        private static Stopwatch LogStart(string testName, int cycles, int cyclesPerMinute)
        {
            double estMinutes = (double)cycles / cyclesPerMinute;
            string estStr = estMinutes < 1
                ? $"{estMinutes * 60:F0}s"
                : $"{estMinutes:F1}min";
            TestContext.Progress.WriteLine($"[{DateTime.Now:HH:mm:ss}] ▶ {testName}  cycles={cycles:N0}  est={estStr}");
            return Stopwatch.StartNew();
        }

        /// <summary>
        /// 印出測試完成資訊。
        /// </summary>
        private static void LogEnd(string testName, Stopwatch sw)
        {
            TestContext.Progress.WriteLine($"[{DateTime.Now:HH:mm:ss}] ✔ {testName}  elapsed={sw.Elapsed.TotalSeconds:F1}s");
        }

        /// <summary>
        /// 每 10% 印一次進度。回傳下一個要報告的門檻。
        /// </summary>
        private static int LogProgress(string testName, int i, int total, int nextThreshold)
        {
            if (i >= nextThreshold)
            {
                int pct = (int)((long)i * 100 / total);
                TestContext.Progress.WriteLine($"  [{DateTime.Now:HH:mm:ss}]   {testName}  {pct}%  ({i:N0}/{total:N0})");
                return nextThreshold + total / 10;
            }
            return nextThreshold;
        }

        // =====================================================================
        //  每分鐘基準 cycle 數（STRESS_MINUTES=10 實測校準）：
        //    PLC FSM:     16,000 /min
        //    PLC Fault:   17,000 /min
        //    CSV Write:   670,000 /min
        //    Settings RW: 15,600 /min
        //    CsvConfig:   2,070,000 /min
        //    Statistics:  2,500,000 /min (真實場景：每日14K筆CSV生成+Compute)
        // =====================================================================

        // ── PLC FSM 循環壓力測試 ──

        [Test]
        public async Task PlcFsm_1000000Cycles_NoStateCorruption()
        {
            const int Rate = 16_000;
            int Cycles = Scale(Rate);
            var sw = LogStart("PlcFsm", Cycles, Rate);
            int nextLog = Cycles / 10;

            var mockPlc = new Mock<IModbusTcpClient>();
            mockPlc.SetupProperty(p => p.ReadWriteTimeoutMs, 2000);
            mockPlc.Setup(p => p.WriteDo(It.IsAny<int>(), It.IsAny<bool>())).Returns(Task.CompletedTask);
            mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);
            mockPlc.Setup(p => p.IsConnected).Returns(true);

            using (var ctrl = new IoGrabController(mockPlc.Object))
            {
                int startCount = 0, stopCount = 0;
                ctrl.OnStartRequested += () => startCount++;
                ctrl.OnStopRequested  += () => stopCount++;

                await ctrl.StartAsync("192.168.255.1");
                Assert.That(ctrl.CurrentState, Is.EqualTo(IoState.Idle));

                for (int i = 0; i < Cycles; i++)
                {
                    // Rising edge → Running
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, true, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(IoState.Running),
                        $"Cycle {i}: expected Running after rising edge");

                    // Falling edge → Idle
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(IoState.Idle),
                        $"Cycle {i}: expected Idle after falling edge");

                    nextLog = LogProgress("PlcFsm", i, Cycles, nextLog);
                }

                Assert.That(startCount, Is.EqualTo(Cycles), $"Expected {Cycles} start events");
                Assert.That(stopCount, Is.EqualTo(Cycles), $"Expected {Cycles} stop events");
            }

            LogEnd("PlcFsm", sw);
        }

        [Test]
        public async Task PlcFsm_FaultRecoveryCycles_500000_NoLeak()
        {
            const int Rate = 17_000;
            int Cycles = Scale(Rate);
            var sw = LogStart("PlcFaultRecovery", Cycles, Rate);
            int nextLog = Cycles / 10;

            var mockPlc = new Mock<IModbusTcpClient>();
            mockPlc.SetupProperty(p => p.ReadWriteTimeoutMs, 2000);
            mockPlc.Setup(p => p.WriteDo(It.IsAny<int>(), It.IsAny<bool>())).Returns(Task.CompletedTask);
            mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);
            mockPlc.Setup(p => p.IsConnected).Returns(true);

            using (var ctrl = new IoGrabController(mockPlc.Object))
            {
                await ctrl.StartAsync("192.168.255.1");

                for (int i = 0; i < Cycles; i++)
                {
                    // PLC ALIVE lost → Faulted
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { false, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(IoState.Faulted),
                        $"Cycle {i}: expected Faulted");

                    // PLC ALIVE restored → Idle
                    mockPlc.Setup(p => p.ReadDiStatuses())
                        .ReturnsAsync(new bool[] { true, false, false, false, false, false, false, false });
                    await ctrl.PollTick();
                    Assert.That(ctrl.CurrentState, Is.EqualTo(IoState.Idle),
                        $"Cycle {i}: expected Idle after recovery");

                    nextLog = LogProgress("PlcFaultRecovery", i, Cycles, nextLog);
                }
            }

            LogEnd("PlcFaultRecovery", sw);
        }

        // ── CSV 寫入耐久測試 ──

        [Test]
        public void CsvWrite_500000Records_NoDataloss()
        {
            const int Rate = 670_000;
            int RecordCount = Scale(Rate);
            var sw = LogStart("CsvWrite", RecordCount, Rate);
            int nextLog = RecordCount / 10;

            string tempRoot = Path.Combine(Path.GetTempPath(), "StressCsv_" + Guid.NewGuid().ToString("N"));

            try
            {
                Directory.CreateDirectory(tempRoot);
                var svc = new InspectionLogService(() => tempRoot);
                var ts = new DateTime(2026, 3, 30, 10, 0, 0, 0);
                var config = new CsvConfigSnapshot(
                    new double[7], new double[7], null, null, null, 1.0f, 0.5f, 0.8f, ts);

                for (int i = 0; i < RecordCount; i++)
                {
                    var grabTs = ts.AddSeconds(i);
                    string grabId = InspectionLogService.FormatGrabId(grabTs);
                    int camId = (i % 7) + 1;
                    string fileName = $"{ts:yyyyMMdd_HHmmss}.{i:D3}-{camId}";
                    float meanPeak = (i % 2 == 0) ? 0.3f : 0.7f;
                    float maxPeak  = (i % 3 == 0) ? 1.0f : 0.5f;

                    svc.AppendRecord(grabId, fileName, meanPeak, maxPeak,
                        0.5f, 0.8f, 3001, 3001.0, 149.0, config, ts);

                    nextLog = LogProgress("CsvWrite", i, RecordCount, nextLog);
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

            LogEnd("CsvWrite", sw);
        }

        // ── AcquisitionSettings Save/Load 循環壓力測試 ──

        [Test]
        public void AcquisitionSettings_145000SaveLoadCycles_NoDataloss()
        {
            const int Rate = 15_600;
            int Cycles = Scale(Rate);
            var sw = LogStart("SettingsRW", Cycles, Rate);
            int nextLog = Cycles / 10;

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

                    nextLog = LogProgress("SettingsRW", i, Cycles, nextLog);
                }
            }
            finally
            {
                try { Directory.Delete(tempDir, true); } catch { }
            }

            LogEnd("SettingsRW", sw);
        }

        // ── CsvConfigSnapshot ToCsvLine/TryParse 大量循環 ──

        [Test]
        public void CsvConfigSnapshot_1000000RoundTrips_NoCorruption()
        {
            const int Rate = 2_070_000;
            int Cycles = Scale(Rate);
            var sw = LogStart("CsvConfigRoundTrip", Cycles, Rate);
            int nextLog = Cycles / 10;

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
                var grabH = new int[7];
                var expUs = new double[7];
                var lrHz = new double[7];
                for (int c2 = 0; c2 < 7; c2++)
                {
                    grabH[c2] = rng.Next(1000, 10000);
                    expUs[c2] = Math.Round(rng.NextDouble() * 500, 2);
                    lrHz[c2] = Math.Round(rng.NextDouble() * 10000, 2);
                }
                float hessian = (float)Math.Round(rng.NextDouble() * 5, 4);
                float errMean = (float)Math.Round(rng.NextDouble() * 2, 4);
                float errMax  = (float)Math.Round(rng.NextDouble() * 3, 4);
                var ts = new DateTime(2026, 1, 1).AddSeconds(rng.Next(0, 31536000));
                // Truncate to millisecond precision
                ts = new DateTime(ts.Year, ts.Month, ts.Day, ts.Hour, ts.Minute, ts.Second, ts.Millisecond);

                var snap = new CsvConfigSnapshot(ops, pos, grabH, expUs, lrHz, hessian, errMean, errMax, ts);
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
                    Assert.That(parsed.CamGrabHeight[c], Is.EqualTo(grabH[c]),
                        $"Cycle {i}, cam {c}: CamGrabHeight mismatch");
                    Assert.That(parsed.CamExposureUs[c], Is.EqualTo(expUs[c]).Within(0.01),
                        $"Cycle {i}, cam {c}: CamExposureUs mismatch");
                    Assert.That(parsed.CamLineRateHz[c], Is.EqualTo(lrHz[c]).Within(0.01),
                        $"Cycle {i}, cam {c}: CamLineRateHz mismatch");
                }

                nextLog = LogProgress("CsvConfigRoundTrip", i, Cycles, nextLog);
            }

            LogEnd("CsvConfigRoundTrip", sw);
        }

        // ── InspectionStatisticsService 真實月資料量壓力測試 ──
        //
        //  模擬真實場景：每天 2000 次 grab × 7 台相機 = 14,000 筆/天
        //  一個月 30 天 = 420,000 筆，分散在 30 個 CSV 檔案中。
        //  Scale 控制月數：STRESS_MINUTES=6 → 1 個月，60 → 10 個月。

        [Test]
        public void Statistics_RealisticMonthly_MultiDayCsv()
        {
            const int GrabsPerDay = 2000;
            const int NumCameras = 7;
            const int RecordsPerDay = GrabsPerDay * NumCameras; // 14,000
            // Rate 校準：1 分鐘可生成+計算 ~2,500,000 筆
            // STRESS_MINUTES=1 → 30 天(420K筆), =10 → 300 天, =60 → 1800 天
            const int Rate = 2_500_000;

            int totalRecords = Scale(Rate);
            int totalDays = Math.Max(1, totalRecords / RecordsPerDay);
            totalRecords = totalDays * RecordsPerDay;

            var sw = LogStart("Statistics", totalRecords, Rate);
            TestContext.Progress.WriteLine(
                $"  [{DateTime.Now:HH:mm:ss}]   Statistics  {totalDays} days x {RecordsPerDay:N0} records/day = {totalRecords:N0} total");

            string tempRoot = Path.Combine(Path.GetTempPath(), "StressStat_" + Guid.NewGuid().ToString("N"));

            try
            {
                // ── Phase 1: 生成每日 CSV ──
                var genSw = Stopwatch.StartNew();
                var baseDate = new DateTime(2026, 1, 1);
                int globalGrabIdx = 0;
                var expectedPerCam = new Dictionary<int, int[]>(); // camId → [pass, fail]
                for (int c = 1; c <= NumCameras; c++)
                    expectedPerCam[c] = new int[2]; // [0]=pass, [1]=fail

                for (int day = 0; day < totalDays; day++)
                {
                    var date = baseDate.AddDays(day);
                    string dir = Path.Combine(tempRoot,
                        date.ToString("yyyy"),
                        date.ToString("yyyyMM"));
                    Directory.CreateDirectory(dir);
                    string csvPath = Path.Combine(dir, date.ToString("yyyyMMdd") + ".csv");

                    var sb = new System.Text.StringBuilder();
                    sb.AppendLine("Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs");

                    for (int g = 0; g < GrabsPerDay; g++)
                    {
                        globalGrabIdx++;
                        // 每次 grab 產生 7 張照片（7 台相機）
                        var ts = date.AddSeconds(g * 30); // 每 30 秒一次 grab
                        string grabId = InspectionLogService.FormatGrabId(ts);
                        string tsStr = ts.ToString("yyyyMMdd_HHmmss") + ".000";

                        for (int c = 1; c <= NumCameras; c++)
                        {
                            int maxExceed  = ((globalGrabIdx + c) % 5 == 0) ? 1 : 0;
                            int meanExceed = ((globalGrabIdx + c) % 7 == 0) ? 1 : 0;
                            if (maxExceed == 0 && meanExceed == 0)
                                expectedPerCam[c][0]++;
                            else
                                expectedPerCam[c][1]++;

                            sb.AppendLine($"{grabId},{tsStr}-{c},{maxExceed},{meanExceed},0.3,0.6,3001,3001.0,149.0");
                        }
                    }

                    File.WriteAllText(csvPath, sb.ToString());

                    // 每 10% 報告一次
                    if (totalDays >= 10 && (day + 1) % (totalDays / 10) == 0)
                    {
                        int pct = (int)((long)(day + 1) * 100 / totalDays);
                        TestContext.Progress.WriteLine(
                            $"  [{DateTime.Now:HH:mm:ss}]   Statistics  gen {pct}%  ({day + 1}/{totalDays} days)");
                    }
                }
                genSw.Stop();
                TestContext.Progress.WriteLine(
                    $"  [{DateTime.Now:HH:mm:ss}]   Statistics  CSV generation done ({genSw.Elapsed.TotalSeconds:F1}s)");

                // ── Phase 2: Compute 統計 ──
                var compSw = Stopwatch.StartNew();
                var endDate = baseDate.AddDays(totalDays);
                var stats = InspectionStatisticsService.Compute(tempRoot, baseDate, endDate);
                compSw.Stop();
                TestContext.Progress.WriteLine(
                    $"  [{DateTime.Now:HH:mm:ss}]   Statistics  Compute done ({compSw.Elapsed.TotalSeconds:F1}s, " +
                    $"{totalRecords / Math.Max(0.001, compSw.Elapsed.TotalSeconds):F0} records/sec)");

                // ── Phase 3: 驗證每台相機統計 ──
                for (int c = 1; c <= NumCameras; c++)
                {
                    Assert.That(stats[c].Pass, Is.EqualTo(expectedPerCam[c][0]),
                        $"CAM{c} pass count mismatch");
                    Assert.That(stats[c].Fail, Is.EqualTo(expectedPerCam[c][1]),
                        $"CAM{c} fail count mismatch");
                    Assert.That(stats[c].Total, Is.EqualTo(GrabsPerDay * totalDays),
                        $"CAM{c} total count mismatch");
                }
            }
            finally
            {
                try { Directory.Delete(tempRoot, true); } catch { }
            }

            LogEnd("Statistics", sw);
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
