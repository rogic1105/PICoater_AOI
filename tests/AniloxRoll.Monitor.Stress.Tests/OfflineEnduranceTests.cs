using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using IoBridge.Core;
using NUnit.Framework;
using StorageBridge.Core;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Category("Soak")]
    [NonParallelizable]
    public sealed class OfflineEnduranceTests
    {
        private static TimeSpan Duration
        {
            get
            {
                string value = Environment.GetEnvironmentVariable("SOAK_MINUTES");
                if (double.TryParse(
                        value,
                        NumberStyles.Float,
                        CultureInfo.InvariantCulture,
                        out double minutes) &&
                    minutes > 0)
                {
                    return TimeSpan.FromMinutes(minutes);
                }

                return TimeSpan.FromMinutes(120);
            }
        }

        [Test]
        public async Task MixedIoStorageAndPersistence_RemainsBounded()
        {
            string tempRoot = Path.Combine(
                Path.GetTempPath(),
                "AniloxOfflineSoak_" + Guid.NewGuid().ToString("N"));
            string captureRoot = Path.Combine(tempRoot, "captures");
            string localRoot = Path.Combine(tempRoot, "local");
            string remoteRoot = Path.Combine(tempRoot, "remote");
            Directory.CreateDirectory(captureRoot);
            Directory.CreateDirectory(localRoot);
            Directory.CreateDirectory(remoteRoot);

            var fakePlc = new SoakModbusTcpClient();

            var stopwatch = Stopwatch.StartNew();
            var samples = new List<ResourceSample>();
            DateTime nextSample = DateTime.UtcNow;
            int cycles = 0;
            int startCount = 0;
            int stopCount = 0;
            int enqueuedFiles = 0;
            int statisticsPasses = 0;
            var baseTime = new DateTime(2026, 7, 29, 0, 0, 0);
            var config = new CsvConfigSnapshot(
                new double[7],
                new double[7],
                new int[7],
                new double[7],
                new double[7],
                0.3f,
                0.3f,
                0.2f,
                0.6f,
                0.2f,
                0.6f,
                0.0,
                0.0,
                baseTime);
            var logService = new InspectionLogService(() => captureRoot);

            TestContext.Progress.WriteLine(
                $"Offline soak started: budget={Duration.TotalMinutes:F2} min");

            try
            {
                using (var controller = new IoGrabController(fakePlc)
                {
                    AutoBackgroundLoop = false
                })
                using (var remoteCopy = new RemoteCopyService(
                    () => remoteRoot,
                    () => localRoot))
                {
                    controller.OnStartRequested += () => startCount++;
                    controller.OnStopRequested += reason => stopCount++;
                    await controller.StartAsync("127.0.0.1");

                    while (cycles < 100 || stopwatch.Elapsed < Duration)
                    {
                        cycles++;

                        fakePlc.StartSignal = true;
                        await controller.PollTick();
                        fakePlc.StartSignal = false;
                        await controller.PollTick();
                        Assert.That(
                            controller.CurrentState,
                            Is.EqualTo(IoState.Idle),
                            $"IO state did not return to Idle at cycle {cycles}");

                        string configLine = config.ToCsvLine();
                        Assert.That(
                            CsvConfigSnapshot.TryParse(
                                configLine,
                                out CsvConfigSnapshot parsed),
                            Is.True,
                            $"CFG parse failed at cycle {cycles}");
                        Assert.That(parsed.Timestamp, Is.EqualTo(config.Timestamp));

                        if (cycles % 40 == 0)
                        {
                            DateTime timestamp = baseTime.AddSeconds(cycles / 40);
                            string grabId = InspectionLogService.FormatGrabId(timestamp);
                            for (int camId = 1; camId <= 7; camId++)
                            {
                                logService.AppendRecord(
                                    grabId,
                                    $"{timestamp:yyyyMMdd_HHmmss}.000-{camId}",
                                    0.3f,
                                    0.6f,
                                    0.2f,
                                    0.6f,
                                    3000,
                                    3000.0,
                                    50.0,
                                    config,
                                    timestamp);
                            }
                        }

                        if (cycles == 10 || cycles % 400 == 0)
                        {
                            string dayDirectory = Path.Combine(
                                localRoot,
                                "2026",
                                "202607",
                                "20260729");
                            Directory.CreateDirectory(dayDirectory);
                            for (int fileIndex = 0; fileIndex < 5; fileIndex++)
                            {
                                string source = Path.Combine(
                                    dayDirectory,
                                    $"capture-{cycles:D8}-{fileIndex}.bin");
                                File.WriteAllText(
                                    source,
                                    $"cycle={cycles};file={fileIndex}");
                                remoteCopy.EnqueueFile(source);
                                enqueuedFiles++;
                            }
                        }

                        if (cycles == 20 || cycles % 4000 == 0)
                        {
                            var statistics = InspectionStatisticsService.Compute(
                                captureRoot,
                                baseTime.Date,
                                baseTime.Date.AddDays(1));
                            Assert.That(statistics.Count, Is.EqualTo(7));
                            statisticsPasses++;
                        }

                        if (DateTime.UtcNow >= nextSample)
                        {
                            ResourceSample sample = CaptureResourceSample(
                                stopwatch.Elapsed.TotalSeconds);
                            samples.Add(sample);
                            TestContext.Progress.WriteLine(
                                $"  elapsed={stopwatch.Elapsed.TotalMinutes:F1}min " +
                                $"cycles={cycles:N0} queue={remoteCopy.QueueCount:N0} " +
                                $"private={sample.PrivateMB:F1}MB " +
                                $"handles={sample.Handles} threads={sample.Threads}");
                            nextSample = DateTime.UtcNow.AddMinutes(1);
                        }

                        await Task.Delay(25);
                    }

                    await WaitUntilQueueEmpty(remoteCopy, TimeSpan.FromMinutes(2));
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                    samples.Add(CaptureResourceSample(
                        stopwatch.Elapsed.TotalSeconds));

                    Assert.That(startCount, Is.EqualTo(cycles));
                    Assert.That(stopCount, Is.EqualTo(cycles));
                    Assert.That(remoteCopy.QueueCount, Is.Zero);
                    Assert.That(
                        remoteCopy.TotalCopiedFiles,
                        Is.EqualTo(enqueuedFiles));
                    Assert.That(
                        Directory.GetFiles(
                            remoteRoot,
                            "*.part-*",
                            SearchOption.AllDirectories),
                        Is.Empty);

                    AssertResourceTrend(samples, Duration);
                }
            }
            finally
            {
                stopwatch.Stop();
                try
                {
                    Directory.Delete(tempRoot, true);
                }
                catch (Exception ex)
                {
                    TestContext.Progress.WriteLine(
                        "Soak cleanup failed: " + ex.Message);
                }
            }

            TestContext.Progress.WriteLine(
                $"Offline soak completed: elapsed={stopwatch.Elapsed.TotalSeconds:F1}s " +
                $"cycles={cycles:N0} copied={enqueuedFiles:N0} " +
                $"statistics={statisticsPasses:N0}");
        }

        private static async Task WaitUntilQueueEmpty(
            RemoteCopyService service,
            TimeSpan timeout)
        {
            var stopwatch = Stopwatch.StartNew();
            while (service.QueueCount > 0 && stopwatch.Elapsed < timeout)
                await Task.Delay(25);

            Assert.That(
                service.QueueCount,
                Is.Zero,
                "Remote copy queue did not drain before the endurance timeout.");
        }

        private static ResourceSample CaptureResourceSample(
            double elapsedSeconds)
        {
            using (Process process = Process.GetCurrentProcess())
            {
                process.Refresh();
                ProcessThreadCollection threads = process.Threads;
                int threadCount = threads.Count;
                foreach (ProcessThread thread in threads)
                    thread.Dispose();

                return new ResourceSample
                {
                    ElapsedSeconds = elapsedSeconds,
                    PrivateMB = process.PrivateMemorySize64 / 1024.0 / 1024.0,
                    Handles = process.HandleCount,
                    Threads = threadCount
                };
            }
        }

        private static void AssertResourceTrend(
            IList<ResourceSample> samples,
            TimeSpan duration)
        {
            Assert.That(samples.Count, Is.GreaterThanOrEqualTo(2));

            double warmupSeconds = Math.Min(
                duration.TotalSeconds / 4.0,
                300.0);
            ResourceSample first = null;
            for (int index = 0; index < samples.Count; index++)
            {
                if (samples[index].ElapsedSeconds >= warmupSeconds)
                {
                    first = samples[index];
                    break;
                }
            }
            if (first == null)
                first = samples[0];

            ResourceSample last = samples[samples.Count - 1];
            double privateDelta = last.PrivateMB - first.PrivateMB;
            int handleDelta = last.Handles - first.Handles;
            int threadDelta = last.Threads - first.Threads;

            TestContext.Progress.WriteLine(
                $"Resource trend after warmup: private={first.PrivateMB:F1}" +
                $"->{last.PrivateMB:F1}MB ({privateDelta:+0.0;-0.0;0.0}), " +
                $"handles={first.Handles}->{last.Handles} ({handleDelta:+#;-#;0}), " +
                $"threads={first.Threads}->{last.Threads} ({threadDelta:+#;-#;0})");

            Assert.That(
                privateDelta,
                Is.LessThanOrEqualTo(512.0),
                "Offline endurance Private Bytes grew beyond the 512 MB guard.");
            Assert.That(
                handleDelta,
                Is.LessThanOrEqualTo(50),
                "Offline endurance handles grew beyond the 50-handle guard.");
            Assert.That(
                threadDelta,
                Is.LessThanOrEqualTo(15),
                "Offline endurance threads grew beyond the 15-thread guard.");
        }

        private sealed class ResourceSample
        {
            public double ElapsedSeconds { get; set; }
            public double PrivateMB { get; set; }
            public int Handles { get; set; }
            public int Threads { get; set; }
        }

        private sealed class SoakModbusTcpClient : IModbusTcpClient
        {
            private readonly bool[] _diStatuses = new bool[8];
            private readonly bool[] _doStatuses = new bool[8];

            public SoakModbusTcpClient()
            {
                _diStatuses[0] = true;
                IsConnected = true;
                ReadWriteTimeoutMs = 2000;
            }

            public int ReadWriteTimeoutMs { get; set; }
            public bool IsConnected { get; private set; }

            public bool StartSignal
            {
                set { _diStatuses[1] = value; }
            }

            public Task<bool> ConnectAsync(
                string ip,
                int port = 502,
                int timeoutMs = 5000)
            {
                IsConnected = true;
                return Task.FromResult(true);
            }

            public Task<bool[]> ReadDoStatuses()
            {
                return Task.FromResult(_doStatuses);
            }

            public Task<bool[]> ReadDiStatuses()
            {
                return Task.FromResult(_diStatuses);
            }

            public Task WriteDo(int index, bool value)
            {
                _doStatuses[index] = value;
                return Task.CompletedTask;
            }

            public void Dispose()
            {
                IsConnected = false;
            }
        }
    }
}
