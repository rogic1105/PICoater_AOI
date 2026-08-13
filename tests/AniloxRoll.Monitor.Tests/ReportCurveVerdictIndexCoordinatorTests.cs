using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReportCurveVerdictIndexCoordinatorTests
    {
        [Test]
        public void Start_NewGenerationPreventsOlderResultFromApplying()
        {
            var index = ReadyIndex("D:\\data", "old", "new");
            var firstStarted = new ManualResetEventSlim();
            var releaseFirst = new ManualResetEventSlim();
            int call = 0;
            BuildCurvePeakSummaries build = (root, infos, hm, configs, count, token) =>
            {
                if (Interlocked.Increment(ref call) == 1)
                {
                    firstStarted.Set();
                    releaseFirst.Wait(2000);
                    return Result("old");
                }
                return Result("new");
            };

            using (var coordinator = Create(index, () => "D:\\data", build))
            {
                coordinator.Start(false, "D:\\data", Infos("old"), EmptyHm(), EmptyConfigs(), 1);
                Assert.That(firstStarted.Wait(1000), Is.True, "first generation should start");
                coordinator.Start(false, "D:\\data", Infos("new"), EmptyHm(), EmptyConfigs(), 1);
                Assert.That(SpinWait.SpinUntil(
                    () => index.ColumnPeaks.ContainsKey("new"), 2000), Is.True);

                releaseFirst.Set();
                Thread.Sleep(100);

                Assert.That(index.ColumnPeaks.ContainsKey("new"), Is.True);
                Assert.That(index.ColumnPeaks.ContainsKey("old"), Is.False);
            }
        }

        [Test]
        public void Dispose_BeforePostedCompletion_PreventsResultFromApplying()
        {
            var index = ReadyIndex("D:\\data", "g1");
            Action posted = null;
            var postedReady = new ManualResetEventSlim();
            var coordinator = Create(
                index,
                () => "D:\\data",
                (root, infos, hm, configs, count, token) => Result("g1"),
                action =>
                {
                    posted = action;
                    postedReady.Set();
                    return true;
                });

            coordinator.Start(false, "D:\\data", Infos("g1"), EmptyHm(), EmptyConfigs(), 1);
            Assert.That(postedReady.Wait(1000), Is.True, "completion should be queued");
            coordinator.Dispose();

            posted();

            Assert.That(index.ColumnPeaks.ContainsKey("g1"), Is.False);
        }

        [Test]
        public void Start_BackgroundFailure_LeavesExplicitFlowEvidence()
        {
            var index = ReadyIndex("D:\\data", "g1");
            var logs = new ConcurrentQueue<string>();
            using (var coordinator = Create(
                index,
                () => "D:\\data",
                (root, infos, hm, configs, count, token) =>
                    throw new InvalidOperationException("broken"),
                action => { action(); return true; },
                logs.Enqueue))
            {
                coordinator.Start(false, "D:\\data", Infos("g1"), EmptyHm(), EmptyConfigs(), 1);
                Assert.That(SpinWait.SpinUntil(() => !logs.IsEmpty, 2000), Is.True);
            }

            Assert.That(logs, Has.Some.EqualTo(
                "DT verdict index apply=failed gen=1 stage=summaries error=InvalidOperationException"));
            Assert.That(index.ColumnPeaks, Is.Empty);
        }

        [Test]
        public void Start_BinFallbackFailure_LeavesExplicitFlowEvidence()
        {
            var index = ReadyIndex("D:\\data", "g1");
            var logs = new ConcurrentQueue<string>();
            BuildCurvePeakSummaries summaries = (root, infos, hm, configs, count, token) =>
            {
                var result = new ColumnCurvePeakIndexResult { RequestedGrabCount = 1 };
                result.PendingBinGrabInfos.Add(new GrabIdInfo { GrabId = "g1" });
                return result;
            };
            using (var coordinator = Create(
                index,
                () => "D:\\data",
                summaries,
                action => { action(); return true; },
                logs.Enqueue,
                (root, infos, hm, configs, count, token, progress, batch) =>
                    throw new InvalidDataException("broken bin")))
            {
                coordinator.Start(false, "D:\\data", Infos("g1"), EmptyHm(), EmptyConfigs(), 1);
                Assert.That(SpinWait.SpinUntil(
                    () => Contains(logs, "stage=bins"), 2000), Is.True);
            }

            Assert.That(logs, Has.Some.EqualTo(
                "DT verdict index apply=failed gen=1 stage=bins error=InvalidDataException"));
            Assert.That(index.ColumnPeaks, Is.Empty);
        }

        [Test]
        public void Start_AfterDispose_DoesNotRestartBackgroundWork()
        {
            var index = ReadyIndex("D:\\data", "g1");
            int builds = 0;
            var coordinator = Create(
                index,
                () => "D:\\data",
                (root, infos, hm, configs, count, token) =>
                {
                    Interlocked.Increment(ref builds);
                    return Result("g1");
                });

            coordinator.Dispose();
            coordinator.Start(false, "D:\\data", Infos("g1"), EmptyHm(), EmptyConfigs(), 1);
            Thread.Sleep(100);

            Assert.That(builds, Is.Zero);
            Assert.That(index.ColumnPeaks, Is.Empty);
        }

        private static ReportCurveVerdictIndexCoordinator Create(
            ReportCurveVerdictIndex index,
            Func<string> currentRoot,
            BuildCurvePeakSummaries build,
            Func<Action, bool> post = null,
            Action<string> log = null,
            BuildCurvePeakBinFallback buildBins = null)
        {
            return new ReportCurveVerdictIndexCoordinator(
                index,
                currentRoot,
                () => Context(),
                post ?? (action => { action(); return true; }),
                () => { },
                () => { },
                log ?? (_ => { }),
                _ => { },
                build,
                buildBins ?? ((root, infos, hm, configs, count, token, progress, batch) =>
                    new ColumnCurvePeakIndexResult()));
        }

        private static ReportCurveVerdictIndex ReadyIndex(string root, params string[] grabIds)
        {
            var details = new Dictionary<string, GrabDetail>();
            foreach (string grabId in grabIds)
                details[grabId] = new GrabDetail { GrabId = grabId };
            var index = new ReportCurveVerdictIndex();
            index.ReplaceDetails(root, details, Context());
            return index;
        }

        private static ColumnCurvePeakIndexResult Result(string grabId)
        {
            var result = new ColumnCurvePeakIndexResult
            {
                RequestedGrabCount = 1,
                SummaryGrabCount = 1,
                CameraCount = 1
            };
            result.ByGrabId[grabId] = new[]
            {
                new ColumnCurvePeakRecord { GrabId = grabId, CameraId = 1 }
            };
            result.RowByGrabId[grabId] = new RowCurvePeakRecord { GrabId = grabId };
            return result;
        }

        private static List<GrabIdInfo> Infos(string grabId)
        {
            return new List<GrabIdInfo> { new GrabIdInfo { GrabId = grabId } };
        }

        private static Dictionary<string, float> EmptyHm()
        {
            return new Dictionary<string, float>();
        }

        private static Dictionary<string, CsvConfigSnapshot> EmptyConfigs()
        {
            return new Dictionary<string, CsvConfigSnapshot>();
        }

        private static ThresholdContext Context()
        {
            return new ThresholdContext(
                1f, 0.5f, 0.5f,
                1f, 0.5f, 0.5f,
                ColumnCurveDisplayMode.Both, RidgeDirection.Both);
        }

        private static bool Contains(IEnumerable<string> messages, string value)
        {
            foreach (string message in messages)
            {
                if (message.Contains(value)) return true;
            }
            return false;
        }
    }
}
