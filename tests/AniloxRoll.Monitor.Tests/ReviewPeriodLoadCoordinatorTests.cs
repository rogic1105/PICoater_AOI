using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReviewPeriodLoadCoordinatorTests
    {
        [Test]
        public async Task Enqueue_BurstOfDistinctPeriods_RunsSeriallyAndDeduplicates()
        {
            var firstRelease = NewSignal();
            var started = new List<DateTime>();
            int active = 0;
            int maxActive = 0;
            DateTime p1 = new DateTime(2026, 7, 13, 10, 0, 1);
            DateTime p2 = p1.AddSeconds(1);
            DateTime p3 = p1.AddSeconds(2);

            var coordinator = new ReviewPeriodLoadCoordinator(async (request, canApply) =>
            {
                int current = Interlocked.Increment(ref active);
                maxActive = Math.Max(maxActive, current);
                lock (started) started.Add(request.Period);
                if (request.Period == p1) await firstRelease.Task;
                Interlocked.Decrement(ref active);
            });

            Task drain = coordinator.Enqueue(p1, false);
            _ = coordinator.Enqueue(p1, false); // running duplicate
            _ = coordinator.Enqueue(p2, false);
            _ = coordinator.Enqueue(p2, false); // queued duplicate
            _ = coordinator.Enqueue(p3, false);
            firstRelease.SetResult(true);
            await drain;

            Assert.That(started, Is.EqualTo(new[] { p1, p2, p3 }));
            Assert.That(maxActive, Is.EqualTo(1));
        }

        [Test]
        public async Task Invalidate_RunningAndQueuedRequests_PreventsApplyAndClearsQueue()
        {
            var started = NewSignal();
            var release = NewSignal();
            var applied = new List<DateTime>();
            DateTime p1 = new DateTime(2026, 7, 13, 10, 0, 1);
            DateTime p2 = p1.AddSeconds(1);

            var coordinator = new ReviewPeriodLoadCoordinator(async (request, canApply) =>
            {
                started.TrySetResult(true);
                await release.Task;
                if (canApply()) applied.Add(request.Period);
            });

            Task drain = coordinator.Enqueue(p1, false);
            await started.Task;
            _ = coordinator.Enqueue(p2, false);
            coordinator.Invalidate();
            release.SetResult(true);
            await drain;

            Assert.That(applied, Is.Empty);
        }

        private static TaskCompletionSource<bool> NewSignal()
            => new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
