using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class LatestCurveLoadCoordinatorTests
    {
        [Test]
        public async Task Enqueue_BurstWhileFirstRuns_LoadsFirstAndLatestOnly()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var started = new List<string>();
            int active = 0;
            int maxActive = 0;

            var coordinator = new LatestCurveLoadCoordinator(async request =>
            {
                int current = Interlocked.Increment(ref active);
                maxActive = Math.Max(maxActive, current);
                lock (started) started.Add(request.GrabId);
                if (request.GrabId == "A")
                {
                    firstStarted.TrySetResult(true);
                    await firstRelease.Task;
                }
                Interlocked.Decrement(ref active);
            });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await firstStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            _ = coordinator.Enqueue("C", DateTime.MinValue, DateTime.MaxValue);
            firstRelease.SetResult(true);
            await drain;

            Assert.That(started, Is.EqualTo(new[] { "A", "C" }));
            Assert.That(maxActive, Is.EqualTo(1));
        }

        [Test]
        public async Task IsCurrent_NewerRequestArrives_RejectsOldResult()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var applicability = new List<string>();
            LatestCurveLoadCoordinator coordinator = null;

            coordinator = new LatestCurveLoadCoordinator(async request =>
            {
                if (request.GrabId == "A")
                {
                    firstStarted.TrySetResult(true);
                    await firstRelease.Task;
                }

                lock (applicability)
                    applicability.Add($"{request.GrabId}:{coordinator.IsCurrent(request)}");
            });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await firstStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            firstRelease.SetResult(true);
            await drain;

            Assert.That(applicability, Is.EqualTo(new[] { "A:False", "B:True" }));
        }

        [Test]
        public async Task Invalidate_DropsPendingAndRejectsRunningResult()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var started = new List<string>();
            var applicability = new List<bool>();
            LatestCurveLoadCoordinator coordinator = null;

            coordinator = new LatestCurveLoadCoordinator(async request =>
            {
                lock (started) started.Add(request.GrabId);
                if (request.GrabId == "A")
                {
                    firstStarted.TrySetResult(true);
                    await firstRelease.Task;
                }
                lock (applicability) applicability.Add(coordinator.IsCurrent(request));
            });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await firstStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            coordinator.Invalidate();
            firstRelease.SetResult(true);
            await drain;

            Assert.That(started, Is.EqualTo(new[] { "A" }));
            Assert.That(applicability, Is.EqualTo(new[] { false }));
        }

        private static TaskCompletionSource<bool> NewSignal()
            => new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
