using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReviewCurveLoadCoordinatorTests
    {
        [Test]
        public async Task Enqueue_BurstWhileFirstRuns_LoadsFirstAndLatestOnly()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var started = new List<string>();
            int active = 0;
            int maxActive = 0;

            var coordinator = new ReviewCurveLoadCoordinator(async request =>
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
            ReviewCurveLoadCoordinator coordinator = null;

            coordinator = new ReviewCurveLoadCoordinator(async request =>
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

        private static TaskCompletionSource<bool> NewSignal()
            => new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
