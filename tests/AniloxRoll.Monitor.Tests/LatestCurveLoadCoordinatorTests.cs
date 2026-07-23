using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class LatestGrabLoadCoordinatorTests
    {
        [Test]
        public async Task Enqueue_BurstWhileFirstRuns_LoadsFirstAndLatestOnly()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var started = new List<string>();
            int active = 0;
            int maxActive = 0;

            var coordinator = new LatestGrabLoadCoordinator(async request =>
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
            LatestGrabLoadCoordinator coordinator = null;

            coordinator = new LatestGrabLoadCoordinator(async request =>
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
        public async Task CanApplyStarted_NewerRequestArrives_AllowsSerializedRunningResult()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var applicability = new List<string>();
            LatestGrabLoadCoordinator coordinator = null;

            coordinator = new LatestGrabLoadCoordinator(async request =>
            {
                if (request.GrabId == "A")
                {
                    firstStarted.TrySetResult(true);
                    await firstRelease.Task;
                }

                lock (applicability)
                    applicability.Add($"{request.GrabId}:{coordinator.CanApplyStarted(request)}");
            });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await firstStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            firstRelease.SetResult(true);
            await drain;

            Assert.That(applicability, Is.EqualTo(new[] { "A:True", "B:True" }));
        }

        [Test]
        public async Task Invalidate_DropsPendingAndRejectsRunningResult()
        {
            var firstStarted = NewSignal();
            var firstRelease = NewSignal();
            var started = new List<string>();
            var applicability = new List<bool>();
            var startedApplicability = new List<bool>();
            LatestGrabLoadCoordinator coordinator = null;

            coordinator = new LatestGrabLoadCoordinator(async request =>
            {
                lock (started) started.Add(request.GrabId);
                if (request.GrabId == "A")
                {
                    firstStarted.TrySetResult(true);
                    await firstRelease.Task;
                }
                lock (applicability) applicability.Add(coordinator.IsCurrent(request));
                lock (startedApplicability)
                    startedApplicability.Add(coordinator.CanApplyStarted(request));
            });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await firstStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            coordinator.Invalidate();
            firstRelease.SetResult(true);
            await drain;

            Assert.That(started, Is.EqualTo(new[] { "A" }));
            Assert.That(applicability, Is.EqualTo(new[] { false }));
            Assert.That(startedApplicability, Is.EqualTo(new[] { false }));
        }

        [Test]
        public async Task Enqueue_DuringMinimumCycle_CoalescesToLatestRequest()
        {
            var cooldownStarted = NewSignal();
            var cooldownRelease = NewSignal();
            var started = new List<string>();
            var coalesced = new List<int>();
            int delayCalls = 0;

            var coordinator = new LatestGrabLoadCoordinator(
                request =>
                {
                    lock (started)
                    {
                        started.Add(request.GrabId);
                        coalesced.Add(request.CoalescedCount);
                    }
                    return Task.CompletedTask;
                },
                minimumCycleMs: 33,
                delayAsync: _ =>
                {
                    if (Interlocked.Increment(ref delayCalls) == 1)
                    {
                        cooldownStarted.TrySetResult(true);
                        return cooldownRelease.Task;
                    }
                    return Task.CompletedTask;
                });

            Task drain = coordinator.Enqueue("A", DateTime.MinValue, DateTime.MaxValue);
            await cooldownStarted.Task;
            _ = coordinator.Enqueue("B", DateTime.MinValue, DateTime.MaxValue);
            _ = coordinator.Enqueue("C", DateTime.MinValue, DateTime.MaxValue);

            Assert.That(started, Is.EqualTo(new[] { "A" }));
            cooldownRelease.SetResult(true);
            await drain;

            Assert.That(started, Is.EqualTo(new[] { "A", "C" }));
            Assert.That(coalesced, Is.EqualTo(new[] { 0, 1 }));
        }

        private static TaskCompletionSource<bool> NewSignal()
            => new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
    }
}
