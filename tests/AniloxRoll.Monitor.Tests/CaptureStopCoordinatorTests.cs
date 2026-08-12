using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CaptureStopCoordinatorTests
    {
        private List<CaptureStopRequest> _requests;
        private CaptureStopCoordinator _coordinator;

        [SetUp]
        public void SetUp()
        {
            _requests = new List<CaptureStopRequest>();
            _coordinator = new CaptureStopCoordinator(
                request => _requests.Add(request));
        }

        [TearDown]
        public void TearDown()
        {
            _coordinator.Dispose();
        }

        [TestCase(IoStopRequestReason.StartLow, true)]
        [TestCase(IoStopRequestReason.PlcAliveLost, false)]
        [TestCase(IoStopRequestReason.CommunicationLost, false)]
        public void IoMode_AcceptsOneIoStop_AndDrainsOnlyStartLow(
            IoStopRequestReason reason,
            bool shouldDrain)
        {
            bool waits = _coordinator.Arm(
                CaptureStopCondition.IoSignal,
                true,
                10,
                2,
                30000,
                "grab-io");

            Assert.That(waits, Is.False);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.ArmedIo));

            CaptureStopRequest request;
            Assert.That(
                _coordinator.TryRequestIoStop(reason, out request),
                Is.True);
            Assert.That(request.Condition, Is.EqualTo(CaptureStopCondition.IoSignal));
            Assert.That(request.DrainIoTail, Is.EqualTo(shouldDrain));
            Assert.That(request.NotifyFixedGrabCompleted, Is.False);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.StopPending));

            CaptureStopRequest duplicate;
            Assert.That(
                _coordinator.TryRequestIoStop(reason, out duplicate),
                Is.False);
            Assert.That(duplicate, Is.Null);
        }

        [Test]
        public void IoMode_IgnoresConfiguredTime_AndWaitsForStartLow()
        {
            _coordinator.Arm(
                CaptureStopCondition.IoSignal,
                true,
                5,
                2,
                30000,
                "grab-io-long-high");

            _coordinator.HandleTimerElapsed(7);

            Assert.That(_requests, Is.Empty);
            Assert.That(_coordinator.State, Is.EqualTo(CaptureStopState.ArmedIo));

            CaptureStopRequest request;
            Assert.That(
                _coordinator.TryRequestIoStop(
                    IoStopRequestReason.StartLow,
                    out request),
                Is.True);
            Assert.That(request.Trigger, Is.EqualTo(CaptureStopTrigger.IoRequest));
            Assert.That(request.DrainIoTail, Is.True);
        }

        [Test]
        public void TimeMode_WaitsForFirstSet_IgnoresIo_ThenStopsOnTimer()
        {
            bool waits = _coordinator.Arm(
                CaptureStopCondition.Time,
                true,
                10,
                2,
                30000,
                "grab-time");

            Assert.That(waits, Is.True);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.WaitingForFirstSet));

            CaptureStopRequest ignored;
            Assert.That(
                _coordinator.TryRequestIoStop(
                    IoStopRequestReason.StartLow,
                    out ignored),
                Is.False);
            Assert.That(
                _coordinator.ActivateTimeAfterFirstSet(),
                Is.True);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.ArmedTime));

            _coordinator.HandleTimerElapsed(10);

            Assert.That(_requests, Has.Count.EqualTo(1));
            Assert.That(
                _requests[0].Trigger,
                Is.EqualTo(CaptureStopTrigger.TimerElapsed));
            Assert.That(_requests[0].Limit, Is.EqualTo(10));
            Assert.That(_requests[0].NotifyFixedGrabCompleted, Is.True);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.StopPending));

            _coordinator.HandleTimerElapsed(10);
            Assert.That(_requests, Has.Count.EqualTo(1));
        }

        [Test]
        public void HeightMode_StopsOnceWhenCommonRowsReachSnapshot()
        {
            _coordinator.Arm(
                CaptureStopCondition.Height,
                true,
                10,
                2,
                30000,
                "grab-height");

            _coordinator.ObserveCommonRows(29999);
            Assert.That(_requests, Is.Empty);

            _coordinator.ObserveCommonRows(30000);
            _coordinator.ObserveCommonRows(31000);

            Assert.That(_requests, Has.Count.EqualTo(1));
            Assert.That(
                _requests[0].Trigger,
                Is.EqualTo(CaptureStopTrigger.HeightReached));
            Assert.That(_requests[0].Limit, Is.EqualTo(30000));
            Assert.That(_requests[0].Observed, Is.EqualTo(30000));
            Assert.That(_requests[0].NotifyFixedGrabCompleted, Is.True);
        }

        [Test]
        public void FirstSetFailure_MovesToStopPending_AndRejectsTimer()
        {
            _coordinator.Arm(
                CaptureStopCondition.Time,
                false,
                10,
                2,
                30000,
                "grab-failed");

            Assert.That(_coordinator.FailFirstSet(), Is.True);
            _coordinator.HandleTimerElapsed(10);

            Assert.That(_requests, Is.Empty);
            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.StopPending));
        }

        [Test]
        public void CompleteStop_ResetsState_ForNextCapture()
        {
            _coordinator.Arm(
                CaptureStopCondition.Height,
                true,
                10,
                2,
                100,
                "first");
            _coordinator.ObserveCommonRows(100);

            _coordinator.CompleteStop();

            Assert.That(
                _coordinator.State,
                Is.EqualTo(CaptureStopState.Idle));
            bool waits = _coordinator.Arm(
                CaptureStopCondition.Time,
                false,
                5,
                0,
                100,
                "second");
            Assert.That(waits, Is.True);
        }
    }
}
