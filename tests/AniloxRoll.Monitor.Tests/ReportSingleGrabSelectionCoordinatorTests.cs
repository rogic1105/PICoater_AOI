using System.Collections.Generic;
using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class ReportSingleGrabSelectionCoordinatorTests
    {
        [Test]
        public void ApplyPendingNow_MultipleRequests_AppliesLatestOnce()
        {
            string selected = "g1";
            int applied = 0;
            var logs = new List<string>();
            using (var coordinator = new ReportSingleGrabSelectionCoordinator(
                () => selected, () => applied++, logs.Add))
            {
                coordinator.Schedule();
                selected = "g2";
                coordinator.Schedule();
                selected = "g3";
                coordinator.Schedule();

                coordinator.ApplyPendingNow();
            }

            Assert.That(applied, Is.EqualTo(1));
            Assert.That(logs[0], Is.EqualTo("ui:【報表序號】→ g3"));
            Assert.That(logs[1], Does.Contain("skipped=2"));
            Assert.That(logs[1], Does.Contain("intervalMs=33"));
        }

        [Test]
        public void Cancel_DropsPendingSelection()
        {
            int applied = 0;
            var logs = new List<string>();
            using (var coordinator = new ReportSingleGrabSelectionCoordinator(
                () => "g1", () => applied++, logs.Add))
            {
                coordinator.Schedule();
                coordinator.Cancel();

                coordinator.ApplyPendingNow();
            }

            Assert.That(applied, Is.Zero);
            Assert.That(logs, Is.Empty);
        }

        [Test]
        public void EmptySelection_DoesNotApplyOrLog()
        {
            int applied = 0;
            var logs = new List<string>();
            using (var coordinator = new ReportSingleGrabSelectionCoordinator(
                () => string.Empty, () => applied++, logs.Add))
            {
                coordinator.Schedule();
                coordinator.ApplyPendingNow();
            }

            Assert.That(applied, Is.Zero);
            Assert.That(logs, Is.Empty);
        }

        [Test]
        public void Dispose_DropsPendingAndCannotBeRescheduled()
        {
            int applied = 0;
            var logs = new List<string>();
            var coordinator = new ReportSingleGrabSelectionCoordinator(
                () => "g1", () => applied++, logs.Add);
            coordinator.Schedule();

            coordinator.Dispose();
            coordinator.Schedule();
            coordinator.ApplyPendingNow();

            Assert.That(applied, Is.Zero);
            Assert.That(logs, Is.Empty);
        }
    }
}
