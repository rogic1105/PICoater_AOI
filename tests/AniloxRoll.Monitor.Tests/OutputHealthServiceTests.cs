using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class OutputHealthServiceTests
    {
        [Test]
        public void ActiveProblem_CannotBeAcknowledgedAway()
        {
            var service = new OutputHealthService();
            service.Report("Remote", OutputHealthSeverity.Critical, "remote offline");

            service.AcknowledgeResolved("Remote");

            Assert.That(service.Snapshot.Severity, Is.EqualTo(OutputHealthSeverity.Critical));
            Assert.That(service.Snapshot.IsActive, Is.True);
        }

        [Test]
        public void ResolvedProblem_RemainsUntilAcknowledged()
        {
            var service = new OutputHealthService();
            service.Report("Write", OutputHealthSeverity.OutputFault, "write failed");
            service.Resolve("Write", "write recovered");

            Assert.That(service.Snapshot.Severity, Is.EqualTo(OutputHealthSeverity.OutputFault));
            Assert.That(service.Snapshot.IsActive, Is.False);

            service.AcknowledgeResolved("Write");

            Assert.That(service.Snapshot.Severity, Is.EqualTo(OutputHealthSeverity.Normal));
        }

        [Test]
        public void HighestSeverityWins_ThenFallsBackToRemainingIncident()
        {
            var service = new OutputHealthService();
            service.Report("Backlog", OutputHealthSeverity.Notice, "backlog");
            service.Report("Network", OutputHealthSeverity.Critical, "offline");

            service.Resolve("Network");
            service.AcknowledgeResolved("Network");

            Assert.That(service.Snapshot.Code, Is.EqualTo("Backlog"));
            Assert.That(service.Snapshot.Severity, Is.EqualTo(OutputHealthSeverity.Notice));
            Assert.That(service.Snapshot.IsActive, Is.True);
        }

        [Test]
        public void ResolvingOneCameraWriteFault_LeavesOtherCameraFaultActive()
        {
            var service = new OutputHealthService();
            service.Report(
                "CaptureWriteFailure.CAM1",
                OutputHealthSeverity.OutputFault,
                "CAM1 write failed");
            service.Report(
                "CaptureWriteFailure.CAM2",
                OutputHealthSeverity.OutputFault,
                "CAM2 write failed");

            service.Resolve("CaptureWriteFailure.CAM2");

            Assert.That(service.Snapshot.Code, Is.EqualTo("CaptureWriteFailure.CAM1"));
            Assert.That(service.Snapshot.IsActive, Is.True);
        }

        [Test]
        public void DuplicateReport_DoesNotRaiseChangedAgain()
        {
            var service = new OutputHealthService();
            int changes = 0;
            service.Changed += _ => changes++;

            service.Report("Backlog", OutputHealthSeverity.Notice, "backlog");
            service.Report("Backlog", OutputHealthSeverity.Notice, "backlog");

            Assert.That(changes, Is.EqualTo(1));
        }

        [Test]
        public void NonTopIncidentChange_RaisesChangedAndAppearsInIncidentList()
        {
            var service = new OutputHealthService();
            int changes = 0;
            service.Changed += _ => changes++;

            service.Report("Network", OutputHealthSeverity.Critical, "offline");
            service.Report("Backlog", OutputHealthSeverity.Notice, "backlog");
            service.Resolve("Backlog", "backlog recovered");

            Assert.That(changes, Is.EqualTo(3));
            Assert.That(service.Snapshot.Code, Is.EqualTo("Network"));
            Assert.That(service.Incidents.Length, Is.EqualTo(2));
            Assert.That(service.Incidents[1].Code, Is.EqualTo("Backlog"));
            Assert.That(service.Incidents[1].IsActive, Is.False);
        }

        [Test]
        public void AcknowledgeResolvedCode_RemovesOnlySelectedIncident()
        {
            var service = new OutputHealthService();
            int changes = 0;
            service.Changed += _ => changes++;
            service.Report("WriteA", OutputHealthSeverity.OutputFault, "write A failed");
            service.Report("WriteB", OutputHealthSeverity.OutputFault, "write B failed");
            service.Resolve("WriteA");
            service.Resolve("WriteB");

            service.AcknowledgeResolved("WriteA");

            Assert.That(changes, Is.EqualTo(5));
            Assert.That(service.Incidents.Length, Is.EqualTo(1));
            Assert.That(service.Incidents[0].Code, Is.EqualTo("WriteB"));
            Assert.That(service.Incidents[0].IsActive, Is.False);
        }

        [Test]
        public void AcknowledgeResolvedCode_DoesNotRemoveActiveIncident()
        {
            var service = new OutputHealthService();
            service.Report("Network", OutputHealthSeverity.Critical, "offline");

            service.AcknowledgeResolved("Network");

            Assert.That(service.Incidents.Length, Is.EqualTo(1));
            Assert.That(service.Incidents[0].IsActive, Is.True);
        }
    }
}
