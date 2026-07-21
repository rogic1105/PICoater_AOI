using AniloxRoll.Monitor.UI.Coordinators;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class ReviewImageLoadGateTests
    {
        [Test]
        public void Invalidate_ActiveLoad_ReleasesBusyLeaseAndRejectsResult()
        {
            var gate = new ReviewImageLoadGate();
            int load = gate.Begin();

            Assert.That(gate.Invalidate(), Is.True);
            Assert.That(gate.IsCurrent(load), Is.False);
            Assert.That(gate.Complete(load), Is.False);
        }

        [Test]
        public void OldCompletion_AfterNewLoad_DoesNotReleaseNewBusyLease()
        {
            var gate = new ReviewImageLoadGate();
            int oldLoad = gate.Begin();
            int newLoad = gate.Begin();

            Assert.That(gate.Complete(oldLoad), Is.False);
            Assert.That(gate.IsCurrent(newLoad), Is.True);
            Assert.That(gate.Complete(newLoad), Is.True);
        }

        [Test]
        public void Complete_CurrentLoad_ReleasesBusyLeaseOnce()
        {
            var gate = new ReviewImageLoadGate();
            int load = gate.Begin();

            Assert.That(gate.Complete(load), Is.True);
            Assert.That(gate.Complete(load), Is.False);
        }
    }
}
