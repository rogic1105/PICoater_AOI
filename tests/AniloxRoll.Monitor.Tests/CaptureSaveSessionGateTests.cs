using System;
using NUnit.Framework;
using AniloxRoll.Monitor.Core.Camera;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class CaptureSaveSessionGateTests
    {
        [Test]
        public void Close_RejectsNewWorkAndWaitsForAcceptedWork()
        {
            var gate = new CaptureSaveSessionGate();
            gate.Begin("grab-1");
            Assert.That(gate.TryEnter("grab-1"), Is.True);
            Assert.That(gate.TryEnter("grab-1"), Is.True);

            var drain = gate.Close();
            Assert.That(drain.IsCompleted, Is.False);
            Assert.That(gate.TryEnter("grab-1"), Is.False);

            gate.Complete();
            Assert.That(drain.IsCompleted, Is.False);
            gate.Complete();
            Assert.That(drain.Wait(1000), Is.True);
        }

        [Test]
        public void Begin_RequiresPreviousSessionToBeDrained()
        {
            var gate = new CaptureSaveSessionGate();
            gate.Begin("grab-1");
            Assert.That(gate.TryEnter("grab-1"), Is.True);
            Assert.Throws<InvalidOperationException>(() => gate.Begin("grab-2"));

            gate.Close();
            gate.Complete();
            gate.Begin("grab-2");
            Assert.That(gate.TryEnter("grab-2"), Is.True);
            gate.Complete();
        }
    }
}
