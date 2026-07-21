using System.Diagnostics;
using System.IO;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [NonParallelizable]
    public class FlowTraceTests
    {
        private StringWriter _writer;
        private TextWriterTraceListener _listener;

        [SetUp]
        public void SetUp()
        {
            _writer = new StringWriter();
            _listener = new TextWriterTraceListener(_writer);
            Trace.Listeners.Add(_listener);
        }

        [TearDown]
        public void TearDown()
        {
            Trace.Listeners.Remove(_listener);
            _listener.Dispose();
            _writer.Dispose();
            FlowTrace.Configure(LogRecordingMode.Operational);
        }

        [Test]
        public void Operational_KeepsLifecycleButDropsDvtAndStats()
        {
            FlowTrace.Configure(LogRecordingMode.Operational);
            _writer.GetStringBuilder().Clear();

            FlowTrace.Log("lifecycle");
            FlowTrace.Dvt("coordinate");
            FlowTrace.Display("IC", "stats paints=20/s");
            FlowTrace.Display("RV", "visiblePaint ready=True");
            Trace.Flush();

            string output = _writer.ToString();
            StringAssert.Contains("lifecycle", output);
            StringAssert.Contains("RV visiblePaint ready=True", output);
            StringAssert.DoesNotContain("coordinate", output);
            StringAssert.DoesNotContain("stats paints", output);
        }

        [Test]
        public void FlowVerification_AddsCoordinatesButNotStats()
        {
            FlowTrace.Configure(LogRecordingMode.FlowVerification);
            _writer.GetStringBuilder().Clear();

            FlowTrace.Dvt("coordinate");
            FlowTrace.Display("IC", "state viewX 0~1 viewY 0~1");
            FlowTrace.Display("IC", "stats paints=20/s");
            Trace.Flush();

            string output = _writer.ToString();
            StringAssert.Contains("coordinate", output);
            StringAssert.Contains("IC state viewX", output);
            StringAssert.DoesNotContain("stats paints", output);
        }

        [Test]
        public void FullDiagnostic_AddsStats()
        {
            FlowTrace.Configure(LogRecordingMode.FullDiagnostic);
            _writer.GetStringBuilder().Clear();

            FlowTrace.Display("WF", "stats paints=20/s");
            Trace.Flush();

            StringAssert.Contains("WF stats paints=20/s", _writer.ToString());
        }
    }
}
