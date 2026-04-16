using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Moq;
using NUnit.Framework;
using PlcBridge.Core;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    public class IoGrabControllerTests
    {
        private Mock<IModbusTcpClient> _mockPlc;
        private IoGrabController _ctrl;
        private List<IoState> _stateLog;
        private int _startCount;
        private int _stopCount;

        [SetUp]
        public void SetUp()
        {
            _mockPlc = new Mock<IModbusTcpClient>();
            _mockPlc.SetupProperty(p => p.ReadWriteTimeoutMs, 2000);
            _mockPlc.Setup(p => p.WriteDo(It.IsAny<int>(), It.IsAny<bool>())).Returns(Task.CompletedTask);

            _ctrl = new IoGrabController(_mockPlc.Object);
            _stateLog = new List<IoState>();
            _startCount = 0;
            _stopCount = 0;

            _ctrl.OnStateChanged += s => _stateLog.Add(s);
            _ctrl.OnStartRequested += () => _startCount++;
            _ctrl.OnStopRequested += () => _stopCount++;
        }

        [TearDown]
        public void TearDown()
        {
            _ctrl.Dispose();
        }

        private void SetupDiStatuses(bool plcAlive, bool start)
        {
            _mockPlc.Setup(p => p.ReadDiStatuses())
                .ReturnsAsync(new bool[] { plcAlive, start, false, false, false, false, false, false });
        }

        // ── 連線測試 ──

        [Test]
        public async Task StartAsync_ConnectSuccess_EntersIdle()
        {
            _mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);

            await _ctrl.StartAsync("192.168.255.1");
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Idle));
            Assert.That(_stateLog, Does.Contain(IoState.Idle));
        }

        [Test]
        public async Task StartAsync_ConnectFail_StaysDisconnected()
        {
            _mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(false);

            await _ctrl.StartAsync("192.168.255.1");
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Disconnected));
        }

        // ── START 上升/下降緣 ──

        [Test]
        public async Task PollTick_StartRisingEdge_FiresOnStartRequested()
        {
            await ConnectAndEnterIdle();

            SetupDiStatuses(true, true); // PLC_ALIVE=true, START=true (rising edge)
            await _ctrl.PollTick();

            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Running));
            Assert.That(_startCount, Is.EqualTo(1));
        }

        [Test]
        public async Task PollTick_StartFallingEdge_FiresOnStopRequested()
        {
            await ConnectAndEnterIdle();

            // Rising edge → Running
            SetupDiStatuses(true, true);
            await _ctrl.PollTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Running));

            // Falling edge → Idle (via Stopping)
            SetupDiStatuses(true, false);
            await _ctrl.PollTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Idle));
            Assert.That(_stopCount, Is.EqualTo(1));
        }

        [Test]
        public async Task PollTick_StartHigh_NoRepeatedStart()
        {
            await ConnectAndEnterIdle();

            SetupDiStatuses(true, true);
            await _ctrl.PollTick(); // first rising edge
            await _ctrl.PollTick(); // still high → no second start

            Assert.That(_startCount, Is.EqualTo(1), "Should not fire start twice for held-high signal");
        }

        // ── PLC ALIVE 故障 ──

        [Test]
        public async Task PollTick_PlcAliveLost_EntersFaulted()
        {
            await ConnectAndEnterIdle();

            SetupDiStatuses(false, false); // PLC_ALIVE lost
            await _ctrl.PollTick();

            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Faulted));
            Assert.That(_stopCount, Is.EqualTo(1), "Should fire stop on fault");
        }

        [Test]
        public async Task PollTick_PlcAliveRestored_ReturnsToIdle()
        {
            await ConnectAndEnterIdle();

            // Fault
            SetupDiStatuses(false, false);
            await _ctrl.PollTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Faulted));

            // Restore
            SetupDiStatuses(true, false);
            await _ctrl.PollTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Idle));
        }

        // ── CommLost ──

        [Test]
        public async Task PollTick_ReadThrows_EntersCommLost()
        {
            await ConnectAndEnterIdle();

            _mockPlc.Setup(p => p.ReadDiStatuses())
                .ThrowsAsync(new TimeoutException("Read timeout"));

            await _ctrl.PollTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.CommLost));
            Assert.That(_stopCount, Is.EqualTo(1));
        }

        // ── ReconnectTick ──

        [Test]
        public async Task ReconnectTick_Success_EntersIdle()
        {
            _mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);

            await _ctrl.ReconnectTick();
            Assert.That(_ctrl.CurrentState, Is.EqualTo(IoState.Idle));
        }

        [Test]
        public async Task ReconnectTick_Fail_StaysInCurrentState()
        {
            _mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(false);

            await _ctrl.ReconnectTick();
            // State shouldn't change from whatever it was
            Assert.That(_ctrl.CurrentState, Is.Not.EqualTo(IoState.Idle));
        }

        // ── DO 通知 ──

        [Test]
        public async Task NotifyGrabStarted_WritesDoPcInspect()
        {
            _mockPlc.Setup(p => p.IsConnected).Returns(true);
            await _ctrl.NotifyGrabStarted();
            _mockPlc.Verify(p => p.WriteDo(2, true), Times.Once);
        }

        [Test]
        public async Task NotifyMuraDetected_WritesMuraDetected()
        {
            _mockPlc.Setup(p => p.IsConnected).Returns(true);
            await _ctrl.NotifyMuraDetected();
            _mockPlc.Verify(p => p.WriteDo(1, true), Times.Once);
        }

        // ── IO 快照 ──

        [Test]
        public async Task PollTick_FiresIoSnapshot()
        {
            await ConnectAndEnterIdle();

            IoSnapshot? captured = null;
            _ctrl.OnIoUpdated += snap => captured = snap;

            SetupDiStatuses(true, false);
            await _ctrl.PollTick();

            Assert.That(captured, Is.Not.Null);
            Assert.That(captured.Value.DiNakanAlive, Is.True);
            Assert.That(captured.Value.DoPcAlive, Is.True);
        }

        // ── 完整交握循環 ──

        [Test]
        public async Task FullCycle_Idle_Running_Stop_Idle()
        {
            await ConnectAndEnterIdle();
            _stateLog.Clear();

            // START rising → Running
            SetupDiStatuses(true, true);
            await _ctrl.PollTick();

            // START falling → Stopping → Idle
            SetupDiStatuses(true, false);
            await _ctrl.PollTick();

            Assert.That(_stateLog, Is.EqualTo(new[]
            {
                IoState.Running,
                IoState.Stopping,
                IoState.Idle
            }));
        }

        // ── Helper ──

        private async Task ConnectAndEnterIdle()
        {
            _mockPlc.Setup(p => p.ConnectAsync(It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(true);
            _mockPlc.Setup(p => p.IsConnected).Returns(true);
            await _ctrl.StartAsync("192.168.255.1");
        }
    }
}
