using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using IoBridge.Core;

namespace IoBridge.Automation
{
    public class MainProcess
    {
        private readonly IcpDasModbusTcpClient _plc;
        private readonly Action<string> _updateStatusCallback;

        // IO Mapping
        public const int DI_NAKAN_ALIVE   = 0;
        public const int DI_INSPECT_START = 1;

        public const int DO_PC_ALIVE      = 0;
        public const int DO_MURA_DETECTED = 1;
        public const int DO_PC_INSPECT    = 2;

        private string _currentStatus = "Initial";
        private bool _lastDiStart = false;
        private bool _isInspecting = false;
        private int _inspectCount = 0;
        private bool _isPcAlive = false;

        public MainProcess(IcpDasModbusTcpClient plc, Action<string> updateStatusCallback)
        {
            _plc = plc;
            _updateStatusCallback = updateStatusCallback;
        }

        private void SetStatus(string status)
        {
            IoLogger.Info($"State: {_currentStatus} -> {status}");
            _currentStatus = status;
            _updateStatusCallback?.Invoke(status);
        }

        public async Task StartSystemAsync()
        {
            if (_plc.IsConnected)
            {
                await _plc.WriteDo(DO_PC_ALIVE, true);
                _isPcAlive = true;
                await _plc.WriteDo(DO_MURA_DETECTED, false);
                await _plc.WriteDo(DO_PC_INSPECT, false);
                SetStatus("Ready");
            }
        }

        public async Task TriggerMuraAsync()
        {
            if (_currentStatus == "Run" && _plc.IsConnected)
            {
                await _plc.WriteDo(DO_MURA_DETECTED, true);
                SetStatus("MURA");
            }
        }

        public void TriggerPcError()
        {
            IoLogger.Warn("PC connection lost, entering PCErr state.");
            SetStatus("PCErr");
            _isInspecting = false;
            _isPcAlive = false;
        }

        public async Task RecoverFromPcErrorAsync()
        {
            if (_plc.IsConnected)
            {
                IoLogger.Info("Recovering from PCErr.");
                await _plc.WriteDo(DO_PC_ALIVE, true);
                _isPcAlive = true;
                await _plc.WriteDo(DO_MURA_DETECTED, false);
                await _plc.WriteDo(DO_PC_INSPECT, false);
                SetStatus("Ready");
            }
        }

        public Task ProcessSignalAsync(bool[] diStates)
        {
            if (diStates == null || diStates.Length < 8) return Task.CompletedTask;

            bool currentDiStart = diStates[DI_INSPECT_START];
            bool currentPlcAlive = diStates[DI_NAKAN_ALIVE];

            if (_isPcAlive && !currentPlcAlive && _currentStatus != "PLCErr" && _currentStatus != "PCErr")
            {
                _ = ExecutePlcErrorSequence();
                return Task.CompletedTask;
            }

            if (_currentStatus == "PLCErr" && currentPlcAlive && _isPcAlive)
            {
                IoLogger.Info("PLC ALIVE restored, recovering from PLCErr.");
                SetStatus("Ready");
                return Task.CompletedTask;
            }

            if (_currentStatus == "PLCErr" || _currentStatus == "PCErr")
                return Task.CompletedTask;

            if (!_lastDiStart && currentDiStart && _currentStatus == "Ready")
                _ = ExecuteStartSequence();

            if (_lastDiStart && !currentDiStart && (_currentStatus == "Run" || _currentStatus == "MURA"))
                _ = ExecuteStopSequence();

            _lastDiStart = currentDiStart;
            return Task.CompletedTask;
        }

        private async Task ExecutePlcErrorSequence()
        {
            try
            {
                IoLogger.Warn("PLC ALIVE lost, entering PLCErr state.");
                SetStatus("PLCErr");
                _isInspecting = false;

                if (_plc.IsConnected)
                {
                    await _plc.WriteDo(DO_MURA_DETECTED, false);
                    await _plc.WriteDo(DO_PC_INSPECT, false);
                }
            }
            catch (SocketException ex) { IoLogger.Error("ExecutePlcErrorSequence socket error", ex); }
            catch (Exception ex) { IoLogger.Error("ExecutePlcErrorSequence failed", ex); }
        }

        private async Task ExecuteStartSequence()
        {
            try
            {
                IoLogger.Info("START signal detected, beginning inspection.");
                SetStatus("Start");
                if (_plc.IsConnected) await _plc.WriteDo(DO_PC_INSPECT, true);

                _isInspecting = true;
                _inspectCount = 0;
                SetStatus("Run");

                await SimulationLoop();
            }
            catch (SocketException ex) { IoLogger.Error("ExecuteStartSequence socket error", ex); }
            catch (Exception ex) { IoLogger.Error("ExecuteStartSequence failed", ex); }
        }

        private async Task ExecuteStopSequence()
        {
            try
            {
                IoLogger.Info("STOP signal detected, clearing inspection.");
                SetStatus("Stop");
                _isInspecting = false;
                SetStatus("Clear");

                if (_plc.IsConnected)
                {
                    await _plc.WriteDo(DO_MURA_DETECTED, false);
                    await _plc.WriteDo(DO_PC_INSPECT, false);
                }

                await Task.Delay(1000);
                SetStatus("Ready");
            }
            catch (SocketException ex) { IoLogger.Error("ExecuteStopSequence socket error", ex); }
            catch (Exception ex) { IoLogger.Error("ExecuteStopSequence failed", ex); }
        }

        private async Task SimulationLoop()
        {
            IoLogger.Info("=== Simulation Started ===");
            while (_isInspecting)
            {
                _inspectCount++;
                IoLogger.Info($"Simulating... Count: {_inspectCount}");
                await Task.Delay(1000);
            }
            IoLogger.Info("=== Simulation Stopped ===");
        }

        public async Task StopSystemAsync()
        {
            _isInspecting = false;
            if (_plc.IsConnected)
            {
                await _plc.WriteDo(DO_PC_ALIVE, false);
                _isPcAlive = false;
                await _plc.WriteDo(DO_MURA_DETECTED, false);
                await _plc.WriteDo(DO_PC_INSPECT, false);
            }
            SetStatus("Closed");
        }
    }
}
