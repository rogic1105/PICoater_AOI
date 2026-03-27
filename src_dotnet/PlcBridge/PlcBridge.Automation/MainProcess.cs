using System;
using System.Net.Sockets;
using System.Threading.Tasks;
using PlcBridge.Core;

namespace PlcBridge.Automation
{
    public class MainProcess
    {
        private readonly IcpDasModbusTcpClient _plc;
        private readonly Action<string> _updateStatusCallback;

        // IO Mapping
        public const int DI_PLC_ALIVE = 0;
        public const int DI_START = 1;

        public const int DO_PC_ALIVE = 0;
        public const int DO_MURA = 1;
        public const int DO_PC_BUSY = 2;

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
            PlcLogger.Info($"State: {_currentStatus} -> {status}");
            _currentStatus = status;
            _updateStatusCallback?.Invoke(status);
        }

        public async Task StartSystemAsync()
        {
            if (_plc.IsConnected)
            {
                await _plc.WriteDo(DO_PC_ALIVE, true);
                _isPcAlive = true;
                await _plc.WriteDo(DO_MURA, false);
                await _plc.WriteDo(DO_PC_BUSY, false);
                SetStatus("Ready");
            }
        }

        public async Task TriggerMuraAsync()
        {
            if (_currentStatus == "Run" && _plc.IsConnected)
            {
                await _plc.WriteDo(DO_MURA, true);
                SetStatus("MURA");
            }
        }

        public void TriggerPcError()
        {
            PlcLogger.Warn("PC connection lost, entering PCErr state.");
            SetStatus("PCErr");
            _isInspecting = false;
            _isPcAlive = false;
        }

        public async Task RecoverFromPcErrorAsync()
        {
            if (_plc.IsConnected)
            {
                PlcLogger.Info("Recovering from PCErr.");
                await _plc.WriteDo(DO_PC_ALIVE, true);
                _isPcAlive = true;
                await _plc.WriteDo(DO_MURA, false);
                await _plc.WriteDo(DO_PC_BUSY, false);
                SetStatus("Ready");
            }
        }

        public Task ProcessSignalAsync(bool[] diStates)
        {
            if (diStates == null || diStates.Length < 8) return Task.CompletedTask;

            bool currentDiStart = diStates[DI_START];
            bool currentPlcAlive = diStates[DI_PLC_ALIVE];

            if (_isPcAlive && !currentPlcAlive && _currentStatus != "PLCErr" && _currentStatus != "PCErr")
            {
                _ = ExecutePlcErrorSequence();
                return Task.CompletedTask;
            }

            if (_currentStatus == "PLCErr" && currentPlcAlive && _isPcAlive)
            {
                PlcLogger.Info("PLC ALIVE restored, recovering from PLCErr.");
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
                PlcLogger.Warn("PLC ALIVE lost, entering PLCErr state.");
                SetStatus("PLCErr");
                _isInspecting = false;

                if (_plc.IsConnected)
                {
                    await _plc.WriteDo(DO_MURA, false);
                    await _plc.WriteDo(DO_PC_BUSY, false);
                }
            }
            catch (SocketException ex) { PlcLogger.Error("ExecutePlcErrorSequence socket error", ex); }
            catch (Exception ex) { PlcLogger.Error("ExecutePlcErrorSequence failed", ex); }
        }

        private async Task ExecuteStartSequence()
        {
            try
            {
                PlcLogger.Info("START signal detected, beginning inspection.");
                SetStatus("Start");
                if (_plc.IsConnected) await _plc.WriteDo(DO_PC_BUSY, true);

                _isInspecting = true;
                _inspectCount = 0;
                SetStatus("Run");

                await SimulationLoop();
            }
            catch (SocketException ex) { PlcLogger.Error("ExecuteStartSequence socket error", ex); }
            catch (Exception ex) { PlcLogger.Error("ExecuteStartSequence failed", ex); }
        }

        private async Task ExecuteStopSequence()
        {
            try
            {
                PlcLogger.Info("STOP signal detected, clearing inspection.");
                SetStatus("Stop");
                _isInspecting = false;
                SetStatus("Clear");

                if (_plc.IsConnected)
                {
                    await _plc.WriteDo(DO_MURA, false);
                    await _plc.WriteDo(DO_PC_BUSY, false);
                }

                await Task.Delay(1000);
                SetStatus("Ready");
            }
            catch (SocketException ex) { PlcLogger.Error("ExecuteStopSequence socket error", ex); }
            catch (Exception ex) { PlcLogger.Error("ExecuteStopSequence failed", ex); }
        }

        private async Task SimulationLoop()
        {
            PlcLogger.Info("=== Simulation Started ===");
            while (_isInspecting)
            {
                _inspectCount++;
                PlcLogger.Info($"Simulating... Count: {_inspectCount}");
                await Task.Delay(1000);
            }
            PlcLogger.Info("=== Simulation Stopped ===");
        }

        public async Task StopSystemAsync()
        {
            _isInspecting = false;
            if (_plc.IsConnected)
            {
                await _plc.WriteDo(DO_PC_ALIVE, false);
                _isPcAlive = false;
                await _plc.WriteDo(DO_MURA, false);
                await _plc.WriteDo(DO_PC_BUSY, false);
            }
            SetStatus("Closed");
        }
    }
}
