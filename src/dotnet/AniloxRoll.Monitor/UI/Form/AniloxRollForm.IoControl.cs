using System;
using System.ComponentModel;
using System.IO;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>IO connection, capture requests, and IO status presentation.</summary>
    public partial class AniloxRollForm
    {
        private int _pendingIoSnapshotBits = -1;
        private int _appliedIoSnapshotBits = -1;

        /// <summary>初始化 IO 連動：自動偵測連線，連上後以 DI START 控制 Grab。</summary>
        private void InitIoController()
        {
            // Disabled startup owns no idle coordinator. The first later enable
            // is therefore a clean generation instead of restarting an empty one.
            if (_settings == null || !_settings.IoEnabled)
                return;

            if (_ioConnectionCoordinator == null)
            {
                _ioConnectionCoordinator =
                    new IoConnectionCoordinator();
                _ioConnectionCoordinator.StartRequested +=
                    OnIoControllerStartRequested;
                _ioConnectionCoordinator.StopRequested +=
                    OnIoControllerStopRequested;
                _ioConnectionCoordinator.StateChanged +=
                    (controller, generation, state) =>
                        DispatchCurrentIoController(
                            controller,
                            generation,
                            () => UpdateIoStateLabel(state));
                _ioConnectionCoordinator.ConnectionChanged +=
                    OnIoControllerConnectionChanged;
                _ioConnectionCoordinator.IoUpdated +=
                    OnIoControllerSnapshotUpdated;
            }

            _ = _ioConnectionCoordinator.StartAsync(
                CreateIoConnectionOptions());
        }

        private IoConnectionOptions CreateIoConnectionOptions()
        {
            return new IoConnectionOptions(
                _settings.IoEnabled,
                _settings.IoModel,
                _settings.IoIp,
                _settings.IoPort,
                _settings.CaptureStopCondition ==
                    CaptureStopCondition.IoSignal);
        }

        private IoGrabController CurrentIoController =>
            _ioConnectionCoordinator?.CurrentController;

        private void OnIoControllerStartRequested(
            IoGrabController controller,
            int generation)
        {
            if (!IsCurrentIoController(controller, generation))
                return;

            int requestGeneration =
                System.Threading.Interlocked.Increment(
                    ref _ioGrabRequestGeneration);
            DispatchCurrentIoController(
                controller,
                generation,
                () =>
                {
                    FlowTrace.Log("io:DI START 上升緣 → 抓取請求");
                    _ = IoStartGrabAsync(
                        controller,
                        generation,
                        requestGeneration);
                });
        }

        private void OnIoControllerStopRequested(
            IoGrabController controller,
            int generation,
            IoStopRequestReason reason)
        {
            if (!IsCurrentIoController(controller, generation))
                return;

            System.Threading.Interlocked.Increment(
                ref _ioGrabRequestGeneration);
            DispatchCurrentIoController(
                controller,
                generation,
                () => _ = IoStopGrabAsync(
                    controller,
                    generation,
                    reason));
        }

        private void OnIoControllerConnectionChanged(
            IoGrabController controller,
            int generation,
            bool connected)
        {
            if (!IsCurrentIoController(controller, generation))
                return;

            if (!connected)
            {
                System.Threading.Interlocked.Increment(
                    ref _ioGrabRequestGeneration);
            }
            DispatchCurrentIoController(
                controller,
                generation,
                () => UpdateIoConnectionUi(connected));
        }

        private void OnIoControllerSnapshotUpdated(
            IoGrabController controller,
            int generation,
            IoSnapshot snapshot)
        {
            if (!IsCurrentIoController(controller, generation))
                return;

            System.Threading.Interlocked.Exchange(
                ref _pendingIoSnapshotBits,
                PackIoSnapshot(snapshot));
        }

        private bool IsCurrentIoController(IoGrabController controller, int generation)
        {
            return !_shutdownInProgress &&
                   _ioConnectionCoordinator?.IsCurrent(
                       controller,
                       generation) == true;
        }

        private string GetIoGrabRequestInvalidReason(
            IoGrabController controller,
            int controllerGeneration,
            int requestGeneration)
        {
            if (!IsCurrentIoController(controller, controllerGeneration))
                return "controller-stale";
            if (!controller.IsConnected)
                return "io-disconnected";
            if (requestGeneration != System.Threading.Volatile.Read(ref _ioGrabRequestGeneration))
                return "request-cancelled";
            if (controller.CurrentState != IoState.Running)
                return "start-not-running:" + controller.CurrentState;
            return null;
        }

        private bool IsCurrentIoGrabRequest(
            IoGrabController controller,
            int controllerGeneration,
            int requestGeneration)
        {
            return GetIoGrabRequestInvalidReason(
                controller,
                controllerGeneration,
                requestGeneration) == null;
        }

        private void DispatchCurrentIoController(
            IoGrabController controller, int generation, Action action)
        {
            if (!IsCurrentIoController(controller, generation)) return;
            SafeBeginInvoke(() =>
            {
                if (IsCurrentIoController(controller, generation)) action();
            });
        }

        private async Task IoStartGrabAsync(
            IoGrabController controller,
            int generation,
            int requestGeneration)
        {
            await _ioGrabTransitionGate.WaitAsync();
            try
            {
                string invalidReason = GetIoGrabRequestInvalidReason(
                    controller,
                    generation,
                    requestGeneration);
                if (invalidReason != null)
                {
                    await RejectIoGrabStartAsync(controller, generation, invalidReason);
                    return;
                }
                if (_isIoSuspended)
                {
                    await RejectIoGrabStartAsync(controller, generation, "io-suspended");
                    return;
                }
                if (_liveCameraManager == null)
                {
                    await RejectIoGrabStartAsync(controller, generation, "camera-manager-unavailable");
                    return;
                }
                if (_liveCameraManager.IsCaptureTailDrainActive)
                {
                    await RejectIoGrabStartAsync(
                        controller,
                        generation,
                        "capture-not-ready:tail-drain");
                    return;
                }
                if (_liveCameraManager.IsLiveGrabbing)
                {
                    await controller.NotifyGrabStarted();
                    FlowTrace.Log("IO grab accepted busy=on state=already-grabbing");
                    return;
                }
                string standbyReason;
                if (!_liveCameraManager.TryGetCaptureStandbyReady(out standbyReason))
                {
                    await RejectIoGrabStartAsync(
                        controller,
                        generation,
                        "capture-not-ready:" + standbyReason);
                    return;
                }
                if (IsStandardBgSubEnabled && !IsBgBinReady())
                {
                    System.Diagnostics.Trace.TraceWarning("[IoStartGrab] StandardBgSub 無背景 bin，自動取得背景後接續 grab");
                    _autoStartGrabAfterBg = true;
                    _autoStartGrabIoGeneration = generation;
                    _autoStartGrabIoRequestGeneration = requestGeneration;
                    btnLiveGetBackground_Click(null, null);
                    return;
                }

                try
                {
                    CaptureStopCondition stopCondition =
                        _settings?.CaptureStopCondition ??
                        InspectionDefaults.DefaultCaptureStopCondition;
                    controller.StopCaptureOnStartLow =
                        stopCondition == CaptureStopCondition.IoSignal;
                    FlowTrace.Log(
                        $"IO grab request stopCondition={stopCondition} " +
                        $"stopOnLow={controller.StopCaptureOnStartLow}");
                    bool started = await ToggleLiveGrabAsync(
                        "io:DI START 上升緣 → 開始抓取",
                        ioControlled: true,
                        captureStartStillValid: () => IsCurrentIoGrabRequest(
                            controller,
                            generation,
                            requestGeneration));
                    invalidReason = GetIoGrabRequestInvalidReason(
                        controller,
                        generation,
                        requestGeneration);
                    if (invalidReason != null)
                    {
                        await RejectIoGrabStartAsync(controller, generation, invalidReason);
                        return;
                    }
                    if (started && _liveCameraManager.IsLiveGrabbing)
                    {
                        await controller.NotifyGrabStarted();
                        FlowTrace.Log("IO grab accepted busy=on");
                        return;
                    }

                    await RejectIoGrabStartAsync(controller, generation, "capture-start-failed");
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[IO.GrabStart] {ex.GetType().Name}: {ex.Message}");
                    await RejectIoGrabStartAsync(controller, generation, "exception");
                }
            }
            finally
            {
                _ioGrabTransitionGate.Release();
            }
        }

        private async Task RejectIoGrabStartAsync(
            IoGrabController controller, int generation, string reason)
        {
            if (controller != null && IsCurrentIoController(controller, generation))
                await controller.NotifyGrabStartRejected();
            FlowTrace.Log($"IO grab rejected busy=off reason={reason}");
        }

        private async Task IoStopGrabAsync(
            IoGrabController controller,
            int generation,
            IoStopRequestReason reason)
        {
            await _ioGrabTransitionGate.WaitAsync();
            try
            {
                if (!IsCurrentIoController(controller, generation) || _isIoSuspended) return;
                if (_liveCameraManager == null || !_liveCameraManager.IsLiveGrabbing) return;
                CaptureStopRequest stopRequest;
                if (_captureStopCoordinator == null ||
                    !_captureStopCoordinator.TryRequestIoStop(
                        reason,
                        out stopRequest))
                {
                    CaptureStopCondition stopCondition =
                        _captureStopCoordinator?.Condition ??
                        CaptureStopCondition.IoSignal;
                    FlowTrace.Log(
                        $"IO grab stop ignored reason={reason} stopCondition={stopCondition} " +
                        "captureContinues=True");
                    return;
                }

                FlowTrace.Log(
                    $"IO grab stop accepted reason={reason} " +
                    $"stopCondition={stopRequest.Condition} " +
                    $"drainTail={stopRequest.DrainIoTail}");
                bool stopped = await ToggleLiveGrabAsync(
                    stopRequest.CreateIntentLine(),
                    drainIoTail: stopRequest.DrainIoTail);
                if (stopped && IsCurrentIoController(controller, generation))
                    await controller.NotifyGrabStopped();
            }
            finally
            {
                _ioGrabTransitionGate.Release();
            }
        }


        /// <summary>IO 設定變更（IP/Port/型號/啟用）→ 重啟 IO controller，改完數值立即生效（不用重開程式）。</summary>
        private void HandleIoSettingsChanged(string changedPropertyName)
        {
            switch (changedPropertyName)
            {
                case nameof(InspectionSettings.IoEnabled):
                case nameof(InspectionSettings.IoIp):
                case nameof(InspectionSettings.IoPort):
                case nameof(InspectionSettings.IoModel):
                    System.Threading.Interlocked.Increment(ref _ioGrabRequestGeneration);
                    UpdateIoConnectionUi(false);
                    if (_ioConnectionCoordinator == null)
                        InitIoController();
                    else
                        _ = _ioConnectionCoordinator.RestartAsync(
                            CreateIoConnectionOptions());
                    break;
            }
        }

        private async Task ShutdownIoControllerAsync()
        {
            System.Threading.Interlocked.Increment(ref _ioGrabRequestGeneration);
            if (_ioConnectionCoordinator == null)
                return;

            await _ioConnectionCoordinator.ShutdownAsync();
            _ioConnectionCoordinator = null;
        }

        private void UpdateIoStateLabel(IoState state)
        {
            if (_isIoSuspended) return;
            string text;
            Color bgColor;
            switch (state)
            {
                case IoState.Idle:      text = "待機";   bgColor = IecGreen;  break;
                case IoState.Running:   text = "取像";   bgColor = IecBlue;   break;
                case IoState.AwaitingStartLow: text = "等待復歸"; bgColor = IecYellow; break;
                case IoState.Stopping:  text = "停止";   bgColor = IecYellow; break;
                case IoState.Faulted:   text = "故障";   bgColor = IecRed;    break;
                case IoState.CommLost:  text = "斷線";   bgColor = IecRed;    break;
                case IoState.Closed:    text = "關閉";   bgColor = IecGray;   break;
                default:                text = "未連線"; bgColor = IecGray;   break;  // Disconnected
            }
            lblIoState.Text = text;
            lblIoState.BackColor = bgColor;
        }

        private void UpdateIoConnectionUi(bool connected)
        {
            if (_isIoSuspended) return;
            // 「按鈕是否變 IO 控制中」交給 RefreshGrabButtonState 統一決定（相機未就緒前不被 IO 搶顯示）。
            lblIoConn.ForeColor = Color.White;   // 統一白字
            if (connected)
            {
                lblIoConn.Text = "● IO 已連線";
                lblIoConn.BackColor = IecGreen;
            }
            else
            {
                // 尚未排出下一次重連時間＝目前正在嘗試，倒數明確落在 0s。
                lblIoConn.Text = "● IO 重連中 0s…";
                lblIoConn.BackColor = IecRed;
            }
            RefreshGrabButtonState();
        }

        private void RefreshIoConnLabel()
        {
            IoGrabController controller = CurrentIoController;
            if (controller == null || _isIoSuspended) return;  // 手動暫停 → 保留「IO 暫停 ⏸」
            if (controller.IsConnected)
            {
                if (lblIoConn.BackColor != IecGreen) UpdateIoConnectionUi(true);  // 連上 → 綠（idempotent）
                return;
            }
            var next = controller.NextReconnectAtUtc;
            if (!next.HasValue) return;  // 尚未排程重連（初始連線進行中）→ 維持「初始化中」
            int sec = (int)Math.Ceiling((next.Value - DateTime.UtcNow).TotalSeconds);
            sec = Math.Max(0, Math.Min(sec, controller.ReconnectIntervalMs / 1000));
            lblIoConn.Text = $"● IO 重連中 {sec}s…";
            lblIoConn.BackColor = IecRed;
            lblIoConn.ForeColor = Color.White;
        }

        private void UpdateIoLeds(IoSnapshot io)
        {
            if (_isIoSuspended) return;
            SetIoLed(lblIoDiAlive,   io.DiNakanAlive);
            SetIoLed(lblIoDiStart,   io.DiInspectStart);
            SetIoLed(lblIoDoPcAlive, io.DoPcAlive);
            UpdateMuraLed(io.DoMuraDetected);
            SetIoLed(lblIoDoPcBusy,  io.DoPcInspect);
        }

        private void ApplyPendingIoSnapshot()
        {
            int bits = System.Threading.Volatile.Read(
                ref _pendingIoSnapshotBits);
            if (bits < 0 || bits == _appliedIoSnapshotBits)
                return;

            _appliedIoSnapshotBits = bits;
            UpdateIoLeds(new IoSnapshot
            {
                DiNakanAlive = (bits & 1) != 0,
                DiInspectStart = (bits & 2) != 0,
                DoPcAlive = (bits & 4) != 0,
                DoMuraDetected = (bits & 8) != 0,
                DoPcInspect = (bits & 16) != 0
            });
        }

        private static int PackIoSnapshot(IoSnapshot snapshot)
        {
            return (snapshot.DiNakanAlive ? 1 : 0) |
                   (snapshot.DiInspectStart ? 2 : 0) |
                   (snapshot.DoPcAlive ? 4 : 0) |
                   (snapshot.DoMuraDetected ? 8 : 0) |
                   (snapshot.DoPcInspect ? 16 : 0);
        }

        private static void SetIoLed(Label lbl, bool on)
        {
            string[] parts = lbl.Text.Split(new[] { "\r\n" }, StringSplitOptions.None);
            string id   = parts[0].TrimStart('◎', '×', ' ');
            string name = parts.Length > 1 ? parts[1] : "";
            lbl.Text = (on ? "◎ " : "× ") + id + "\r\n" + name;
            lbl.BackColor = on ? IecGreen : IecDarkGray;
        }


        /// <summary>Mura pause is an output policy: clear the edge state and force DO1 low.</summary>
        private void HandleMuraPauseSettingsChanged(string name)
        {
            if (name != nameof(InspectionSettings.MuraDetectPaused)) return;

            _muraExceedLatch[0] = false;
            _muraExceedLatch[1] = false;
            _outputHealthService?.Resolve("MuraExceed.v");
            _outputHealthService?.Resolve("MuraExceed.h");
            UpdateMuraLed(false);

            if (_settings.MuraDetectPaused)
            {
                FlowTrace.Log("MURA 暫停 → 清除 DO1");
                _ = CurrentIoController?.ClearMura();
            }
        }

        private void UpdateMuraLed(bool doMuraOn)
        {
            if (_settings.MuraDetectPaused)
            {
                lblIoDoMura.BackColor = IecYellow;
                lblIoDoMura.ForeColor = Color.Black;
                lblIoDoMura.Text = "⏸ DO1\r\nMURA_DET";
            }
            else
            {
                lblIoDoMura.BackColor = doMuraOn ? IecGreen : IecDarkGray;
                lblIoDoMura.ForeColor = Color.White;
                lblIoDoMura.Text = (doMuraOn ? "◎ " : "× ") + "DO1\r\nMURA_DET";
            }
        }

        private void lblIoDoMura_Click(object sender, EventArgs e)
        {
            // intent 行：此鈕走 Hub（程式化通道）→ 後續 set:[MuraDetectPaused] 記程式來源，
            // 這行先蓋「使用者親手做的」章（孤兒判讀規則的主人）。
            FlowTrace.Log("ui:【暫停Mura檢測】鈕");
            // 走 SettingsHub → Changed event → OnSettingChanged 接管 UpdateMuraLed
            _settingsHub.Set(s => s.MuraDetectPaused, !_settings.MuraDetectPaused);
        }

        private void lblIoConn_Click(object sender, EventArgs e)
        {
            IoGrabController controller = CurrentIoController;
            if (controller == null) return;
            _isIoSuspended = !_isIoSuspended;
            FlowTrace.Log($"ui:【IO暫停】鈕 → {(_isIoSuspended ? "暫停" : "恢復")}");   // intent 行（原本完全無痕＝盲區）
            if (_isIoSuspended)
            {
                lblIoConn.BackColor = IecYellow;
                lblIoConn.ForeColor = Color.White;   // 統一白字（原黑字）
                lblIoConn.Text = "● IO 暫停 ⏸";
                btnLiveGrab.Enabled = true;
                UpdateGrabButton(_liveCameraManager?.IsLiveGrabbing ?? false);
                btnLiveGrab.BackColor = SystemColors.Control;
                btnLiveGrab.ForeColor = SystemColors.ControlText;
                // 暫停 = 等同 IO 離線：重置狀態燈和所有 IO 燈號
                lblIoState.Text = "關閉";
                lblIoState.BackColor = IecGray;
                SetIoLed(lblIoDiAlive,   false);
                SetIoLed(lblIoDiStart,   false);
                SetIoLed(lblIoDoPcAlive, false);
                SetIoLed(lblIoDoPcBusy,  false);
                UpdateMuraLed(false);
            }
            else
            {
                UpdateIoConnectionUi(controller.IsConnected);
            }
        }
    }
}
