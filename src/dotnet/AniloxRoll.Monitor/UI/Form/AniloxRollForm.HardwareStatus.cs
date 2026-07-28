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
    /// <summary>AniloxRollForm IO / 光源 / 儲存硬體狀態（初始化 + 連線標籤 + LED）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        /// <summary>初始化 IO 連動：自動偵測連線，連上後以 DI START 控制 Grab。</summary>
        private void InitIoController()
        {
            int generation = System.Threading.Interlocked.Increment(ref _ioControllerGeneration);
            StartIoController(generation);
        }

        private void StartIoController(int generation)
        {
            if (!_settings.IoEnabled || generation != System.Threading.Volatile.Read(ref _ioControllerGeneration))
                return;

            var controller = new IoGrabController(_settings.IoModel)
            {
                ReconnectIntervalMs = 3000,
                ReadWriteTimeoutMs = 500,
                StopCaptureOnStartLow =
                    _settings.CaptureStopCondition == CaptureStopCondition.IoSignal
            };
            string ip = _settings.IoIp;
            int port = _settings.IoPort;
            _ioGrabController = controller;
            _ioControllerActiveGeneration = generation;

            controller.OnStartRequested += () =>
            {
                if (!IsCurrentIoController(controller, generation)) return;
                int requestGeneration = System.Threading.Interlocked.Increment(
                    ref _ioGrabRequestGeneration);
                DispatchCurrentIoController(
                    controller, generation, () =>
                    {
                        FlowTrace.Log("io:DI START 上升緣 → 抓取請求");
                        _ = IoStartGrabAsync(controller, generation, requestGeneration);
                    });
            };
            controller.OnStopRequested += reason =>
            {
                if (!IsCurrentIoController(controller, generation)) return;
                System.Threading.Interlocked.Increment(ref _ioGrabRequestGeneration);
                DispatchCurrentIoController(
                    controller, generation, () => _ = IoStopGrabAsync(controller, generation, reason));
            };
            controller.OnStateChanged += state => DispatchCurrentIoController(
                controller, generation, () => UpdateIoStateLabel(state));
            controller.OnConnectionChanged += connected =>
            {
                if (!IsCurrentIoController(controller, generation)) return;
                if (!connected)
                    System.Threading.Interlocked.Increment(ref _ioGrabRequestGeneration);
                DispatchCurrentIoController(
                    controller, generation, () => UpdateIoConnectionUi(connected));
            };
            controller.OnIoUpdated += snapshot => DispatchCurrentIoController(
                controller, generation, () => UpdateIoLeds(snapshot));

            FlowTrace.Log($"IO controller start generation={generation} endpoint={ip}:{port}");
            _ioControllerStartTask = StartIoControllerAsync(controller, generation, ip, port);
        }

        private async Task StartIoControllerAsync(
            IoGrabController controller, int generation, string ip, int port)
        {
            try
            {
                await controller.StartAsync(ip, port);
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[IO.Start generation={generation}] {ex.GetType().Name}: {ex.Message}");
                DispatchCurrentIoController(
                    controller, generation, () => UpdateIoConnectionUi(false));
            }
        }

        private bool IsCurrentIoController(IoGrabController controller, int generation)
        {
            return !_shutdownInProgress &&
                   generation == System.Threading.Volatile.Read(ref _ioControllerGeneration) &&
                   ReferenceEquals(_ioGrabController, controller);
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

        private void InitLightController()
        {
            if (!_settings.LightEnabled) return;
            if (_lightConnectionCoordinator == null)
            {
                _lightConnectionCoordinator =
                    new LightConnectionCoordinator(TelemetryTickMs);
                _lightConnectionCoordinator.StateChanged += () =>
                    SafeBeginInvoke(UpdateLightConnLabel);
                _lightConnectionCoordinator.ActivePortChanged += found =>
                    SafeBeginInvoke(() =>
                    {
                        if (_settings == null ||
                            !_settings.LightEnabled ||
                            string.Equals(
                                found,
                                _settings.LightComPort,
                                StringComparison.OrdinalIgnoreCase))
                            return;

                        _settingsHub.SetBatch(s => s.LightComPort = found);
                        RefreshGridItem(nameof(InspectionSettings.LightComPort));
                    });
            }

            _lightConnectionCoordinator.Start(
                _settings.LightComPort,
                _settings.LightChannel);
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

        private void LightTurnOn()
        {
            _lightConnectionCoordinator?.TurnOn(
                _settings.LightChannel,
                _settings.LightBrightness);
        }

        private void LightTurnOff()
        {
            _lightConnectionCoordinator?.TurnOff(_settings.LightChannel);
        }

        /// <summary>
        /// 光源 PropertyGrid 變更 → 立即生效：
        /// - LightEnabled false→true：啟動偵測；true→false：關閉連線
        /// - COM Port / 通道變更：重新偵測
        /// - 亮度變更：立即套用到硬體（若正在點燈，連同 TurnOn 更新輸出）
        /// </summary>
        private void HandleLightSettingsChanged(string changedPropertyName)
        {
            switch (changedPropertyName)
            {
                case nameof(InspectionSettings.LightEnabled):
                    if (_settings.LightEnabled)
                    {
                        InitLightController();
                    }
                    else
                    {
                        _lightConnectionCoordinator?.Disable();
                    }
                    break;

                case nameof(InspectionSettings.LightComPort):
                case nameof(InspectionSettings.LightChannel):
                    if (_settings.LightEnabled)
                        InitLightController();
                    break;

                case nameof(InspectionSettings.LightBrightness):
                    _lightConnectionCoordinator?.SetBrightness(
                        _settings.LightChannel,
                        _settings.LightBrightness);
                    UpdateLightConnLabel();
                    break;
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
                    int generation = System.Threading.Interlocked.Increment(ref _ioControllerGeneration);
                    _ = RestartIoControllerAsync(generation);
                    break;
            }
        }

        /// <summary>序列化停舊與重建；快速連續改設定只建立最後一代 controller。</summary>
        private async Task RestartIoControllerAsync(int requestedGeneration)
        {
            await _ioControllerLifecycleGate.WaitAsync();
            try
            {
                if (requestedGeneration != System.Threading.Volatile.Read(ref _ioControllerGeneration))
                {
                    FlowTrace.Log($"IO controller restart coalesced generation={requestedGeneration}");
                    return;
                }

                var oldController = _ioGrabController;
                var oldStartTask = _ioControllerStartTask;
                int oldGeneration = _ioControllerActiveGeneration;
                _ioGrabController = null;
                _ioControllerActiveGeneration = 0;
                _ioControllerStartTask = Task.CompletedTask;
                if (oldController != null)
                {
                    FlowTrace.Log($"IO controller stop generation={oldGeneration} reason=settings");
                    try { await oldStartTask; } catch { }
                    try { await oldController.StopAsync(); } catch { }
                    oldController.Dispose();
                }

                if (requestedGeneration != System.Threading.Volatile.Read(ref _ioControllerGeneration) ||
                    _shutdownInProgress)
                    return;

                UpdateIoConnectionUi(false);
                StartIoController(requestedGeneration);
            }
            finally
            {
                _ioControllerLifecycleGate.Release();
            }
        }

        private async Task ShutdownIoControllerAsync()
        {
            System.Threading.Interlocked.Increment(ref _ioGrabRequestGeneration);
            System.Threading.Interlocked.Increment(ref _ioControllerGeneration);
            await _ioControllerLifecycleGate.WaitAsync();
            try
            {
                var controller = _ioGrabController;
                var startTask = _ioControllerStartTask;
                int activeGeneration = _ioControllerActiveGeneration;
                _ioGrabController = null;
                _ioControllerActiveGeneration = 0;
                _ioControllerStartTask = Task.CompletedTask;
                if (controller == null) return;

                FlowTrace.Log($"IO controller stop generation={activeGeneration} reason=shutdown");
                try { await startTask; } catch { }
                try { await controller.StopAsync(); }
                catch (Exception ex)
                {
                    Trace.TraceWarning(
                        $"[Shutdown.IO] {ex.GetType().Name}: {ex.Message}");
                }
                try { controller.Dispose(); } catch { }
            }
            finally
            {
                _ioControllerLifecycleGate.Release();
            }
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

        /// <summary>啟動時硬體狀態列先顯示「初始化中」（灰）—— 各硬體連線/偵測完成後由各自 Update*Label 接手。
        /// 只對「已啟用/已設定」的硬體顯示，避免停用項目卡在「初始化中」。</summary>
        private void ShowHardwareStatusInitializing()
        {
            if (_settings == null) return;
            if (_settings.IoEnabled)
            {
                lblIoConn.Text = "● IO: 初始化中…";  lblIoConn.BackColor = IecGray;
            }
            if (_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源: 初始化中…";  lblLightConn.BackColor = IecGray;
            }
            if (!string.IsNullOrWhiteSpace(_settings.RemotePath))
            {
                lblStorageConn.Text = "● 儲存電腦: 初始化中…";  lblStorageConn.BackColor = IecGray;
            }
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

        /// <summary>由 TelemetryTimer 每 tick 呼叫：IO 斷線時顯示重連倒數（秒數源自 IoGrabController）。
        /// 手動暫停（_isIoSuspended）不覆蓋；初始連線中（尚未排程重連）維持「初始化中」。</summary>
        // ── H 系列：硬體連線邊緣留痕（IO/光源/儲存；狀態轉變才記一行，斷線/恢復現場排障關鍵）──
        private bool? _lastFlowIoConn, _lastFlowLightConn, _lastFlowStorageShareConn;

        private void FlowHardwareEdges()
        {
            void Edge(ref bool? last, bool now, string name)
            {
                if (last == now) return;
                // 首次觀測（null→值）只記基線斷線（開機就連不上也值得留痕）；恢復/斷線轉變一律記
                if (last.HasValue)
                    FlowTrace.Log(now ? $"{name} 恢復連線" : $"⚠ {name} 斷線");
                else if (!now)
                    FlowTrace.Log($"⚠ {name} 未連線（開機基線）");
                last = now;
            }
            Edge(ref _lastFlowIoConn, _ioGrabController?.IsConnected == true, "IO");
            if (_settings?.LightEnabled == true)
                Edge(
                    ref _lastFlowLightConn,
                    _lightConnectionCoordinator?.Snapshot.Connected == true,
                    "光源");
            if (!string.IsNullOrWhiteSpace(_settings?.RemotePath))
                Edge(
                    ref _lastFlowStorageShareConn,
                    _storageHealthCoordinator?.Snapshot.RemoteShareConnected == true,
                    "儲存分享");

            if (_settings?.IoEnabled == true && _ioGrabController != null)
            {
                if (_ioGrabController.IsConnected)
                    _outputHealthService?.Resolve("IoConnection");
                else
                    _outputHealthService?.Report(
                        "IoConnection", OutputHealthSeverity.Critical, "IO 未連線");
            }
            else
            {
                _outputHealthService?.Resolve("IoConnection");
            }

            LightConnectionSnapshot light =
                _lightConnectionCoordinator?.Snapshot;
            if (_settings?.LightEnabled == true && light?.HasProbed == true)
            {
                if (light.Connected)
                    _outputHealthService?.Resolve("LightConnection");
                else
                    _outputHealthService?.Report(
                        "LightConnection", OutputHealthSeverity.Critical, "光源未連線");
            }
            else if (_settings?.LightEnabled != true)
            {
                _outputHealthService?.Resolve("LightConnection");
            }
        }

        private void RefreshIoConnLabel()
        {
            if (_ioGrabController == null || _isIoSuspended) return;  // 手動暫停 → 保留「IO 暫停 ⏸」
            if (_ioGrabController.IsConnected)
            {
                if (lblIoConn.BackColor != IecGreen) UpdateIoConnectionUi(true);  // 連上 → 綠（idempotent）
                return;
            }
            var next = _ioGrabController.NextReconnectAtUtc;
            if (!next.HasValue) return;  // 尚未排程重連（初始連線進行中）→ 維持「初始化中」
            int sec = (int)Math.Ceiling((next.Value - DateTime.UtcNow).TotalSeconds);
            sec = Math.Max(0, Math.Min(sec, _ioGrabController.ReconnectIntervalMs / 1000));
            lblIoConn.Text = $"● IO 重連中 {sec}s…";
            lblIoConn.BackColor = IecRed;
            lblIoConn.ForeColor = Color.White;
        }

        private void UpdateLightConnLabel()
        {
            if (_settings == null || !_settings.LightEnabled)
            {
                lblLightConn.Text = "● 光源 停用";
                lblLightConn.BackColor = IecGray;
                return;
            }

            LightConnectionSnapshot light =
                _lightConnectionCoordinator?.Snapshot;
            if (light?.Connected == true)
            {
                lblLightConn.Text = $"● 光源 已連線 ({_settings.LightBrightness})";
                lblLightConn.BackColor = IecGreen;
            }
            else if (light == null || !light.HasProbed)
            {
                // 初次偵測還沒回來 → 維持「初始化中」（與 IO/儲存一致）
                lblLightConn.Text = "● 光源: 初始化中…";
                lblLightConn.BackColor = IecGray;
            }
            else
            {
                // 斷線 → 顯示 coordinator 提供的 probe 狀態與倒數。
                lblLightConn.Text = light.ProbeInFlight
                    ? "● 光源 探測中…"
                    : $"● 光源 重連中 {light.ReconnectSeconds}s…";
                lblLightConn.BackColor = IecRed;
            }

            UpdateStandardBgSubLockState();
        }

        internal const int TelemetryTickMs = 500;          // = SettingsTabs 的 _telemetryTimer.Interval
        private const long RemoteBacklogWarningBytes = 20L * 1024 * 1024 * 1024;

        private void RefreshOutputCapacityHealth()
        {
            if (_outputHealthService == null) return;

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            long minFreeBytes = GetStorageMinFreeBytes();
            if (storage?.LocalFreeBytes >= 0 &&
                storage.LocalTotalBytes > 0)
            {
                if (minFreeBytes >= storage.LocalTotalBytes)
                {
                    _outputHealthService.Report(
                        "StorageThresholdInvalid",
                        OutputHealthSeverity.OutputFault,
                        "預留空間設定超過磁碟容量，已停止自動清理");
                    _outputHealthService.Resolve("LocalLowSpace");
                }
                else
                {
                    _outputHealthService.Resolve("StorageThresholdInvalid");
                    if (storage.LocalFreeBytes < minFreeBytes)
                    {
                        _outputHealthService.Report(
                            "LocalLowSpace",
                            OutputHealthSeverity.Notice,
                            (_appMode?.Role == MachineRole.Storage ? "儲存電腦" : "檢測電腦") +
                            "空間低於預留值，正在清理最舊資料");
                    }
                    else
                    {
                        _outputHealthService.Resolve("LocalLowSpace");
                    }
                }
            }

            long pendingBytes = _remoteCopyService?.PendingBytes ?? 0;
            if (pendingBytes >= RemoteBacklogWarningBytes)
            {
                _outputHealthService.Report(
                    "RemoteBacklog",
                    OutputHealthSeverity.Notice,
                    "遠端待傳已超過 20 GB");
            }
            else
            {
                _outputHealthService.Resolve("RemoteBacklog");
            }
        }

        private void HandleSettingsStoreIssue(SettingsStoreIssue issue)
        {
            if (issue == null || _outputHealthService == null) return;

            string file = Path.GetFileName(issue.Path);
            if (issue.Kind == SettingsStoreIssueKind.RebuiltDefaults)
            {
                _outputHealthService.Report(
                    "ConfigRebuilt." + file,
                    OutputHealthSeverity.OutputFault,
                    $"{file} 損壞，已用預設值重建");
                _outputHealthService.Resolve("ConfigRebuilt." + file);
                return;
            }

            _outputHealthService.Report(
                "ConfigSaveFailed." + file,
                OutputHealthSeverity.OutputFault,
                $"{file} 寫入失敗：{issue.Reason}");
        }

        private void HandleStorageSettingsChanged(string changedPropertyName)
        {
            if (changedPropertyName != nameof(InspectionSettings.LocalMinFreeGB)) return;

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.RefreshLocalCapacity();
            long totalBytes = storage?.LocalTotalBytes ?? 0;

            int requestedGb = _settings.LocalMinFreeGB;
            int maxGb = totalBytes > 0
                ? Math.Max(1, (int)(totalBytes / (1024L * 1024L * 1024L)) - 1)
                : requestedGb;
            if (totalBytes > 0 && requestedGb > maxGb)
            {
                _settingsHub.SetBatch(s => s.LocalMinFreeGB = maxGb);
                requestedGb = maxGb;
                RefreshGridItem(nameof(InspectionSettings.LocalMinFreeGB));
                _outputHealthService?.Report(
                    "StorageThresholdAdjusted",
                    OutputHealthSeverity.OutputFault,
                    $"預留空間超過磁碟容量，已調整為 {maxGb} GB");
                _outputHealthService?.Resolve("StorageThresholdAdjusted");
            }

            if (_appMode?.Role == MachineRole.Storage &&
                _appMode.StorageMinFreeGB != requestedGb)
            {
                _appMode.StorageMinFreeGB = requestedGb;
                _appMode.Save();
            }

            RefreshOutputCapacityHealth();
            Task.Run(() => _retentionService?.RunCleanup());
        }

        private int CancelRemoteCopyForDay(string dayDirectory)
        {
            if (_remoteCopyService == null || string.IsNullOrWhiteSpace(dayDirectory)) return 0;

            int canceled = _remoteCopyService.CancelPendingFilesUnder(dayDirectory);
            string monthDirectory = Path.GetDirectoryName(dayDirectory);
            string dailyCsv = monthDirectory == null
                ? null
                : Path.Combine(monthDirectory, Path.GetFileName(dayDirectory) + ".csv");
            if (_remoteCopyService.CancelPendingFile(dailyCsv)) canceled++;
            return canceled;
        }

        private void HandleRetentionCleanupCompleted(RetentionCleanupResult result)
        {
            if (result == null) return;
            _storageHeartbeatService?.RecordCleanup(result.FreedBytes);
            if (result.DeletedDayFolders > 0 &&
                _appMode?.Role != MachineRole.Storage)
            {
                CleanupInactiveBackgroundVersions();
            }

            if (result.CanceledPendingFiles > 0)
            {
                const string code = "RetentionDiscardedPending";
                _outputHealthService?.Report(
                    code,
                    OutputHealthSeverity.OutputFault,
                    $"空間不足，已刪除最舊資料（含 {result.CanceledPendingFiles} 個未傳檔案）");
                _outputHealthService?.Resolve(code);
            }
            else if (result.DeletedDayFolders > 0)
            {
                const string code = "RetentionCleanup";
                _outputHealthService?.Report(
                    code,
                    OutputHealthSeverity.Notice,
                    $"空間不足，已清理最舊 {result.DeletedDayFolders} 天資料");
                _outputHealthService?.Resolve(code);
            }
        }

        private static string FormatCapacity(string computerName, long freeBytes, long totalBytes)
        {
            if (freeBytes < 0 || totalBytes <= 0)
                return computerName + "：無法讀取";

            double freeGb = freeBytes / (1024.0 * 1024 * 1024);
            double totalGb = totalBytes / (1024.0 * 1024 * 1024);
            return $"{computerName}：剩餘 {freeGb:N1} / {totalGb:N1} GB";
        }

        private void RefreshCapacityInfoLabel()
        {
            if (lblInfo == null) return;

            RefreshOutputCapacityHealth();

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            string capacityText = _appMode?.Role == MachineRole.Storage
                ? FormatCapacity(
                    "儲存電腦",
                    storage?.LocalFreeBytes ?? -1,
                    storage?.LocalTotalBytes ?? 0)
                : FormatCapacity(
                    "檢測電腦",
                    storage?.LocalFreeBytes ?? -1,
                    storage?.LocalTotalBytes ?? 0) +
                  " ｜ " + FormatCapacity(
                    "儲存電腦",
                    storage?.RemoteFreeBytes ?? -1,
                    storage?.RemoteTotalBytes ?? 0);

            if (_appMode?.Role != MachineRole.Storage && _remoteCopyService != null)
            {
                capacityText += $" ｜ 待傳：{_remoteCopyService.PendingBytes / (1024.0 * 1024 * 1024):N1} GB" +
                    $"（{_remoteCopyService.QueueCount} 檔）";
                long localTicks = System.Threading.Interlocked.Read(
                    ref _lastLocalSaveUtcTicks);
                DateTime? remoteUtc = _remoteCopyService.LastSuccessfulCopyUtc;
                capacityText += " ｜ 最近存檔：" +
                    (localTicks > 0
                        ? new DateTime(localTicks, DateTimeKind.Utc).ToLocalTime().ToString("HH:mm:ss")
                        : "--");
                capacityText += " ｜ 最近遠傳：" +
                    (remoteUtc.HasValue
                        ? remoteUtc.Value.ToLocalTime().ToString("HH:mm:ss")
                        : "--");
            }

            if (!string.Equals(lblInfo.Text, capacityText, StringComparison.Ordinal))
                lblInfo.Text = capacityText;
        }

        private void UpdateStorageConnLabel()
        {
            string path = _settings?.RemotePath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(path))
            {
                lblStorageConn.Text = "● 儲存電腦 停用";
                lblStorageConn.BackColor = IecGray;
                _outputHealthService?.Resolve("StorageConnection");
                _outputHealthService?.Resolve("StorageHeartbeat");
                return;
            }

            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;
            if (storage?.RemoteShareConnected == true)
            {
                _outputHealthService?.Resolve("StorageConnection");
                if (storage.RemoteAppAlive == true)
                {
                    lblStorageConn.Text = "● 儲存電腦 已連線";
                    lblStorageConn.BackColor = IecGreen;
                    _outputHealthService?.Resolve("StorageHeartbeat");
                }
                else
                {
                    lblStorageConn.Text = "● 儲存分享可用 / 程式未回報";
                    lblStorageConn.BackColor = IecYellow;
                    _outputHealthService?.Report(
                        "StorageHeartbeat",
                        OutputHealthSeverity.Critical,
                        "儲存電腦程式未回報");
                }
            }
            else if (storage?.RemoteShareConnected == false)
            {
                _outputHealthService?.Resolve("StorageHeartbeat");
                _outputHealthService?.Report(
                    "StorageConnection",
                    OutputHealthSeverity.Critical,
                    "儲存電腦連線中斷，本機持續存檔");
                lblStorageConn.Text = storage.RemoteProbeInFlight
                    ? "● 儲存電腦 探測中…"
                    : $"● 儲存電腦 重連中 {storage.ReconnectSeconds}s…";
                lblStorageConn.BackColor = IecRed;
            }
            // 尚未 probe 過時維持「初始化中」。
        }

        /// <summary>
        /// 由 TelemetryTimer_Tick 每 500ms 呼叫。各 coordinator 推進自己的
        /// 連線生命週期，Form 只依快照更新控制項與產品告警。
        /// </summary>
        private void UpdateConnectionStatusLabels()
        {
            _storageHealthCoordinator?.Tick();
            StorageHealthSnapshot storage =
                _storageHealthCoordinator?.Snapshot;

            if (_appMode?.Role == MachineRole.Storage)
            {
                if (storage?.LocalFreeBytes >= 0 &&
                    storage.LocalTotalBytes > 0)
                {
                    if (_storageDiskFreeRow != null)
                    {
                        double freeGb =
                            storage.LocalFreeBytes /
                            (1024.0 * 1024 * 1024);
                        double totalGb =
                            storage.LocalTotalBytes /
                            (1024.0 * 1024 * 1024);
                        _storageDiskFreeRow.SubItems[1].Text = $"{freeGb:F1} / {totalGb:F1} GB";
                    }
                }
                RefreshCapacityInfoLabel();
                return;
            }

            RefreshCapacityInfoLabel();

            // Grab watchdog：取像中超過 30 秒沒有 result callback → 觸發循環儲存
            if (_liveCameraManager?.IsLiveGrabbing == true &&
                _lastGrabEventTime != DateTime.MinValue &&
                (DateTime.UtcNow - _lastGrabEventTime).TotalSeconds > 30)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                Task.Run(() => _retentionService?.RunCleanup());
            }

            // 光源連線生命週期由 coordinator 管理；Form 只推進 timer 並畫快照。
            _lightConnectionCoordinator?.Tick();
            UpdateLightConnLabel();
            RefreshIoConnLabel();          // IO 重連倒數每 tick 刷新（源自 IoGrabController）
            UpdateStorageConnLabel();
            FlowHardwareEdges();           // H 系列：IO/光源/儲存 斷線/恢復 邊緣留痕
        }

        /// <summary>本機網路介面變動（拔/插網路線）→ 立即觸發儲存重探（下一個 telemetry tick ≤500ms），
        /// 不必等整個探測週期。事件驅動、零輪詢成本。
        /// 注意：遠端 PC 自己關機/更新時本機網卡不變、此事件不觸發，那種情況仍靠週期探測。</summary>
        private void OnNetworkAddressChanged(object sender, EventArgs e)
        {
            _storageHealthCoordinator?.ForceRemoteProbe();
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
                _ = _ioGrabController?.ClearMura();
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

        private void TriggerRetentionAndFlagAsync()
        {
            Task.Run(() => _retentionService?.RunCleanup());
            WriteFlagToRemoteAsync();
        }

        private void WriteFlagToRemoteAsync()
        {
            // JSON 有設定就用，否則從 RemotePath 推算（同 IP，固定 AniloxConfig share）
            string configPath = _settings?.RemoteConfigPath ?? string.Empty;
            if (string.IsNullOrWhiteSpace(configPath))
                configPath = DeriveFlagSharePath(_settings?.RemotePath);
            if (string.IsNullOrWhiteSpace(configPath)) return;

            Task.Run(() =>
            {
                try
                {
                    string flagPath = Path.Combine(configPath, "cleanup-request.flag");
                    File.WriteAllText(flagPath, DateTime.UtcNow.ToString("O"),
                        System.Text.Encoding.UTF8);
                }
                catch (Exception ex)
                {
                    Trace.TraceWarning($"[RetentionFlag] 寫旗標失敗: {ex.Message}");
                }
            });
        }

        private bool _storageModeLayoutApplied;

        private void ApplyStorageModeUi()
        {
            if (_appMode?.Role != MachineRole.Storage || _storageModeLayoutApplied) return;
            _storageModeLayoutApplied = true;

            tabMain.TabPages.Remove(tabPageLiveView);
            tabControlRight.TabPages.Remove(tabPageCamera);

            // PropertyGrid：隱藏 IO / 相機 / 光源三個大類
            TypeDescriptor.AddProvider(
                new StorageModeSettingsFilter(TypeDescriptor.GetProvider(_settings)), _settings);
            propertyGridSettings.Refresh();

            // The remaining panes are anchored at a fixed Y instead of Dock.Fill, so
            // hiding the parent alone does not reclaim its row. Compact both panes
            // before ProportionalScaler captures the storage-mode baseline.
            int releasedHeight = panelStatusBar.Height;
            panelStatusBar.Visible = false;
            tabMain.SetBounds(
                tabMain.Left,
                tabMain.Top - releasedHeight,
                tabMain.Width,
                tabMain.Height + releasedHeight);
            tabControlRight.SetBounds(
                tabControlRight.Left,
                tabControlRight.Top - releasedHeight,
                tabControlRight.Width,
                tabControlRight.Height + releasedHeight);
        }

        // \\server\share → \\server\AniloxConfig（cleanup-request.flag 目標）
        private static string DeriveFlagSharePath(string remotePath)
        {
            if (string.IsNullOrWhiteSpace(remotePath)) return "";
            var parts = remotePath.TrimStart('\\').Split('\\');
            return parts.Length < 1 || string.IsNullOrEmpty(parts[0])
                ? "" : $@"\\{parts[0]}\AniloxConfig";
        }

        private string GetStorageRetentionRoot()
        {
            if (_appMode?.Role == MachineRole.Storage &&
                !string.IsNullOrWhiteSpace(_appMode.StorageMachineDataPath))
                return _appMode.StorageMachineDataPath;
            return _settings?.CaptureRootPath ?? string.Empty;
        }

        private long GetStorageMinFreeBytes()
        {
            int minFreeGb = _settings?.LocalMinFreeGB ?? InspectionDefaults.LocalMinFreeGB;
            if (_appMode?.Role == MachineRole.Storage)
            {
                minFreeGb = _appMode.StorageMinFreeGB > 0
                    ? _appMode.StorageMinFreeGB
                    : AppModeDefaults.StorageMinFreeGB;
            }
            return (long)minFreeGb * 1024L * 1024L * 1024L;
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
            if (_ioGrabController == null) return;
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
                UpdateIoConnectionUi(_ioGrabController.IsConnected);
            }
        }
    }
}
