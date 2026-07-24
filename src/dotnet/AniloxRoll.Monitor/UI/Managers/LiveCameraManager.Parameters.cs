using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        private bool _enableAutoCapture;
        private bool _saveOriginalBmp;
        private string _captureRootPath = string.Empty;
        private int[] _cameraGrabHeight = new int[7];
        private double[] _cameraExposureTimeUs = new double[7];
        private double[] _cameraLineRateHz = new double[7];
        private int _saveResizeScale = InspectionEngineConfig.DefaultSaveResizeScale;
        private int _saveJpgQuality = InspectionEngineConfig.DefaultSaveJpgQuality;
        private float _hessianMaxFactor = InspectionEngineConfig.DefaultHessianMaxFactor;
        private float _ridgeSigma = InspectionEngineConfig.DefaultRidgeSigma;
        private string _ridgeMode = InspectionEngineConfig.DefaultRidgeMode;
        private volatile bool _isParameterReconfiguring;
        private InspectionSettings _inspectionSettings;

        public void SetCaptureSettings(InspectionSettings settings)
        {
            if (settings == null) return;
            _inspectionSettings = settings;

            UpdateCaptureSettingsCache(settings);
            ApplyCapturePolicyToCameras();

            foreach (var cam in _cameras)
            {
                int camIdx = cam.CameraId - 1;
                cam.CameraGrabHeight     = _cameraGrabHeight[camIdx]; // 已在 UpdateCaptureSettingsCache clamp 到 MaxGrabHeightPx
                // 曝光：走 CLProtocol-aware SetExposureUs（CLProtocol 未就緒時記錄，就緒後自動重套）
                cam.SetExposureUs(_cameraExposureTimeUs[camIdx]);
                // 線掃速率：同上，CLProtocol 未就緒時記錄，就緒後自動重套
                cam.SetLineRateHz(_cameraLineRateHz[camIdx]);
            }
        }

        /// <summary>
        /// 套用執行期存檔/演算法政策，不重送曝光、線掃速率或擷取高度。
        /// PropertyGrid 設定變更走這條；完整相機 timing 只在初始化或專用相機參數流程套用。
        /// </summary>
        public void RefreshCapturePolicy(InspectionSettings settings)
        {
            if (settings == null) return;
            _inspectionSettings = settings;
            UpdateCaptureSettingsCache(settings);
            ApplyCapturePolicyToCameras();
        }

        private void ApplyCapturePolicyToCameras()
        {
            foreach (var cam in _cameras)
            {
                cam.EnableAutoCapture    = _enableAutoCapture;
                cam.SaveOriginalBmp      = _saveOriginalBmp;
                cam.CaptureRootPath      = _captureRootPath;
                cam.HessianSigma         = _ridgeSigma;
                cam.HessianFixedMax      = _hessianMaxFactor;
                cam.RidgeMode            = _ridgeMode;
                cam.SaveResizeScale      = _saveResizeScale;
                cam.SaveJpgQuality       = _saveJpgQuality;
                cam.TimestampCoordinator = _timestampCoordinator;
            }
        }

        /// <summary>
        /// 對指定相機設定曝光時間（μs）。CLProtocol 就緒後走 Feature API，否則 fallback MdigControl。
        /// 同 CameraSession.SetExposureForCamera()。
        /// </summary>
        public void SetExposureForCamera(int camId, double exposureUs)
        {
            FindCamera(camId)?.SetExposureUs(exposureUs);
        }

        /// <summary>
        /// 對指定相機設定 Line Rate（Hz），需 CLProtocol 已啟用。
        /// 同 CameraSession.SetLineRateForCamera()。
        /// </summary>
        public void SetLineRateForCamera(int camId, double hz)
        {
            if (IsLiveGrabbing && !_isParameterReconfiguring)
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=cam{camId} param=LineRate reason=GrabActive");
                throw new InvalidOperationException("LineRate cannot change during an active Grab.");
            }

            var cam = FindCamera(camId);
            if (cam == null)
            {
                FlowTrace.Log(
                    $"parameter hardware deferred scope=cam{camId} param=LineRate requested={hz:F1} reason=Unavailable");
                return;
            }

            cam.SetLineRateHz(hz);
            double measured = cam.IsConnected ? cam.GetLineRateHz() : 0;
            double applied = measured > 0 ? measured : cam.AppliedLineRateHz;
            double tolerance = Math.Max(5.0, hz * 0.02);
            if (cam.IsConnected && measured > 0 && Math.Abs(measured - hz) > tolerance)
                throw new InvalidOperationException(
                    $"CAM{camId} LineRate mismatch: requested={hz:F1}, applied={measured:F1}.");

            FlowTrace.Log(
                $"parameter hardware applied scope=cam{camId} param=LineRate " +
                $"requested={hz:F1} applied={applied:F1}");
        }

        /// <summary>
        /// 對指定相機變更 Grab 高度（px）。
        /// 內部走 Stop → Free → Realloc → Restart 完整流程（由 AniloxCamera.SetGrabHeight 處理）。
        /// 同 CameraSession.SetGrabHeightForCamera()。
        /// </summary>
        public void SetGrabHeightForCamera(int camId, int height)
        {
            if (IsLiveGrabbing && !_isParameterReconfiguring)
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=cam{camId} param=Height reason=GrabActive");
                throw new InvalidOperationException("Height cannot change during an active Grab.");
            }
            // grab 中拉大到 ~12062 會 stall → 一律 cap 在 MaxGrabHeightPx(12000) 以下（per-camera 固定，不分台數）。
            if (height > AcquisitionDefaults.MaxGrabHeightPx) height = AcquisitionDefaults.MaxGrabHeightPx;
            var cam = FindCamera(camId);
            if (cam == null)
            {
                FlowTrace.Log(
                    $"parameter hardware deferred scope=cam{camId} param=Height requested={height} reason=Unavailable");
                return;
            }

            if (_isParameterReconfiguring)
            {
                cam.SetGrabHeight(height);
            }
            else
            {
                cam.PauseAcquisition();
                try { cam.SetGrabHeight(height); }
                finally { cam.ResumeAcquisition(); }
            }

            int applied = cam.FrameHeight;
            if (cam.IsConnected && applied != height)
                throw new InvalidOperationException(
                    $"CAM{camId} Height mismatch: requested={height}, applied={applied}.");

            FlowTrace.Log(
                $"parameter hardware applied scope=cam{camId} param=Height " +
                $"requested={height} applied={applied}");
        }

        /// <summary>
        /// Applies one camera parameter, then restarts every connected digitizer as one physical
        /// generation. Restarting only the edited camera would shift its phase against the rest.
        /// </summary>
        public Task<bool> ApplyParamCoordinatedAsync(int camId, Action write)
        {
            var cam = FindCamera(camId);
            var targets = cam == null
                ? new AniloxCamera[0]
                : _cameras.ToArray();
            return ApplyParamCoordinatedCoreAsync("cam" + camId, targets, write);
        }

        /// <summary>
        /// Applies an All parameter as one reconfiguration generation. All connected digitizers
        /// are drained, written, resumed, and observed warm before product frames are accepted.
        /// </summary>
        public Task<bool> ApplyParamCoordinatedAsync(Action write)
        {
            return ApplyParamCoordinatedCoreAsync("All", _cameras.ToArray(), write);
        }

        /// <summary>
        /// Applies exposure without restarting acquisition. Exposure changes integration time but
        /// does not change line timing, so closing the capture gate and rebuilding the physical
        /// frame generation would only add several seconds of avoidable latency.
        /// </summary>
        public Task<bool> ApplyExposureFastAsync(int camId, Action write)
        {
            return ApplyExposureFastCoreAsync("cam" + camId, write);
        }

        /// <summary>Applies the all-camera exposure command while the current grab keeps flowing.</summary>
        public Task<bool> ApplyExposureFastAsync(Action write)
        {
            return ApplyExposureFastCoreAsync("All", write);
        }

        private async Task<bool> ApplyExposureFastCoreAsync(string scope, Action write)
        {
            if (write == null) return true;

            await _allocationGate.WaitAsync();
            try
            {
                if (IsReleasing) return false;

                bool live = IsLiveGrabbing && _captureGateOpen;
                var sw = Stopwatch.StartNew();
                if (live)
                    FlowTrace.Log($"exposure live apply begin scope={scope} gate=open");

                await Task.Run(write);

                if (live)
                {
                    FlowTrace.Log(
                        $"exposure live apply complete scope={scope} " +
                        $"gate={(_captureGateOpen ? "open" : "closed")} " +
                        $"elapsedMs={sw.ElapsedMilliseconds}");
                }
                return true;
            }
            catch (Exception ex)
            {
                FlowTrace.Log(
                    $"exposure live apply failed scope={scope} gate={(_captureGateOpen ? "open" : "closed")} " +
                    $"error={ex.GetType().Name}");
                Trace.TraceWarning(
                    $"[ApplyExposureFastCoreAsync.{scope}] {ex.GetType().Name}: {ex.Message}");
                return false;
            }
            finally
            {
                _allocationGate.Release();
            }
        }

        private async Task<bool> ApplyParamCoordinatedCoreAsync(
            string scope, AniloxCamera[] requestedTargets, Action write)
        {
            if (write == null) return true;

            await _allocationGate.WaitAsync();
            bool enteredReconfiguration = false;
            try
            {
                if (IsReleasing) return false;

                var targets = (requestedTargets ?? new AniloxCamera[0])
                    .Where(cam => cam != null && cam.IsConnected)
                    .Distinct()
                    .ToArray();
                bool wasCapturing = IsLiveGrabbing;
                bool acquisitionRunning = targets.Any(cam => cam.IsLive);
                if (!acquisitionRunning || targets.Length == 0)
                {
                    await Task.Run(write);
                    return true;
                }

                _isParameterReconfiguring = true;
                enteredReconfiguration = true;
                // This must precede every physical stop. Callbacks already inside the old
                // generation are discarded by the display/curve reset before the gate reopens.
                _captureGateOpen = false;
                FlowTrace.Log(
                    $"parameter reconfigure begin scope={scope} gate=closed targets={targets.Length}");

                AcquisitionSyncResult sync = await SynchronizeAcquisitionAsync(
                    "parameter:" + scope,
                    targets,
                    () =>
                    {
                        FlowTrace.Log(
                            $"parameter reconfigure paused scope={scope} cams={targets.Length}");
                        write();
                        FlowTrace.Log($"parameter reconfigure applied scope={scope}");
                    },
                    () => ReapplyLineRatesForSynchronization(
                        "parameter:" + scope, targets),
                    () => IsReleasing || (wasCapturing && !IsLiveGrabbing),
                    validateFramePeriod: true);

                if (!sync.Succeeded)
                {
                    if (sync.Canceled || (wasCapturing && !IsLiveGrabbing) || IsReleasing)
                    {
                        if (!IsReleasing)
                        {
                            ClearUserGrabIntents();
                            FlowTrace.Log(
                                $"parameter stop intent-clear complete scope={scope}");
                        }
                        FlowTrace.Log(
                            $"parameter reconfigure canceled scope={scope} " +
                            $"gate=closed reason=StopGrab");
                        return true;
                    }
                    IsLiveGrabbing = false;
                    ClearUserGrabIntents();
                    FlowTrace.Log(
                        $"parameter reconfigure failed scope={scope} gate=closed " +
                        $"error={sync.Error}");
                    FlowTrace.Log("capture gate closed standby=on");
                    return false;
                }

                foreach (AcquisitionWarmSample sample in sync.Samples)
                {
                    FlowTrace.Log(
                        $"parameter warm ready scope={scope} cam{sample.CameraId} " +
                        $"tick={sample.FrameStartTicks}");
                }

                if (!wasCapturing)
                {
                    FlowTrace.Log(
                        $"parameter reconfigure complete scope={scope} gate=closed warm=True");
                    return true;
                }

                // Stop/close may run while reconfiguration is awaiting a frame. Never reopen a
                // generation that the user has already ended.
                if (!IsLiveGrabbing || IsReleasing)
                {
                    if (!IsReleasing)
                    {
                        ClearUserGrabIntents();
                        FlowTrace.Log(
                            $"parameter stop intent-clear complete scope={scope}");
                    }
                    FlowTrace.Log(
                        $"parameter reconfigure canceled scope={scope} gate=closed reason=StopGrab");
                    return true;
                }

                _display.ResetFlowFirstFrame();
                _display.ResetWaterfallIfActive();
                OnCaptureSequenceReset?.Invoke();
                FlowTrace.Log($"parameter sequence reset scope={scope}");

                _captureGateOpen = true;
                FlowTrace.Log(
                    $"parameter reconfigure complete scope={scope} gate=open warm=True");
                return true;
            }
            finally
            {
                if (enteredReconfiguration)
                {
                    _isParameterReconfiguring = false;
                    // Covers stop/release races that exit through a failure path before the
                    // normal StopGrab cancellation branch above.
                    if (!IsLiveGrabbing && !IsReleasing)
                        ClearUserGrabIntents();
                }
                _allocationGate.Release();
            }
        }

        private AniloxCamera FindCamera(int camId)
        {
            for (int i = 0; i < _cameras.Count; i++)
                if (_cameras[i].CameraId == camId) return _cameras[i];
            return null;
        }

        /// <summary>
        /// Returns the capture-time values owned by the camera/MIL layer. These values are used
        /// for persisted inspection records so a rejected UI command cannot make CSV metadata lie.
        /// </summary>
        public bool TryGetAppliedCaptureParameters(
            int camId, out int frameHeight, out double lineRateHz, out double exposureUs)
        {
            frameHeight = 0;
            lineRateHz = 0;
            exposureUs = 0;

            AniloxCamera cam = FindCamera(camId);
            if (cam == null) return false;

            frameHeight = cam.FrameHeight;
            lineRateHz = cam.AppliedLineRateHz;
            exposureUs = cam.GetExposureUs();
            return frameHeight > 0;
        }

        /// <summary>改參數窗口：暫停/恢復「全部相機」存檔（不影響 grab/檢測/顯示）。
        /// 套用參數時設 true → 等全部相機恢復同步（UI 解鎖時）設 false → 存出的序列不含重啟空檔、各台齊全。</summary>
        public void SetCaptureSuppressed(bool suppressed)
        {
            foreach (var cam in _cameras)
                if (cam != null) cam.SuppressCapture = suppressed;
        }

        /// <summary>所有在抓相機中「最大的幀週期」(ms)＝FrameHeight/AppliedLineRateHz。
        /// 供改參數後的參數鎖：至少鎖住一個完整幀週期，確保改完的相機跑完一張乾淨幀才放行下一次改（防連改太快 stall）。</summary>
        public int GetMaxFramePeriodMs()
        {
            return GetMaxFramePeriodMs(_cameras);
        }

        private static int GetMaxFramePeriodMs(IEnumerable<AniloxCamera> cameras)
        {
            int maxMs = 0;
            foreach (var cam in cameras)
            {
                if (cam == null || !cam.IsLive) continue;
                double lr = cam.AppliedLineRateHz;
                int h = cam.FrameHeight;
                if (lr > 0 && h > 0)
                {
                    int ms = (int)System.Math.Ceiling(h / lr * 1000.0);
                    if (ms > maxMs) maxMs = ms;
                }
            }
            return maxMs;
        }

        private void UpdateCaptureSettingsCache(InspectionSettings settings)
        {
            if (settings == null) return;
            _enableAutoCapture    = settings.EnableAutoCapture;
            _saveOriginalBmp = settings.Storage?.SaveOriginalBmp ?? false;
            _captureRootPath      = settings.CaptureRootPath ?? string.Empty;
            // grab 高度 clamp 到硬上限 MaxGrabHeightPx（grab 中拉大超過 ~12062 會 stall；per-camera 固定不分台數）。
            // clamp 後是純數字、走與 json 同路徑設給 cam.CameraGrabHeight（不碰 MIL init 查詢）→ 不會 stall。
            int maxH = AcquisitionDefaults.MaxGrabHeightPx;
            var srcH = settings.Acquisition.CameraGrabHeight;
            _cameraGrabHeight = new int[srcH.Length];
            for (int i = 0; i < srcH.Length; i++)
            {
                int h = srcH[i] > 0 ? srcH[i] : AcquisitionDefaults.GrabHeight;
                _cameraGrabHeight[i] = h > maxH ? maxH : h;
            }
            _cameraExposureTimeUs = settings.Acquisition.CameraExposureTimeUs;
            _cameraLineRateHz     = settings.Acquisition.CameraLineRateHz;
            _saveResizeScale      = settings.Recipe?.SaveResizeScale ?? InspectionEngineConfig.DefaultSaveResizeScale;
            _saveJpgQuality       = settings.Recipe?.SaveJpgQuality  ?? InspectionEngineConfig.DefaultSaveJpgQuality;
            // capture-time HM 用 V（baked 進 .bin）；H 為 view-time only，不送進 native
            _hessianMaxFactor     = settings.HessianMaxFactorV > 0
                ? settings.HessianMaxFactorV
                : InspectionEngineConfig.DefaultHessianMaxFactor;
            _ridgeSigma           = settings.RidgeSigma > 0
                ? settings.RidgeSigma
                : InspectionEngineConfig.DefaultRidgeSigma;
            _ridgeMode            = InspectionRecipe.RidgeDirectionToNative(settings.RidgeDir);
            _dcfPath              = settings.DcfPath ?? string.Empty;
        }
    }
}
