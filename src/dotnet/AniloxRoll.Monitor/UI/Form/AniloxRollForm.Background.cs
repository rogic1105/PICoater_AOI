using System;
using System.ComponentModel;
using System.IO;
using System.Drawing;
using System.Globalization;
using System.Runtime.InteropServices;
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
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>AniloxRollForm 背景（取得/載入/預覽）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        private bool? _lastFlowBackgroundCaptureReady;

        /// <summary>
        /// 取得背景：啟動 grab → 採集 N 秒 → 多幀平均 column mean → 存 MCBF bin。
        /// </summary>
        private async void btnLiveGetBackground_Click(object sender, EventArgs e)
        {
            FlowTrace.Log("ui:【取得背景】鈕");   // intent 行（孤兒判讀規則）
            if (!IsStandardBgSubEnabled)
            {
                MessageBox.Show("請先將去背演算法切換為「標準去背」。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            // 先清除舊的背景預覽（釋放 overlay + 恢復 MIL display）
            if (IsBgPreviewActive) ClearBackgroundPreview();

            // 背景採樣只借用相機 grab 與演算法，不是產品擷取，不得產生圖片/CSV。
            _liveCameraManager.SetCaptureSuppressed(true);
            FlowTrace.Log("background capture begin output=disabled");

            // 確保相機已 allocate
            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    bool started =
                        await _liveCameraManager.EnsureAllocatedAndToggleGrabAsync(false);
                    if (!_liveCameraManager.IsAllocated || !started)
                    {
                        _liveCameraManager.SetCaptureSuppressed(false);
                        FlowTrace.Log("background capture end output=disabled result=start-failed");
                        return;
                    }
                }
                catch (Exception ex)
                {
                    _liveCameraManager.SetCaptureSuppressed(false);
                    FlowTrace.Log("background capture end output=disabled result=start-failed");
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }

            // 確保 grab 中；開燈命令完成後直接開始，無固定暖機等待。
            if (!_liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOn();
                bool started = await _liveCameraManager.ToggleGrabAsync();
                if (!started)
                {
                    _liveCameraManager.SetCaptureSuppressed(false);
                    FlowTrace.Log("background capture end output=disabled result=start-failed");
                    return;
                }
                UpdateGrabButton(true);
            }

            btnLiveGetBackground.Enabled = false;
            btnLiveGrab.Enabled = false;

            int sampleSeconds = Math.Max(1, _settings.Recipe.BackgroundSampleSeconds);
            string bgDir = _settings.Storage.BackgroundPath;
            var backgroundRepository = new BackgroundProfileRepository(bgDir);
            var captureCoordinator = new BackgroundCaptureCoordinator(
                _liveCameraManager,
                backgroundRepository);
            bool captureSucceeded = false;

            try
            {
                await captureCoordinator.CaptureAndActivateAsync(
                    sampleSeconds,
                    _settings.LightBrightness,
                    remaining =>
                        btnLiveGetBackground.Text = $"採集中 {remaining}s");
                // 載入到各相機
                LoadBackgroundBins();
                captureSucceeded = true;
                _outputHealthService?.Resolve("BackgroundCaptureFailure");
            }
            catch (Exception ex)
            {
                _outputHealthService?.Report(
                    "BackgroundCaptureFailure",
                    OutputHealthSeverity.OutputFault,
                    "背景取得失敗，繼續使用上一組背景：" + ex.Message);
                _outputHealthService?.Resolve("BackgroundCaptureFailure");
            }
            finally
            {
                btnLiveGetBackground.Text = "取得背景";
                btnLiveGetBackground.Enabled = true;

                // 採集完成後一律停止 grab
                if (_liveCameraManager.IsLiveGrabbing)
                {
                    _liveCameraManager.StopGrab();
                    LightTurnOff();
                    UpdateGrabButton(false);
                }

                _liveCameraManager.SetCaptureSuppressed(false);
                FlowTrace.Log(
                    $"background capture end output=disabled result={(captureSucceeded ? "ok" : "failed")}");
                UpdateStandardBgSubLockState();
            }

            if (!captureSucceeded)
            {
                bool wasIoContinuation = _autoStartGrabAfterBg;
                int ioGeneration = _autoStartGrabIoGeneration;
                var ioController = CurrentIoController;
                _autoStartGrabAfterBg = false;
                _autoStartGrabIoGeneration = 0;
                _autoStartGrabIoRequestGeneration = 0;
                if (wasIoContinuation)
                    await RejectIoGrabStartAsync(
                        ioController,
                        ioGeneration,
                        "background-capture-failed");
                return;
            }

            if (_autoStartGrabAfterBg)
            {
                int ioGeneration = _autoStartGrabIoGeneration;
                int ioRequestGeneration = _autoStartGrabIoRequestGeneration;
                var ioController = CurrentIoController;
                _autoStartGrabAfterBg = false;
                _autoStartGrabIoGeneration = 0;
                _autoStartGrabIoRequestGeneration = 0;

                await _ioGrabTransitionGate.WaitAsync();
                try
                {
                    if (!IsCurrentIoGrabRequest(
                        ioController,
                        ioGeneration,
                        ioRequestGeneration))
                    {
                        await RejectIoGrabStartAsync(
                            ioController,
                            ioGeneration,
                            "background-continuation-cancelled");
                        return;
                    }
                    await _liveCameraManager.ReleaseAsync();
                    bool started = await ToggleLiveGrabAsync(
                        "io:背景取得完成 → 開始抓取",
                        ioControlled: true,
                        captureStartStillValid: () => IsCurrentIoGrabRequest(
                            ioController,
                            ioGeneration,
                            ioRequestGeneration));
                    if (started && IsCurrentIoGrabRequest(
                        ioController,
                        ioGeneration,
                        ioRequestGeneration))
                    {
                        await ioController.NotifyGrabStarted();
                        FlowTrace.Log("IO grab accepted busy=on source=background");
                    }
                    else
                    {
                        await RejectIoGrabStartAsync(
                            ioController,
                            ioGeneration,
                            "background-continuation-failed");
                    }
                }
                finally
                {
                    _ioGrabTransitionGate.Release();
                }
                return;
            }

            // 採集完成後直接預覽（先清除舊預覽，確保每次都重新開啟）
            if (IsBgPreviewActive) ClearBackgroundPreview();
            btnLiveViewBackground_Click(btnLiveViewBackground, EventArgs.Empty);
        }

        /// <summary>去背演算法 setting 變更 → 重載背景 bin + 更新 StandardBgSub 鎖定狀態。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入。）</summary>
        private void HandleAlgorithmSettingsChanged(string name)
        {
            if (name == "db_Algorithm" || name == nameof(InspectionRecipe.Algorithm) || name == "去背演算法")
            {
                if (_liveCameraManager.IsAllocated) LoadBackgroundBins();
                UpdateStandardBgSubLockState();
            }
        }

        /// 從 BackgroundPath 載入各相機的 bg bin，驗證後原子替換相機持有的 pinned 背景。
        /// </summary>
        private void LoadBackgroundBins()
        {
            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub 模式：每幀自行計算背景。
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    cam.ClearPrecomputedColumnMean();
                    _outputHealthService?.Resolve(
                        "BackgroundLoad.cam" + cam.CameraId);
                    FlowTrace.Log(
                        $"background bind cam{cam.CameraId} mode=single " +
                        "source=per-frame status=ready");
                }
                return;
            }

            string bgDir = _settings.Storage.BackgroundPath;
            var backgroundRepository = new BackgroundProfileRepository(bgDir);
            if (!backgroundRepository.DirectoryExists)
            {
                ReportBackgroundLoadFailure(
                    "directory-missing", Path.GetFileName(bgDir));
                return;
            }

            BackgroundManifestSnapshot manifest =
                backgroundRepository.ReadManifest();
            if (manifest.Status == BackgroundManifestStatus.Invalid)
            {
                _outputHealthService?.Report(
                    "BackgroundManifestInvalid",
                    OutputHealthSeverity.OutputFault,
                    "背景啟用清單損壞，未切換背景");
                ReportBackgroundLoadFailure(
                    "manifest-invalid", CaptureFileNaming.BgActiveManifest);
                return;
            }
            _outputHealthService?.Resolve("BackgroundManifestInvalid");

            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (!cam.IsConnected || cam.FrameWidth <= 0)
                {
                    cam.ClearPrecomputedColumnMean();
                    _outputHealthService?.Resolve(
                        "BackgroundLoad.cam" + cam.CameraId);
                    FlowTrace.Log(
                        $"background bind cam{cam.CameraId} mode=standard " +
                        "source=none status=skipped reason=offline");
                    continue;
                }

                string binPath = backgroundRepository.ResolveCameraProfilePath(
                    cam.FrameWidth, cam.CameraId);
                float[] colMean = backgroundRepository.LoadProfile(binPath);
                if (TryDescribeBackground(
                    colMean, cam.FrameWidth,
                    out float minimum, out float maximum, out double mean))
                {
                    IntPtr pinned = NativeMethods.TanukiCv_AllocPinned((ulong)(cam.FrameWidth * sizeof(float)));
                    if (pinned != IntPtr.Zero)
                    {
                        Marshal.Copy(colMean, 0, pinned, colMean.Length);
                        cam.ReplacePrecomputedColumnMean(pinned);
                        _outputHealthService?.Resolve(
                            "BackgroundLoad.cam" + cam.CameraId);
                        FlowTrace.Log(
                            $"background bind cam{cam.CameraId} mode=standard " +
                            $"source={Path.GetFileName(binPath)} status=ready " +
                            $"width={cam.FrameWidth} samples={colMean.Length} " +
                            $"min={minimum.ToString("0.###", CultureInfo.InvariantCulture)} " +
                            $"max={maximum.ToString("0.###", CultureInfo.InvariantCulture)} " +
                            $"mean={mean.ToString("0.###", CultureInfo.InvariantCulture)}");
                        continue;
                    }

                    ReportBackgroundLoadFailure(
                        cam, "alloc-failed", Path.GetFileName(binPath));
                }
                else
                {
                    ReportBackgroundLoadFailure(
                        cam, "invalid-bin", Path.GetFileName(binPath));
                }
            }

            UpdateViewBackgroundButtonText();
        }

        private static bool TryDescribeBackground(
            float[] values,
            int expectedLength,
            out float minimum,
            out float maximum,
            out double mean)
        {
            minimum = 0f;
            maximum = 0f;
            mean = 0.0;
            if (values == null || values.Length != expectedLength || values.Length == 0)
                return false;

            minimum = values[0];
            maximum = values[0];
            double sum = 0.0;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (float.IsNaN(value) || float.IsInfinity(value))
                    return false;
                if (value < minimum) minimum = value;
                if (value > maximum) maximum = value;
                sum += value;
            }
            mean = sum / values.Length;
            return true;
        }

        private void ReportBackgroundLoadFailure(
            string reason,
            string source)
        {
            foreach (var cam in _liveCameraManager.Cameras)
                ReportBackgroundLoadFailure(cam, reason, source);
        }

        private void ReportBackgroundLoadFailure(
            AniloxCamera cam,
            string reason,
            string source)
        {
            bool retained = cam.HasPrecomputedColumnMean;
            _outputHealthService?.Report(
                "BackgroundLoad.cam" + cam.CameraId,
                OutputHealthSeverity.OutputFault,
                $"CAM{cam.CameraId} 標準背景載入失敗 ({reason})");
            FlowTrace.Log(
                $"background bind cam{cam.CameraId} mode=standard " +
                $"source={(string.IsNullOrWhiteSpace(source) ? "none" : source)} " +
                $"status=failed reason={reason} retained={retained}");
        }

        private void UpdateViewBackgroundButtonText()
        {
            // lblBgBinInfo 已刪除（2026-06-12 使用者刪除清單）；保留空方法給既有呼叫點，待 #13 收尾一併清。
        }

        /// <summary>釋放所有相機持有的預算背景 pinned buffer。</summary>
        private void FreePrecomputedColMeanBuffers()
        {
            if (_liveCameraManager == null) return;
            foreach (var cam in _liveCameraManager.Cameras)
                cam.ClearPrecomputedColumnMean();
        }

        /// <summary>
        /// StandardBgSub 時檢查是否有 bin → 控制按鈕鎖定狀態。
        /// </summary>
        private void UpdateStandardBgSubLockState()
        {
            // 相機未就緒（CLProtocol/buffer 還沒配好）→ 一律不解鎖 btnLiveGrab：此方法原本會繞過
            // RefreshGrabButtonState 的 camReady gate 直接 Enabled=true → 使用者可在「沒配置好」時點 grab → stall。
            // 故所有 enable 都 AND camReady（與 RefreshGrabButtonState 一致）。
            bool camReady = _liveCameraManager?.AreCamerasHwReady ?? false;

            // 背景鈕不歸 IO 管（借 grab 取樣：光源+相機就緒、非抓取中即可）——原本放在 IO early-return
            // 之後 → IO 開機即連線的機台每 tick 提前返回、開機鎖死到第一次 grab 才被 UpdateGrabButton 解
            // （2026-07-09 使用者回報）。
            bool lightReady = IsLightReadyForBg;
            bool isGrabbing =
                _liveCameraManager?.IsLiveGrabbing ?? false;
            bool backgroundCaptureReady =
                lightReady && camReady && !isGrabbing;
            btnLiveGetBackground.Enabled = backgroundCaptureReady;
            if (_lastFlowBackgroundCaptureReady != backgroundCaptureReady)
            {
                FlowTrace.Log(
                    $"background capture ready={backgroundCaptureReady} " +
                    $"camReady={camReady} lightReady={lightReady} " +
                    $"grabbing={isGrabbing}");
                _lastFlowBackgroundCaptureReady = backgroundCaptureReady;
            }

            // IO 已連線且未暫停：btnLiveGrab 由 IO 連線邏輯控制，不覆寫
            if (CurrentIoController?.IsConnected == true && !_isIoSuspended) return;
            // IO 暫停模式：交由使用者手動控制，不受 StandardBgSub bin 限制
            if (_isIoSuspended) { btnLiveGrab.Enabled = camReady; return; }

            if (!IsStandardBgSubEnabled)
            {
                btnLiveGrab.Enabled = camReady;
                return;
            }

            btnLiveGrab.Enabled = camReady && IsBgBinReady();
        }

        // --- 背景預覽狀態 ---
        /// <summary>預覽狀態唯讀轉發（唯一真相在 LiveDisplayCoordinator 靜音鍵，form 不自存＝不會分歧）。</summary>
        private bool IsBgPreviewActive => _liveCameraManager?.IsBgPreviewActive ?? false;

        /// <summary>
        /// 預覽背景（顯示鐵則0：主畫面＝7 台背景合圖，走 grab 同一個 ImageDisplayView 共用路）：
        /// 讀各相機 bg bin → 擴成灰階 bytes → PushStaticFrame 餵共用顯示（合圖/縮圖/縮放/overlay 全免費）。
        /// 再按一次＝清除預覽。瀑布模式下暫用即時 view（設定不動，離開預覽即還原）。
        /// </summary>
        private void btnLiveViewBackground_Click(object sender, EventArgs e)
        {
            FlowTrace.Log("ui:【預覽背景】鈕");   // intent 行（孤兒判讀規則）
            if (IsBgPreviewActive) { ClearBackgroundPreview(); return; }

            string bgDir = _settings.Storage.BackgroundPath;
            var backgroundRepository = new BackgroundProfileRepository(bgDir);
            if (!backgroundRepository.DirectoryExists)
            {
                MessageBox.Show("背景目錄不存在。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            int[] grabHeights = _settings.Acquisition.CameraGrabHeight;
            _liveCameraManager.EnterBackgroundPreview();
            ClearLiveRowChartForBackgroundPreview();
            int pushed = 0;
            for (int i = 0; i < CameraCount; i++)
            {
                int camId = i + 1;
                string binPath =
                    backgroundRepository.ResolvePreviewProfilePath(camId);
                if (binPath == null) continue;
                float[] colMean = backgroundRepository.LoadProfile(binPath);
                if (colMean == null || colMean.Length == 0) continue;
                int height = (i < grabHeights.Length && grabHeights[i] > 0) ? grabHeights[i] : 3000;
                _liveCameraManager.PushStaticFrame(camId,
                    ExpandColMeanToGray(colMean, colMean.Length, height), colMean.Length, height);
                pushed++;
            }
            if (pushed == 0)
            {
                _liveCameraManager.ExitBackgroundPreview();
                MessageBox.Show("未找到背景 bin 檔。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }
        }

        /// <summary>清除背景預覽：清共用顯示的幀 + 回設定模式（coordinator 負責）——不再自建/銷毀 canvas、
        /// 不 Free 相機（舊「預覽後 grab 重配」路徑已退場）。</summary>
        private void ClearBackgroundPreview()
        {
            _liveCameraManager?.ExitBackgroundPreview();
        }

        private bool IsStandardBgSubEnabled =>
            _settings?.Recipe?.Algorithm == BackgroundAlgorithm.StandardBgSub;

        private bool IsLightReadyForBg =>
            !(_settings?.LightEnabled == true) ||
            (_lightConnectionCoordinator?.Snapshot.Connected == true);

        private bool _autoStartGrabAfterBg;
        private int _autoStartGrabIoGeneration;
        private int _autoStartGrabIoRequestGeneration;

        private bool IsBgBinReady()
        {
            if (!IsStandardBgSubEnabled) return true;
            string bgDir = _settings.Storage.BackgroundPath;
            var backgroundRepository = new BackgroundProfileRepository(bgDir);
            if (_liveCameraManager?.IsAllocated == true)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (!cam.IsConnected) continue;
                    if (cam.FrameWidth <= 0) continue;
                    if (!cam.HasPrecomputedColumnMean) return false;
                }
                return true;
            }
            if (!backgroundRepository.DirectoryExists) return false;
            BackgroundManifestSnapshot manifest =
                backgroundRepository.ReadManifest();
            if (manifest.Status == BackgroundManifestStatus.Invalid)
            {
                _outputHealthService?.Report(
                    "BackgroundManifestInvalid",
                    OutputHealthSeverity.OutputFault,
                    "背景啟用清單損壞，無法確認有效背景");
                return false;
            }
            _outputHealthService?.Resolve("BackgroundManifestInvalid");
            return backgroundRepository.HasAnyProfile();
        }

        private void CleanupInactiveBackgroundVersions()
        {
            string bgDir = _settings?.Storage?.BackgroundPath;
            new BackgroundProfileRepository(bgDir).CleanupInactiveVersions();
        }
    }
}
