using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using LightBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
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

            // 確保相機已 allocate
            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    await _liveCameraManager.EnsureAllocatedAndToggleGrabAsync(false);
                    if (!_liveCameraManager.IsAllocated) return;
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
            }

            // 確保 grab 中，先開燈等穩定再開始
            if (!_liveCameraManager.IsLiveGrabbing)
            {
                LightTurnOn();
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
                _liveCameraManager.ToggleGrab();
                UpdateGrabButton(true);
            }

            btnLiveGetBackground.Enabled = false;
            btnLiveGrab.Enabled = false;

            int sampleSeconds = Math.Max(1, _settings.Recipe.BackgroundSampleSeconds);
            string bgDir = _settings.Storage.BackgroundPath;
            string version = DateTime.Now.ToString("yyyyMMdd-HHmmssfff");
            var newVersionFiles = new List<string>();
            bool captureSucceeded = false;

            try
            {
                if (!Directory.Exists(bgDir))
                    Directory.CreateDirectory(bgDir);

                var cameras = _liveCameraManager.Cameras;
                int camCount = cameras.Count;

                double[][] accum = new double[camCount][];
                int[] frameCount = new int[camCount];

                // 採集 sampleSeconds 秒，按鈕顯示倒數
                var sw = Stopwatch.StartNew();
                int lastShown = -1;
                while (sw.Elapsed.TotalSeconds < sampleSeconds)
                {
                    int remaining = sampleSeconds - (int)sw.Elapsed.TotalSeconds;
                    if (remaining != lastShown)
                    {
                        lastShown = remaining;
                        btnLiveGetBackground.Text = $"採集中 {remaining}s";
                    }

                    await Task.Delay(100);

                    for (int i = 0; i < camCount; i++)
                    {
                        var cam = cameras[i];
                        if (!cam.IsConnected || cam.FrameWidth <= 0) continue;

                        if (accum[i] == null)
                            accum[i] = new double[cam.FrameWidth];

                        float[] colMean = new float[cam.FrameWidth];
                        if (cam.TryComputeColumnMean(colMean))
                        {
                            for (int c = 0; c < cam.FrameWidth; c++)
                                accum[i][c] += colMean[c];
                            frameCount[i]++;
                        }
                    }
                }

                // 平均並存檔
                for (int i = 0; i < camCount; i++)
                {
                    var cam = cameras[i];
                    if (!cam.IsConnected || cam.FrameWidth <= 0) continue;
                    if (frameCount[i] == 0 || accum[i] == null)
                        throw new IOException($"CAM{cam.CameraId} 沒有取得有效背景樣本");

                    float[] avgColMean = new float[cam.FrameWidth];
                    double invN = 1.0 / frameCount[i];
                    for (int c = 0; c < cam.FrameWidth; c++)
                        avgColMean[c] = (float)(accum[i][c] * invN);

                    string binPath = Path.Combine(
                        bgDir,
                        CaptureFileNaming.BgVersionedBin(
                            cam.FrameWidth, cam.CameraId, version));
                    SaveBackgroundBin(avgColMean, binPath, _settings.LightBrightness, (float)cam.CameraExposureTimeUs); // LightBrightness = light controller level (0-255)
                    newVersionFiles.Add(binPath);
                }

                if (newVersionFiles.Count == 0)
                    throw new IOException("沒有任何相機產生背景檔");

                ActivateBackgroundVersion(bgDir, version);
                // 載入到各相機
                LoadBackgroundBins();
                captureSucceeded = true;
                _outputHealthService?.Resolve("BackgroundCaptureFailure");
            }
            catch (Exception ex)
            {
                foreach (string path in newVersionFiles)
                {
                    try { if (File.Exists(path)) File.Delete(path); } catch { }
                }
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
                    _liveCameraManager.ToggleGrab();
                    LightTurnOff();
                    UpdateGrabButton(false);
                }

                UpdateStandardBgSubLockState();
            }

            if (!captureSucceeded)
            {
                _autoStartGrabAfterBg = false;
                return;
            }

            if (_autoStartGrabAfterBg)
            {
                _autoStartGrabAfterBg = false;
                await _liveCameraManager.ReleaseAsync();
                btnLiveGrab_Click(null, null);
                _ = _ioGrabController?.NotifyGrabStarted();
                return;
            }

            // 採集完成後直接預覽（先清除舊預覽，確保每次都重新開啟）
            if (IsBgPreviewActive) ClearBackgroundPreview();
            btnLiveViewBackground_Click(btnLiveViewBackground, EventArgs.Empty);
        }

        /// <summary>MCBF v2 格式存 background column mean（含光源等級與曝光時間）。</summary>
        private static void SaveBackgroundBin(float[] data, string path, int lightLevel, float exposureUs)
        {
            using (var stream = new FileStream(
                path, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                FileOptions.WriteThrough))
            using (var bw = new BinaryWriter(stream))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(2);                    // version 2
                bw.Write(1.0f);                 // scale_factor (1 = 全解析度)
                bw.Write(lightLevel);           // light controller level (0-255)
                bw.Write(exposureUs);           // camera exposure (µs)
                bw.Write(data.Length);          // array_length
                foreach (float v in data) bw.Write(v);
                bw.Flush();
                stream.Flush(true);
            }
        }

        private static void ActivateBackgroundVersion(string bgDir, string version)
        {
            string manifest = Path.Combine(bgDir, CaptureFileNaming.BgActiveManifest);
            string temp = manifest + ".tmp-" + Guid.NewGuid().ToString("N");
            try
            {
                byte[] bytes = System.Text.Encoding.UTF8.GetBytes(
                    "{\r\n  \"Version\": \"" + version + "\"\r\n}");
                using (var stream = new FileStream(
                    temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 4096,
                    FileOptions.WriteThrough))
                {
                    stream.Write(bytes, 0, bytes.Length);
                    stream.Flush(true);
                }
                if (File.Exists(manifest))
                    File.Replace(temp, manifest, null, true);
                else
                    File.Move(temp, manifest);
            }
            finally
            {
                try { if (File.Exists(temp)) File.Delete(temp); } catch { }
            }
        }

        private static string ReadActiveBackgroundVersion(string bgDir)
        {
            try
            {
                string manifest = Path.Combine(bgDir, CaptureFileNaming.BgActiveManifest);
                if (!File.Exists(manifest)) return null;
                string json = File.ReadAllText(manifest, System.Text.Encoding.UTF8);
                return SettingsStoreHelper.GetString(json, "Version", null);
            }
            catch
            {
                return null;
            }
        }

        private static bool HasActiveBackgroundManifest(string bgDir)
        {
            return !string.IsNullOrWhiteSpace(bgDir) &&
                   File.Exists(Path.Combine(bgDir, CaptureFileNaming.BgActiveManifest));
        }

        private static string ResolveBackgroundBinPath(
            string bgDir, int width, int camId)
        {
            string version = ReadActiveBackgroundVersion(bgDir);
            if (!string.IsNullOrWhiteSpace(version))
            {
                return Path.Combine(
                    bgDir, CaptureFileNaming.BgVersionedBin(width, camId, version));
            }
            if (HasActiveBackgroundManifest(bgDir))
                return Path.Combine(bgDir, "__invalid-active-background__.bin");
            return Path.Combine(bgDir, CaptureFileNaming.BgBin(width, camId));
        }

        private static string ResolveBackgroundBinPathForPreview(
            string bgDir, int camId)
        {
            string version = ReadActiveBackgroundVersion(bgDir);
            if (string.IsNullOrWhiteSpace(version) && HasActiveBackgroundManifest(bgDir))
                return null;
            string pattern = string.IsNullOrWhiteSpace(version)
                ? CaptureFileNaming.BgGlobForCam(camId)
                : CaptureFileNaming.BgVersionedGlobForCam(camId, version);
            string[] matches = Directory.GetFiles(bgDir, pattern);
            return matches.Length == 0 ? null : matches[0];
        }

        /// <summary>
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

        /// 從 BackgroundPath 載入各相機的 bg bin → pinned buffer → 設定到 AniloxCamera.PrecomputedColMean。
        /// </summary>
        private void LoadBackgroundBins()
        {
            if (!IsStandardBgSubEnabled)
            {
                // 非 StandardBgSub 模式：清除所有預算背景
                foreach (var cam in _liveCameraManager.Cameras)
                    cam.PrecomputedColMean = IntPtr.Zero;
                return;
            }

            string bgDir = _settings.Storage.BackgroundPath;
            if (!Directory.Exists(bgDir)) return;

            if (HasActiveBackgroundManifest(bgDir) &&
                string.IsNullOrWhiteSpace(ReadActiveBackgroundVersion(bgDir)))
            {
                _outputHealthService?.Report(
                    "BackgroundManifestInvalid",
                    OutputHealthSeverity.OutputFault,
                    "背景啟用清單損壞，未切換背景");
                return;
            }
            _outputHealthService?.Resolve("BackgroundManifestInvalid");

            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.FrameWidth <= 0) continue;

                string binPath = ResolveBackgroundBinPath(
                    bgDir, cam.FrameWidth, cam.CameraId);
                float[] colMean = CurveBinFile.Load(binPath);
                if (colMean != null && colMean.Length == cam.FrameWidth)
                {
                    // 分配 pinned memory 並複製
                    IntPtr pinned = NativeMethods.TanukiCv_AllocPinned((ulong)(cam.FrameWidth * sizeof(float)));
                    if (pinned != IntPtr.Zero)
                    {
                        Marshal.Copy(colMean, 0, pinned, colMean.Length);

                        // 釋放舊的（如果有）
                        if (cam.PrecomputedColMean != IntPtr.Zero)
                            NativeMethods.TanukiCv_FreePinned(cam.PrecomputedColMean);

                        cam.PrecomputedColMean = pinned;
                    }
                }
                else if (cam.PrecomputedColMean != IntPtr.Zero)
                {
                    NativeMethods.TanukiCv_FreePinned(cam.PrecomputedColMean);
                    cam.PrecomputedColMean = IntPtr.Zero;
                }
            }

            UpdateViewBackgroundButtonText();
        }

        private void UpdateViewBackgroundButtonText()
        {
            // lblBgBinInfo 已刪除（2026-06-12 使用者刪除清單）；保留空方法給既有呼叫點，待 #13 收尾一併清。
        }

        /// <summary>釋放所有相機的 PrecomputedColMean pinned buffer。</summary>
        private void FreePrecomputedColMeanBuffers()
        {
            if (_liveCameraManager == null) return;
            foreach (var cam in _liveCameraManager.Cameras)
            {
                if (cam.PrecomputedColMean != IntPtr.Zero)
                {
                    NativeMethods.TanukiCv_FreePinned(cam.PrecomputedColMean);
                    cam.PrecomputedColMean = IntPtr.Zero;
                }
            }
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
            btnLiveGetBackground.Enabled = IsLightReadyForBg && camReady
                && !(_liveCameraManager?.IsLiveGrabbing ?? false);

            // IO 已連線且未暫停：btnLiveGrab 由 IO 連線邏輯控制，不覆寫
            if (_ioGrabController?.IsConnected == true && !_isIoSuspended) return;
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
            if (!Directory.Exists(bgDir))
            {
                MessageBox.Show("背景目錄不存在。", "提示", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return;
            }

            int[] grabHeights = _settings.Acquisition.CameraGrabHeight;
            _liveCameraManager.EnterBackgroundPreview();
            int pushed = 0;
            for (int i = 0; i < CameraCount; i++)
            {
                int camId = i + 1;
                string binPath = ResolveBackgroundBinPathForPreview(bgDir, camId);
                if (binPath == null) continue;
                float[] colMean = CurveBinFile.Load(binPath);
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
            !(_settings?.LightEnabled == true) || (_lightController != null && _lightController.IsConnected);

        private bool _autoStartGrabAfterBg;

        private bool IsBgBinReady()
        {
            if (!IsStandardBgSubEnabled) return true;
            string bgDir = _settings.Storage.BackgroundPath;
            if (_liveCameraManager?.IsAllocated == true)
            {
                foreach (var cam in _liveCameraManager.Cameras)
                {
                    if (!cam.IsConnected) continue;
                    if (cam.FrameWidth <= 0) continue;
                    string binPath = ResolveBackgroundBinPath(
                        bgDir, cam.FrameWidth, cam.CameraId);
                    if (!File.Exists(binPath)) return false;
                }
                return true;
            }
            if (!Directory.Exists(bgDir)) return false;
            string activeVersion = ReadActiveBackgroundVersion(bgDir);
            if (string.IsNullOrWhiteSpace(activeVersion) &&
                HasActiveBackgroundManifest(bgDir))
            {
                _outputHealthService?.Report(
                    "BackgroundManifestInvalid",
                    OutputHealthSeverity.OutputFault,
                    "背景啟用清單損壞，無法確認有效背景");
                return false;
            }
            _outputHealthService?.Resolve("BackgroundManifestInvalid");
            return !string.IsNullOrWhiteSpace(activeVersion)
                ? Directory.GetFiles(
                    bgDir, CaptureFileNaming.BgVersionedGlobForCam(1, activeVersion)).Length > 0
                : Directory.GetFiles(bgDir, CaptureFileNaming.BgGlob).Length > 0;
        }

        private void CleanupInactiveBackgroundVersions()
        {
            string bgDir = _settings?.Storage?.BackgroundPath;
            if (string.IsNullOrWhiteSpace(bgDir) || !Directory.Exists(bgDir)) return;

            string activeVersion = ReadActiveBackgroundVersion(bgDir);
            if (string.IsNullOrWhiteSpace(activeVersion)) return;

            foreach (string path in Directory.GetFiles(bgDir, CaptureFileNaming.BgGlob))
            {
                string name = Path.GetFileNameWithoutExtension(path);
                if (name.EndsWith("_" + activeVersion, StringComparison.OrdinalIgnoreCase))
                    continue;
                try { File.Delete(path); }
                catch (Exception ex)
                {
                    Trace.TraceWarning(
                        $"[BackgroundRetention] delete failed {path}: {ex.Message}");
                }
            }
        }
    }
}
