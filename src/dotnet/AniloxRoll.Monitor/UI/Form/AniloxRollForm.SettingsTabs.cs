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
using TanukiCv.Core; // SystemInfo（CPU/GPU/RAM/螢幕 唯一來源）
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
    /// <summary>AniloxRollForm 右側設定面板 tab 建構（相機參數 / 系統）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        private bool _cameraParameterControlsReady;

        // ==========================================
        // --- 右側面板：初始化 ---
        // ==========================================

        private void InitializeRightPanelControls()
        {
            SetupCameraTab();
            SetupSystemTab();
        }

        /// <summary>
        /// 用 reflection 調整 PropertyGrid 標籤欄寬度至最長屬性名稱剛好容納。
        /// PropertyGrid 無公開 API 可設欄寬，透過內部 gridView.MoveSplitterTo() 實現。
        /// 注意：MoveSplitterTo 的參數是整個標籤欄寬（含左側 indent 區域），
        /// 因此需在純文字寬度之外加上 indent（約 16px）＋ 右側留白。
        /// </summary>
        private static void AutoFitPropertyGridLabelColumn(System.Windows.Forms.PropertyGrid grid)
        {
            try
            {
                var gridViewField = grid.GetType().GetField("gridView",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var gridView = gridViewField?.GetValue(grid);
                if (gridView == null) return;

                // 以所有可見屬性的 DisplayName 量測最大文字寬度
                int maxTextWidth = 0;
                foreach (System.ComponentModel.PropertyDescriptor pd in
                    System.ComponentModel.TypeDescriptor.GetProperties(grid.SelectedObject))
                {
                    if (!pd.IsBrowsable) continue;
                    string label = pd.DisplayName ?? pd.Name;
                    int w = System.Windows.Forms.TextRenderer.MeasureText(
                        label, grid.Font).Width;
                    if (w > maxTextWidth) maxTextWidth = w;
                }

                // indent：PropertyGrid 標籤欄左側的展開箭頭區域固定約 16px
                // rightMargin：文字右側留白，避免緊貼分隔線
                const int indent      = 16;
                const int rightMargin = 8;
                var moveSplitter = gridView.GetType().GetMethod("MoveSplitterTo",
                    System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                moveSplitter?.Invoke(gridView, new object[] { indent + maxTextWidth + rightMargin });
            }
            catch { /* reflection 失敗時保留預設欄寬，不影響功能 */ }
        }

        private void SetupCameraTab()
        {
            _cameraParameterControlsReady = false;
            const int ExpMin    =     1;   // μs
            const int ExpMaxCap = 10000;   // μs 硬上限
            const int LrMin     =   100;   // Hz
            const int LrMax     = 10000;   // Hz
            const int HtMin     =   100;   // px
            // grab 高度硬上限：grab 中拉大超過 ~12062 會 stall → 一律 cap 12000（per-camera 固定，不分台數；
            // 板載是 4 path 各自獨立到 ~12062。來源見 AcquisitionDefaults.MaxGrabHeightPx）。
            int HtMax = AcquisitionDefaults.MaxGrabHeightPx;

            // ── 7 台相機控制項陣列（存為 Form 欄位，供 SyncFromCamera 存取）────
            var acq = _settings.Acquisition;
            _expBars = new[] { trackBarExpCam1, trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7 };
            _expNums = new[] { numExpCam1,      numExpCam2,      numExpCam3,      numExpCam4,      numExpCam5,      numExpCam6,      numExpCam7      };
            _lrBars  = new[] { trackBarLrCam1,  trackBarLrCam2,  trackBarLrCam3,  trackBarLrCam4,  trackBarLrCam5,  trackBarLrCam6,  trackBarLrCam7  };
            _lrNums  = new[] { numLrCam1,       numLrCam2,       numLrCam3,       numLrCam4,       numLrCam5,       numLrCam6,       numLrCam7       };
            _htBars  = new[] { trackBarHtCam1,  trackBarHtCam2,  trackBarHtCam3,  trackBarHtCam4,  trackBarHtCam5,  trackBarHtCam6,  trackBarHtCam7  };
            _htNums  = new[] { numHtCam1,       numHtCam2,       numHtCam3,       numHtCam4,       numHtCam5,       numHtCam6,       numHtCam7       };

            // ── CAM All 控制項事件綁定（控制項已在 Designer.cs 定義）──────────
            _expAllBar = trackBarExpAll; _expAllNum = numExpAll;
            _lrAllBar  = trackBarLrAll;  _lrAllNum  = numLrAll;
            _htAllBar  = trackBarHtAll;  _htAllNum  = numHtAll;

            BindAllSync(_expAllBar, _expAllNum, _expBars, _expNums, "ExpAll",
                (j, v) => _liveCameraManager?.SetExposureForCamera(j + 1, v),
                (j, v) => { acq.CameraExposureTimeUs[j] = v; ConfigManager.SaveAcquisitionSettings(acq); });

            BindAllSync(_lrAllBar, _lrAllNum, _lrBars, _lrNums, "LineRateAll",
                (j, v) => _liveCameraManager?.SetLineRateForCamera(j + 1, v),
                (j, v) => { acq.CameraLineRateHz[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    // 同步所有 cam 的 exp max（每台 LR 都同值，算一次套到所有 cam）
                    int newMax = (int)acq.CameraLineRateHz[0];
                    int expMax = MilCameraParams.CalcExposureMaxUs(newMax, ExpMin, ExpMaxCap);
                    for (int i = 0; i < CameraCount; i++) UpdateExpMaxAndClampColor(i, expMax);
                    UpdateRowChartPitch();
                });

            BindAllSync(_htAllBar, _htAllNum, _htBars, _htNums, "HeightAll",
                (j, v) => _liveCameraManager?.SetGrabHeightForCamera(j + 1, v),
                (j, v) => { acq.CameraGrabHeight[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsGlobalMergeActive == true)
                        _liveCameraManager.RefreshGlobalMergeLayout(_settings.Ops.ToArray(), _settings.StartPosition.ToArray());
                });

            for (int i = 0; i < CameraCount; i++)
            {
                int idx   = i;
                int camId = i + 1;

                // 動態曝光上限（依各台自己的 LR）
                int CalcExpMax()
                {
                    int lrHz = (int)acq.CameraLineRateHz[idx];
                    return MilCameraParams.CalcExposureMaxUs(lrHz, ExpMin, ExpMaxCap);
                }

                // ── 曝光時間 ────────────────────────────────────────────
                int expMax = CalcExpMax();
                BindBidirectionalSync(_expBars[idx], _expNums[idx], camId,
                    ExpMin, expMax, (int)acq.CameraExposureTimeUs[idx],
                    v => { acq.CameraExposureTimeUs[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => ApplyCamParamAsync(camId, "Exp", v, () => _liveCameraManager.SetExposureForCamera(camId, v)),
                    debounceMs: 200);

                // ── 線掃速率 ────────────────────────────────────────────
                BindBidirectionalSync(_lrBars[idx], _lrNums[idx], camId,
                    LrMin, LrMax, (int)acq.CameraLineRateHz[idx],
                    v => { acq.CameraLineRateHz[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => ApplyCamParamAsync(camId, "LineRate", v, () => _liveCameraManager.SetLineRateForCamera(camId, v)),
                    () => { UpdateExpMaxAndClampColor(idx, CalcExpMax()); if (idx == 0) UpdateRowChartPitch(); });

                // ── 擷取高度 ────────────────────────────────────────────
                BindBidirectionalSync(_htBars[idx], _htNums[idx], camId,
                    HtMin, HtMax, Math.Max(HtMin, Math.Min(HtMax, acq.CameraGrabHeight[idx])),
                    v => { acq.CameraGrabHeight[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => ApplyCamParamAsync(camId, "Height", v, () => _liveCameraManager.SetGrabHeightForCamera(camId, v)),
                    () => {
                        if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsGlobalMergeActive == true)
                            _liveCameraManager.RefreshGlobalMergeLayout(_settings.Ops.ToArray(), _settings.StartPosition.ToArray());
                    });
                _htBars[idx].SmallChange = 64; _htBars[idx].LargeChange = 512;
            }

            // ── CAM All 範圍設定 ──────────────────────────────────────────
            int expAllMax = ExpMaxCap;
            for (int i = 0; i < CameraCount; i++)
            {
                int lrHz = (int)acq.CameraLineRateHz[i];
                int m = MilCameraParams.CalcExposureMaxUs(lrHz, ExpMin, ExpMaxCap);
                if (m < expAllMax) expAllMax = m;
            }
            _expAllBar.Minimum = ExpMin; _expAllBar.Maximum = expAllMax;
            _expAllNum.Minimum = ExpMin; _expAllNum.Maximum = expAllMax;
            _expAllBar.Value = Math.Max(ExpMin, Math.Min(expAllMax, (int)acq.CameraExposureTimeUs[0]));
            _expAllNum.Value = _expAllBar.Value;

            _lrAllBar.Minimum = LrMin; _lrAllBar.Maximum = LrMax;
            _lrAllNum.Minimum = LrMin; _lrAllNum.Maximum = LrMax;
            _lrAllBar.Value = Math.Max(LrMin, Math.Min(LrMax, (int)acq.CameraLineRateHz[0]));
            _lrAllNum.Value = _lrAllBar.Value;

            _htAllBar.Minimum = HtMin; _htAllBar.Maximum = HtMax;
            _htAllNum.Minimum = HtMin; _htAllNum.Maximum = HtMax;
            _htAllBar.Value = Math.Max(HtMin, Math.Min(HtMax, acq.CameraGrabHeight[0]));
            _htAllNum.Value = _htAllBar.Value;
            _htAllBar.SmallChange = 64; _htAllBar.LargeChange = 512;

            // 滾輪每格移動 1（攔截原生 3 格行為）
            RegisterWheelInterceptors(_expBars);
            RegisterWheelInterceptors(_lrBars);
            RegisterWheelInterceptors(_htBars);
            RegisterWheelInterceptors(new[] { _expAllBar, _lrAllBar, _htAllBar });
            _cameraParameterControlsReady = true;
            RefreshCameraParameterControlState();
        }

        /// <summary>
        /// TrackBar ↔ NumericUpDown 雙向同步綁定：
        /// - 拖曳中：抑制硬體寫入，MouseUp 立即寫入。
        /// - 滾輪 / 鍵盤箭頭 / NUD 輸入：1 秒 debounce 才寫硬體（避免高頻 MIL 寫入造成卡頓）。
        /// </summary>
        private void BindBidirectionalSync(
            TrackBar bar, NumericUpDown num, int camId,
            int min, int max, int initialValue,
            Action<int> saveSetting, Func<int, Task> writeHardwareAsync,
            Action postAction = null,
            int debounceMs = 1000)
        {
            int clamped = Math.Max(min, Math.Min(max, initialValue));
            bar.Minimum = min; bar.Maximum = max; bar.TickFrequency = TickFreq;
            num.Minimum = min; num.Maximum = max;
            bar.Value = clamped; num.Value = clamped;

            bool syncing = false;

            // 滾輪/鍵盤/NUD 等非拖曳輸入 → 1s debounce 才寫硬體
            var debounce = new System.Windows.Forms.Timer { Interval = debounceMs };
            int pendingValue = clamped;
            bool hasPending = false;
            debounce.Tick += async (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                await writeHardwareAsync(pendingValue);
                postAction?.Invoke();   // 硬體寫完後再 refresh（例：HT 改變後重新載入主畫面 buffer）
            };
            void ScheduleWrite(int v)
            {
                pendingValue = v;
                hasPending = true;
                debounce.Stop();
                debounce.Start();
            }

            bar.MouseDown += (s, e) => _dragging.Add(bar);
            bar.MouseUp += async (s, e) =>
            {
                _dragging.Remove(bar);
                // 拖曳結束 → 立即寫入並取消 debounce
                debounce.Stop();
                hasPending = false;
                await writeHardwareAsync(bar.Value);
                postAction?.Invoke();   // 硬體寫完後再 refresh（例：HT 改變後重新載入主畫面 buffer）
            };
            bar.ValueChanged += (s, e) =>
            {
                if (!_cameraParameterControlsReady || syncing || _syncingFromHw) return; syncing = true;
                num.Value = bar.Value;
                saveSetting(bar.Value);
                if (!_dragging.Contains(bar)) ScheduleWrite(bar.Value);
                syncing = false;
            };
            num.ValueChanged += (s, e) =>
            {
                if (!_cameraParameterControlsReady || syncing || _syncingFromHw) return; syncing = true;
                int v = (int)num.Value;
                bar.Value = Math.Max(min, Math.Min(max, v));
                saveSetting(v);
                ScheduleWrite(v);
                syncing = false;
            };
        }

        /// <summary>
        /// CAM All → CAM1~7 同步：
        /// - 拖曳 All：MouseUp 才寫硬體；滾輪/鍵盤/NUD：1s debounce 才寫硬體。
        /// - 寫硬體完成後才同步 CAM1~7 的 bar/num 顯示（避免 UI 比硬體快）。
        /// </summary>
        private void BindAllSync(TrackBar barAll, NumericUpDown numAll,
            TrackBar[] bars, NumericUpDown[] nums,
            string paramLabel,                       // 參數名（param-change log 用）
            Action<int, int> writeHardwareForCam,    // (camIdx0based, value)
            Action<int, int> saveSettingForCam,      // (camIdx0based, value)
            Action postWriteAll = null)
        {
            bool allSyncing = false;
            var debounce = new System.Windows.Forms.Timer
            {
                Interval = IsExposureParameter(paramLabel) ? 200 : 1000
            };
            int pendingValue = barAll.Value;
            bool hasPending = false;

            async Task ApplyAsync(int v)
            {
                // 1. 寫硬體（所有 7 台）：關產品 gate → 全部 drain/write/resume → raw 新幀到齊才放行。
                await ApplyAllCamParamAsync(paramLabel, v, () =>
                {
                    for (int j = 0; j < bars.Length; j++)
                        writeHardwareForCam(j, v);
                });
                // 2. 寫 settings
                for (int j = 0; j < bars.Length; j++)
                    saveSettingForCam(j, v);
                // 3. 同步 cam 的 bar/num 顯示（用 _syncingFromHw 跳過 BindBidirectionalSync 的 ScheduleWrite/saveSetting）
                _syncingFromHw = true;
                try
                {
                    for (int j = 0; j < bars.Length; j++)
                    {
                        int clamped = Math.Max(bars[j].Minimum, Math.Min(bars[j].Maximum, v));
                        bars[j].Value = clamped;
                        nums[j].Value = clamped;
                    }
                }
                finally { _syncingFromHw = false; }
                postWriteAll?.Invoke();
            }

            debounce.Tick += async (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                await ApplyAsync(pendingValue);
            };
            void Schedule(int v)
            {
                pendingValue = v;
                hasPending = true;
                debounce.Stop();
                debounce.Start();
            }

            barAll.MouseDown += (s, e) => _dragging.Add(barAll);
            barAll.MouseUp += async (s, e) =>
            {
                _dragging.Remove(barAll);
                debounce.Stop();
                hasPending = false;
                await ApplyAsync(barAll.Value);
            };
            barAll.ValueChanged += (s, e) => {
                if (!_cameraParameterControlsReady || allSyncing || _syncingFromHw) return; allSyncing = true;
                numAll.Value = barAll.Value;
                if (!_dragging.Contains(barAll)) Schedule(barAll.Value);
                allSyncing = false;
            };
            numAll.ValueChanged += (s, e) => {
                if (!_cameraParameterControlsReady || allSyncing || _syncingFromHw) return; allSyncing = true;
                int v = (int)numAll.Value;
                barAll.Value = Math.Max(barAll.Minimum, Math.Min(barAll.Maximum, v));
                Schedule(v);
                allSyncing = false;
            };
        }

        // ==================== 相機參數鎖（套用期間真正 disable，防空拉跳值）====================
        // 套用相機參數期間把所有參數控制項 Enabled=false：
        //   ① disable 的控制項「拒絕輸入、不排隊」→ 不會解鎖後 replay 跳到空拉位置（你看到的暴力漏洞）。
        //   ② manager 先等 raw 新幀再放行；UI 只保留非阻塞 cooldown，不在 Timer 查 MIL。
        private System.Windows.Forms.Timer _paramUnlockTimer;
        private DateTime _paramLockMinReleaseUtc;
        private bool _cameraParameterOperationLocked;

        private static bool IsExposureParameter(string param)
        {
            return string.Equals(param, "Exp", StringComparison.Ordinal)
                || string.Equals(param, "ExpAll", StringComparison.Ordinal);
        }

        /// <summary>套用單一相機參數：記 log → 鎖控制項 → 非同步重配置 → cooldown 解鎖。</summary>
        private async Task ApplyCamParamAsync(int camId, string param, int value, Action write)
        {
            if (_liveCameraManager?.IsLiveGrabbing == true && !IsExposureParameter(param))
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=cam{camId} param={param} reason=GrabActive");
                RefreshCameraParameterControlState(true);
                return;
            }

            FlowTrace.Log($"ui:【相機參數】cam{camId} {param}={value}");   // intent 帶參數名+值（單行自足）
            LogParamChange("cam", camId, param, value);
            if (_liveCameraManager == null) { write?.Invoke(); return; }
            bool live = _liveCameraManager.IsLiveGrabbing;
            bool fastExposure = live && IsExposureParameter(param);
            if (live) { SetParamControlsLocked(true); _liveCameraManager.SetCaptureSuppressed(true); }
            try
            {
                bool applied = fastExposure
                    ? await _liveCameraManager.ApplyExposureFastAsync(camId, write)
                    : await _liveCameraManager.ApplyParamCoordinatedAsync(camId, write);
                if (!applied)
                    FlowTrace.Log(
                        $"parameter ui apply failed scope=cam{camId} param={param} value={value}");
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[ApplyCamParamAsync.cam{camId}] {ex.GetType().Name}: {ex.Message}");
                FlowTrace.Log(
                    $"parameter ui apply failed scope=cam{camId} param={param} " +
                    $"value={value} error={ex.GetType().Name}");
            }
            finally
            {
                if (fastExposure)
                {
                    _liveCameraManager.SetCaptureSuppressed(false);
                    SetParamControlsLocked(false);
                }
                else if (live)
                {
                    BeginParamUnlockPoll();
                }
            }
        }

        /// <summary>套用全部相機參數：記 log → 鎖控制項 → 同一代重配置 → cooldown 解鎖。</summary>
        private async Task ApplyAllCamParamAsync(string param, int value, Action write)
        {
            if (_liveCameraManager?.IsLiveGrabbing == true && !IsExposureParameter(param))
            {
                FlowTrace.Log(
                    $"parameter change blocked scope=All param={param} reason=GrabActive");
                RefreshCameraParameterControlState(true);
                return;
            }

            FlowTrace.Log($"ui:【相機參數】All {param}={value}");   // intent 帶參數名+值（單行自足；開機初始還原三連發亦經此=同值可辨識）
            LogParamChange("all", 0, param, value);
            if (_liveCameraManager == null) { write?.Invoke(); return; }
            bool live = _liveCameraManager.IsLiveGrabbing;
            bool fastExposure = live && IsExposureParameter(param);
            if (live) { SetParamControlsLocked(true); _liveCameraManager.SetCaptureSuppressed(true); }
            try
            {
                bool applied = fastExposure
                    ? await _liveCameraManager.ApplyExposureFastAsync(write)
                    : await _liveCameraManager.ApplyParamCoordinatedAsync(write);
                if (!applied)
                    FlowTrace.Log(
                        $"parameter ui apply failed scope=All param={param} value={value}");
            }
            catch (Exception ex)
            {
                Trace.TraceWarning(
                    $"[ApplyAllCamParamAsync] {ex.GetType().Name}: {ex.Message}");
                FlowTrace.Log(
                    $"parameter ui apply failed scope=All param={param} " +
                    $"value={value} error={ex.GetType().Name}");
            }
            finally
            {
                if (fastExposure)
                {
                    _liveCameraManager.SetCaptureSuppressed(false);
                    SetParamControlsLocked(false);
                }
                else if (live)
                {
                    BeginParamUnlockPoll();
                }
            }
        }

        // ── 參數變更 log（diag：對齊 phaselog 或 archive tick，定位掉偵 vs 改參數）────
        /// <summary>參數變更 log 路徑（AutoAllocateCameras 設）；null=不記。</summary>
        public static string ParamChangeLogPath;
        private static readonly object _paramChangeLogLock = new object();

        /// <summary>記一筆參數變更：time,scope,cam,param,value。路徑由 ParamChangeLogPath（AutoAllocateCameras 設）控制。</summary>
        private static void LogParamChange(string scope, int camId, string param, int value)
        {
            string p = ParamChangeLogPath;
            if (p == null) return;
            try
            {
                lock (_paramChangeLogLock)
                {
                    if (!System.IO.File.Exists(p))
                        System.IO.File.AppendAllText(p, "time,scope,cam,param,value\r\n");
                    System.IO.File.AppendAllText(p, $"{DateTime.Now:HH:mm:ss.fff},{scope},{camId},{param},{value}\r\n");
                }
            }
            catch { }
        }

        /// <summary>套用參數期間鎖住全部控制項；解鎖後仍遵守 Grab 中只開放曝光的產品規則。</summary>
        private void SetParamControlsLocked(bool locked)
        {
            _cameraParameterOperationLocked = locked;
            RefreshCameraParameterControlState();
        }

        /// <summary>
        /// 相機參數控制項唯一狀態計算點：停止時三種皆可改；Grab 中只有曝光可改。
        /// </summary>
        private void RefreshCameraParameterControlState(bool? isGrabbingOverride = null)
        {
            bool isGrabbing = isGrabbingOverride
                ?? (_liveCameraManager?.IsLiveGrabbing == true);
            bool exposureEnabled = !_cameraParameterOperationLocked;
            bool timingEnabled = exposureEnabled && !isGrabbing;

            void SetArr(System.Windows.Forms.Control[] controls, bool enabled)
            {
                if (controls == null) return;
                foreach (System.Windows.Forms.Control control in controls)
                    if (control != null) control.Enabled = enabled;
            }

            SetArr(_expBars, exposureEnabled);
            SetArr(_expNums, exposureEnabled);
            SetArr(_lrBars, timingEnabled);
            SetArr(_lrNums, timingEnabled);
            SetArr(_htBars, timingEnabled);
            SetArr(_htNums, timingEnabled);

            if (_expAllBar != null) _expAllBar.Enabled = exposureEnabled;
            if (_expAllNum != null) _expAllNum.Enabled = exposureEnabled;
            if (_lrAllBar != null) _lrAllBar.Enabled = timingEnabled;
            if (_lrAllNum != null) _lrAllNum.Enabled = timingEnabled;
            if (_htAllBar != null) _htAllBar.Enabled = timingEnabled;
            if (_htAllNum != null) _htAllNum.Enabled = timingEnabled;
        }

        /// <summary>新幀已由 manager 確認；再保留短 cooldown，避免連續重啟 digitizer。</summary>
        private void BeginParamUnlockPoll()
        {
            // 禁止在 UI timer 內 MdigInquire；實際新幀等待已在 manager 的 raw callback 狀態完成。
            int periodMs = _liveCameraManager.GetMaxFramePeriodMs();
            int minHoldMs = periodMs > 0
                ? System.Math.Min(4000, periodMs * 2 + 150)
                : 500;
            _paramLockMinReleaseUtc = DateTime.UtcNow.AddMilliseconds(minHoldMs);
            if (_paramUnlockTimer == null)
            {
                _paramUnlockTimer = new System.Windows.Forms.Timer { Interval = 120 };
                _paramUnlockTimer.Tick += ParamUnlockTimer_Tick;
            }
            _paramUnlockTimer.Start();
        }

        private void ParamUnlockTimer_Tick(object sender, EventArgs e)
        {
            if (DateTime.UtcNow < _paramLockMinReleaseUtc) return;
            _paramUnlockTimer.Stop();
            SetParamControlsLocked(false);
            _liveCameraManager?.SetCaptureSuppressed(false);
        }

        /// <summary>
        /// 更新曝光 TrackBar/NUD 的 Maximum；若現有值被夾緊則將 NUD 背景色改為 OrangeRed，
        /// 否則恢復預設白色。由 LR ValueChanged 呼叫。
        /// </summary>
        private void UpdateExpMaxAndClampColor(int idx, int newMax)
        {
            _expBars[idx].Maximum = newMax;
            _expNums[idx].Maximum = newMax;
            if (_expBars[idx].Value > newMax)
            {
                _expBars[idx].Value = newMax;
                _expNums[idx].Value = newMax;
                _expNums[idx].BackColor = Color.OrangeRed;   // 夾緊警告
            }
            else
            {
                _expNums[idx].BackColor = SystemColors.Window;
            }
            UpdateExpAllMax();
        }

        private void UpdateExpAllMax()
        {
            if (_expAllBar == null || _expBars == null) return;
            int minMax = _expBars[0].Maximum;
            for (int i = 1; i < _expBars.Length; i++)
                if (_expBars[i].Maximum < minMax) minMax = _expBars[i].Maximum;
            _expAllBar.Maximum = minMax;
            _expAllNum.Maximum = minMax;
            if (_expAllBar.Value > minMax)
            {
                _expAllBar.Value = minMax;
                _expAllNum.Value = minMax;
            }
        }

        private void SetupSystemTab()
        {
            // ── 即時 Telemetry ListView（取代靜態 5 欄設定表）─────────────
            _telemetryPresenter = new LiveTelemetryPresenter(listViewCameras);
            _telemetryPresenter.Initialize(SystemSettings.CreateDefault().CameraDevices);

            listViewSystemParameters.Columns.Add("參數", 160);
            listViewSystemParameters.Columns.Add("值", 120);

            // ── 影像／圖表引擎 ────────────────────────────────────────────
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "───", "── 影像引擎 ──" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "MaxWidth",            InspectionEngineConfig.MaxWidth.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "MaxHeight",           InspectionEngineConfig.MaxHeight.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",    InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",      InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "PerFrameBgSigma",     InspectionEngineConfig.PerFrameBgSigma.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma",   InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "DefaultHessianMax",   InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",    InspectionEngineConfig.DefaultRidgeMode }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "SaveResizeScale",     InspectionEngineConfig.DefaultSaveResizeScale.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "SaveJpgQuality",      InspectionEngineConfig.DefaultSaveJpgQuality.ToString() }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "───", "── 圖表引擎 ──" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "MaxOverviewPoints", "2000" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "TelemetryInterval", "500 ms" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "OverviewRefresh",   "FPS-sync" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "DownsampleMode",    "Max-Window" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "OverlapMean",       "Average" }));
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "OverlapMax",        "Maximum" }));

            // ── 硬體參數 ─────────────────────────────────────────────────
            listViewSystemParameters.Items.Add(new ListViewItem(new[] { "───", "── 硬體 ──" }));

            // ── CPU / RAM / GPU（通用硬體，收進 TanukiCv.Core.SystemInfo 唯一來源）──
            foreach (var kv in SystemInfo.GetGenericHardwareRows())
                listViewSystemParameters.Items.Add(new ListViewItem(new[] { kv.Key, kv.Value }));

            listViewSystemParameters.Items.Add(new ListViewItem(new[] {
                "IO_Model", _settings?.IoModel ?? InspectionDefaults.IoModel }));

            // ── Grabber（PCIe frame grabber）──
            try
            {
                using (var grabSearcher = new ManagementObjectSearcher(
                    "SELECT Name, DeviceID FROM Win32_PnPEntity WHERE Name LIKE '%frame grabber%' OR Name LIKE '%Frame Grabber%'"))
                foreach (ManagementObject obj in grabSearcher.Get())
                {
                    string grabName = obj["Name"]?.ToString() ?? "N/A";
                    string devId = obj["DeviceID"]?.ToString() ?? "";
                    listViewSystemParameters.Items.Add(new ListViewItem(new[] { "Grabber", grabName }));

                    if (!devId.StartsWith("PCI\\", StringComparison.OrdinalIgnoreCase)) continue;

                    // PCIe link speed/width via PowerShell Get-PnpDeviceProperty
                    try
                    {
                        var psi = new System.Diagnostics.ProcessStartInfo
                        {
                            FileName = "powershell.exe",
                            Arguments = $"-NoProfile -Command \"Get-PnpDeviceProperty -InstanceId '{devId}' | " +
                                "Where-Object { $_.KeyName -match 'CurrentLinkSpeed|CurrentLinkWidth' } | " +
                                "ForEach-Object { $_.KeyName + '=' + $_.Data }\"",
                            RedirectStandardOutput = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        using (var proc = System.Diagnostics.Process.Start(psi))
                        {
                            string output = proc.StandardOutput.ReadToEnd();
                            proc.WaitForExit(5000);

                            int linkSpeed = 0, linkWidth = 0;
                            foreach (string line in output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
                            {
                                var parts = line.Split('=');
                                if (parts.Length != 2) continue;
                                if (parts[0].Contains("CurrentLinkSpeed")) int.TryParse(parts[1].Trim(), out linkSpeed);
                                if (parts[0].Contains("CurrentLinkWidth")) int.TryParse(parts[1].Trim(), out linkWidth);
                            }

                            if (linkSpeed > 0 && linkWidth > 0)
                            {
                                string[] genNames = { "?", "Gen1", "Gen2", "Gen3", "Gen4", "Gen5" };
                                double[] genGTs = { 0, 2.5, 5, 8, 16, 32 };
                                string gen = linkSpeed < genNames.Length ? genNames[linkSpeed] : $"Gen{linkSpeed}";
                                double bwGBs = linkSpeed < genGTs.Length
                                    ? genGTs[linkSpeed] * linkWidth * 0.8 / 8.0   // 8b/10b for Gen1-2, 128b/130b for Gen3+
                                    : 0;
                                if (linkSpeed >= 3 && linkSpeed < genGTs.Length)
                                    bwGBs = genGTs[linkSpeed] * linkWidth * (128.0 / 130.0) / 8.0;

                                listViewSystemParameters.Items.Add(new ListViewItem(new[] {
                                    "Grabber_PCIe", $"{gen} x{linkWidth} ({bwGBs:F1} GB/s)" }));
                            }
                        }
                    }
                    catch { /* PowerShell 非關鍵 */ }
                }
            }
            catch { }

            // ── 磁碟（所有固定碟） ──
            try
            {
                string capRoot = _settings?.CaptureRootPath ??
                    Path.Combine(
                        InspectionDefaults.AniloxRootPath,
                        InspectionDefaults.CaptureDirectoryName);
                string capDrive = Path.GetPathRoot(capRoot)?.TrimEnd('\\') ?? "";
                foreach (var di in DriveInfo.GetDrives())
                {
                    if (di.DriveType != DriveType.Fixed || !di.IsReady) continue;
                    double totalGb = di.TotalSize / (1024.0 * 1024 * 1024);
                    double freeGb  = di.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                    string label   = di.Name.TrimEnd('\\');
                    string suffix  = label.Equals(capDrive, StringComparison.OrdinalIgnoreCase) ? " [存圖]" : "";
                    listViewSystemParameters.Items.Add(new ListViewItem(new[] {
                        $"Disk_{label}", $"{di.DriveFormat}  {freeGb:F1} / {totalGb:F1} GB free{suffix}" }));
                }
            }
            catch { /* 非關鍵，忽略 */ }

            // ── 螢幕（收進 TanukiCv.Core.SystemInfo 唯一來源；mm/px 同步給座標/倍率計算）──
            foreach (var kv in SystemInfo.GetScreenRows())
                listViewSystemParameters.Items.Add(new ListViewItem(new[] { kv.Key, kv.Value }));
            var screen = SystemInfo.GetScreenMetrics();
            if (screen.HorzPx > 0)
            {
                _reviewRuntimeState.ScreenMmPerPixel = screen.MmPerPx;
                _liveCameraManager?.SetScreenMmPerPixel(screen.MmPerPx);
            }

            // ── Storage 模式：磁碟 + 清理狀態（即時，Timer 更新）──
            if (_appMode?.Role == MachineRole.Storage)
            {
                listViewSystemParameters.Items.Add(new ListViewItem(new[] { "───", "── Storage 狀態 ──" }));
                _storageDiskFreeRow  = AddResMonItem("Disk_Free",    "—");
                _storageLastCleanRow = AddResMonItem("Last_Cleanup", "—");
                _retentionService.OnCleanupCompleted += r =>
                {
                    if (_storageLastCleanRow == null) return;
                    string text = r.FreedBytes > 0
                        ? $"{r.DeletedDayFolders} folders, {r.FreedBytes / (1024.0 * 1024):F1} MB  ({DateTime.Now:HH:mm:ss})"
                        : $"OK  ({DateTime.Now:HH:mm:ss})";
                    SafeBeginInvoke(() => { if (_storageLastCleanRow != null) _storageLastCleanRow.SubItems[1].Text = text; });
                };
            }
            else
            {
                // ── Resource Monitor（即時資源用量，Timer 更新）──
                listViewSystemParameters.Items.Add(new ListViewItem(new[] { "───", "── Resource Monitor ──" }));
                _resMonRawSize     = AddResMonItem("RawSize",     "—");
                _resMonGpuTime     = AddResMonItem("GPU_Time",    "—");
                _resMonSaveSize    = AddResMonItem("Save/Frame",  "—");
                _resMonDiskWrite   = AddResMonItem("DiskWrite",   "—");
                _resMonFrames      = AddResMonItem("Frames",      "—");
                _resMonRamUsed     = AddResMonItem("RAM_Used",    "—");
                _resMonVramEst     = AddResMonItem("VRAM_Est",    "—");
            }
            AutoFitListViewColumns(listViewSystemParameters);

            // ── Telemetry Timer（每 TelemetryTickMs 更新 ListView + SyncFromHardware + 重連倒數）─
            _telemetryTimer = new System.Windows.Forms.Timer { Interval = TelemetryTickMs };
            _telemetryTimer.Tick += TelemetryTimer_Tick;
            _telemetryTimer.Start();

            // ── Live Overview Timer（chartLiveColumn 全覽圖，動態跟隨最大 FPS）──
            _liveOverviewTimer = new System.Windows.Forms.Timer { Interval = 100 };
            _liveOverviewTimer.Tick += LiveOverviewTimer_Tick;
            _liveOverviewTimer.Start();
        }

        // ── TrackBar 滾輪：每格僅移動 1 ──────────────────────────────────
        private void RegisterWheelInterceptors(TrackBar[] bars)
        {
            foreach (var bar in bars)
                _wheelInterceptors.Add(new TrackBarWheelInterceptor(bar));
        }


        /// <summary>硬體參數同步單飛旗標（背景 CLProtocol 讀取進行中不疊發）。</summary>
        private volatile bool _paramSyncInFlight;

        private void SyncCameraParamsFromHardware()
        {
            // CLProtocol feature 讀取（GetMeasuredExposureUs/GetLineRateHz＝MdigInquireFeature）一次可達
            // 數百 ms（2026-07-07 [UiStack] 定罪：TelemetryTick 569ms 卡 UI 元凶）→ 讀取移背景、UI 只套滑桿。
            if (_expBars == null || _lrBars == null || _paramSyncInFlight) return;
            var acq = _settings?.Acquisition;
            if (acq == null) return;

            var targets = new System.Collections.Generic.List<(int idx, Core.Camera.AniloxCamera cam)>();
            for (int idx = 0; idx < CameraCount; idx++)
            {
                var cam = FindCameraById(idx + 1);
                if (cam != null && cam.IsHwParamsStable) targets.Add((idx, cam));
            }
            if (targets.Count == 0) return;

            _paramSyncInFlight = true;
            System.Threading.Tasks.Task.Run(() =>
            {
                var vals = new System.Collections.Generic.List<(int idx, double exp, double lr)>();
                try
                {
                    foreach (var t in targets)
                    {
                        try { vals.Add((t.idx, t.cam.GetMeasuredExposureUs(), t.cam.GetLineRateHz())); }
                        catch (Exception ex) { Trace.WriteLine($"[SyncHw] CAM{t.idx + 1}: {ex.Message}"); }
                    }
                }
                finally { _paramSyncInFlight = false; }
                if (vals.Count == 0 || IsDisposed || Disposing) return;
                try
                {
                    BeginInvoke(new Action(() =>
                    {
                        if (IsDisposed) return;
                        foreach (var v in vals)
                        {
                            SyncHardwareParam(_expBars[v.idx], _expNums[v.idx], v.exp,
                                x => acq.CameraExposureTimeUs[v.idx] = x);
                            SyncHardwareParam(_lrBars[v.idx], _lrNums[v.idx], v.lr,
                                x => acq.CameraLineRateHz[v.idx] = x);
                        }
                    }));
                }
                catch (InvalidOperationException) { }
            });
        }

        // ── Helper Methods ──────────────────────────────────────────

        private void SyncHardwareParam(TrackBar bar, NumericUpDown nud, double hwValue, Action<int> saveSetting)
        {
            if (_dragging.Contains(bar) || hwValue <= 0) return;
            int clamped = Math.Max(bar.Minimum, Math.Min(bar.Maximum, (int)hwValue));
            double diff = Math.Abs(clamped - bar.Value) / (double)Math.Max(1, bar.Value);
            if (diff <= 0.05) return;
            _syncingFromHw = true;
            bar.Value = clamped;
            nud.Value = clamped;
            saveSetting(clamped);
            _syncingFromHw = false;
        }
    }
}
