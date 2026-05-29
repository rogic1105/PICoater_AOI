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
    /// <summary>AniloxRollForm 右側設定面板 tab 建構（相機參數 / 系統）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
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
            const int ExpMin    =     1;   // μs
            const int ExpMaxCap = 10000;   // μs 硬上限
            const int LrMin     =   100;   // Hz
            const int LrMax     = 10000;   // Hz
            const int HtMin     =   100;   // px
            const int HtMax     = 10000;   // px

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

            BindAllSync(_expAllBar, _expAllNum, _expBars, _expNums,
                (j, v) => _liveCameraManager?.SetExposureForCamera(j + 1, v),
                (j, v) => { acq.CameraExposureTimeUs[j] = v; ConfigManager.SaveAcquisitionSettings(acq); });

            BindAllSync(_lrAllBar, _lrAllNum, _lrBars, _lrNums,
                (j, v) => _liveCameraManager?.SetLineRateForCamera(j + 1, v),
                (j, v) => { acq.CameraLineRateHz[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    // 同步所有 cam 的 exp max（每台 LR 都同值，算一次套到所有 cam）
                    int newMax = (int)acq.CameraLineRateHz[0];
                    int expMax = MilCameraParams.CalcExposureMaxUs(newMax, ExpMin, ExpMaxCap);
                    for (int i = 0; i < CameraCount; i++) UpdateExpMaxAndClampColor(i, expMax);
                    UpdateRowChartPitch();
                });

            BindAllSync(_htAllBar, _htAllNum, _htBars, _htNums,
                (j, v) => _liveCameraManager?.SetGrabHeightForCamera(j + 1, v),
                (j, v) => { acq.CameraGrabHeight[j] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                () => {
                    _liveCameraManager?.RefreshMainDisplay();
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
                    v => _liveCameraManager?.SetExposureForCamera(camId, v));

                // ── 線掃速率 ────────────────────────────────────────────
                BindBidirectionalSync(_lrBars[idx], _lrNums[idx], camId,
                    LrMin, LrMax, (int)acq.CameraLineRateHz[idx],
                    v => { acq.CameraLineRateHz[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => _liveCameraManager?.SetLineRateForCamera(camId, v),
                    () => { UpdateExpMaxAndClampColor(idx, CalcExpMax()); if (idx == 0) UpdateRowChartPitch(); });

                // ── 擷取高度 ────────────────────────────────────────────
                BindBidirectionalSync(_htBars[idx], _htNums[idx], camId,
                    HtMin, HtMax, acq.CameraGrabHeight[idx],
                    v => { acq.CameraGrabHeight[idx] = v; ConfigManager.SaveAcquisitionSettings(acq); },
                    v => _liveCameraManager?.SetGrabHeightForCamera(camId, v),
                    () => {
                        _liveCameraManager?.RefreshMainDisplay();
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
        }

        /// <summary>
        /// TrackBar ↔ NumericUpDown 雙向同步綁定：
        /// - 拖曳中：抑制硬體寫入，MouseUp 立即寫入。
        /// - 滾輪 / 鍵盤箭頭 / NUD 輸入：1 秒 debounce 才寫硬體（避免高頻 MIL 寫入造成卡頓）。
        /// </summary>
        private void BindBidirectionalSync(
            TrackBar bar, NumericUpDown num, int camId,
            int min, int max, int initialValue,
            Action<int> saveSetting, Action<int> writeHardware,
            Action postAction = null)
        {
            int clamped = Math.Max(min, Math.Min(max, initialValue));
            bar.Minimum = min; bar.Maximum = max; bar.TickFrequency = TickFreq;
            num.Minimum = min; num.Maximum = max;
            bar.Value = clamped; num.Value = clamped;

            bool syncing = false;

            // 滾輪/鍵盤/NUD 等非拖曳輸入 → 1s debounce 才寫硬體
            var debounce = new System.Windows.Forms.Timer { Interval = 1000 };
            int pendingValue = clamped;
            bool hasPending = false;
            debounce.Tick += (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                writeHardware(pendingValue);
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
            bar.MouseUp += (s, e) =>
            {
                _dragging.Remove(bar);
                // 拖曳結束 → 立即寫入並取消 debounce
                debounce.Stop();
                hasPending = false;
                writeHardware(bar.Value);
                postAction?.Invoke();   // 硬體寫完後再 refresh（例：HT 改變後重新載入主畫面 buffer）
            };
            bar.ValueChanged += (s, e) =>
            {
                if (syncing || _syncingFromHw) return; syncing = true;
                num.Value = bar.Value;
                saveSetting(bar.Value);
                if (!_dragging.Contains(bar)) ScheduleWrite(bar.Value);
                postAction?.Invoke();
                syncing = false;
            };
            num.ValueChanged += (s, e) =>
            {
                if (syncing || _syncingFromHw) return; syncing = true;
                int v = (int)num.Value;
                bar.Value = Math.Max(min, Math.Min(max, v));
                saveSetting(v);
                ScheduleWrite(v);
                postAction?.Invoke();
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
            Action<int, int> writeHardwareForCam,    // (camIdx0based, value)
            Action<int, int> saveSettingForCam,      // (camIdx0based, value)
            Action postWriteAll = null)
        {
            bool allSyncing = false;
            var debounce = new System.Windows.Forms.Timer { Interval = 1000 };
            int pendingValue = barAll.Value;
            bool hasPending = false;

            void Apply(int v)
            {
                // 1. 寫硬體（所有 7 台）
                for (int j = 0; j < bars.Length; j++)
                    writeHardwareForCam(j, v);
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

            debounce.Tick += (s, e) =>
            {
                debounce.Stop();
                if (!hasPending) return;
                hasPending = false;
                Apply(pendingValue);
            };
            void Schedule(int v)
            {
                pendingValue = v;
                hasPending = true;
                debounce.Stop();
                debounce.Start();
            }

            barAll.MouseDown += (s, e) => _dragging.Add(barAll);
            barAll.MouseUp += (s, e) =>
            {
                _dragging.Remove(barAll);
                debounce.Stop();
                hasPending = false;
                Apply(barAll.Value);
            };
            barAll.ValueChanged += (s, e) => {
                if (allSyncing || _syncingFromHw) return; allSyncing = true;
                numAll.Value = barAll.Value;
                if (!_dragging.Contains(barAll)) Schedule(barAll.Value);
                allSyncing = false;
            };
            numAll.ValueChanged += (s, e) => {
                if (allSyncing || _syncingFromHw) return; allSyncing = true;
                int v = (int)numAll.Value;
                barAll.Value = Math.Max(barAll.Minimum, Math.Min(barAll.Maximum, v));
                Schedule(v);
                allSyncing = false;
            };
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

            // ── 影像引擎常數 ──────────────────────────────────────────────
            listViewEngine.Columns.Add("參數", 160);
            listViewEngine.Columns.Add("值",    90);

            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxWidth",            InspectionEngineConfig.MaxWidth.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxHeight",           InspectionEngineConfig.MaxHeight.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxThumbnailSide",    InspectionEngineConfig.MaxThumbnailSide.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultBgSigma",      InspectionEngineConfig.DefaultBgSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeSigma",   InspectionEngineConfig.DefaultRidgeSigma.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultHessianMax",   InspectionEngineConfig.DefaultHessianMaxFactor.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DefaultRidgeMode",    InspectionEngineConfig.DefaultRidgeMode }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "SaveResizeScale",     InspectionEngineConfig.DefaultSaveResizeScale.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "SaveJpgQuality",      InspectionEngineConfig.DefaultSaveJpgQuality.ToString() }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "───", "── 圖表引擎 ──" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "MaxOverviewPoints", "2000" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "TelemetryInterval", "500 ms" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverviewRefresh",   "FPS-sync" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "DownsampleMode",    "Max-Window" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverlapMean",       "Average" }));
            listViewEngine.Items.Add(new ListViewItem(new[] { "OverlapMax",        "Maximum" }));
            AutoFitListViewColumns(listViewEngine);

            // ── 硬體參數 ─────────────────────────────────────────────────
            listViewHardware.Columns.Add("參數", 120);
            listViewHardware.Columns.Add("值",   120);

            // ── CPU / RAM ──
            try
            {
                using (var cpuSearcher = new ManagementObjectSearcher("SELECT Name, NumberOfCores, NumberOfLogicalProcessors FROM Win32_Processor"))
                foreach (var obj in cpuSearcher.Get())
                {
                    listViewHardware.Items.Add(new ListViewItem(new[] { "CPU",       obj["Name"]?.ToString().Trim() ?? "N/A" }));
                    listViewHardware.Items.Add(new ListViewItem(new[] { "CPU_Cores",  $"{obj["NumberOfCores"]}C / {obj["NumberOfLogicalProcessors"]}T" }));
                    break; // 只取第一顆
                }

                using (var memSearcher = new ManagementObjectSearcher("SELECT Capacity, Speed, SMBIOSMemoryType FROM Win32_PhysicalMemory"))
                {
                    var sticks = memSearcher.Get().Cast<ManagementObject>().ToArray();
                    int count = sticks.Length;
                    ulong totalBytes = 0;
                    int speed = 0;
                    int memType = 0;
                    foreach (var stick in sticks)
                    {
                        totalBytes += (ulong)stick["Capacity"];
                        if (speed == 0 && stick["Speed"] != null)
                            speed = Convert.ToInt32(stick["Speed"]);
                        if (memType == 0 && stick["SMBIOSMemoryType"] != null)
                            memType = Convert.ToInt32(stick["SMBIOSMemoryType"]);
                    }
                    double perStickGb = count > 0 ? (totalBytes / (double)count) / (1024.0 * 1024 * 1024) : 0;
                    string ddrGen = memType == 34 ? "DDR5" : memType == 26 ? "DDR4" : memType == 24 ? "DDR3" : "DDR";
                    string speedStr = speed > 0 ? $"-{speed}" : "";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "RAM",
                        $"{totalBytes / (1024.0 * 1024 * 1024):F0} GB ({count}×{perStickGb:F0}GB {ddrGen}{speedStr})" }));
                }
            }
            catch { /* WMI 非關鍵，忽略 */ }

            // ── GPU ──
            try
            {
                // Registry 查 64-bit VRAM（qwMemorySize），避免 WMI uint32 溢位
                var regVram = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
                try
                {
                    using (var videoKey = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"))
                    if (videoKey != null)
                    {
                        foreach (string sub in videoKey.GetSubKeyNames())
                        {
                            if (!int.TryParse(sub, out _)) continue;
                            using (var sk = videoKey.OpenSubKey(sub))
                            {
                                if (sk == null) continue;
                                string desc = sk.GetValue("DriverDesc") as string;
                                if (string.IsNullOrEmpty(desc)) continue;
                                object qw = sk.GetValue("HardwareInformation.qwMemorySize");
                                if (qw is long qwVal && qwVal > 0)
                                    regVram[desc] = qwVal;
                                else if (qw is byte[] qwBytes && qwBytes.Length >= 8)
                                    regVram[desc] = BitConverter.ToInt64(qwBytes, 0);
                            }
                        }
                    }
                }
                catch { /* registry 非關鍵 */ }

                using (var gpuSearcher = new ManagementObjectSearcher("SELECT Name, AdapterRAM FROM Win32_VideoController"))
                foreach (ManagementObject obj in gpuSearcher.Get())
                {
                    string gpuName = obj["Name"]?.ToString() ?? "N/A";
                    long vramBytes;
                    if (regVram.TryGetValue(gpuName, out long regBytes) && regBytes > 0)
                        vramBytes = regBytes;
                    else
                        vramBytes = Convert.ToUInt32(obj["AdapterRAM"]);

                    double vramGb = vramBytes / (1024.0 * 1024 * 1024);
                    string vramStr = vramGb >= 1.0 ? $"{vramGb:F1} GB" : $"{vramBytes / (1024.0 * 1024):F0} MB";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "GPU",      gpuName }));
                    listViewHardware.Items.Add(new ListViewItem(new[] { "GPU_VRAM", vramStr }));
                }
            }
            catch { /* WMI 非關鍵，忽略 */ }

            // ── Grabber（PCIe frame grabber）──
            try
            {
                using (var grabSearcher = new ManagementObjectSearcher(
                    "SELECT Name, DeviceID FROM Win32_PnPEntity WHERE Name LIKE '%frame grabber%' OR Name LIKE '%Frame Grabber%'"))
                foreach (ManagementObject obj in grabSearcher.Get())
                {
                    string grabName = obj["Name"]?.ToString() ?? "N/A";
                    string devId = obj["DeviceID"]?.ToString() ?? "";
                    listViewHardware.Items.Add(new ListViewItem(new[] { "Grabber", grabName }));

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

                                listViewHardware.Items.Add(new ListViewItem(new[] {
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
                string capRoot = _settings?.CaptureRootPath ?? @"D:\AniloxCaptures";
                string capDrive = Path.GetPathRoot(capRoot)?.TrimEnd('\\') ?? "";
                foreach (var di in DriveInfo.GetDrives())
                {
                    if (di.DriveType != DriveType.Fixed || !di.IsReady) continue;
                    double totalGb = di.TotalSize / (1024.0 * 1024 * 1024);
                    double freeGb  = di.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                    string label   = di.Name.TrimEnd('\\');
                    string suffix  = label.Equals(capDrive, StringComparison.OrdinalIgnoreCase) ? " [存圖]" : "";
                    listViewHardware.Items.Add(new ListViewItem(new[] {
                        $"Disk_{label}", $"{di.DriveFormat}  {freeGb:F1} / {totalGb:F1} GB free{suffix}" }));
                }
            }
            catch { /* 非關鍵，忽略 */ }

            // ── 螢幕 ──
            try
            {
                IntPtr hdc = GetDC(IntPtr.Zero);
                int horzMm   = GetDeviceCaps(hdc, 4);   // HORZSIZE (mm)
                int vertMm   = GetDeviceCaps(hdc, 6);   // VERTSIZE (mm)
                int horzPx   = GetDeviceCaps(hdc, 8);   // HORZRES (px, 含 DPI 縮放)
                int vertPx   = GetDeviceCaps(hdc, 10);  // VERTRES (px, 含 DPI 縮放)
                int logDpiX  = GetDeviceCaps(hdc, 88);  // LOGPIXELSX
                int logDpiY  = GetDeviceCaps(hdc, 90);  // LOGPIXELSY
                ReleaseDC(IntPtr.Zero, hdc);

                int nativeW  = (int)Math.Round(horzPx * logDpiX / 96.0);
                int nativeH  = (int)Math.Round(vertPx * logDpiY / 96.0);
                int scalePct = (int)Math.Round(logDpiX / 96.0 * 100);

                double screenMmPerPx = (double)horzMm / horzPx;
                listViewHardware.Items.Add(new ListViewItem(new[] { "ScreenSize",   $"{horzMm / 10.0:F1} × {vertMm / 10.0:F1} cm" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "NativeRes",    $"{nativeW} × {nativeH}" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "EffectiveRes", $"{horzPx} × {vertPx}" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "DpiScale",     $"{scalePct}%" }));
                listViewHardware.Items.Add(new ListViewItem(new[] { "mm/px",        $"{screenMmPerPx:F4}" }));

                _interactionHelper?.SetScreenMmPerPixel(screenMmPerPx);
                _liveCameraManager?.SetScreenMmPerPixel(screenMmPerPx);
            }
            catch { /* 非關鍵資訊，忽略 */ }

            // ── Storage 模式：磁碟 + 清理狀態（即時，Timer 更新）──
            if (_appMode?.Role == MachineRole.Storage)
            {
                listViewHardware.Items.Add(new ListViewItem(new[] { "───", "── Storage 狀態 ──" }));
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
                listViewHardware.Items.Add(new ListViewItem(new[] { "───", "── Resource Monitor ──" }));
                _resMonRawSize     = AddResMonItem("RawSize",     "—");
                _resMonGpuTime     = AddResMonItem("GPU_Time",    "—");
                _resMonSaveSize    = AddResMonItem("Save/Frame",  "—");
                _resMonDiskWrite   = AddResMonItem("DiskWrite",   "—");
                _resMonFrames      = AddResMonItem("Frames",      "—");
                _resMonRamUsed     = AddResMonItem("RAM_Used",    "—");
                _resMonVramEst     = AddResMonItem("VRAM_Est",    "—");
            }
            AutoFitListViewColumns(listViewHardware);

            // ── Telemetry Timer（每 500ms 更新 ListView + SyncFromHardware）─
            _telemetryTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _telemetryTimer.Tick += TelemetryTimer_Tick;
            _telemetryTimer.Start();

            // ── Live Overview Timer（chartLivePatch 全覽圖，動態跟隨最大 FPS）──
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


        private void SyncCameraParamsFromHardware()
        {
            if (_expBars == null || _lrBars == null) return;

            var cameras = _liveCameraManager.Cameras;
            var acq     = _settings?.Acquisition;
            if (acq == null) return;

            for (int idx = 0; idx < CameraCount; idx++)
            {
                try
                {
                    var cam = FindCameraById(idx + 1);
                    if (cam == null) continue;
                    if (!cam.IsHwParamsStable) continue;

                    SyncHardwareParam(_expBars[idx], _expNums[idx],
                        cam.GetMeasuredExposureUs(), v => acq.CameraExposureTimeUs[idx] = v);

                    SyncHardwareParam(_lrBars[idx], _lrNums[idx],
                        cam.GetLineRateHz(), v => acq.CameraLineRateHz[idx] = v);
                }
                catch (Exception ex) { Debug.WriteLine($"[SyncHw] CAM{idx + 1}: {ex.Message}"); }
            }
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
