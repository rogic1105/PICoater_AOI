using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Web.Script.Serialization; // JavaScriptSerializer 解析 system-settings.json
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary; // MApp 仍由本範例管理
using MilGrabber.Core;             // 已封裝的單相機 MIL library

namespace MilGrabber.Monitor
{
    /// <summary>
    /// 多相機即時監控 UI 範例。
    /// 取像/顯示細節全部委派給 <see cref="MilCamera"/>（sdk/MIL/MilGrabber.Core）；
    /// 本 Form 只負責 MApp 生命週期、UI 佈局、相機選擇與 telemetry 顯示。
    /// MilCamera 內建 grab hook：未訂閱 FrameReady 時自動把原圖顯示到 primary panel，
    /// 故本純顯示範例不訂閱 FrameReady。
    /// </summary>
    public partial class MilGrabberForm : Form
    {
        // 固定畫 8 個子畫面（不管實際幾台相機）。Designer 已逐一宣告 panelCam0..7（容器）；
        // displayPanel + status label 在本檔 runtime 建（仿主程式 SetupLivePanel）。
        // 此常數供 label / telemetry 迴圈使用。
        private const int SubPanelCount = 8;

        // ==================== Config（system-settings.json 反序列化目標） ====================
        /// <summary>JSON 根物件：CameraDevices 陣列。</summary>
        private sealed class SystemConfig
        {
            public List<CameraDeviceConfig> CameraDevices { get; set; }
        }

        /// <summary>單一相機硬體配置（對應主程式 CameraHardwareConfig）。</summary>
        private sealed class CameraDeviceConfig
        {
            public int Id { get; set; }
            public string SystemDescriptor { get; set; }
            public int SystemNum { get; set; }
            public int DevNum { get; set; }
            public string DcfPath { get; set; }
        }

        // 讀到的相機 device 清單（依 btnInit 讀 config 或 fallback 填入）。
        private List<CameraDeviceConfig> _devices;
        // 是否使用了內建 fallback 預設（檔不存在 / 解析失敗）→ lvEngine / Trace 標示。
        private bool _usedFallbackConfig = false;

        // ==================== MIL Application（MApp 由本範例管理；MilCamera 不碰 MApp） ====================
        private MIL_ID _milApplication = MIL.M_NULL;
        // 按 SystemNum 快取已分配的擷取卡 system（每張卡一個 MsysAlloc）。
        private readonly Dictionary<int, MIL_ID> _systems = new Dictionary<int, MIL_ID>();

        // ==================== Cameras ====================
        private MilCamera[] _cams;          // 實際建立的相機（長度 = 實際 device 數）
        private int _selectedCam = -1;      // 目前選中的相機索引（顯示於主畫面）

        // ==================== Param Tabs（Designer 固定 8 相機：每 tab 全部相機列 + Cam1~8 列） ====================
        // 控制項本體在 Designer.cs 逐一宣告（panel + label + trackBar + NUD）；
        // 此處只把 Designer 具名控制項組成陣列（索引 0..7 = 子畫面 / _cams 索引），仿 panelCam0..7 模式。
        private TrackBar[] _tbExposure, _tbLineRate, _tbHeight;
        private NumericUpDown[] _nudExposure, _nudLineRate, _nudHeight;

        private Panel[] _panelExp, _panelLr, _panelHt;   // 每相機列 panel（enable/disable 整列用）

        // 每 tab 頂部「全部相機」統一控制列（Designer 宣告 panelExpAll/...）。
        private TrackBar _tbExpAll, _tbLrAll, _tbHtAll;
        private NumericUpDown _nudExpAll, _nudLrAll, _nudHtAll;

        // 8 個子畫面容器：Designer 逐一宣告 panelCam0..7（純宣告式，不在 InitializeComponent 用迴圈），
        // 否則 VS 設計工具的 XML parser 無法解析 → 設計階段載入失敗。
        // 每個容器內部的 displayPanel（MIL 顯示）+ status label（狀態）由 SetupCamPanel runtime 建（仿主程式 SetupLivePanel）。
        private Panel[] _camContainers;     // Designer 宣告的 panelCam0..7（容器）
        private Panel[] _displayPanels;     // runtime 建：MilCamera 顯示目標（panelHandle 來源）
        private Label[] _statusLabels;      // runtime 建：取代原 lblCam，顯示 Online/Offline/No Camera

        // ==================== State ====================
        private bool _userWantsGrab = false; // btnGrab toggle 狀態
        private volatile bool _isReleasing = false;

        // ==================== Telemetry Timer ====================
        private readonly System.Windows.Forms.Timer _statusTimer;

        // 抑制 NumericUpDown ValueChanged 在「程式碼設值」時回灌相機（只允許使用者手動變更觸發套用）
        private bool _suppressParamEvents = false;

        // 拖曳中的 TrackBar（MouseDown 進 / MouseUp 出）：拖曳過程不寫硬體，放掉才套用（仿 AniloxRoll.Monitor）
        private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

        public MilGrabberForm()
        {
            InitializeComponent();

            // 容器陣列：在 Form 建構式從 Designer 具名控制項 panelCam0..7 組成（陣列在 .cs 組，不在 Designer）。
            _camContainers = new Panel[] {
                panelCam0, panelCam1, panelCam2, panelCam3,
                panelCam4, panelCam5, panelCam6, panelCam7 };
            _displayPanels = new Panel[SubPanelCount];
            _statusLabels  = new Label[SubPanelCount];

            // 每個容器 runtime 建內部 displayPanel + status（仿主程式 SetupLivePanel）。
            // 一律先建好（即使未綁相機），未綁的 status 顯示 "No Camera"。
            for (int i = 0; i < SubPanelCount; i++)
                SetupCamPanel(_camContainers[i], i);

            // 參數控制項陣列：從 Designer 具名控制項組成（陣列在 .cs 組、控制項在 Designer 宣告，同 panelCam0..7 模式）。
            _tbExposure = new[] { trackBarExpCam1, trackBarExpCam2, trackBarExpCam3, trackBarExpCam4, trackBarExpCam5, trackBarExpCam6, trackBarExpCam7, trackBarExpCam8 };
            _nudExposure = new[] { numExpCam1, numExpCam2, numExpCam3, numExpCam4, numExpCam5, numExpCam6, numExpCam7, numExpCam8 };
            _panelExp = new[] { panelExpCam1, panelExpCam2, panelExpCam3, panelExpCam4, panelExpCam5, panelExpCam6, panelExpCam7, panelExpCam8 };
            _tbLineRate = new[] { trackBarLrCam1, trackBarLrCam2, trackBarLrCam3, trackBarLrCam4, trackBarLrCam5, trackBarLrCam6, trackBarLrCam7, trackBarLrCam8 };
            _nudLineRate = new[] { numLrCam1, numLrCam2, numLrCam3, numLrCam4, numLrCam5, numLrCam6, numLrCam7, numLrCam8 };
            _panelLr = new[] { panelLrCam1, panelLrCam2, panelLrCam3, panelLrCam4, panelLrCam5, panelLrCam6, panelLrCam7, panelLrCam8 };
            _tbHeight = new[] { trackBarHtCam1, trackBarHtCam2, trackBarHtCam3, trackBarHtCam4, trackBarHtCam5, trackBarHtCam6, trackBarHtCam7, trackBarHtCam8 };
            _nudHeight = new[] { numHtCam1, numHtCam2, numHtCam3, numHtCam4, numHtCam5, numHtCam6, numHtCam7, numHtCam8 };
            _panelHt = new[] { panelHtCam1, panelHtCam2, panelHtCam3, panelHtCam4, panelHtCam5, panelHtCam6, panelHtCam7, panelHtCam8 };

            _tbExpAll = trackBarExpAll; _nudExpAll = numExpAll;
            _tbLrAll = trackBarLrAll; _nudLrAll = numLrAll;
            _tbHtAll = trackBarHtAll; _nudHtAll = numHtAll;

            // 接線（每相機列雙向同步 + 套用 _cams；全部相機列套用全部 + 同步各列）。控制項初值/enable 在 btnInit 後 InitParamControls。
            WireParamControls();

            _statusTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _statusTimer.Tick += StatusTimer_Tick;

            UpdateButtonsEnabled(initialized: false);
        }

        // =========================================================================
        // SetupCamPanel：把 Designer 宣告的容器 panelCam{idx} 變成「顯示 + 狀態」複合控制項。
        //   - displayPanel（Dock=Fill, 黑底）← MilCamera 顯示到這（panelHandle = displayPanel.Handle）
        //   - status Label（Dock=Bottom, h=18, 置中, 深色底）← 取代原 lblCam
        //   - 容器 Paint → 畫選中邊框（選中橘色、其他深灰；仿主程式 OnLivePanelPaint）
        //   - displayPanel / status 的 MouseClick → 該相機 SelectCamera
        //   仿 LiveCameraManager.SetupLivePanel；結果存入 _displayPanels / _statusLabels（索引對齊 _cams）。
        // =========================================================================
        private void SetupCamPanel(Panel container, int idx)
        {
            container.BackColor = Color.Black;
            container.Padding = new Padding(2);
            container.Controls.Clear();

            var displayPanel = new Panel
            {
                Dock = DockStyle.Fill,
                BackColor = Color.Black
            };

            var status = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 18,
                ForeColor = Color.LightGray,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Text = "No Camera"
            };

            displayPanel.MouseClick += (s, e) => SelectCamera(idx);
            status.MouseClick += (s, e) => SelectCamera(idx);
            container.Paint += (s, e) => OnCamPanelPaint(s, e, idx);

            container.Controls.Add(displayPanel);
            container.Controls.Add(status);
            displayPanel.BringToFront();

            _displayPanels[idx] = displayPanel;
            _statusLabels[idx] = status;
        }

        // 容器選中邊框：選中橘色（粗）、其他深灰（細）。仿主程式 OnLivePanelPaint。
        private void OnCamPanelPaint(object sender, PaintEventArgs e, int idx)
        {
            if (!(sender is Panel panel)) return;
            bool isSelected = idx == _selectedCam;
            Color borderColor = isSelected ? Color.Orange : Color.FromArgb(60, 60, 60);
            int borderWidth = isSelected ? 3 : 1;
            ControlPaint.DrawBorder(e.Graphics, panel.ClientRectangle,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid);
        }

        // =========================================================================
        // btnInit: 初始化
        //   1. 讀 system-settings.json（或 fallback 預設 7 相機）
        //   2. MappAlloc（只配 App context，不用 MappAllocDefault）
        //   3. 按 SystemNum 去重，每張 Radient 卡一次 MsysAlloc → _systems dict
        //   4. 每 device 建 MilCamera（綁子 panel）+ Initialize
        //   5. 建 telemetry 列 + 參數 tab + 預設選第 0 台
        // =========================================================================
        private void btnInit_Click(object sender, EventArgs e)
        {
            if (_milApplication != MIL.M_NULL) return; // 已初始化

            _isReleasing = false;
            _userWantsGrab = false;

            // 1. 讀 config（檔不存在 / 解析失敗 → fallback 內建 7 相機，並標示）
            _devices = LoadDeviceConfig();
            int camCount = Math.Min(_devices.Count, SubPanelCount); // 子畫面只有 8 個，超過不綁

            // 2. MappAlloc：只配 Application Context（不要 MappAllocDefault，避免它自動配預設 system 干擾多卡分配）
            MIL.MappAlloc(MIL.M_NULL, MIL.M_DEFAULT, ref _milApplication);
            MIL.MappControl(MIL.M_DEFAULT, MIL.M_ERROR, MIL.M_PRINT_DISABLE);

            // 3. 按 SystemNum 去重，每張卡 MsysAlloc 一次
            _systems.Clear();
            _cams = new MilCamera[camCount];

            for (int i = 0; i < camCount; i++)
            {
                CameraDeviceConfig dev = _devices[i];

                MIL_ID sysId;
                if (!_systems.TryGetValue(dev.SystemNum, out sysId))
                {
                    sysId = MIL.M_NULL;
                    MIL.MsysAlloc(_milApplication, MapDescriptor(dev.SystemDescriptor), dev.SystemNum, MIL.M_DEFAULT, ref sysId);
                    if (sysId == MIL.M_NULL)
                    {
                        Trace.WriteLine($"[Init] MsysAlloc 失敗：SystemNum={dev.SystemNum} ({dev.SystemDescriptor})。CAM {dev.Id} 跳過。");
                        SetSubLabel(i, "No Camera", Color.FromArgb(32, 32, 32), Color.LightGray); // 與多餘子畫面一致（診斷見 Trace）
                        continue; // 該卡分配失敗，跳過此 device（陣列該格保持 null）
                    }
                    _systems[dev.SystemNum] = sysId;
                }

                IntPtr panelHandle = _displayPanels[i].Handle; // 第 i 台綁第 i 個容器內的 displayPanel（primary display）
                var cam = new MilCamera(sysId, dev.Id, MIL.M_DEV0 + dev.DevNum, dev.DcfPath, panelHandle);
                int idx = i; // 迴圈變數捕捉（綁子畫面索引，與 _cams / lvCameras Tag 一致）
                cam.OnCameraClicked += _ => SelectCamera(idx);
                cam.Initialize();
                _cams[i] = cam;
            }

            // 多餘子畫面 status 顯示 "No Camera"（深底淺灰字）
            for (int i = camCount; i < SubPanelCount; i++)
                SetSubLabel(i, "No Camera", Color.FromArgb(32, 32, 32), Color.LightGray);

            // 依實際建立的相機數建立 16 欄 telemetry 列（每列 Tag = 子畫面 index），未建立的不加
            InitCamerasListView(camCount);
            // 取像/系統常數表（多為靜態，初始化填一次；選中相機那列由 SelectCamera/Timer 動態更新）
            InitEngineListView(camCount);

            // 4. 參數 tab：Designer 固定 8 相機，依實際相機數填初值 + enable/disable
            InitParamControls(camCount);

            // 預設選中第 0 台
            if (camCount > 0)
                SelectCamera(0);

            _statusTimer.Start();
            btnGrab.Text = "開始抓取";
            UpdateButtonsEnabled(initialized: true);
        }

        // =========================================================================
        // Config 讀取：exe 同目錄 system-settings.json → JavaScriptSerializer 反序列化。
        // 檔不存在 / 解析失敗 / 空清單 → fallback 內建預設 7 相機（_usedFallbackConfig=true）。
        // =========================================================================
        private List<CameraDeviceConfig> LoadDeviceConfig()
        {
            _usedFallbackConfig = false;
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "system-settings.json");

            try
            {
                if (File.Exists(path))
                {
                    string json = File.ReadAllText(path);
                    var cfg = new JavaScriptSerializer().Deserialize<SystemConfig>(json);
                    if (cfg?.CameraDevices != null && cfg.CameraDevices.Count > 0)
                    {
                        Trace.WriteLine($"[Config] 讀取 {path}：{cfg.CameraDevices.Count} 台相機。");
                        return cfg.CameraDevices;
                    }
                    Trace.WriteLine($"[Config] {path} 內容為空 / 無 CameraDevices，改用 fallback 預設。");
                }
                else
                {
                    Trace.WriteLine($"[Config] 找不到 {path}，改用 fallback 預設。");
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[Config] 解析 {path} 失敗（{ex.GetType().Name}: {ex.Message}），改用 fallback 預設。");
            }

            _usedFallbackConfig = true;
            return CreateFallbackDevices();
        }

        /// <summary>內建預設：照 system-settings.json 那 7 筆硬編（SystemNum 0 有 4 台、1 有 3 台）。</summary>
        private static List<CameraDeviceConfig> CreateFallbackDevices()
        {
            const string desc = "M_SYSTEM_RADIENTEVCL";
            const string dcf = @"D:\Anilox\Dcf\Radient_Config.dcf";
            return new List<CameraDeviceConfig>
            {
                new CameraDeviceConfig { Id = 1, SystemDescriptor = desc, SystemNum = 0, DevNum = 0, DcfPath = dcf },
                new CameraDeviceConfig { Id = 2, SystemDescriptor = desc, SystemNum = 0, DevNum = 1, DcfPath = dcf },
                new CameraDeviceConfig { Id = 3, SystemDescriptor = desc, SystemNum = 0, DevNum = 2, DcfPath = dcf },
                new CameraDeviceConfig { Id = 4, SystemDescriptor = desc, SystemNum = 0, DevNum = 3, DcfPath = dcf },
                new CameraDeviceConfig { Id = 5, SystemDescriptor = desc, SystemNum = 1, DevNum = 0, DcfPath = dcf },
                new CameraDeviceConfig { Id = 6, SystemDescriptor = desc, SystemNum = 1, DevNum = 1, DcfPath = dcf },
                new CameraDeviceConfig { Id = 7, SystemDescriptor = desc, SystemNum = 1, DevNum = 2, DcfPath = dcf },
            };
        }

        /// <summary>
        /// config 的 SystemDescriptor 字串 → MIL system descriptor。
        /// 照主程式：MsysAlloc 收字串描述子；"M_SYSTEM_RADIENTEVCL" → MIL.M_SYSTEM_RADIENTEVCL 常數
        /// （該常數本身即字串 "M_SYSTEM_RADIENTEVCL"）。未知值原樣傳回。
        /// </summary>
        private static string MapDescriptor(string descriptor)
        {
            if (string.IsNullOrWhiteSpace(descriptor)) return MIL.M_SYSTEM_RADIENTEVCL;
            if (descriptor == "M_SYSTEM_RADIENTEVCL") return MIL.M_SYSTEM_RADIENTEVCL;
            return descriptor; // 其他擷取卡描述子原樣傳遞給 MsysAlloc
        }

        // =========================================================================
        // btnFetchInfo: 手動抓相機資訊
        //   逐台抓 line rate 上限（CLProtocol 就緒回 >0）：
        //     - 線掃 slider/NUD value = clamp(上限, LrMin, LrMax)（SetRowValue + _suppressParamEvents 防遞迴）
        //     - cam.SetLineRateHz(上限)
        //     - 重算該台曝光 slider/NUD Maximum = CalcExpMax(上限)（含當前曝光值夾緊）
        //   抓不到（CLProtocol 未就緒回 0）跳過該台。最後刷新 17 欄表格。
        //   取代原 timer 自動套用（TryApplyLineRateMax）。
        // =========================================================================
        // 一鍵：確保 grab → 等 CLProtocol 就緒 → 抓 line rate 上限 → 設線掃 slider 上限(Maximum)=硬體上限（方案 A，
        // 當前值維持使用者預設）→ 重算曝光上限。等待期間 async 不凍 UI。
        private async void btnFetchInfo_Click(object sender, EventArgs e)
        {
            if (_cams == null) return;
            btnFetchInfo.Enabled = false;
            string origText = btnFetchInfo.Text;
            try
            {
                // 抓取期間完全不顯示（grab 但 panel 不亮）：停所有子畫面(primary) + 選中主畫面(secondary)
                foreach (var c in _cams) if (c != null) c.SetPrimaryDisplayVisible(false);
                if (_selectedCam >= 0 && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                    _cams[_selectedCam].SetSecondaryDisplay(IntPtr.Zero);

                // 1. 確保 grab（沒在 grab 就自動啟動，CLProtocol 才會背景啟用）
                if (!_userWantsGrab)
                {
                    _userWantsGrab = true;
                    foreach (var c in _cams) if (c != null) c.SetUserGrabIntent(true);
                    btnGrab.Text = "停止抓取";
                }

                // 2. 等 CLProtocol 就緒（連線的相機都 enabled；最多 ~12 秒），UI 不凍
                btnFetchInfo.Text = "等待 CLProtocol…";
                bool ready = false;
                for (int t = 0; t < 24 && !ready; t++)
                {
                    await System.Threading.Tasks.Task.Delay(500);
                    ready = true;
                    foreach (var c in _cams)
                        if (c != null && c.IsConnected && !c.IsClProtocolEnabled) { ready = false; break; }
                }

                // 3. 抓上限 → 設線掃 slider 上限範圍(Maximum) = 硬體上限（方案 A，當前值維持）+ 重算曝光上限
                int applied = 0, allMax = LrMin;
                for (int i = 0; i < _cams.Length; i++)
                {
                    var cam = _cams[i];
                    if (cam == null) continue;
                    double max = cam.GetLineRateMaxHz();
                    if (max <= 0) continue;

                    int maxHz = ClampInt((int)Math.Floor(max), LrMin, LrMax, LrMax);
                    _suppressParamEvents = true;
                    try
                    {
                        if (_tbLineRate != null && i < _tbLineRate.Length && _tbLineRate[i] != null)
                        {
                            _tbLineRate[i].Maximum = maxHz;
                            if (_tbLineRate[i].Value > maxHz) _tbLineRate[i].Value = maxHz;
                        }
                        if (_nudLineRate != null && i < _nudLineRate.Length && _nudLineRate[i] != null)
                        {
                            _nudLineRate[i].Maximum = maxHz;
                            if (_nudLineRate[i].Value > maxHz) _nudLineRate[i].Value = maxHz;
                        }
                    }
                    finally { _suppressParamEvents = false; }

                    int curLr = (_tbLineRate != null && i < _tbLineRate.Length && _tbLineRate[i] != null) ? _tbLineRate[i].Value : maxHz;
                    ApplyExposureMax(i, curLr); // 曝光上限依當前線掃值重算
                    if (maxHz > allMax) allMax = maxHz;
                    applied++;
                }

                // 全部相機線掃列上限也放大到最大值
                if (applied > 0)
                {
                    _suppressParamEvents = true;
                    try
                    {
                        if (_tbLrAll != null) { _tbLrAll.Maximum = allMax; if (_tbLrAll.Value > allMax) _tbLrAll.Value = allMax; }
                        if (_nudLrAll != null) { _nudLrAll.Maximum = allMax; if (_nudLrAll.Value > allMax) _nudLrAll.Value = allMax; }
                    }
                    finally { _suppressParamEvents = false; }
                }

                UpdateCamerasListView();

                // 抓取相機資訊只為取得上限，不需持續取像顯示 → 抓完停 grab（panel 不顯示影像）
                _userWantsGrab = false;
                foreach (var c in _cams) if (c != null) c.SetUserGrabIntent(false);
                btnGrab.Text = "開始抓取";

                // 清掉停 grab 後殘留的最後一幀 → 恢復顯示綁定 → panel 顯示黑（不顯示殘影）
                foreach (var c in _cams) if (c != null) { c.ClearDisplay(); c.SetPrimaryDisplayVisible(true); }
                if (_selectedCam >= 0 && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                    _cams[_selectedCam].SetSecondaryDisplay(panelMain.Handle);
            }
            finally
            {
                btnFetchInfo.Text = origText;
                btnFetchInfo.Enabled = true;
            }
        }

        // =========================================================================
        // btnGrab: 開始抓取 ↔ 停止抓取（所有相機 SetUserGrabIntent）
        // =========================================================================
        private void btnGrab_Click(object sender, EventArgs e)
        {
            if (_cams == null) return;

            _userWantsGrab = !_userWantsGrab;
            foreach (var cam in _cams)
                cam?.SetUserGrabIntent(_userWantsGrab);

            btnGrab.Text = _userWantsGrab ? "停止抓取" : "開始抓取";
        }

        // =========================================================================
        // btnRelease: 釋放（停 timer + 旗標 → 所有 Dispose → MappFreeDefault）
        // =========================================================================
        private void btnRelease_Click(object sender, EventArgs e)
        {
            ReleaseAll();
            UpdateButtonsEnabled(initialized: false);
            ClearSysInfo();
        }

        private void ReleaseAll()
        {
            _isReleasing = true;
            _statusTimer.Stop();

            // 1. 所有 MilCamera Dispose（會 stop grab + free digitizer/display/buffer）
            if (_cams != null)
            {
                for (int i = 0; i < _cams.Length; i++)
                {
                    try { _cams[i]?.Dispose(); } catch { /* 釋放期間忽略 */ }
                    _cams[i] = null;
                }
                _cams = null;
            }

            // 2. 每張卡 MsysFree（與 MsysAlloc 對稱）
            foreach (var sysId in _systems.Values)
            {
                try { if (sysId != MIL.M_NULL) MIL.MsysFree(sysId); } catch { /* 釋放期間忽略 */ }
            }
            _systems.Clear();

            // 3. App context 釋放
            if (_milApplication != MIL.M_NULL)
            {
                MIL.MappFreeDefault(_milApplication, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
                _milApplication = MIL.M_NULL;
            }

            // 4. 釋放後參數控制項全部 disable（Designer 控制項保留，不 Dispose）
            DisableAllParamControls();

            _selectedCam = -1;
            _userWantsGrab = false;
            btnGrab.Text = "開始抓取";

            // 子畫面 status 全部回到 No Camera（深底淺灰字）+ 重畫邊框
            for (int i = 0; i < SubPanelCount; i++)
                SetSubLabel(i, "No Camera", Color.FromArgb(32, 32, 32), Color.LightGray);
            if (_camContainers != null)
                foreach (var c in _camContainers)
                    c?.Invalidate();
        }

        // =========================================================================
        // 選擇相機：舊選中解除副顯示、新選中接到主畫面、同步參數 UI
        // =========================================================================
        private void SelectCamera(int idx)
        {
            if (_cams == null || idx < 0 || idx >= _cams.Length || _cams[idx] == null) return;

            // 舊選中：解除副顯示（回到只在自己的子 panel）
            if (_selectedCam >= 0 && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                _cams[_selectedCam].SetSecondaryDisplay(IntPtr.Zero);

            _selectedCam = idx;

            // 新選中：把這台顯示到主畫面
            _cams[idx].SetSecondaryDisplay(panelMain.Handle);

            // 重畫所有容器邊框（選中橘色、其他深灰）
            if (_camContainers != null)
                foreach (var c in _camContainers)
                    c?.Invalidate();

            // 更新「選中相機」常數列
            UpdateEngineSelectedCam();
        }

        // =========================================================================
        // 參數 Tab（Designer 固定 8 相機）：每個 tab（曝光/線掃/高度）有「全部相機」列 + Cam1~8 列，
        //   控制項本體在 Designer.cs 逐一宣告；本檔負責接線（雙向同步 + 套用 _cams）與初值/enable。
        //   - 每相機列：trackBar.Scroll / nud.ValueChanged → 雙向同步（_suppressParamEvents 防遞迴）→ 套用 _cams[i]。
        //   - 全部相機列：→ 套用全部 _cams + 同步各 Cam 列顯示。
        //   - InitParamControls：btnInit 後依相機數填初值 + enable（第 i 列只有 i<camCount 且 _cams[i]!=null 才 enable）。
        //   控制項陣列索引 0..7 = 子畫面 / _cams 索引。
        // =========================================================================

        private const int ExpMin = 1, ExpMaxCap = 10000;       // 曝光 μs（ExpMaxCap = 絕對上限；動態上限走 CalcExpMax）
        private const int LrMin = 100, LrMax = 100000;          // 線掃 Hz
        private const int HtMin = 1, HtMax = 10000;             // 高度 px

        private enum ParamKind { Exposure, LineRate, Height }

        /// <summary>
        /// 依線掃速率算曝光動態上限（仿主程式）：lrHz≤0 → 絕對上限 ExpMaxCap；
        /// 否則 clamp(floor(900000 / lrHz), ExpMin, ExpMaxCap)。
        /// </summary>
        private int CalcExpMax(int lrHz) => MilCameraParams.CalcExposureMaxUs(lrHz, ExpMin, ExpMaxCap); // 公式單一真相在 MilGrabber.Core

        /// <summary>
        /// 依新線掃值重算第 i 台曝光 slider/NUD 的 Maximum = CalcExpMax(lrHz)；
        /// 當前曝光值 &gt; 新 Maximum 時夾到 Maximum（並套用到相機）。包 _suppressParamEvents 防遞迴。
        /// </summary>
        private void ApplyExposureMax(int camIdx, int lrHz)
        {
            if (_tbExposure == null || camIdx < 0 || camIdx >= _tbExposure.Length) return;
            TrackBar tb = _tbExposure[camIdx];
            NumericUpDown nud = _nudExposure[camIdx];
            if (tb == null || nud == null) return;

            int expMax = Math.Max(ExpMin, CalcExpMax(lrHz));
            bool overflow = tb.Value > expMax;

            _suppressParamEvents = true;
            try
            {
                tb.Maximum = expMax;
                nud.Maximum = expMax;
                if (overflow)
                    SetRowValue(tb, nud, expMax); // 當前值超出新上限 → 夾到上限
            }
            finally { _suppressParamEvents = false; }

            if (overflow)
                ApplyParam(camIdx, ParamKind.Exposure, expMax); // 夾緊後實際套用到相機

            SyncExpAllMax(); // 全部相機曝光列上限 = 各台上限最小值（全部都不超過）
        }

        /// <summary>全部相機曝光列(trackBarExpAll/nudExpAll)上限 = 各 active 台曝光上限的最小值；當前值超出則夾緊。</summary>
        private void SyncExpAllMax()
        {
            if (_tbExpAll == null || _nudExpAll == null || _tbExposure == null) return;
            int m = ExpMaxCap; bool any = false;
            for (int i = 0; i < _tbExposure.Length; i++)
                if (_panelExp != null && i < _panelExp.Length && _panelExp[i] != null && _panelExp[i].Enabled)
                    { m = Math.Min(m, _tbExposure[i].Maximum); any = true; }
            m = any ? Math.Max(ExpMin, m) : ExpMaxCap;
            _suppressParamEvents = true;
            try
            {
                _tbExpAll.Maximum = m; if (_tbExpAll.Value > m) _tbExpAll.Value = m;
                _nudExpAll.Maximum = m; if (_nudExpAll.Value > m) _nudExpAll.Value = m;
            }
            finally { _suppressParamEvents = false; }
        }

        /// <summary>建構式呼叫一次：把每相機列與全部相機列的事件接到對應 setter（控制項由 Designer 宣告）。</summary>
        private void WireParamControls()
        {
            for (int i = 0; i < SubPanelCount; i++)
            {
                WireParamRow(_tbExposure[i], _nudExposure[i], i, ParamKind.Exposure);
                WireParamRow(_tbLineRate[i], _nudLineRate[i], i, ParamKind.LineRate);
                WireParamRow(_tbHeight[i], _nudHeight[i], i, ParamKind.Height);
            }

            WireAllRow(_tbExpAll, _nudExpAll, ParamKind.Exposure);
            WireAllRow(_tbLrAll, _nudLrAll, ParamKind.LineRate);
            WireAllRow(_tbHtAll, _nudHtAll, ParamKind.Height);
        }

        /// <summary>單一相機列接線：trackBar↔NUD 雙向同步（_suppressParamEvents 防遞迴），值變更套用 _cams[camIdx]。
        /// 拖曳中只更新 NUD 顯示 + 曝光上限視覺，放掉(MouseUp)才寫硬體；鍵盤/點軌道/NUD 即時寫（仿 AniloxRoll.Monitor）。</summary>
        private void WireParamRow(TrackBar tb, NumericUpDown nud, int camIdx, ParamKind kind)
        {
            tb.MouseDown += (s, e) => _dragging.Add(tb);
            tb.MouseUp += (s, e) =>
            {
                if (!_dragging.Remove(tb)) return;
                ApplyParam(camIdx, kind, tb.Value);                                   // 拖曳放掉 → 立即寫硬體
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, tb.Value);
            };

            tb.Scroll += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { nud.Value = ClampDecimal(tb.Value, nud.Minimum, nud.Maximum); } // NUD 顯示即時跟動
                finally { _suppressParamEvents = false; }
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, tb.Value);   // 曝光上限視覺即時跟動
                if (!_dragging.Contains(tb)) ApplyParam(camIdx, kind, tb.Value);      // 拖曳中不寫硬體；鍵盤/點軌道即時寫
            };

            nud.ValueChanged += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { tb.Value = ClampInt((int)nud.Value, tb.Minimum, tb.Maximum, tb.Minimum); }
                finally { _suppressParamEvents = false; }
                ApplyParam(camIdx, kind, (double)nud.Value);
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, (int)nud.Value); // 線掃變更 → 重算該台曝光上限
            };
        }

        /// <summary>全部相機列接線：套用到全部 _cams + 同步各 Cam 列顯示（_suppressParamEvents 防遞迴）。
        /// 拖曳中只更新 NUD 顯示 + 曝光上限視覺，放掉(MouseUp)才寫全部硬體；鍵盤/點軌道/NUD 即時寫（仿 AniloxRoll.Monitor）。</summary>
        private void WireAllRow(TrackBar tb, NumericUpDown nud, ParamKind kind)
        {
            tb.MouseDown += (s, e) => _dragging.Add(tb);
            tb.MouseUp += (s, e) =>
            {
                if (!_dragging.Remove(tb)) return;
                ApplyParamAll(kind, tb.Value);                                  // 拖曳放掉 → 立即寫全部硬體
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll(tb.Value);
            };

            tb.Scroll += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { nud.Value = ClampDecimal(tb.Value, nud.Minimum, nud.Maximum); } // NUD 顯示即時跟動
                finally { _suppressParamEvents = false; }
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll(tb.Value);   // 曝光上限視覺即時跟動
                if (!_dragging.Contains(tb)) ApplyParamAll(kind, tb.Value);      // 拖曳中不寫硬體；鍵盤/點軌道即時寫
            };

            nud.ValueChanged += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { tb.Value = ClampInt((int)nud.Value, tb.Minimum, tb.Maximum, tb.Minimum); }
                finally { _suppressParamEvents = false; }
                ApplyParamAll(kind, (double)nud.Value);
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll((int)nud.Value); // 全部相機線掃 → 全部重算曝光上限
            };
        }

        /// <summary>全部相機列線掃變更：對所有相機重算曝光上限（仿主程式）。</summary>
        private void ApplyExposureMaxAll(int lrHz)
        {
            if (_cams == null) return;
            for (int i = 0; i < _cams.Length; i++)
            {
                if (_cams[i] == null) continue;
                ApplyExposureMax(i, lrHz);
            }
        }

        // 固定初值（btnInit 即套到硬體）：曝光 100 μs、線掃 3000 Hz、高度 3000 px。
        private const int ExpInitDefault = 100, LrInitDefault = 3000, HtInitDefault = 3000;

        /// <summary>
        /// btnInit 建相機後：依實際相機數填固定初值 + enable/disable。
        /// 第 i 列若 i&lt;camCount 且 _cams[i]!=null → enable + 套固定預設到硬體（曝光 100、線掃 3000、高度 3000）；
        /// 曝光 slider/NUD Maximum = CalcExpMax(線掃預設)，value = 曝光預設。否則整列 disable（灰掉）。全部相機列：有相機才 enable。
        /// </summary>
        private void InitParamControls(int camCount)
        {
            _suppressParamEvents = true;   // 設初值期間抑制事件回灌相機
            try
            {
                for (int i = 0; i < SubPanelCount; i++)
                {
                    MilCamera cam = (_cams != null && i < _cams.Length) ? _cams[i] : null;
                    bool active = i < camCount && cam != null;

                    if (active)
                    {
                        int camId = (_devices != null && i < _devices.Count) ? _devices[i].Id : i + 1;
                        _lblExpLabel(i).Text = $"CAM{camId}";
                        _lblLrLabel(i).Text = $"CAM{camId}";
                        _lblHtLabel(i).Text = $"CAM{camId}";

                        // 固定預設值（不再讀相機回值）
                        int expInit = ExpInitDefault;
                        int lrInit = LrInitDefault;
                        int htInit = HtInitDefault;

                        // 曝光動態上限 = CalcExpMax(線掃預設)（= 300）；value = 曝光預設（= 100）
                        int expMax = Math.Max(ExpMin, CalcExpMax(lrInit));
                        _tbExposure[i].Maximum = expMax;
                        _nudExposure[i].Maximum = expMax;

                        SetRowValue(_tbExposure[i], _nudExposure[i], expInit);
                        SetRowValue(_tbLineRate[i], _nudLineRate[i], lrInit);
                        SetRowValue(_tbHeight[i], _nudHeight[i], htInit);

                        // init 即把固定預設套到硬體
                        cam.SetExposureUs(expInit);
                        cam.SetLineRateHz(lrInit);
                        cam.SetGrabHeight(htInit);
                    }

                    _panelExp[i].Enabled = active;
                    _panelLr[i].Enabled = active;
                    _panelHt[i].Enabled = active;
                }

                bool any = camCount > 0;
                panelExpAll.Enabled = any;
                panelLrAll.Enabled = any;
                panelHtAll.Enabled = any;
            }
            finally { _suppressParamEvents = false; }

            SyncExpAllMax(); // 全部相機曝光列上限 = 各 active 台上限最小值
        }

        /// <summary>釋放後：所有參數控制項列 disable（Designer 控制項保留，不 Dispose）。</summary>
        private void DisableAllParamControls()
        {
            if (_panelExp == null) return;
            for (int i = 0; i < SubPanelCount; i++)
            {
                _panelExp[i].Enabled = false;
                _panelLr[i].Enabled = false;
                _panelHt[i].Enabled = false;
            }
            panelExpAll.Enabled = false;
            panelLrAll.Enabled = false;
            panelHtAll.Enabled = false;
        }

        // 取各 tab 第 i 列的 CAM 標籤（Designer 具名，索引 0..7）。
        private Label _lblExpLabel(int i) => new[] { lblExpCam1, lblExpCam2, lblExpCam3, lblExpCam4, lblExpCam5, lblExpCam6, lblExpCam7, lblExpCam8 }[i];
        private Label _lblLrLabel(int i) => new[] { lblLrCam1, lblLrCam2, lblLrCam3, lblLrCam4, lblLrCam5, lblLrCam6, lblLrCam7, lblLrCam8 }[i];
        private Label _lblHtLabel(int i) => new[] { lblHtCam1, lblHtCam2, lblHtCam3, lblHtCam4, lblHtCam5, lblHtCam6, lblHtCam7, lblHtCam8 }[i];

        /// <summary>同步設 trackBar + NUD 值（夾在各自範圍內）；呼叫端需先設 _suppressParamEvents。</summary>
        private static void SetRowValue(TrackBar tb, NumericUpDown nud, int value)
        {
            tb.Value = ClampInt(value, tb.Minimum, tb.Maximum, tb.Minimum);
            nud.Value = ClampDecimal(value, nud.Minimum, nud.Maximum);
        }

        /// <summary>「全部相機」套用：對每台相機 ApplyParam，並同步該 kind 對應的每相機列控制項顯示。</summary>
        private void ApplyParamAll(ParamKind kind, double value)
        {
            if (_cams == null) return;

            // 依 kind 選對應的每相機列控制項陣列
            TrackBar[] tbs;
            NumericUpDown[] nuds;
            switch (kind)
            {
                case ParamKind.Exposure: tbs = _tbExposure; nuds = _nudExposure; break;
                case ParamKind.LineRate: tbs = _tbLineRate; nuds = _nudLineRate; break;
                case ParamKind.Height: tbs = _tbHeight; nuds = _nudHeight; break;
                default: return;
            }

            for (int i = 0; i < _cams.Length; i++)
            {
                ApplyParam(i, kind, value);

                // 同步該相機列的 trackBar/NUD 顯示（包 _suppressParamEvents 防遞迴觸發各列事件）。
                _suppressParamEvents = true;
                try
                {
                    if (tbs != null && i < tbs.Length && tbs[i] != null)
                        tbs[i].Value = ClampInt((int)Math.Round(value), tbs[i].Minimum, tbs[i].Maximum, tbs[i].Minimum);
                    if (nuds != null && i < nuds.Length && nuds[i] != null)
                        nuds[i].Value = ClampDecimal((int)Math.Round(value), nuds[i].Minimum, nuds[i].Maximum);
                }
                finally { _suppressParamEvents = false; }
            }
        }

        /// <summary>把參數值套用到對應相機（依 ParamKind 呼 SetExposureUs / SetLineRateHz / SetGrabHeight）。</summary>
        private void ApplyParam(int camIdx, ParamKind kind, double value)
        {
            MilCamera cam = (_cams != null && camIdx >= 0 && camIdx < _cams.Length) ? _cams[camIdx] : null;
            if (cam == null) return;
            switch (kind)
            {
                case ParamKind.Exposure: cam.SetExposureUs(value); break;
                case ParamKind.LineRate: cam.SetLineRateHz(value); break;
                case ParamKind.Height: cam.SetGrabHeight((int)Math.Round(value)); break;
            }
        }

        private static int ClampInt(int value, int min, int max, int fallbackWhenNonPositive)
        {
            if (value <= 0) value = fallbackWhenNonPositive;
            if (value < min) value = min;
            if (value > max) value = max;
            return value;
        }

        private static decimal ClampDecimal(int value, decimal min, decimal max)
        {
            decimal d = value;
            if (d < min) d = min;
            if (d > max) d = max;
            return d;
        }


        // =========================================================================
        // Timer（500ms）：更新子畫面狀態 label + 選中相機 telemetry
        // tick 在 UI 執行緒，故可直接更新 UI
        // =========================================================================
        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            if (_isReleasing || _cams == null) return;

            // 子畫面狀態：Online 綠 / Offline 紅
            for (int i = 0; i < _cams.Length; i++)
            {
                var cam = _cams[i];
                if (cam == null) continue;
                bool online = cam.CheckPresence();
                int camId = CamIdForIndex(i);
                if (online)
                    SetSubLabel(i, $"Cam {camId}: Online", Color.LightGreen, Color.Black);
                else
                    SetSubLabel(i, $"Cam {camId}: Offline", Color.Red, Color.White);
            }

            // 每 tick 更新所有相機列（17 欄 telemetry）
            UpdateCamerasListView();
            // 選中相機那列（動態）
            UpdateEngineSelectedCam();
        }

        // =========================================================================
        // lvCameras：17 欄「所有相機資訊」表（每相機一列）
        // 欄位與資料來源照 LiveTelemetryPresenter；新增「Max Line Rate(Hz)」欄（CLProtocol 上限，只顯示不自動套用）。
        // =========================================================================

        /// <summary>
        /// 依實際建立的相機數新增初始列（每列 Tag = 相機 index，各欄填 "N/A"）。
        /// 未建立的相機不加列。在 btnInit 建相機後呼叫一次。
        /// </summary>
        private void InitCamerasListView(int camCount)
        {
            lvCameras.BeginUpdate();
            try
            {
                lvCameras.Items.Clear();
                for (int i = 0; i < camCount; i++)
                {
                    // 顯示文字用 config 的 CameraId；Tag 用子畫面 index（= _cams 索引，UpdateCamerasListView 依此查相機）
                    var item = new ListViewItem($"CAM {CamIdForIndex(i)}");
                    for (int c = 1; c < 17; c++) item.SubItems.Add("N/A"); // 欄 1~16
                    item.Tag = i;
                    lvCameras.Items.Add(item);
                }
            }
            finally
            {
                lvCameras.EndUpdate();
            }
        }

        /// <summary>
        /// Timer（500ms）更新所有相機列。相機未建立/釋放後該列顯示 "-"。
        /// 17 欄索引：0 Camera / 1 FPS / 2 Target FPS / 3 Line Rate / 4 Max Line Rate /
        /// 5 Exp Set / 6 Exp Meas / 7 Frames / 8 Missed / 9 Grab Miss / 10 Resolution /
        /// 11 Scan Mode / 12 FPGA / 13 Cam Temp / 14 Mem Free / 15 PCIe Lanes / 16 PCIe Speed。
        /// Max Line Rate(欄 4)：cam.GetLineRateMaxHz()，&gt;0 顯示整數，否則 "-"（CLProtocol 未就緒即 "-"）。
        /// </summary>
        private void UpdateCamerasListView()
        {
            if (lvCameras.Items.Count == 0) return;

            lvCameras.BeginUpdate();
            try
            {
                foreach (ListViewItem item in lvCameras.Items)
                {
                    int idx = (int)item.Tag;
                    MilCamera cam = (_cams != null && idx >= 0 && idx < _cams.Length) ? _cams[idx] : null;

                    if (cam == null)
                    {
                        for (int c = 1; c <= 16; c++) item.SubItems[c].Text = "-";
                        continue;
                    }

                    item.SubItems[1].Text = $"{cam.CurrentFps:F2}";
                    item.SubItems[2].Text = $"{cam.GetSelectedFrameRate():F2}";

                    double lineRate = cam.GetLineRateHz();
                    item.SubItems[3].Text = lineRate > 0 ? $"{lineRate:F1}" : "-";

                    // Max Line Rate(Hz)：CLProtocol 上限；>0 顯示整數，未就緒/抓不到顯示 "-"。
                    double lineRateMax = cam.GetLineRateMaxHz();
                    item.SubItems[4].Text = lineRateMax > 0 ? $"{(long)Math.Floor(lineRateMax)}" : "-";

                    double expUs = cam.GetExposureUs();
                    item.SubItems[5].Text = expUs > 0 ? $"{expUs:F1}" : "-";

                    double measUs = cam.GetMeasuredExposureUs();
                    item.SubItems[6].Text = measUs > 0 ? $"{measUs:F1}" : "-";

                    item.SubItems[7].Text = $"{cam.GetFrameCount()}";
                    item.SubItems[8].Text = $"{cam.GetFrameMissed()}";
                    item.SubItems[9].Text = $"{cam.GetGrabFrameMissed()}";
                    item.SubItems[10].Text = $"{cam.FrameWidth}×{cam.FrameHeight}";
                    item.SubItems[11].Text = cam.GetScanMode();

                    double fpgaTemp = cam.GetFpgaTemperature();
                    item.SubItems[12].Text = double.IsNaN(fpgaTemp) ? "-" : $"{fpgaTemp:F1}";

                    double camTemp = cam.GetCameraTemperature();
                    item.SubItems[13].Text = double.IsNaN(camTemp) ? "-" : $"{camTemp:F1}";

                    long memFree = cam.GetMemoryFreeMB();
                    item.SubItems[14].Text = memFree >= 0 ? $"{memFree}" : "-";

                    int lanes = cam.GetPcieNumberOfLanes();
                    item.SubItems[15].Text = lanes >= 0 ? $"{lanes}" : "-";

                    item.SubItems[16].Text = cam.GetPcieSpeed();
                }
            }
            finally
            {
                lvCameras.EndUpdate();
            }
        }

        /// <summary>所有相機列欄 1~16 重置為 "-"（相機釋放後呼叫）。</summary>
        private void ResetCamerasListView()
        {
            foreach (ListViewItem item in lvCameras.Items)
                for (int c = 1; c <= 16; c++) item.SubItems[c].Text = "-";
        }

        // 維持原 btnRelease 呼叫名稱：釋放後把相機資訊表全部重置為 "-"。
        private void ClearSysInfo()
        {
            ResetCamerasListView();
        }

        // =========================================================================
        // lvEngine：取像/系統常數（非檢測引擎；本範例無檢測）
        // 多為靜態，初始化填一次；「選中相機」那列由 SelectCamera/Timer 動態更新。
        // =========================================================================

        private const int EngineSelectedCamRow = 6; // 「選中相機」列索引（動態更新）

        private void InitEngineListView(int camCount)
        {
            lvEngine.BeginUpdate();
            try
            {
                lvEngine.Items.Clear();
                // Config 來源：讀到幾台 + 是否 fallback（DevNum 分布 SystemNum 0/1）
                string cfgSource = _usedFallbackConfig ? "fallback 預設" : "system-settings.json";
                AddEngineRow("Config", $"{_devices?.Count ?? 0} 台（{cfgSource}）");
                AddEngineRow("相機數", camCount.ToString());
                AddEngineRow("擷取卡(System)", _systems.Count.ToString());
                AddEngineRow("Grab Buffer", "2");          // MilCamera 內部雙 buffer（固定）
                AddEngineRow("Telemetry Interval", $"{_statusTimer.Interval} ms");
                AddEngineRow("MIL Version", GetMilVersion());
                AddEngineRow("選中相機", "-");             // index EngineSelectedCamRow，動態
            }
            finally
            {
                lvEngine.EndUpdate();
            }
        }

        private void AddEngineRow(string param, string value)
        {
            var item = new ListViewItem(param);
            item.SubItems.Add(value);
            lvEngine.Items.Add(item);
        }

        private void UpdateEngineSelectedCam()
        {
            if (lvEngine.Items.Count <= EngineSelectedCamRow) return;
            int camId = (_selectedCam >= 0 && _devices != null && _selectedCam < _devices.Count)
                ? _devices[_selectedCam].Id : _selectedCam;
            string text = (_selectedCam >= 0) ? $"CAM {camId}" : "-";
            lvEngine.Items[EngineSelectedCamRow].SubItems[1].Text = text;
        }

        /// <summary>取 MIL 版本（M_VERSION）；取不到回 "MIL.NET"。</summary>
        private static string GetMilVersion()
        {
            try
            {
                var sb = new System.Text.StringBuilder(256);
                MIL.MappInquire(MIL.M_DEFAULT, MIL.M_VERSION, sb);
                string v = sb.ToString();
                return string.IsNullOrWhiteSpace(v) ? "MIL.NET" : v;
            }
            catch { return "MIL.NET"; }
        }

        // =========================================================================
        // 子畫面狀態 label 更新（容器內 runtime 建的 status；去重避免重繪閃爍）
        // =========================================================================
        private void SetSubLabel(int idx, string text, Color back, Color fore)
        {
            if (_statusLabels == null || idx < 0 || idx >= _statusLabels.Length) return;
            var l = _statusLabels[idx];
            if (l == null) return;
            if (l.Text != text) l.Text = text;
            if (l.BackColor != back) l.BackColor = back;
            if (l.ForeColor != fore) l.ForeColor = fore;
        }

        private void UpdateButtonsEnabled(bool initialized)
        {
            btnInit.Enabled = !initialized;
            btnGrab.Enabled = initialized;
            btnRelease.Enabled = initialized;
            btnFetchInfo.Enabled = initialized; // 已初始化才可手動抓相機資訊
            tabParams.Enabled = initialized;
        }

        /// <summary>子畫面 index → config 的 CameraId（無 config 時退回 index）。</summary>
        private int CamIdForIndex(int index)
        {
            return (_devices != null && index >= 0 && index < _devices.Count) ? _devices[index].Id : index;
        }

        // =========================================================================
        // Form 關閉：安全清理
        // =========================================================================
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            ReleaseAll();
            base.OnFormClosing(e);
        }
    }
}
