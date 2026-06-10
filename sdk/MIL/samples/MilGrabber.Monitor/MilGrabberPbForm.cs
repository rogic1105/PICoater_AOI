using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Web.Script.Serialization; // JavaScriptSerializer 解析 system-settings.json
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary; // MApp 仍由本範例管理
using MilGrabber.Core;             // 已封裝的單相機 MIL library
using TanukiCv.Controls;           // ThumbView（共用無閃縮圖控制項）

namespace MilGrabber.Monitor
{
    /// <summary>
    /// 多相機即時監控 UI 範例。
    /// 取像/顯示細節全部委派給 <see cref="MilCamera"/>（sdk/MIL/MilGrabber.Core）；
    /// 本 Form 只負責 MApp 生命週期、UI 佈局、相機選擇與 telemetry 顯示。
    /// MilCamera 內建 grab hook：未訂閱 FrameReady 時自動把原圖顯示到 primary panel，
    /// 故本純顯示範例不訂閱 FrameReady。
    /// </summary>
    public partial class MilGrabberPbForm : Form
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

        public MilGrabberPbForm()
        {
            InitializeComponent();

            // 容器陣列：在 Form 建構式從 Designer 具名控制項 panelCam0..7 組成（陣列在 .cs 組，不在 Designer）。
            _camContainers = new Panel[] {
                panelCam0, panelCam1, panelCam2, panelCam3,
                panelCam4, panelCam5, panelCam6, panelCam7 };
            _statusLabels  = new Label[SubPanelCount];

            // 每個容器 runtime 建內部 status + 邊框（仿主程式 SetupLivePanel）。
            // 一律先建好（即使未綁相機），未綁的 status 顯示 "No Camera"。
            for (int i = 0; i < SubPanelCount; i++)
                SetupCamPanel(_camContainers[i], i);

            SetupPbMain(); // 建共用 LiveDisplayView（主畫面 SmartCanvas + 各容器疊縮圖；含合圖/LOD）
            SetupMergeTab(); // tabParams 新增「合圖」tab：ops/start 表格 + 重疊演算法選擇

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
            // 注意：不對「容器」開雙緩衝。對容器（有子控制項）開雙緩衝是經典 WinForms 陷阱：
            // 容器雙緩衝與子控制項各自繪製對不上 → 反而讓子控制項閃。葉子 ThumbView 自己已雙緩衝（正解）。
            // 對齊主程式 SetupLivePanel（不對 parentPanel 開雙緩衝 → camLive 不閃）。

            // 縮圖 ThumbView 由共用 ThumbStrip 統一建（ctor 後一次建好）；此處只建 status + 邊框。
            var status = new Label
            {
                Dock = DockStyle.Bottom,
                Height = 18,
                ForeColor = Color.LightGray,
                BackColor = Color.FromArgb(32, 32, 32),
                TextAlign = ContentAlignment.MiddleCenter,
                Text = "No Camera"
            };

            status.MouseClick += (s, e) => SelectCamera(idx);
            container.Paint += (s, e) => OnCamPanelPaint(s, e, idx);

            container.Controls.Add(status);

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

            // 依「初始化前選的模式」建立顯示控制項（MIL→Panel 直繪 / PictureBox→現成 PictureBox+SmartCanvas）
            CreateDisplaysForMode();

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

                // 顯示目標 handle：MIL 模式 = 該容器的 Panel（MIL 直繪）；PictureBox 模式 = PictureBox（之後 detach 純用 FrameReady 畫）。
                // 不可傳 IntPtr.Zero，否則 MIL 會自己開顯示視窗（每台一個彈窗）。
                // PictureBox 模式：MIL 主顯示稍後 detach（純用 FrameReady→ThumbStrip 畫），h 只需任一有效視窗 handle → 用容器。
                IntPtr h = _milMode ? _displayPanels[i].Handle : _camContainers[i].Handle;
                var cam = new MilCamera(sysId, dev.Id, dev.DevNum, dev.DcfPath, h);
                int idx = i; // 迴圈變數捕捉（綁子畫面索引，與 _cams / lvCameras Tag 一致）
                cam.OnCameraClicked += _ => SelectCamera(idx);
                if (!_milMode)
                    cam.FrameReady += (c, buf) => OnCameraFrame(idx, c, buf); // PictureBox 模式：每幀縮圖→繪
                cam.Initialize();
                if (!_milMode)
                    cam.SetPrimaryDisplayVisible(false); // PictureBox 模式 detach MIL；MIL 模式保留 MIL 直繪
                _cams[i] = cam;
            }

            // CLProtocol 背景啟用：相機都分配完成後，只對在線相機啟用。線掃/曝光是 CLProtocol(GenICam) feature，
            // 不啟用就只記錄不寫硬體 → 取樣頻率/曝光調了沒反應。非阻塞（內部 Task.Run），~2-5s 後自動套已記錄參數。
            foreach (var c in _cams)
                if (c != null && c.CheckPresence()) c.BeginCLProtocolInit();

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
            // 主畫面/合圖刷新由 LiveDisplayView 內部 timer 自管（不需這裡的 _displayTimer）
            btnGrab.Text = "開始抓取";
            UpdateButtonsEnabled(initialized: true);

            // 自動跑一次 FetchInfo（等 CLProtocol 就緒 + 套線掃到所有相機）→ 相機進入受控同頻線掃 → 取像同步，
            // 省去手動按 btnFetchInfo（實測：先按 FetchInfo 就不會偶發不同步）。async void，背景跑不凍 UI。
            if (camCount > 0)
                btnFetchInfo_Click(null, null);
        }

        // ===== Config 載入（LoadDeviceConfig / CreateFallbackDevices / MapDescriptor）→ MilGrabberPbForm.Config.cs =====

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
                // PictureBox 版不用 MIL display（已 detach），顯示全走 FrameReady → PictureBox/SmartCanvas，不需開關。

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

                UpdateCamerasListView(slow: true); // 一次性完整刷（含慢欄位）

                // 抓取相機資訊只為取得上限，不需持續取像顯示 → 抓完停 grab（panel 不顯示影像）
                _userWantsGrab = false;
                foreach (var c in _cams) if (c != null) c.SetUserGrabIntent(false);
                btnGrab.Text = "開始抓取";

                // PictureBox 版：停 grab 後 FrameReady 自然停止，畫面留最後一幀；不需 MIL display 還原。
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
        // chkFlipVertical: 上下翻轉顯示（線掃相機由下往上拍 → 顯示需反轉）。
        // 即時套用到所有相機（MilCamera 的 grab hook 用 MimFlip 翻轉到 display buffer）。
        // =========================================================================
        private void chkFlipVertical_CheckedChanged(object sender, EventArgs e)
        {
            _flipDisplay = chkFlipVertical.Checked; // PictureBox 模式：LiveDisplayView 主畫面+縮圖一起翻
            if (_live != null) _live.FlipVertical = _flipDisplay;
            if (_cams == null) return;
            foreach (var cam in _cams)
                if (cam != null) cam.FlipVertical = chkFlipVertical.Checked; // MIL 模式：MIL 翻轉
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

            // 0. 先所有 MilCamera Dispose（stop grab）→ 確保不再有 FrameReady→OnCameraFrame 在背景跑。
            // **順序關鍵**：必須先停相機，才能釋放它正在讀寫的 pinned 緩衝（步驟 1）。反過來會 use-after-free：
            // OnCameraFrame（背景）讀到已釋放的 _dstPinned → AccessViolationException（關窗崩潰）。
            if (_cams != null)
            {
                for (int i = 0; i < _cams.Length; i++)
                {
                    try { _cams[i]?.Dispose(); } catch { /* 釋放期間忽略 */ }
                    _cams[i] = null;
                }
                _cams = null;
            }

            // 1. 釋放 PictureBox 縮圖緩衝（pinned）+ 清縮圖（此時 grab 已停，無背景 callback → 安全）
            ReleasePictureBoxDisplays();

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

            if (_milMode)
            {
                // MIL 模式：舊選中解除副顯示、新選中接主畫面（MIL secondary 直繪到 _mainPanelMil）
                if (_selectedCam >= 0 && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                    _cams[_selectedCam].SetSecondaryDisplay(IntPtr.Zero);
                _selectedCam = idx;
                if (_mainPanelMil != null) _cams[idx].SetSecondaryDisplay(_mainPanelMil.Handle);
            }
            else
            {
                // PictureBox 模式：主畫面由 LiveDisplayView 依選中相機顯示
                _selectedCam = idx;
                _live?.SetSelected(idx + 1);   // 0-based idx → 1-based camId
                ApplyLayout();                 // 選中相機變 → FOV→ops 基準（FrameWidth）可能變
            }

            // 重畫所有容器邊框（選中橘色、其他深灰）
            if (_camContainers != null)
                foreach (var c in _camContainers)
                    c?.Invalidate();

            // 更新「選中相機」常數列
            UpdateEngineSelectedCam();
        }

        // ===== 參數 Tab 接線（ParamKind / Wire* / ApplyParam* / InitParamControls / Clamp*）→ MilGrabberPbForm.Params.cs =====

        // ===== Telemetry / ListView（StatusTimer_Tick / lvCameras / lvEngine / CamIdForIndex）→ MilGrabberPbForm.Telemetry.cs =====

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

        // =========================================================================
        // Form 關閉：安全清理
        // =========================================================================
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            ReleaseAll();
            _live?.Dispose(); _live = null; // 共用顯示元件存活到關窗才釋放（ReleaseAll 會在重 init 呼叫，故不放那）
            base.OnFormClosing(e);
        }
    }
}
