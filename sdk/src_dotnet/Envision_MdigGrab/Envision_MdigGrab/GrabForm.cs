using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;

namespace Envision_MdigGrab
{
    public partial class GrabForm : Form
    {
        private class CameraConfig
        {
            public int Id { get; set; }

            // [新增] 該相機使用哪種 System (卡片驅動類型)
            public string SystemDescriptor { get; set; }

            // [新增] 該相機使用第幾張卡 (System Index)
            public int SystemNum { get; set; }

            // [修改] 在該 System 上的 Device 編號 (因為每個 System 都是獨立的，通常都是 M_DEV0)
            public MIL_INT DevNum { get; set; }

            public string DcfPath { get; set; }
            public Panel DisplayPanel { get; set; }
            public Label StatusLabel { get; set; }
        }

        private List<MilCameraUnit> _cameras = new List<MilCameraUnit>();
        private List<CameraConfig> _configs;

        // [新增] 用來管理已開啟的 System (Key: SystemNum, Value: MIL_ID)
        // 這樣若有多支相機在同一張卡上，不會重複開啟
        private Dictionary<int, MIL_ID> _allocatedSystems = new Dictionary<int, MIL_ID>();

        private System.Windows.Forms.Timer statusTimer;
        private bool _userWantsGrab = false;
        private volatile bool _isReleasing = false;
        private Label _sharedCoordLabel;

        public GrabForm()
        {
            InitializeComponent();
            _sharedCoordLabel = labelCoord1;

            _configs = new List<CameraConfig>
            {
                // === 相機 1：第一張卡 (System 0), Digitizer 0 ===
                new CameraConfig
                {
                    Id = 1,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, // 或 M_SYSTEM_SOLIOS 等
                    SystemNum = 0,                               // 第一張卡
                    DevNum = MIL.M_DEV0,                         // 第一個 Port
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    DisplayPanel = panel1,
                    StatusLabel = label1
                },

                // === 相機 2：第二張卡 (System 1), Digitizer 0 ===
                new CameraConfig
                {
                    Id = 2,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, // 相同的驅動
                    SystemNum = 1,                               // [重點] 第二張卡
                    DevNum = MIL.M_DEV0,                         // [重點] 該卡上的第一個 Port
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    DisplayPanel = panel2,
                    StatusLabel = label2
                }
            };

            statusTimer = new System.Windows.Forms.Timer();
            statusTimer.Interval = 500;
            statusTimer.Tick += StatusTimer_Tick;

            foreach (var cfg in _configs) UpdateStatusUI(cfg.Id, false);
            UpdateGlobalCoordLabel("Ready");
            ApplyImageProcessingSetting();
            UpdateFpsStatusStrip();
        }

        // ================= Button Events =================

        private void button1_Click(object sender, EventArgs e)
        {
            if (_cameras.Count > 0) return;

            _isReleasing = false;
            _userWantsGrab = false;

            // 1. 初始化 Application
            MilSystemManager.Initialize();

            // 2. 遍歷設定檔，分配 System 並建立相機
            foreach (var cfg in _configs)
            {
                MIL_ID currentSysId = MIL.M_NULL;

                // 檢查該張卡 (SystemNum) 是否已經開啟過？
                if (_allocatedSystems.ContainsKey(cfg.SystemNum))
                {
                    currentSysId = _allocatedSystems[cfg.SystemNum];
                }
                else
                {
                    // 尚未開啟，呼叫 Manager 分配新 System
                    currentSysId = MilSystemManager.AllocateSystem(cfg.SystemDescriptor, cfg.SystemNum);

                    if (currentSysId != MIL.M_NULL)
                    {
                        _allocatedSystems.Add(cfg.SystemNum, currentSysId);
                    }
                    else
                    {
                        MessageBox.Show($"Failed to allocate System Index: {cfg.SystemNum}");
                        continue; // 跳過此相機
                    }
                }

                // 3. 建立相機 (傳入 System ID)
                var cam = new MilCameraUnit(
                    currentSysId,
                    cfg.Id,
                    cfg.DevNum,
                    cfg.DcfPath,
                    cfg.DisplayPanel.Handle,
                    checkBoxEnableImageProcessing.Checked);
                cam.OnMouseDataChanged += UpdateGlobalCoordLabel_FromCamera;
                cam.Initialize();
                _cameras.Add(cam);
            }

            statusTimer.Start();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            _userWantsGrab = !_userWantsGrab;
            foreach (var cam in _cameras)
            {
                cam.SetUserGrabIntent(_userWantsGrab);
            }
        }

        private async void button3_Click(object sender, EventArgs e)
        {
            _isReleasing = true;
            statusTimer.Stop();
            SetButtonsEnabled(false);

            await Task.Run(() =>
            {
                // 1. 先釋放所有相機 (MdigFree, MbufFree)
                foreach (var cam in _cameras)
                {
                    cam.Free();
                }
                _cameras.Clear();

                // 2. 釋放所有 System (MsysFree)
                foreach (var kvp in _allocatedSystems)
                {
                    MilSystemManager.FreeSystem(kvp.Value);
                }
                _allocatedSystems.Clear();

                // 3. 最後釋放 Application (MappFree)
                MilSystemManager.FreeApplication();
            });

            ResetUI();
            SetButtonsEnabled(true);
        }

        // ================= UI Update Logic =================

        private void UpdateGlobalCoordLabel_FromCamera(int camId, int x, int y, int val)
        {
            string text = (val != -1)
                ? $"[CAM {camId}] X: {x}, Y: {y} | Val: {val}"
                : $"[CAM {camId}] Out of Range";

            UpdateGlobalCoordLabel(text);
        }

        private void UpdateGlobalCoordLabel(string text)
        {
            if (_sharedCoordLabel != null)
            {
                if (_sharedCoordLabel.InvokeRequired)
                {
                    _sharedCoordLabel.BeginInvoke(new Action(() => UpdateGlobalCoordLabel(text)));
                    return;
                }

                _sharedCoordLabel.Text = text;
            }

            UpdateCoordStatusStrip(text);
        }

        private void UpdateCoordStatusStrip(string text)
        {
            if (toolStripStatusLabelCoord == null) return;

            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action(() => UpdateCoordStatusStrip(text)));
                return;
            }

            toolStripStatusLabelCoord.Text = text;
        }

        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            if (_isReleasing) return;

            foreach (var cam in _cameras)
            {
                bool isConnected = cam.CheckPresence();
                UpdateStatusUI(cam.CameraId, isConnected);

                if (isConnected && cam.UserWantsGrab && !cam.IsLive)
                {
                    cam.ApplyGrabState();
                }
            }

            UpdateFpsStatusStrip();
        }

        private void UpdateStatusUI(int id, bool isConnected)
        {
            // 1. 找到對應的設定檔
            var cfg = _configs.Find(c => c.Id == id);
            if (cfg == null || cfg.StatusLabel == null) return;

            // 2. 處理跨執行緒呼叫 (Invoke)
            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action(() => UpdateStatusUI(id, isConnected)));
                return;
            }

            // 3. [修改] 組合顯示字串，加入 SystemNum 和 DevNum
            // 顯示格式例如: "CAM 1 [Sys:0 Dev:0]: Online"
            string hardwareInfo = $"[Sys:{cfg.SystemNum} Dev:{cfg.DevNum}]";

            string statusText = isConnected
                ? $"CAM {id} {hardwareInfo}: Online"
                : $"CAM {id} {hardwareInfo}: Offline";

            Color backColor = isConnected ? Color.LightGreen : Color.Red;
            Color foreColor = isConnected ? Color.Black : Color.White;

            // 4. 更新 Label 屬性
            if (cfg.StatusLabel.Text != statusText)
            {
                cfg.StatusLabel.Text = statusText;
                cfg.StatusLabel.BackColor = backColor;
                cfg.StatusLabel.ForeColor = foreColor;
            }
        }

        private void ResetUI()
        {
            foreach (var cfg in _configs) UpdateStatusUI(cfg.Id, false);
            _userWantsGrab = false;
            UpdateGlobalCoordLabel("System Released");
            UpdateFpsStatusStrip();
        }

        private void SetButtonsEnabled(bool enabled)
        {
            button1.Enabled = enabled;
            button2.Enabled = enabled;
            button3.Enabled = enabled;
        }

        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            ApplyImageProcessingSetting();
        }

        private void ApplyImageProcessingSetting()
        {
            bool enableImageProcessing = checkBoxEnableImageProcessing.Checked;

            foreach (var cam in _cameras)
            {
                cam.EnableImageProcessing = enableImageProcessing;
            }
        }


        private void UpdateFpsStatusStrip()
        {
            if (toolStripStatusLabelFps == null) return;

            if (_cameras.Count == 0)
            {
                toolStripStatusLabelFps.Text = "FPS: N/A";
                return;
            }

            var fpsParts = new List<string>();
            double totalFps = 0;

            foreach (var cam in _cameras)
            {
                double fps = cam.CurrentFps;
                totalFps += fps;
                fpsParts.Add($"CAM {cam.CameraId}: {fps:F1}");
            }

            toolStripStatusLabelFps.Text = $"FPS | {string.Join(" | ", fpsParts)} | Total: {totalFps:F1}";
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _isReleasing = true;
            statusTimer.Stop();
            // Form 關閉時的簡易清理，最好是呼叫 button3 的邏輯
            foreach (var cam in _cameras) cam.Free();
            foreach (var kvp in _allocatedSystems) MilSystemManager.FreeSystem(kvp.Value);
            MilSystemManager.FreeApplication();
            base.OnFormClosing(e);
        }
    }
}
