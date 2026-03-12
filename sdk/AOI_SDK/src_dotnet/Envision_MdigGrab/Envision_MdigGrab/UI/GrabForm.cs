using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 主視窗，負責 UI 佈局與按鈕事件。
    /// MIL 相機生命週期委由 <see cref="CameraSession"/> 管理，
    /// ListView 更新委由 <see cref="CameraListViewPresenter"/> 管理。
    /// </summary>
    public partial class GrabForm : Form
    {
        private readonly List<CameraConfig> _configs;
        private readonly CameraSession _session = new CameraSession();
        private readonly CameraListViewPresenter _listPresenter;
        private readonly System.Windows.Forms.Timer _statusTimer;
        private readonly Label _sharedCoordLabel;

        /// <summary>
        /// 建構子：初始化相機設定、CameraSession 事件、Timer、ListView Presenter，
        /// 並將所有相機的連線狀態預設為 Offline。
        /// </summary>
        public GrabForm()
        {
            InitializeComponent();
            _sharedCoordLabel = labelCoord1;

            _configs = new List<CameraConfig>
            {
                new CameraConfig
                {
                    Id = 1,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL,
                    SystemNum = 0,
                    DevNum = MIL.M_DEV0,
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    ExposureUs = 150,   // 曝光設定值 (μs)
                    DisplayPanel = panel1,
                    StatusLabel = label1
                },
                new CameraConfig
                {
                    Id = 2,
                    SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL,
                    SystemNum = 1,
                    DevNum = MIL.M_DEV0,
                    DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf",
                    ExposureUs = 150,   // 曝光設定值 (μs)
                    DisplayPanel = panel2,
                    StatusLabel = label2
                }
            };

            _session.OnMouseDataChanged  += UpdateGlobalCoordLabel_FromCamera;
            _session.OnSystemAllocFailed += systemNum =>
                MessageBox.Show($"Failed to allocate System Index: {systemNum}");

            _statusTimer = new System.Windows.Forms.Timer { Interval = 500 };
            _statusTimer.Tick += StatusTimer_Tick;

            _listPresenter = new CameraListViewPresenter(listView1, toolStripStatusLabelFps);
            _listPresenter.Initialize(_configs);

            foreach (var cfg in _configs) UpdateStatusUI(cfg.Id, false);
            UpdateGlobalCoordLabel("Ready");
        }

        // ================= Button Events =================

        /// <summary>
        /// button1（Init MIL）：初始化所有相機並啟動 Timer。
        /// 若已有相機存在則不重複初始化。
        /// </summary>
        private void button1_Click(object sender, EventArgs e)
        {
            if (_session.HasCameras) return;
            _session.Initialize(_configs, checkBoxEnableImageProcessing.Checked);
            _statusTimer.Start();
        }

        /// <summary>
        /// button2（Toggle Grab）：切換所有相機的抓圖狀態（啟動 ↔ 停止）。
        /// </summary>
        private void button2_Click(object sender, EventArgs e)
        {
            _session.ToggleGrab();
        }

        /// <summary>
        /// button3（Free MIL）：非同步釋放所有 MIL 資源。
        /// 停止 Timer → 非同步 Release → 重置 UI → 還原按鈕狀態。
        /// </summary>
        private async void button3_Click(object sender, EventArgs e)
        {
            _session.IsReleasing = true;
            _statusTimer.Stop();
            SetButtonsEnabled(false);

            await _session.ReleaseAsync();

            ResetUI();
            SetButtonsEnabled(true);
        }

        // ================= Timer =================

        /// <summary>
        /// 每 500ms 觸發一次：輪詢相機連線狀態並更新 ListView 數據。
        /// 釋放流程進行中時（IsReleasing）直接跳過，避免存取已釋放資源。
        /// </summary>
        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            if (_session.IsReleasing) return;
            _session.UpdatePresence((camId, connected) => UpdateStatusUI(camId, connected));
            _listPresenter.Update(_session.Cameras);
        }

        // ================= UI Update =================

        /// <summary>
        /// 接收 CameraSession.OnMouseDataChanged 事件，將相機座標/像素值轉為顯示文字。
        /// val == -1 表示滑鼠超出影像範圍。
        /// </summary>
        private void UpdateGlobalCoordLabel_FromCamera(int camId, int x, int y, int val)
        {
            string text = val != -1
                ? $"[CAM {camId}] X: {x}, Y: {y} | Val: {val}"
                : $"[CAM {camId}] Out of Range";
            UpdateGlobalCoordLabel(text);
        }

        /// <summary>
        /// 同時更新共用座標 Label 與 StatusStrip 的座標欄位。
        /// 支援跨執行緒呼叫（MdispHookFunction 在非 UI 執行緒觸發）。
        /// </summary>
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

        /// <summary>
        /// 更新 StatusStrip 的座標欄位（toolStripStatusLabelCoord）。
        /// 支援跨執行緒呼叫。
        /// </summary>
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

        /// <summary>
        /// 更新指定相機的連線狀態 Label（Online = 綠底黑字，Offline = 紅底白字）。
        /// 支援跨執行緒呼叫，並以文字比對避免不必要的重繪。
        /// </summary>
        private void UpdateStatusUI(int id, bool isConnected)
        {
            var cfg = _configs.Find(c => c.Id == id);
            if (cfg == null || cfg.StatusLabel == null) return;

            if (this.InvokeRequired)
            {
                this.BeginInvoke(new Action(() => UpdateStatusUI(id, isConnected)));
                return;
            }

            string hardwareInfo = $"[Sys:{cfg.SystemNum} Dev:{cfg.DevNum}]";
            string statusText = isConnected
                ? $"CAM {id} {hardwareInfo}: Online"
                : $"CAM {id} {hardwareInfo}: Offline";

            Color backColor = isConnected ? Color.LightGreen : Color.Red;
            Color foreColor = isConnected ? Color.Black : Color.White;

            if (cfg.StatusLabel.Text != statusText)
            {
                cfg.StatusLabel.Text = statusText;
                cfg.StatusLabel.BackColor = backColor;
                cfg.StatusLabel.ForeColor = foreColor;
            }
        }

        /// <summary>
        /// 將所有相機狀態重置為 Offline，並重置 ListView 與座標顯示。
        /// 在 Free MIL 完成後呼叫。
        /// </summary>
        private void ResetUI()
        {
            foreach (var cfg in _configs) UpdateStatusUI(cfg.Id, false);
            UpdateGlobalCoordLabel("System Released");
            _listPresenter.ResetAll();
        }

        /// <summary>
        /// 統一設定三個操作按鈕的 Enabled 狀態，避免非同步操作期間重複點擊。
        /// </summary>
        private void SetButtonsEnabled(bool enabled)
        {
            button1.Enabled = enabled;
            button2.Enabled = enabled;
            button3.Enabled = enabled;
        }

        /// <summary>
        /// checkBoxEnableImageProcessing 狀態變更時，即時同步到所有相機的影像處理開關。
        /// </summary>
        private void checkBoxEnableImageProcessing_CheckedChanged(object sender, EventArgs e)
        {
            _session.SetImageProcessing(checkBoxEnableImageProcessing.Checked);
        }

        /// <summary>
        /// 表單關閉時停止 Timer 並同步釋放所有 MIL 資源，確保程式退出前資源完整清除。
        /// </summary>
        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            _statusTimer.Stop();
            _session.Release();
            base.OnFormClosing(e);
        }
    }
}
