using System;
using System.Drawing;
using System.Windows.Forms;
using LightBridge.Core;

namespace LightBridge.Control
{
    /// <summary>
    /// LTS-3DPA24 光源控制 GUI（IoBridge.ManualControl 同風格）。
    /// 連線 + channel + 亮度 + on/off + read。重用 LightBridge.Core 的 LightController。
    ///
    /// UI 原型用途：未來主程式要「光源控制」UI 時以此為藍本。
    /// Channel 控制項刻意包成 _pnlChannel group → 「不顯示 CH」場景設 Visible=false 即可。
    /// </summary>
    public class MainForm : Form
    {
        private readonly LightController _light = new LightController();

        private TextBox _txtPort;
        private Button _btnConnect, _btnDisconnect, _btnAutoDetect, _btnOn, _btnOff, _btnRead;
        private Label _lblStatus, _lblCurrent;
        private Panel _pnlChannel;          // ← 整組 channel 控制項（可隱藏）
        private NumericUpDown _numChannel;
        private TrackBar _trackBrightness;
        private NumericUpDown _numBrightness;
        private bool _syncing;              // 防 trackbar/numeric 同步遞迴

        public MainForm()
        {
            Text = "光源控制 (LTS-3DPA24)";
            try { Icon = System.Drawing.Icon.ExtractAssociatedIcon(System.Windows.Forms.Application.ExecutablePath); } catch { }
            Font = new Font("Microsoft JhengHei", 9F);
            ClientSize = new Size(440, 280);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;

            BuildControls();
            SetConnectedUi(false);

            _txtPort.Text = AppSettings.ComPort;
            _numChannel.Value = AppSettings.Channel;
            _numBrightness.Value = AppSettings.Brightness;
            _trackBrightness.Value = AppSettings.Brightness;

            FormClosed += (s, e) => _light.Dispose();
        }

        private void BuildControls()
        {
            // ── 連線列 ──
            var lblPort = new Label { Text = "COM:", Location = new Point(16, 18), AutoSize = true };
            _txtPort = new TextBox { Location = new Point(60, 15), Width = 80 };
            _btnConnect    = new Button { Text = "連線",     Location = new Point(150, 13), Width = 60 };
            _btnDisconnect = new Button { Text = "斷線",     Location = new Point(216, 13), Width = 60 };
            _btnAutoDetect = new Button { Text = "自動偵測", Location = new Point(282, 13), Width = 90 };
            _btnConnect.Click    += OnConnect;
            _btnDisconnect.Click += OnDisconnect;
            _btnAutoDetect.Click += OnAutoDetect;

            // ── 狀態 ──
            _lblStatus = new Label { Location = new Point(16, 48), AutoSize = true, ForeColor = Color.Gray, Text = "● 未連線" };

            // ── Channel（可隱藏 group）──
            _pnlChannel = new Panel { Location = new Point(12, 78), Size = new Size(416, 36) };
            var lblCh = new Label { Text = "Channel:", Location = new Point(4, 8), AutoSize = true };
            _numChannel = new NumericUpDown { Location = new Point(80, 6), Width = 60, Minimum = 1, Maximum = 8, Value = 1 };
            _pnlChannel.Controls.Add(lblCh);
            _pnlChannel.Controls.Add(_numChannel);

            // ── 亮度 ──
            var lblBright = new Label { Text = "亮度:", Location = new Point(16, 124), AutoSize = true };
            _trackBrightness = new TrackBar { Location = new Point(80, 118), Width = 250, Minimum = 0, Maximum = 255, TickFrequency = 32, Value = 200 };
            _numBrightness = new NumericUpDown { Location = new Point(340, 122), Width = 70, Minimum = 0, Maximum = 255, Value = 200 };
            _trackBrightness.Scroll += (s, e) => SyncBrightness(_trackBrightness.Value, fromTrack: true);
            _numBrightness.ValueChanged += (s, e) => SyncBrightness((int)_numBrightness.Value, fromTrack: false);

            // ── 操作按鈕 ──
            _btnOn   = new Button { Text = "開 ON",     Location = new Point(16, 178), Width = 100, Height = 36, BackColor = Color.LightGreen };
            _btnOff  = new Button { Text = "關 OFF",    Location = new Point(126, 178), Width = 100, Height = 36, BackColor = Color.MistyRose };
            _btnRead = new Button { Text = "讀取亮度", Location = new Point(236, 178), Width = 100, Height = 36 };
            _btnOn.Click   += OnTurnOn;
            _btnOff.Click  += OnTurnOff;
            _btnRead.Click += OnRead;

            // ── 當前亮度 ──
            _lblCurrent = new Label { Location = new Point(16, 230), AutoSize = true, Font = new Font("Microsoft JhengHei", 10F, FontStyle.Bold), Text = "當前亮度: --" };

            Controls.AddRange(new System.Windows.Forms.Control[] {
                lblPort, _txtPort, _btnConnect, _btnDisconnect, _btnAutoDetect,
                _lblStatus, _pnlChannel, lblBright, _trackBrightness, _numBrightness,
                _btnOn, _btnOff, _btnRead, _lblCurrent
            });
        }

        private void SyncBrightness(int value, bool fromTrack)
        {
            if (_syncing) return;
            _syncing = true;
            try
            {
                if (fromTrack) _numBrightness.Value = value;
                else _trackBrightness.Value = value;
                // 已連線時即時送亮度（不改開關狀態）
                if (_light.IsConnected)
                    _light.SetBrightness((int)_numChannel.Value, value);
            }
            finally { _syncing = false; }
        }

        // ── 事件 ──
        private void OnConnect(object sender, EventArgs e)
        {
            string port = _txtPort.Text.Trim();
            SetConnectedUi(_light.Connect(port));
            UpdateStatus();
        }

        private void OnAutoDetect(object sender, EventArgs e)
        {
            string found = _light.AutoDetect(_txtPort.Text.Trim(), (int)_numChannel.Value);
            if (found != null) _txtPort.Text = found;
            SetConnectedUi(_light.IsConnected);
            UpdateStatus();
            if (found == null)
                MessageBox.Show("掃描完所有 COM port 找不到光源 — 檢查接線 / 電源 / Channel。", "自動偵測", MessageBoxButtons.OK, MessageBoxIcon.Warning);
        }

        private void OnDisconnect(object sender, EventArgs e)
        {
            _light.Disconnect();
            SetConnectedUi(false);
            UpdateStatus();
        }

        private void OnTurnOn(object sender, EventArgs e)
        {
            if (!_light.IsConnected) return;
            bool ok = _light.TurnOn((int)_numChannel.Value, (int)_numBrightness.Value);
            if (!ok) MessageBox.Show("開啟失敗", "光源", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void OnTurnOff(object sender, EventArgs e)
        {
            if (!_light.IsConnected) return;
            _light.TurnOff((int)_numChannel.Value);
        }

        private void OnRead(object sender, EventArgs e)
        {
            if (!_light.IsConnected) return;
            int v = _light.ReadBrightness((int)_numChannel.Value);
            _lblCurrent.Text = v >= 0 ? $"當前亮度: {v}" : "當前亮度: 讀取失敗";
        }

        private void UpdateStatus()
        {
            if (_light.IsConnected)
            {
                _lblStatus.Text = $"● 已連線 {_light.ActiveComPort}";
                _lblStatus.ForeColor = Color.SeaGreen;
            }
            else
            {
                _lblStatus.Text = "● 未連線";
                _lblStatus.ForeColor = Color.Gray;
            }
        }

        private void SetConnectedUi(bool connected)
        {
            _btnConnect.Enabled = !connected;
            _btnDisconnect.Enabled = connected;
            _btnOn.Enabled = connected;
            _btnOff.Enabled = connected;
            _btnRead.Enabled = connected;
            _trackBrightness.Enabled = connected;
            _numBrightness.Enabled = connected;
        }
    }
}
