using System;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using StorageBridge.Core;

namespace StorageBridge.Control
{
    /// <summary>
    /// 儲存電腦連線測試 GUI。仿 AniloxRoll.Monitor 的 storage 連線檢測：
    /// 背景 probe 遠端 UNC 路徑實際可寫 → 燈號（綠 已連線 / 紅 離線 / 灰 停用），
    /// 複製測試重用 StorageBridge.Core.RemoteCopyService（主程式同機制）。
    ///
    /// UI 原型用途：未來主程式「儲存設定 / 連線狀態」UI 以此為藍本。
    /// </summary>
    public class MainForm : Form
    {
        // IEC 狀態色（對齊主程式 lblStorageConn / 光源工具）
        private static readonly Color IecGreen = Color.FromArgb(56, 142, 60);
        private static readonly Color IecRed   = Color.FromArgb(198, 40, 40);
        private static readonly Color IecGray  = Color.FromArgb(117, 117, 117);

        private RemoteCopyService _copy;
        private TextBox _txtRemote, _txtLocal;
        private Button _btnProbe, _btnCopyTest;
        private Label _lblStatus, _lblCopyResult;
        private CheckBox _chkAuto;
        private readonly Timer _probeTimer = new Timer();
        private bool _probeInFlight;

        public MainForm()
        {
            Text = "儲存電腦連線測試 (StorageBridge)";
            try { Icon = System.Drawing.Icon.ExtractAssociatedIcon(System.Windows.Forms.Application.ExecutablePath); }
            catch (Exception ex) { Trace.WriteLine("[StorageControl] Icon: " + ex.Message); }
            Font = new Font("Microsoft JhengHei", 9F);
            ClientSize = new Size(480, 250);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;

            BuildControls();
            _txtRemote.Text = AppSettings.RemotePath;
            _txtLocal.Text = AppSettings.LocalPath;
            _probeTimer.Interval = AppSettings.ProbeIntervalMs;
            _probeTimer.Tick += (s, e) => ProbeAsync();
            UpdateStatus(null);

            // 預設開啟自動偵測 → 燈號持續即時反映（同光源工具，launch 即活）
            _chkAuto.Checked = true;

            FormClosed += (s, e) => { _probeTimer.Stop(); _copy?.Dispose(); };
        }

        private void BuildControls()
        {
            var lblR = new Label { Text = "遠端路徑:", Location = new Point(16, 20), AutoSize = true };
            _txtRemote = new TextBox { Location = new Point(96, 17), Width = 360 };
            var lblL = new Label { Text = "本地路徑:", Location = new Point(16, 54), AutoSize = true };
            _txtLocal = new TextBox { Location = new Point(96, 51), Width = 360 };

            _btnProbe = new Button { Text = "連線測試", Location = new Point(16, 90), Width = 90, Height = 30 };
            _btnProbe.Click += (s, e) => ProbeAsync();

            _chkAuto = new CheckBox { Text = "自動偵測（每 5 秒）", Location = new Point(120, 95), AutoSize = true };
            _chkAuto.CheckedChanged += (s, e) => { if (_chkAuto.Checked) { _probeTimer.Start(); ProbeAsync(); } else _probeTimer.Stop(); };

            _lblStatus = new Label
            {
                Location = new Point(16, 134), Size = new Size(448, 32),
                TextAlign = ContentAlignment.MiddleCenter, ForeColor = Color.White,
                BorderStyle = BorderStyle.FixedSingle, Font = new Font("Microsoft JhengHei", 10F, FontStyle.Bold)
            };

            _btnCopyTest = new Button { Text = "選檔複製測試…", Location = new Point(16, 180), Width = 130, Height = 30 };
            _btnCopyTest.Click += OnCopyTest;
            _lblCopyResult = new Label { Location = new Point(156, 186), AutoSize = true, ForeColor = Color.Gray };

            Controls.AddRange(new System.Windows.Forms.Control[] {
                lblR, _txtRemote, lblL, _txtLocal, _btnProbe, _chkAuto, _lblStatus, _btnCopyTest, _lblCopyResult
            });
        }

        // 背景 probe 遠端 UNC 路徑（SMB redirector 可能阻塞，不可在 UI thread）
        private void ProbeAsync()
        {
            string path = _txtRemote.Text.Trim();
            if (string.IsNullOrWhiteSpace(path)) { UpdateStatus(null); return; }
            if (_probeInFlight) return;
            _probeInFlight = true;
            Task.Run(() =>
            {
                string error;
                bool ok = RemoteCopyService.TryProbeRemoteWritable(path, out error);
                if (!ok) Trace.WriteLine("[StorageControl] Probe: " + error);
                _probeInFlight = false;
                if (IsDisposed || Disposing) return;
                try { BeginInvoke(new Action<bool?>(UpdateStatus), (bool?)ok); }
                catch (InvalidOperationException) { }
            });
        }

        private void UpdateStatus(bool? connected)
        {
            if (string.IsNullOrWhiteSpace(_txtRemote.Text))
            {
                _lblStatus.Text = "● 儲存電腦 停用（未設遠端路徑）"; _lblStatus.BackColor = IecGray; return;
            }
            if (connected == true)  { _lblStatus.Text = "● 儲存電腦 已連線"; _lblStatus.BackColor = IecGreen; }
            else if (connected == false) { _lblStatus.Text = "● 儲存電腦 離線（路徑不可達）"; _lblStatus.BackColor = IecRed; }
            else { _lblStatus.Text = "● 儲存電腦 偵測中…"; _lblStatus.BackColor = IecGray; }
        }

        private void OnCopyTest(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog { Multiselect = true, Title = "選擇要複製到遠端的檔案" })
            {
                if (ofd.ShowDialog() != DialogResult.OK) return;
                if (_copy == null)
                    _copy = new RemoteCopyService(() => _txtRemote.Text.Trim(), () => _txtLocal.Text.Trim());
                _copy.EnqueueFiles(ofd.FileNames);
                _lblCopyResult.Text = $"已排入 {ofd.FileNames.Length} 檔複製佇列（背景複製，查遠端路徑確認）";
            }
        }
    }
}
