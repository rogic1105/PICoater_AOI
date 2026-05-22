using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Net;
using System.Net.Sockets;
using System.Windows.Forms;
using System.Threading.Tasks;
using IoBridge.Core;

namespace IoBridge.ManualControl
{
    public partial class MainForm : Form
    {
        private IcpDasModbusTcpClient _plc = new IcpDasModbusTcpClient();
        private Timer _timerScan = new Timer { Interval = AppSettings.PollIntervalMs };

        private readonly Dictionary<string, Panel> _lamps = new Dictionary<string, Panel>();
        private readonly Dictionary<string, CheckBox> _doCheckBoxes = new Dictionary<string, CheckBox>();

        public MainForm()
        {
            InitializeComponent();
            InitDynamicControls();

            _plc.ReadWriteTimeoutMs = AppSettings.ReadWriteTimeoutMs;
            txtIP.Text = AppSettings.PlcIp;
            btnConnect.BackColor = Color.LightSkyBlue;
            btnDisconnect.BackColor = Color.Silver;

            _timerScan.Tick += async (s, e) => await OnPollingTick();

            IoLogger.FilePrefix = "ManualControl";
            IoLogger.Info("Application started.");
        }

        private void InitDynamicControls()
        {
            for (int i = 0; i < 8; i++)
            {
                int yPos = 40 + (i * 42);

                CheckBox chk = new CheckBox
                {
                    Name = "chkDO" + i,
                    Text = "DO-" + i,
                    Location = new Point(20, yPos),
                    Tag = i,
                    AutoSize = true,
                    Font = new Font("Consolas", 10F)
                };
                chk.Click += new EventHandler(DoCheckBox_Click);

                Panel pnlDO = CreateCirclePanel("shpDO" + i, 140, yPos);
                this.grpDO.Controls.Add(chk);
                this.grpDO.Controls.Add(pnlDO);
                _doCheckBoxes["chkDO" + i] = chk;
                _lamps["shpDO" + i] = pnlDO;

                Label lblDI = new Label
                {
                    Text = "DI-" + i,
                    Location = new Point(30, yPos),
                    AutoSize = true,
                    Font = new Font("Consolas", 10F)
                };
                Panel pnlDI = CreateCirclePanel("shpDI" + i, 120, yPos);
                this.grpDI.Controls.Add(lblDI);
                this.grpDI.Controls.Add(pnlDI);
                _lamps["shpDI" + i] = pnlDI;
            }
        }

        private Panel CreateCirclePanel(string name, int x, int y)
        {
            Panel pnl = new Panel
            {
                Name = name,
                Location = new Point(x, y),
                Size = new Size(20, 20),
                BackColor = Color.Silver
            };
            GraphicsPath path = new GraphicsPath();
            path.AddEllipse(0, 0, 20, 20);
            pnl.Region = new Region(path);
            return pnl;
        }

        private async void btnConnect_Click(object sender, EventArgs e)
        {
            string ip = txtIP.Text.Trim();
            if (!IPAddress.TryParse(ip, out _))
            {
                MessageBox.Show("IP 格式不正確，請重新輸入。", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            btnConnect.Enabled = false;
            btnConnect.BackColor = Color.Silver;
            btnConnect.Text = "Connecting...";
            txtIP.Enabled = false;

            bool success = await _plc.ConnectAsync(ip, AppSettings.PlcPort, AppSettings.ConnectTimeoutMs);

            if (success)
            {
                IoLogger.Info($"Connected to {ip}:{AppSettings.PlcPort}");
                // TCP 連上就算連線成功，切換 UI + 啟動 polling。
                // 初始化讀取（下方 try）失敗不該誤判斷線 —— ET-7044 是 4DI+4DO，
                // ReadDoStatuses 讀 8 coils 可能回 Modbus exception，但 TCP 已連上。
                _timerScan.Start();
                btnConnect.Text = "Connect";
                btnDisconnect.Enabled = true;
                btnDisconnect.BackColor = Color.LightPink;

                // 初始化讀取：顯示當前狀態 + 同步 DO checkbox。失敗只記 log，不影響連線。
                try
                {
                    var taskDO = _plc.ReadDoStatuses();
                    var taskDI = _plc.ReadDiStatuses();
                    await Task.WhenAll(taskDO, taskDI);

                    for (int i = 0; i < 8; i++)
                    {
                        if (_doCheckBoxes.TryGetValue("chkDO" + i, out var chk))
                            chk.Checked = taskDO.Result[i];
                        UpdateStatusLamp("shpDO" + i, taskDO.Result[i] ? Color.LimeGreen : Color.White);
                        UpdateStatusLamp("shpDI" + i, taskDI.Result[i] ? Color.LimeGreen : Color.White);
                    }
                }
                catch (Exception ex)
                {
                    IoLogger.Warn($"Initial read failed (connection kept, polling will retry): {ex.Message}");
                }
            }
            else
            {
                IoLogger.Warn($"Connection failed to {ip}:{AppSettings.PlcPort}");
                HandleConnectionFail();
                MessageBox.Show("Connection Failed! (Timeout)", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void HandleConnectionFail()
        {
            btnConnect.Enabled = true;
            btnConnect.BackColor = Color.LightSkyBlue;
            btnConnect.Text = "Connect";
            txtIP.Enabled = true;
            btnDisconnect.Enabled = false;
            btnDisconnect.BackColor = Color.Silver;
            _plc.Dispose();
        }

        private void btnDisconnect_Click(object sender, EventArgs e)
        {
            _timerScan.Stop();
            _plc.Dispose();
            ResetUI();

            IoLogger.Info("Disconnected by user.");
            btnConnect.Enabled = true;
            btnConnect.BackColor = Color.LightSkyBlue;
            btnDisconnect.Enabled = false;
            btnDisconnect.BackColor = Color.Silver;
            txtIP.Enabled = true;
        }

        private void ResetUI()
        {
            for (int i = 0; i < 8; i++)
            {
                UpdateStatusLamp("shpDI" + i, Color.Silver);
                UpdateStatusLamp("shpDO" + i, Color.Silver);
                if (_doCheckBoxes.TryGetValue("chkDO" + i, out var chk))
                    chk.Checked = false;
            }
        }

        private async void DoCheckBox_Click(object sender, EventArgs e)
        {
            if (!_plc.IsConnected) return;
            var chk = (CheckBox)sender;
            try
            {
                await _plc.WriteDo(Convert.ToInt32(chk.Tag), chk.Checked);
            }
            catch (SocketException ex)
            {
                IoLogger.Error($"WriteDo[{chk.Tag}] socket error", ex);
            }
            catch (Exception ex)
            {
                IoLogger.Error($"WriteDo[{chk.Tag}] failed", ex);
            }
        }

        private async Task OnPollingTick()
        {
            // DI 讀取為連線核心：失敗才斷線（ET-7044 DI 在 addr 0，IoGrabController 驗證過正常）
            try
            {
                var diStates = await _plc.ReadDiStatuses();
                for (int i = 0; i < 8; i++)
                    UpdateStatusLamp("shpDI" + i, diStates[i] ? Color.LimeGreen : Color.White);
            }
            catch (SocketException ex)
            {
                IoLogger.Error("DI polling socket error, disconnecting", ex);
                btnDisconnect_Click(null, null);
                return;
            }
            catch (Exception ex)
            {
                IoLogger.Error("DI polling error, disconnecting", ex);
                btnDisconnect_Click(null, null);
                return;
            }

            // DO 讀取容錯：ET-7044 可能不支援 read 8 coils，失敗只 skip DO 燈，不斷線
            // （手動戳 DO 用 WriteDo 不受影響，DO 回讀燈號顯示為止步狀態而已）
            try
            {
                var doStates = await _plc.ReadDoStatuses();
                for (int i = 0; i < 8; i++)
                    UpdateStatusLamp("shpDO" + i, doStates[i] ? Color.LimeGreen : Color.White);
            }
            catch (Exception ex)
            {
                IoLogger.Warn($"DO read failed (skip DO lamps, connection kept): {ex.Message}");
            }
        }

        private void UpdateStatusLamp(string name, Color color)
        {
            if (_lamps.TryGetValue(name, out var pnl))
                pnl.BackColor = color;
        }
    }
}
