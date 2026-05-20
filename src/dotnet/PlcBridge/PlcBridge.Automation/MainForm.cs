using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Net;
using System.Net.Sockets;
using System.Windows.Forms;
using System.Threading.Tasks;
using PlcBridge.Core;

namespace PlcBridge.Automation
{
    public partial class MainForm : Form
    {
        private IcpDasModbusTcpClient _plc = new IcpDasModbusTcpClient();
        private MainProcess _mainProcess;
        private Timer _timerScan = new Timer { Interval = AppSettings.PollIntervalMs };
        private Timer _timerReconnect = new Timer { Interval = AppSettings.ReconnectIntervalMs };

        private readonly Color Col_Connect = Color.FromArgb(0, 122, 204);
        private readonly Color Col_Disconnect = Color.FromArgb(200, 50, 50);
        private readonly Color Col_Mura = Color.FromArgb(255, 140, 0);
        private readonly Color Col_Disabled = Color.FromArgb(60, 60, 60);
        private readonly Color Col_TextDisabled = Color.Gray;
        private readonly Color Lamp_On = Color.FromArgb(0, 255, 127);
        private readonly Color Lamp_Off = Color.FromArgb(50, 50, 50);

        private readonly Dictionary<string, Panel> _lamps = new Dictionary<string, Panel>();

        public MainForm()
        {
            InitializeComponent();
            InitDynamicControls();

            _plc.ReadWriteTimeoutMs = AppSettings.ReadWriteTimeoutMs;
            txtIP.Text = AppSettings.PlcIp;
            SetButtonState(btnConnect, true, Col_Connect);
            SetButtonState(btnDisconnect, false, Col_Disconnect);
            SetButtonState(btnMura, false, Col_Mura);
            UpdateSystemStatusLabel("Closed");

            _timerScan.Tick += async (s, e) => await OnPollingTick();
            _timerReconnect.Tick += async (s, e) => await OnReconnectTick();

            PlcLogger.FilePrefix = "Automation";
            PlcLogger.Info("Application started.");
        }

        private void SetButtonState(Button btn, bool enabled, Color activeColor)
        {
            btn.Enabled = enabled;
            btn.BackColor = enabled ? activeColor : Col_Disabled;
            btn.ForeColor = enabled ? Color.White : Col_TextDisabled;
        }

        private void InitDynamicControls()
        {
            for (int i = 0; i < 8; i++)
            {
                int yPos = 40 + (i * 42);

                string doText = $"DO-{i}";
                if (i == 0) doText += " (PC ALIVE)";
                else if (i == 1) doText += " (MURA)";
                else if (i == 2) doText += " (PC BUSY)";

                string diText = $"DI-{i}";
                if (i == 0) diText += " (PLC ALIVE)";
                else if (i == 1) diText += " (START)";

                Label lblDO = new Label { Text = doText, Location = new Point(20, yPos + 3), AutoSize = true, Font = new Font("Consolas", 10F), ForeColor = Color.WhiteSmoke };
                Panel pnlDO = CreateCirclePanel("shpDO" + i, 300, yPos);
                this.grpDO.Controls.Add(lblDO);
                this.grpDO.Controls.Add(pnlDO);
                _lamps["shpDO" + i] = pnlDO;

                Label lblDI = new Label { Text = diText, Location = new Point(20, yPos + 3), AutoSize = true, Font = new Font("Consolas", 10F), ForeColor = Color.WhiteSmoke };
                Panel pnlDI = CreateCirclePanel("shpDI" + i, 300, yPos);
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
                BackColor = Lamp_Off
            };
            GraphicsPath path = new GraphicsPath();
            path.AddEllipse(0, 0, 20, 20);
            pnl.Region = new Region(path);
            return pnl;
        }

        private async void btnMura_Click(object sender, EventArgs e)
        {
            if (_mainProcess != null) await _mainProcess.TriggerMuraAsync();
        }

        private void UpdateSystemStatusLabel(string status)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new Action<string>(UpdateSystemStatusLabel), status);
                return;
            }

            string upperStatus = status.ToUpper();
            lblSystemStatus.Text = upperStatus;

            bool canMura = (upperStatus == "RUN");
            SetButtonState(btnMura, canMura, Col_Mura);

            switch (upperStatus)
            {
                case "PCERR":
                case "PLCERR":
                    lblSystemStatus.ForeColor = Color.Red;
                    break;
                case "MURA":
                    lblSystemStatus.ForeColor = Color.Orange;
                    break;
                case "RUN":
                    lblSystemStatus.ForeColor = Color.LimeGreen;
                    break;
                case "READY":
                    lblSystemStatus.ForeColor = Color.Cyan;
                    break;
                case "INITIAL":
                case "START":
                case "STOP":
                case "CLEAR":
                    lblSystemStatus.ForeColor = Color.White;
                    break;
                default:
                    lblSystemStatus.ForeColor = Color.Gray;
                    break;
            }
        }

        private async void btnConnect_Click(object sender, EventArgs e)
        {
            string ip = txtIP.Text.Trim();
            if (!IPAddress.TryParse(ip, out _))
            {
                MessageBox.Show("IP 格式不正確，請重新輸入。", "Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }

            SetButtonState(btnConnect, false, Col_Connect);
            btnConnect.Text = "WAIT...";
            txtIP.Enabled = false;
            UpdateSystemStatusLabel("Initial");
            _timerReconnect.Stop();

            bool success = await _plc.ConnectAsync(ip, AppSettings.PlcPort, AppSettings.ConnectTimeoutMs);

            if (success)
            {
                try
                {
                    PlcLogger.Info($"Connected to {ip}:{AppSettings.PlcPort}");
                    _mainProcess = new MainProcess(_plc, UpdateSystemStatusLabel);
                    await _mainProcess.StartSystemAsync();
                    _timerScan.Start();
                    btnConnect.Text = "CONNECT";
                    SetButtonState(btnDisconnect, true, Col_Disconnect);
                }
                catch (SocketException ex)
                {
                    PlcLogger.Error("Post-connect init socket error", ex);
                    HandleConnectionFail();
                }
                catch (Exception ex)
                {
                    PlcLogger.Error("Post-connect init failed", ex);
                    HandleConnectionFail();
                }
            }
            else
            {
                PlcLogger.Warn($"Connection failed to {ip}:{AppSettings.PlcPort}");
                HandleConnectionFail();
                MessageBox.Show("Connection Failed!", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void HandleConnectionFail()
        {
            SetButtonState(btnConnect, true, Col_Connect);
            btnConnect.Text = "CONNECT";
            txtIP.Enabled = true;
            SetButtonState(btnDisconnect, false, Col_Disconnect);
            _plc.Dispose();
            UpdateSystemStatusLabel("Closed");
        }

        private async void btnDisconnect_Click(object sender, EventArgs e)
        {
            _timerScan.Stop();
            _timerReconnect.Stop();

            if (_mainProcess != null)
                await _mainProcess.StopSystemAsync();
            else
                UpdateSystemStatusLabel("Closed");

            PlcLogger.Info("Disconnected by user.");
            _plc.Dispose();
            ResetUI();
            SetButtonState(btnConnect, true, Col_Connect);
            SetButtonState(btnDisconnect, false, Col_Disconnect);
            txtIP.Enabled = true;
        }

        private void ResetUI()
        {
            for (int i = 0; i < 8; i++)
            {
                UpdateStatusLamp("shpDI" + i, Lamp_Off);
                UpdateStatusLamp("shpDO" + i, Lamp_Off);
            }
            if (lblSystemStatus.Text != "CLOSED") UpdateSystemStatusLabel("Closed");
        }

        private async Task OnPollingTick()
        {
            _timerScan.Stop();

            try
            {
                var doStates = await _plc.ReadDoStatuses();
                var diStates = await _plc.ReadDiStatuses();

                if (_mainProcess != null) await _mainProcess.ProcessSignalAsync(diStates);

                for (int i = 0; i < 8; i++)
                {
                    UpdateStatusLamp("shpDO" + i, doStates[i] ? Lamp_On : Lamp_Off);
                    UpdateStatusLamp("shpDI" + i, diStates[i] ? Lamp_On : Lamp_Off);
                }

                _timerScan.Start();
            }
            catch (SocketException ex)
            {
                PlcLogger.Error("Polling socket error, triggering PCErr", ex);
                HandlePollingError();
            }
            catch (Exception ex)
            {
                PlcLogger.Error("Polling error, triggering PCErr", ex);
                HandlePollingError();
            }
        }

        private void HandlePollingError()
        {
            if (_mainProcess != null)
            {
                _mainProcess.TriggerPcError();
                for (int i = 0; i < 8; i++)
                {
                    UpdateStatusLamp("shpDO" + i, Lamp_Off);
                    UpdateStatusLamp("shpDI" + i, Lamp_Off);
                }
            }
            else
            {
                UpdateSystemStatusLabel("PCErr");
            }
            _timerReconnect.Start();
        }

        private async Task OnReconnectTick()
        {
            _timerReconnect.Stop();

            bool success = await _plc.ConnectAsync(txtIP.Text.Trim(), AppSettings.PlcPort, AppSettings.ReconnectTimeoutMs);

            if (success)
            {
                PlcLogger.Info("Reconnected successfully.");
                if (_mainProcess != null) await _mainProcess.RecoverFromPcErrorAsync();
                _timerScan.Start();
            }
            else
            {
                PlcLogger.Warn("Reconnect attempt failed, retrying...");
                _timerReconnect.Start();
            }
        }

        private void UpdateStatusLamp(string name, Color color)
        {
            if (_lamps.TryGetValue(name, out var pnl))
                pnl.BackColor = color;
        }
    }
}
