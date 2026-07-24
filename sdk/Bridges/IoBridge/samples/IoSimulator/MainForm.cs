using System;
using System.Diagnostics;
using System.Drawing;
using System.Windows.Forms;

namespace IoBridge.IoSimulator
{
    /// <summary>
    /// ET-7044 IO 模擬器 GUI（Modbus TCP server）。AniloxRoll.Monitor 把「IO IP」指到本機（同機 127.0.0.1）即連得上。
    /// IO mapping（對齊 app IoGrabController）：
    ///   DI-0 = PLC ALIVE（保持 high，app 才認為 PLC 在線）
    ///   DI-1 = START（grab 訊號；上升緣開抓、下降緣停）← 自動循環這個
    ///   DO-0/1/2 = PC_ALIVE / MURA / PC_BUSY（app 寫，本工具顯示）
    /// 用途：長期測試模擬真實「拍 N 秒 → 停 M 秒」循環取像。
    /// 實體 ET-7044 與本模擬器都使用標準 Modbus TCP Port 502；app 只需切換 IO IP。
    /// </summary>
    public sealed class MainForm : Form
    {
        private readonly ModbusTcpServer _srv = new ModbusTcpServer();
        private TextBox _port, _onSec, _offSec, _log;
        private Button _btnStart, _btnCycle;
        private CheckBox _diAlive, _diStart;
        private Label _do0, _do1, _do2;
        private System.Windows.Forms.Timer _cycleTimer, _doPoll;
        private bool _cycling;
        private int _cycleCount;

        public MainForm()
        {
            Text = "ET-7044 IO 模擬器（Modbus TCP server）";
            ClientSize = new Size(800, 470);
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            Font = new Font("Microsoft JhengHei UI", 9f);
            BuildUi();

            _srv.Log = s => BeginInvoke((Action)(() => AppendLog(s)));
            _srv.OnDoChanged = () => { };   // DO 顯示走 _doPoll 輪詢（避免跨執行緒）
            _srv.SetDi(0, true);            // DI-0 ALIVE 預設 high

            _doPoll = new System.Windows.Forms.Timer { Interval = 200 };
            _doPoll.Tick += (s, e) => RefreshDo();
            _doPoll.Start();
        }

        private void BuildUi()
        {
            int y = 12;
            Controls.Add(new Label { Text = "監聽 Port:", Location = new Point(12, y + 3), AutoSize = true });
            _port = new TextBox { Text = "502", Location = new Point(90, y), Width = 60 };
            Controls.Add(_port);
            _btnStart = new Button { Text = "啟動 server", Location = new Point(160, y - 2), Width = 110 };
            _btnStart.Click += (s, e) => ToggleServer();
            Controls.Add(_btnStart);
            Controls.Add(new Label { Text = "(標準 Modbus TCP Port)", Location = new Point(280, y + 3), AutoSize = true, ForeColor = Color.Gray });

            y += 40;
            Controls.Add(new Label { Text = "── DI（送給 app）──", Location = new Point(12, y), AutoSize = true, ForeColor = Color.DarkBlue });
            y += 26;
            _diAlive = new CheckBox { Text = "DI-0  PLC ALIVE（保持勾）", Location = new Point(20, y), AutoSize = true, Checked = true };
            _diAlive.CheckedChanged += (s, e) => _srv.SetDi(0, _diAlive.Checked);
            Controls.Add(_diAlive);
            y += 26;
            _diStart = new CheckBox { Text = "DI-1  START（grab 訊號；手動切）", Location = new Point(20, y), AutoSize = true };
            _diStart.CheckedChanged += (s, e) => _srv.SetDi(1, _diStart.Checked);
            Controls.Add(_diStart);

            y += 36;
            Controls.Add(new Label { Text = "── 自動循環（DI-1）──", Location = new Point(12, y), AutoSize = true, ForeColor = Color.DarkBlue });
            y += 26;
            Controls.Add(new Label { Text = "開(秒):", Location = new Point(20, y + 3), AutoSize = true });
            _onSec = new TextBox { Text = "10", Location = new Point(70, y), Width = 45 };
            Controls.Add(_onSec);
            Controls.Add(new Label { Text = "停(秒):", Location = new Point(125, y + 3), AutoSize = true });
            _offSec = new TextBox { Text = "1", Location = new Point(175, y), Width = 45 };
            Controls.Add(_offSec);
            _btnCycle = new Button { Text = "開始循環", Location = new Point(235, y - 2), Width = 110 };
            _btnCycle.Click += (s, e) => ToggleCycle();
            Controls.Add(_btnCycle);

            y += 40;
            Controls.Add(new Label { Text = "── DO（app 寫回；亮=High）──", Location = new Point(12, y), AutoSize = true, ForeColor = Color.DarkGreen });
            y += 26;
            _do0 = MakeLed("DO-0 PC_ALIVE", 20, y); Controls.Add(_do0);
            _do1 = MakeLed("DO-1 MURA", 160, y); Controls.Add(_do1);
            _do2 = MakeLed("DO-2 PC_BUSY", 280, y); Controls.Add(_do2);

            y += 38;
            _log = new TextBox { Location = new Point(12, y), Size = new Size(416, 470 - y - 12), Multiline = true, ScrollBars = ScrollBars.Vertical, ReadOnly = true, BackColor = Color.Black, ForeColor = Color.Lime };
            Controls.Add(_log);

            // 右側：使用說明（直接寫在 form 上）
            var help = new TextBox
            {
                Location = new Point(444, 12), Size = new Size(344, 446),
                Multiline = true, ReadOnly = true, ScrollBars = ScrollBars.Vertical,
                BackColor = Color.FromArgb(250, 250, 235), Font = new Font("Microsoft JhengHei UI", 8.5f),
                WordWrap = true,
                // WinForms TextBox 需 \r\n 才換行；原始字串（LF）正規化成 CRLF。
                Text = HelpText.Replace("\r\n", "\n").Replace("\n", "\r\n")
            };
            Controls.Add(help);
        }

        private const string HelpText =
@"【ET-7044 IO 模擬器 — 使用說明】

模擬 PLC 的 IO 訊號連到 AniloxRoll.Monitor，
做長期「拍 N 秒 / 停 M 秒」循環取像測試。

── 步驟 ──
1. 監聽 Port 使用標準 Modbus TCP 502。
   實體 ET-7044 與本模擬器使用相同 Port，
   app 的 IO Port 固定設成 502 即可。
2. 按「啟動 server」。
3. AniloxRoll.Monitor 設定：
   • IO IP = 127.0.0.1（同一台機器）
   • IO Port = 502
   • 啟用 IO = 是
   → app 會連上（DI-0 ALIVE 預設勾，
     app 才認為 PLC 在線）。
4. 按「開始循環」（開 10s / 停 1s）：
   DI-1 START 上升緣 → app 開抓；
   下降緣 → app 停抓。
   = 模擬真實「一次拍 10~20 秒」循環。
5. 下方 DO 燈 = app 寫回的狀態（亮=High）：
   DO-0 PC_ALIVE / DO-1 MURA / DO-2 PC_BUSY。

── 手動模式 ──
不循環時，可直接勾「DI-1 START」
單次手動開 / 停抓。

── IO 對應（ET-7044）──
DI-0 = PLC ALIVE（保持 high）
DI-1 = START（grab 訊號）← 自動循環這個
DO-0 = PC ALIVE（app 開啟=High）
DO-1 = MURA（檢測到瑕疵=High）
DO-2 = PC BUSY（grab 中=High）

── 配合相位量測 ──
循環取像時，MilCamera 會記 phaselog
（Data Latch 硬體時戳）→ 用
tools/python/analyze_phaselog.py 分析
每次循環的相位 offset 是否跨循環漂走。";

        private static Label MakeLed(string text, int x, int y)
            => new Label { Text = text, Location = new Point(x, y), AutoSize = true, BackColor = Color.Gainsboro, Padding = new Padding(4, 2, 4, 2) };

        private void ToggleServer()
        {
            if (_srv.Running) { _srv.Stop(); _btnStart.Text = "啟動 server"; AppendLog("server 停止"); return; }
            try
            {
                _srv.Start(int.Parse(_port.Text));
                _btnStart.Text = "停止 server";
            }
            catch (Exception ex) { AppendLog("啟動失敗：" + ex.Message + "（請檢查 Port 是否已被其他程式占用）"); }
        }

        private void ToggleCycle()
        {
            if (_cycling) { StopCycle(); return; }
            if (!_srv.Running) { AppendLog("請先啟動 server"); return; }
            int onMs = Math.Max(1, ParseInt(_onSec.Text, 10)) * 1000;
            int offMs = Math.Max(1, ParseInt(_offSec.Text, 1)) * 1000;
            _cycling = true; _cycleCount = 0;
            bool high = false;
            long phaseStartedMs = 0;
            long nextTransitionMs = 0;
            var cycleClock = Stopwatch.StartNew();
            _cycleTimer = new System.Windows.Forms.Timer { Interval = 20 };
            _cycleTimer.Tick += (s, e) =>
            {
                long nowMs = cycleClock.ElapsedMilliseconds;
                if (nowMs < nextTransitionMs) return;

                if (_cycleCount > 0)
                {
                    long actualMs = nowMs - phaseStartedMs;
                    AppendLog(
                        $"DI-1 {(high ? "HIGH" : "LOW")} actual={actualMs}ms " +
                        $"target={(high ? onMs : offMs)}ms");
                }

                high = !high;
                phaseStartedMs = nowMs;
                if (high)   // START 上升緣 → app 開抓
                {
                    _cycleCount++;
                    _srv.SetDi(1, true); _diStart.Checked = true;
                    nextTransitionMs = nowMs + onMs;
                    _btnCycle.Text = $"循環中 #{_cycleCount}（拍 {onMs / 1000}s）";
                }
                else        // START 下降緣 → app 停抓
                {
                    _srv.SetDi(1, false); _diStart.Checked = false;
                    nextTransitionMs = nowMs + offMs;
                    _btnCycle.Text = $"循環中 #{_cycleCount}（停 {offMs / 1000}s）";
                }
            };
            _cycleTimer.Start();
            AppendLog($"自動循環啟動：DI-1 開 {onMs / 1000}s / 停 {offMs / 1000}s");
        }

        private void StopCycle()
        {
            _cycling = false;
            _cycleTimer?.Stop(); _cycleTimer = null;
            _srv.SetDi(1, false); _diStart.Checked = false;
            _btnCycle.Text = "開始循環";
            AppendLog($"循環停止（共 {_cycleCount} 次）");
        }

        private void RefreshDo()
        {
            SetLed(_do0, _srv.GetDo(0));
            SetLed(_do1, _srv.GetDo(1));
            SetLed(_do2, _srv.GetDo(2));
        }

        private static void SetLed(Label l, bool on)
            => l.BackColor = on ? Color.LimeGreen : Color.Gainsboro;

        private static int ParseInt(string s, int def) => int.TryParse(s, out int v) ? v : def;

        private void AppendLog(string s)
        {
            _log.AppendText($"{DateTime.Now:HH:mm:ss}  {s}\r\n");
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            try { _cycleTimer?.Stop(); } catch { }
            try { _srv.Stop(); } catch { }
            base.OnFormClosing(e);
        }
    }
}
