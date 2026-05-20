using System;
using System.Diagnostics;
using System.IO.Ports;
using System.Text;

namespace LightBridge.Core
{
    /// <summary>
    /// LTS-3DPA24 光源控制器 RS-232 通訊。
    /// 協定：8-byte ASCII，9600/8N1，格式 $CMD CH 0XX CHECKSUM。
    /// </summary>
    public class LightController : IDisposable
    {
        private const int BaudRate = 9600;
        private const int ResponseTimeoutMs = 1000;
        private const int ProbeTimeoutMs = 400;

        private SerialPort _port;
        private readonly object _lock = new object();

        public bool IsConnected => _port != null && _port.IsOpen;

        /// <summary>連線成功後記錄實際使用的 COM port（可能與 settings 不同，由 AutoDetect 決定）。</summary>
        public string ActiveComPort { get; private set; }

        public bool Connect(string comPort)
        {
            lock (_lock)
            {
                try
                {
                    if (_port != null && _port.IsOpen)
                        _port.Close();

                    _port = new SerialPort(comPort, BaudRate, Parity.None, 8, StopBits.One)
                    {
                        ReadTimeout = ResponseTimeoutMs,
                        WriteTimeout = ResponseTimeoutMs
                    };
                    _port.Open();
                    ActiveComPort = comPort;
                    Trace.WriteLine($"[LightController] Connected: {comPort}");
                    return true;
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[LightController] Connect failed: {ex.Message}");
                    return false;
                }
            }
        }

        /// <summary>
        /// 自動偵測：先試 preferredComPort，失敗則掃描所有可用 port。
        /// 成功會維持連線；失敗回傳 null 且 IsConnected=false。
        /// </summary>
        /// <param name="preferredComPort">檢測設定記錄的 COM（優先嘗試）</param>
        /// <param name="channel">要探測的通道（與設定一致）</param>
        /// <returns>連線成功的 COM port 名稱；NA 回傳 null</returns>
        public string AutoDetect(string preferredComPort, int channel)
        {
            // 1. 先試設定檔記錄的 port
            if (!string.IsNullOrWhiteSpace(preferredComPort) &&
                TryProbeAndConnect(preferredComPort, channel, out _))
            {
                Trace.WriteLine($"[LightController] AutoDetect: 使用設定 {preferredComPort}");
                return preferredComPort;
            }

            // 2. 掃描所有可用 port（排除已試過的 preferred）
            string[] ports = SerialPort.GetPortNames();
            Array.Sort(ports);
            foreach (string p in ports)
            {
                if (string.Equals(p, preferredComPort, StringComparison.OrdinalIgnoreCase)) continue;
                if (TryProbeAndConnect(p, channel, out _))
                {
                    Trace.WriteLine($"[LightController] AutoDetect: 掃描找到 {p}（設定原為 {preferredComPort}）");
                    return p;
                }
            }

            Trace.WriteLine($"[LightController] AutoDetect: 掃描完 {ports.Length} port，無光源回應 (NA)");
            ActiveComPort = null;
            return null;
        }

        /// <summary>
        /// 對指定 port 送出 cmd=4 (read brightness) 並嚴格驗證回應 8-byte。
        /// 通過則保持連線（外部可直接使用），失敗則關閉 port。
        /// </summary>
        private bool TryProbeAndConnect(string comPort, int channel, out byte readBrightness)
        {
            readBrightness = 0;
            SerialPort tmp = null;
            try
            {
                tmp = new SerialPort(comPort, BaudRate, Parity.None, 8, StopBits.One)
                {
                    ReadTimeout = ProbeTimeoutMs,
                    WriteTimeout = ProbeTimeoutMs
                };
                tmp.Open();

                string probe = BuildCommand(4, channel, 0);
                tmp.DiscardInBuffer();
                tmp.Write(probe);

                var buf = new byte[8];
                int read = 0;
                var sw = Stopwatch.StartNew();
                while (read < 8 && sw.ElapsedMilliseconds < ProbeTimeoutMs)
                {
                    if (tmp.BytesToRead > 0)
                        read += tmp.Read(buf, read, Math.Min(tmp.BytesToRead, 8 - read));
                }

                if (!ValidateReadResponse(buf, read, channel, out readBrightness))
                {
                    tmp.Close();
                    tmp.Dispose();
                    return false;
                }

                // 驗證通過 → 取代當前連線
                lock (_lock)
                {
                    if (_port != null && _port.IsOpen) _port.Close();
                    _port = tmp;
                    _port.ReadTimeout = ResponseTimeoutMs;
                    _port.WriteTimeout = ResponseTimeoutMs;
                    ActiveComPort = comPort;
                }
                return true;
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[LightController] Probe {comPort} failed: {ex.GetType().Name}");
                try { tmp?.Close(); } catch { }
                tmp?.Dispose();
                return false;
            }
        }

        /// <summary>
        /// 嚴格驗證 read brightness 回應（PDF §4.1.4 表-4 格式）：
        /// 8 bytes，格式 `$ + '4' + ch + 0XX + checksum`，checksum = XOR 前 6 bytes。
        /// </summary>
        internal static bool ValidateReadResponse(byte[] buf, int length, int expectedChannel, out byte value)
        {
            value = 0;
            if (buf == null || length < 8) return false;
            if (buf[0] != (byte)'$') return false;
            if (buf[1] != (byte)'4') return false;
            if (buf[2] != (byte)('0' + expectedChannel)) return false;
            if (buf[3] != (byte)'0') return false;

            byte xor = 0;
            for (int i = 0; i < 6; i++) xor ^= buf[i];
            string expectedCks = xor.ToString("X2");
            if ((char)buf[6] != expectedCks[0] || (char)buf[7] != expectedCks[1]) return false;

            string hex = "" + (char)buf[4] + (char)buf[5];
            return byte.TryParse(hex, System.Globalization.NumberStyles.HexNumber,
                System.Globalization.CultureInfo.InvariantCulture, out value);
        }

        /// <summary>
        /// 向裝置送 read brightness 並驗證 8-byte 回應。失敗時關閉 port（IsConnected 變 false）。
        /// 用於定期健康檢查：SerialPort.IsOpen 偵測不到實體拔線，必須實際溝通才算數。
        /// </summary>
        public bool Probe(int channel)
        {
            // lock 被佔用 = SendCommand/ReadBrightness 正在執行 = I/O 剛發生 = 線還通
            // 直接跳過這輪，避免與取像期間的光源開關操作競爭（不影響取像時序）
            if (!System.Threading.Monitor.TryEnter(_lock, 0))
                return true;
            try
            {
                if (_port == null || !_port.IsOpen) return false;
                try
                {
                    string cmd = BuildCommand(4, channel, 0);
                    _port.DiscardInBuffer();
                    _port.Write(cmd);

                    var buf = new byte[8];
                    int read = 0;
                    var sw = Stopwatch.StartNew();
                    while (read < 8 && sw.ElapsedMilliseconds < ProbeTimeoutMs)
                    {
                        if (_port.BytesToRead > 0)
                            read += _port.Read(buf, read, Math.Min(_port.BytesToRead, 8 - read));
                    }

                    byte dummy;
                    if (ValidateReadResponse(buf, read, channel, out dummy)) return true;
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[LightController] Probe error: {ex.Message}");
                }

                // 回應不符或 I/O 例外 → 視為斷線，關閉 port
                try { _port.Close(); } catch { }
                return false;
            }
            finally
            {
                System.Threading.Monitor.Exit(_lock);
            }
        }

        public void Disconnect()
        {
            lock (_lock)
            {
                try
                {
                    if (_port != null && _port.IsOpen)
                        _port.Close();
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[LightController] Disconnect error: {ex.Message}");
                }
                finally
                {
                    ActiveComPort = null;
                }
            }
        }

        /// <summary>設定亮度並開啟通道。</summary>
        public bool TurnOn(int channel, int brightness)
        {
            if (brightness < 0) brightness = 0;
            if (brightness > 255) brightness = 255;

            bool ok = SendCommand(3, channel, brightness);
            if (ok) ok = SendCommand(1, channel, brightness);
            return ok;
        }

        /// <summary>關閉通道。</summary>
        public bool TurnOff(int channel)
        {
            return SendCommand(2, channel, 0);
        }

        /// <summary>設定亮度（不改變開關狀態）。</summary>
        public bool SetBrightness(int channel, int brightness)
        {
            if (brightness < 0) brightness = 0;
            if (brightness > 255) brightness = 255;
            return SendCommand(3, channel, brightness);
        }

        /// <summary>讀取通道亮度。回傳 -1 表示失敗。</summary>
        public int ReadBrightness(int channel)
        {
            string cmd = BuildCommand(4, channel, 0);
            lock (_lock)
            {
                try
                {
                    if (!IsConnected) return -1;
                    _port.DiscardInBuffer();
                    _port.Write(cmd);
                    var buf = new byte[8];
                    int read = 0;
                    var sw = Stopwatch.StartNew();
                    while (read < 8 && sw.ElapsedMilliseconds < ResponseTimeoutMs)
                    {
                        if (_port.BytesToRead > 0)
                            read += _port.Read(buf, read, Math.Min(_port.BytesToRead, 8 - read));
                    }
                    if (read >= 8)
                    {
                        string resp = Encoding.ASCII.GetString(buf, 0, read);
                        if (resp[0] == '$' && resp.Length >= 6)
                        {
                            string hexVal = resp.Substring(3, 3);
                            return Convert.ToInt32(hexVal, 16);
                        }
                    }
                    return -1;
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[LightController] ReadBrightness error: {ex.Message}");
                    return -1;
                }
            }
        }

        private bool SendCommand(int cmd, int channel, int value)
        {
            string command = BuildCommand(cmd, channel, value);
            lock (_lock)
            {
                try
                {
                    if (!IsConnected) return false;
                    _port.DiscardInBuffer();
                    _port.Write(command);

                    var sw = Stopwatch.StartNew();
                    while (_port.BytesToRead < 1 && sw.ElapsedMilliseconds < ResponseTimeoutMs) { }

                    if (_port.BytesToRead > 0)
                    {
                        char resp = (char)_port.ReadByte();
                        if (resp == '$') return true;
                        Trace.WriteLine($"[LightController] Command failed: {command} → {resp}");
                        return false;
                    }

                    Trace.WriteLine($"[LightController] Command timeout: {command}");
                    return false;
                }
                catch (Exception ex)
                {
                    Trace.WriteLine($"[LightController] SendCommand error: {ex.Message}");
                    return false;
                }
            }
        }

        /// <summary>
        /// 組合 8-byte ASCII 命令。
        /// 格式：$ + 命令字 + 通道字 + 0XX(hex) + 校驗(2byte)
        /// </summary>
        internal static string BuildCommand(int cmd, int channel, int value)
        {
            string hexValue = value.ToString("X2");
            string data = "0" + hexValue;

            char cCmd = (char)('0' + cmd);
            char cCh = (char)('0' + channel);

            byte xor = (byte)'$';
            xor ^= (byte)cCmd;
            xor ^= (byte)cCh;
            foreach (char c in data)
                xor ^= (byte)c;

            string checksum = xor.ToString("X2");
            return "$" + cCmd + cCh + data + checksum;
        }

        public void Dispose()
        {
            Disconnect();
            _port?.Dispose();
        }
    }
}
