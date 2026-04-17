using System;
using System.Diagnostics;
using System.IO.Ports;
using System.Text;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// LTS-3DPA24 光源控制器 RS-232 通訊。
    /// 協定：8-byte ASCII，9600/8N1，格式 $CMD CH 0XX CHECKSUM。
    /// </summary>
    public class LightController : IDisposable
    {
        private const int BaudRate = 9600;
        private const int ResponseTimeoutMs = 1000;

        private SerialPort _port;
        private readonly object _lock = new object();

        public bool IsConnected => _port != null && _port.IsOpen;

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
