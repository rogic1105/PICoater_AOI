using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace IoBridge.IoSimulator
{
    /// <summary>
    /// 最小 Modbus TCP server，模擬 ET-7044：回應 AniloxRoll.Monitor 的 IcpDasModbusTcpClient。
    /// 支援的 function code（對應 client 的呼叫）：
    ///   FC 0x02 Read Discrete Inputs（client ReadDiStatuses，addr0 count8）→ 回 DI 8 bits
    ///   FC 0x01 Read Coils          （client ReadDoStatuses，addr0 count8）→ 回 DO 8 bits
    ///   FC 0x05 Write Single Coil    （client WriteDo(index,val)）          → 設 DO[index]、echo
    /// DI/DO 各 8 bit，外部（Form）直接讀寫 Di[]/Do[]。執行緒安全靠 volatile byte（單 byte 原子）。
    /// </summary>
    public sealed class ModbusTcpServer
    {
        private TcpListener _listener;
        private Thread _acceptThread;
        private volatile bool _running;

        private volatile int _di;   // 8 bits（bit0=DI0…）；外部設
        private volatile int _do;   // 8 bits；client 寫入後更新

        public bool Running => _running;
        public Action<string> Log;                // UI log callback
        public Action OnDoChanged;                // DO 被 client 寫入時通知 UI

        public bool GetDi(int bit) => (_di & (1 << bit)) != 0;
        public void SetDi(int bit, bool v) { if (v) _di |= (1 << bit); else _di &= ~(1 << bit); }
        public bool GetDo(int bit) => (_do & (1 << bit)) != 0;

        public void Start(int port)
        {
            Stop();
            _listener = new TcpListener(IPAddress.Any, port);
            _listener.Start();
            _running = true;
            _acceptThread = new Thread(AcceptLoop) { IsBackground = true };
            _acceptThread.Start();
            Log?.Invoke($"server 啟動，監聽 {port}（DI-0=ALIVE 預設 high、DI-1=START）");
        }

        public void Stop()
        {
            _running = false;
            try { _listener?.Stop(); } catch { }
            _listener = null;
        }

        private void AcceptLoop()
        {
            while (_running)
            {
                TcpClient client;
                try { client = _listener.AcceptTcpClient(); }
                catch { break; }
                Log?.Invoke($"client 連線：{client.Client.RemoteEndPoint}");
                var t = new Thread(() => Serve(client)) { IsBackground = true };
                t.Start();
            }
        }

        private void Serve(TcpClient client)
        {
            try
            {
                using (client)
                using (var stream = client.GetStream())
                {
                    var req = new byte[12];
                    while (_running)
                    {
                        // Modbus TCP 請求（此 client 固定 12 bytes：MBAP6 + unit + func + addr2 + val2）
                        int got = 0;
                        while (got < 12)
                        {
                            int n = stream.Read(req, got, 12 - got);
                            if (n <= 0) return;   // 連線關閉
                            got += n;
                        }
                        byte[] resp = BuildResponse(req);
                        if (resp != null) stream.Write(resp, 0, resp.Length);
                    }
                }
            }
            catch { /* 連線斷 → 結束此執行緒 */ }
        }

        private byte[] BuildResponse(byte[] r)
        {
            ushort tid = (ushort)((r[0] << 8) | r[1]);
            byte unit = r[6], func = r[7];
            ushort addr = (ushort)((r[8] << 8) | r[9]);
            ushort val = (ushort)((r[10] << 8) | r[11]);

            if (func == 0x02 || func == 0x01)   // Read Discrete Inputs(DI) / Read Coils(DO)
            {
                byte data = (byte)(func == 0x02 ? _di : _do);
                return new byte[]
                {
                    (byte)(tid >> 8), (byte)tid, 0, 0, 0, 4, unit, func, 1, data
                };
            }
            if (func == 0x05)   // Write Single Coil（DO[addr] = val==0xFF00）
            {
                bool on = val == 0xFF00;
                if (addr < 8) { if (on) _do |= (1 << addr); else _do &= ~(1 << addr); }
                OnDoChanged?.Invoke();
                return new byte[]   // echo 12 bytes
                { r[0], r[1], 0, 0, 0, 6, unit, func, r[8], r[9], r[10], r[11] };
            }
            // 不支援的 func：回 exception（func|0x80, code 0x01）
            return new byte[] { (byte)(tid >> 8), (byte)tid, 0, 0, 0, 3, unit, (byte)(func | 0x80), 0x01 };
        }
    }
}
