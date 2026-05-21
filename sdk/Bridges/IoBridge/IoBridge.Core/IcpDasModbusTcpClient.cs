using System;
using System.Collections;
using System.Linq;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace IoBridge.Core
{
    public class IcpDasModbusTcpClient : IModbusTcpClient
    {
        private TcpClient _client;
        private NetworkStream _stream;
        private ushort _transactionId = 0;
        private readonly SemaphoreSlim _txLock = new SemaphoreSlim(1, 1);

        /// <summary>讀寫逾時（ms），預設 2000。由各 App 的 AppSettings 設定。</summary>
        public int ReadWriteTimeoutMs { get; set; } = 2000;

        public bool IsConnected => _client != null && _client.Connected;

        public async Task<bool> ConnectAsync(string ip, int port = 502, int timeoutMs = 5000)
        {
            Dispose();
            var tempClient = new TcpClient();

            try
            {
                var connectTask = tempClient.ConnectAsync(ip, port);
                var timeoutTask = Task.Delay(timeoutMs);
                var completedTask = await Task.WhenAny(connectTask, timeoutTask);

                if (completedTask == timeoutTask)
                {
                    _ = connectTask.ContinueWith(t =>
                    {
                        try { tempClient.Close(); } catch { }
                    });
                    return false;
                }

                await connectTask;
                if (tempClient.Connected)
                {
                    _client = tempClient;
                    _stream = _client.GetStream();
                    return true;
                }

                tempClient.Close();
                return false;
            }
            catch (SocketException)
            {
                try { tempClient.Close(); } catch { }
                return false;
            }
            catch (Exception)
            {
                try { tempClient.Close(); } catch { }
                return false;
            }
        }

        public async Task<bool[]> ReadDoStatuses() => await ReadBits(1, 0, 8);
        public async Task<bool[]> ReadDiStatuses() => await ReadBits(2, 0, 8);

        public async Task WriteDo(int index, bool value)
        {
            byte[] packet = CreateHeader(5, (ushort)index, (ushort)(value ? 0xFF00 : 0x0000));
            await SendAndReceive(packet, 12);
        }

        private byte[] CreateHeader(byte func, ushort addr, ushort val)
        {
            _transactionId++;
            return new byte[]
            {
                (byte)(_transactionId >> 8), (byte)_transactionId,
                0, 0, 0, 6, 1, func,
                (byte)(addr >> 8), (byte)addr,
                (byte)(val >> 8), (byte)val
            };
        }

        private async Task<bool[]> ReadBits(byte func, ushort addr, ushort count)
        {
            byte[] res = await SendAndReceive(CreateHeader(func, addr, count), 10);
            BitArray ba = new BitArray(new byte[] { res[9] });
            return ba.Cast<bool>().Take(count).ToArray();
        }

        private async Task<byte[]> SendAndReceive(byte[] send, int expected)
        {
            if (!IsConnected) throw new InvalidOperationException("Not connected");

            int timeoutMs = ReadWriteTimeoutMs;
            await _txLock.WaitAsync().ConfigureAwait(false);
            try
            {
                // 取得鎖後再次確認連線（可能在等鎖期間被其他執行緒 Dispose）
                var stream = _stream;
                if (stream == null) throw new InvalidOperationException("Not connected");

                var writeTask = stream.WriteAsync(send, 0, send.Length);
                if (await Task.WhenAny(writeTask, Task.Delay(timeoutMs)) != writeTask)
                {
                    Dispose();
                    throw new TimeoutException("Write timeout");
                }
                await writeTask;

                byte[] res = new byte[expected];
                int totalRead = 0;
                while (totalRead < expected)
                {
                    var readTask = stream.ReadAsync(res, totalRead, expected - totalRead);
                    if (await Task.WhenAny(readTask, Task.Delay(timeoutMs)) != readTask)
                    {
                        Dispose();
                        throw new TimeoutException("Read timeout");
                    }
                    int read = await readTask;
                    if (read <= 0)
                    {
                        Dispose();
                        throw new InvalidOperationException("Connection closed by peer");
                    }
                    totalRead += read;
                }
                return res;
            }
            catch (SocketException)
            {
                Dispose();
                throw;
            }
            catch (Exception)
            {
                Dispose();
                throw;
            }
            finally
            {
                _txLock.Release();
            }
        }

        public void Dispose()
        {
            try { _stream?.Dispose(); } catch { }
            try { _client?.Close(); } catch { }
            _stream = null;
            _client = null;
        }
    }
}
