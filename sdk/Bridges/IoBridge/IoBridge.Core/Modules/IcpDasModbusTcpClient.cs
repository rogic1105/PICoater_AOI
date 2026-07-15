using System;
using System.Collections;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace IoBridge.Core
{
    /// <summary>
    /// Serialized Modbus TCP client for ICP DAS ET-series modules. A timed-out
    /// transport is closed immediately and any late task failure is observed.
    /// </summary>
    public class IcpDasModbusTcpClient : IModbusTcpClient
    {
        private Socket _client;
        private NetworkStream _stream;
        private ushort _transactionId = 0;
        private readonly SemaphoreSlim _txLock = new SemaphoreSlim(1, 1);
        private readonly SemaphoreSlim _connectLock = new SemaphoreSlim(1, 1);
        private readonly object _transportSync = new object();
        private long _disconnectGeneration;

        /// <summary>讀寫逾時（ms），預設 2000。由各 App 的 AppSettings 設定。</summary>
        public int ReadWriteTimeoutMs { get; set; } = 2000;

        public bool IsConnected
        {
            get
            {
                lock (_transportSync)
                    return _client != null && _client.Connected;
            }
        }

        /// <summary>Connects one socket at a time; Dispose invalidates an in-flight attempt.</summary>
        public async Task<bool> ConnectAsync(string ip, int port = 502, int timeoutMs = 5000)
        {
            // Capture before waiting: Dispose must also invalidate attempts already
            // queued behind another ConnectAsync, not only the active socket attempt.
            long generation = Interlocked.Read(ref _disconnectGeneration);
            await _connectLock.WaitAsync().ConfigureAwait(false);
            try
            {
                if (generation != Interlocked.Read(ref _disconnectGeneration))
                    return false;

                CloseTransport(invalidatePendingConnect: false);
                var tempClient = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
                NetworkStream tempStream = null;
                SocketAsyncEventArgs connectArgs = null;
                EventHandler<SocketAsyncEventArgs> connectCompleted = null;

                try
                {
                    var connectTcs = new TaskCompletionSource<SocketError>(
                        TaskCreationOptions.RunContinuationsAsynchronously);
                    connectArgs = new SocketAsyncEventArgs
                    {
                        RemoteEndPoint = new DnsEndPoint(ip, port)
                    };
                    connectCompleted = (sender, args) => connectTcs.TrySetResult(args.SocketError);
                    connectArgs.Completed += connectCompleted;

                    bool pending = tempClient.ConnectAsync(connectArgs);
                    if (!pending) connectTcs.TrySetResult(connectArgs.SocketError);
                    Task<SocketError> connectTask = connectTcs.Task;
                    var completedTask = await Task.WhenAny(connectTask, Task.Delay(timeoutMs)).ConfigureAwait(false);

                    if (completedTask != connectTask && !connectTask.IsCompleted)
                    {
                        CloseSocket(null, tempClient);
                        DisposeConnectArgsWhenComplete(connectTask, connectArgs, connectCompleted);
                        connectArgs = null;
                        return false;
                    }

                    SocketError connectError = await connectTask.ConfigureAwait(false);
                    DisposeConnectArgs(connectArgs, connectCompleted);
                    connectArgs = null;
                    if (connectError != SocketError.Success)
                    {
                        CloseSocket(null, tempClient);
                        return false;
                    }

                    if (!tempClient.Connected)
                    {
                        CloseSocket(null, tempClient);
                        return false;
                    }

                    tempStream = new NetworkStream(tempClient, ownsSocket: true);
                    lock (_transportSync)
                    {
                        // Dispose during an in-flight connect invalidates that attempt;
                        // it must not publish a new socket after shutdown/reconfiguration.
                        if (generation != _disconnectGeneration)
                        {
                            CloseSocket(tempStream, tempClient);
                            return false;
                        }

                        _client = tempClient;
                        _stream = tempStream;
                    }
                    return true;
                }
                catch (SocketException)
                {
                    DisposeConnectArgs(connectArgs, connectCompleted);
                    CloseSocket(tempStream, tempClient);
                    return false;
                }
                catch (Exception)
                {
                    DisposeConnectArgs(connectArgs, connectCompleted);
                    CloseSocket(tempStream, tempClient);
                    return false;
                }
            }
            finally
            {
                _connectLock.Release();
            }
        }

        public async Task<bool[]> ReadDoStatuses() => await ReadBits(1, 0, 8);
        public async Task<bool[]> ReadDiStatuses() => await ReadBits(2, 0, 8);

        public async Task WriteDo(int index, bool value)
        {
            await SendAndReceive(5, (ushort)index, (ushort)(value ? 0xFF00 : 0x0000), 12);
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
            byte[] res = await SendAndReceive(func, addr, count, 10);
            BitArray ba = new BitArray(new byte[] { res[9] });
            return ba.Cast<bool>().Take(count).ToArray();
        }

        private async Task<byte[]> SendAndReceive(byte func, ushort addr, ushort value, int expected)
        {
            int timeoutMs = ReadWriteTimeoutMs;
            await _txLock.WaitAsync().ConfigureAwait(false);
            try
            {
                // Build the packet under the same lock as the transaction. This keeps
                // transaction IDs ordered when poll and output notifications overlap.
                byte[] send = CreateHeader(func, addr, value);

                // 取得鎖後再次確認連線（可能在等鎖期間被其他執行緒 Dispose）
                NetworkStream stream;
                lock (_transportSync) stream = _stream;
                if (stream == null) throw new InvalidOperationException("Not connected");

                var writeTask = stream.WriteAsync(send, 0, send.Length);
                if (await Task.WhenAny(writeTask, Task.Delay(timeoutMs)).ConfigureAwait(false) != writeTask)
                {
                    ObserveLateFault(writeTask);
                    Dispose();
                    throw new TimeoutException("Write timeout");
                }
                await writeTask.ConfigureAwait(false);

                byte[] res = new byte[expected];
                int totalRead = 0;
                while (totalRead < expected)
                {
                    var readTask = stream.ReadAsync(res, totalRead, expected - totalRead);
                    if (await Task.WhenAny(readTask, Task.Delay(timeoutMs)).ConfigureAwait(false) != readTask)
                    {
                        ObserveLateFault(readTask);
                        Dispose();
                        throw new TimeoutException("Read timeout");
                    }
                    int read = await readTask.ConfigureAwait(false);
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
            CloseTransport(invalidatePendingConnect: true);
        }

        private void CloseTransport(bool invalidatePendingConnect)
        {
            NetworkStream stream;
            Socket client;
            lock (_transportSync)
            {
                if (invalidatePendingConnect) _disconnectGeneration++;
                stream = _stream;
                client = _client;
                _stream = null;
                _client = null;
            }
            CloseSocket(stream, client);
        }

        private static void CloseSocket(NetworkStream stream, Socket client)
        {
            try { stream?.Dispose(); } catch { }
            try { client?.Dispose(); } catch { }
        }

        private static void DisposeConnectArgsWhenComplete(
            Task<SocketError> connectTask,
            SocketAsyncEventArgs args,
            EventHandler<SocketAsyncEventArgs> completed)
        {
            _ = connectTask.ContinueWith(
                task => DisposeConnectArgs(args, completed),
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }

        private static void DisposeConnectArgs(
            SocketAsyncEventArgs args,
            EventHandler<SocketAsyncEventArgs> completed)
        {
            if (args == null) return;
            if (completed != null) args.Completed -= completed;
            try { args.Dispose(); } catch { }
        }

        private static void ObserveLateFault(Task task)
        {
            // NetworkStream async APIs on .NET Framework have no usable cancellation
            // token. Closing the socket aborts them; this continuation
            // consumes the resulting late exception before the finalizer reports it.
            _ = task.ContinueWith(
                t => { _ = t.Exception; },
                CancellationToken.None,
                TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
    }
}
