using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using IoBridge.Core;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Tests
{
    [TestFixture]
    [Category("BridgeStress")]
    [NonParallelizable]
    public class IoBridgeReconnectStressTests
    {
        [Test]
        public async Task PeerDisconnect_ReconnectsAndHandshakes_OneHundredCycles()
        {
            const int cycles = 100;
            int unobservedCount = 0;
            EventHandler<UnobservedTaskExceptionEventArgs> onUnobserved = (sender, args) =>
            {
                Interlocked.Increment(ref unobservedCount);
                args.SetObserved();
            };

            var listener = new TcpListener(IPAddress.Loopback, 0);
            var client = new IcpDasModbusTcpClient { ReadWriteTimeoutMs = 1000 };
            TaskScheduler.UnobservedTaskException += onUnobserved;
            listener.Start();

            try
            {
                int port = ((IPEndPoint)listener.LocalEndpoint).Port;
                for (int cycle = 0; cycle < cycles; cycle++)
                {
                    Task<TcpClient> accept = listener.AcceptTcpClientAsync();
                    Assert.That(await client.ConnectAsync("127.0.0.1", port, 1000), Is.True,
                        $"connect failed at cycle {cycle}");

                    using (TcpClient serverSide = await accept)
                    {
                        Task serverHandshake = ServeHandshake(serverSide);
                        await client.WriteDo(0, true);
                        await client.WriteDo(1, false);
                        await client.WriteDo(2, false);
                        bool[] di = await client.ReadDiStatuses();
                        await serverHandshake;
                        Assert.That(di[0], Is.True, $"invalid DI handshake at cycle {cycle}");
                    }

                    Assert.CatchAsync<Exception>(async () => await client.ReadDiStatuses(),
                        $"peer disconnect was not detected at cycle {cycle}");
                    Assert.That(client.IsConnected, Is.False, $"stale connected state at cycle {cycle}");
                }

                await Task.Delay(100);
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                Assert.That(unobservedCount, Is.Zero);
            }
            finally
            {
                client.Dispose();
                listener.Stop();
                TaskScheduler.UnobservedTaskException -= onUnobserved;
            }
        }

        private static async Task ServeHandshake(TcpClient client)
        {
            NetworkStream stream = client.GetStream();
            for (int requestIndex = 0; requestIndex < 4; requestIndex++)
            {
                byte[] request = await ReadExactly(stream, 12);
                byte function = request[7];
                byte[] response;
                if (function == 5)
                {
                    response = request;
                }
                else if (function == 2)
                {
                    response = new byte[]
                    {
                        request[0], request[1], 0, 0, 0, 4, 1, function, 1, 1
                    };
                }
                else
                {
                    throw new InvalidOperationException($"Unexpected Modbus function {function}");
                }

                await stream.WriteAsync(response, 0, response.Length);
            }
        }

        private static async Task<byte[]> ReadExactly(NetworkStream stream, int count)
        {
            var bytes = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = await stream.ReadAsync(bytes, offset, count - offset);
                if (read == 0) throw new InvalidOperationException("Client closed during handshake");
                offset += read;
            }
            return bytes;
        }
    }
}
