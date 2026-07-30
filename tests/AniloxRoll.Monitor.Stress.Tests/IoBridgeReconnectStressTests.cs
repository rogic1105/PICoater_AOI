using System;
using System.Collections.Generic;
using System.Diagnostics;
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

        [Test]
        public async Task SustainedPolling_ReusesKernelResources()
        {
            const int warmupPolls = 50;
            const int measuredPolls = 1000;
            var listener = new TcpListener(IPAddress.Loopback, 0);
            var client = new IcpDasModbusTcpClient { ReadWriteTimeoutMs = 1000 };
            listener.Start();

            try
            {
                int port = ((IPEndPoint)listener.LocalEndpoint).Port;
                Task<TcpClient> accept = listener.AcceptTcpClientAsync();
                Assert.That(
                    await client.ConnectAsync("127.0.0.1", port, 1000),
                    Is.True);

                using (TcpClient serverSide = await accept)
                {
                    Task server = Task.Run(
                        () => ServeReadPollsBlocking(
                            serverSide.Client,
                            warmupPolls + measuredPolls));

                    for (int i = 0; i < warmupPolls; i++)
                        await client.ReadDiStatuses();

                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    int handlesBefore;
                    using (Process process = Process.GetCurrentProcess())
                        handlesBefore = process.HandleCount;

                    for (int i = 0; i < measuredPolls; i++)
                        await client.ReadDiStatuses();

                    await server;
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    int handlesAfter;
                    using (Process process = Process.GetCurrentProcess())
                        handlesAfter = process.HandleCount;

                    Assert.That(
                        handlesAfter - handlesBefore,
                        Is.LessThanOrEqualTo(10),
                        $"Sustained Modbus polling accumulated kernel handles: " +
                        $"{handlesBefore} -> {handlesAfter}");
                }
            }
            finally
            {
                client.Dispose();
                listener.Stop();
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

        private static void ServeReadPollsBlocking(
            Socket socket,
            int requestCount)
        {
            for (int requestIndex = 0; requestIndex < requestCount; requestIndex++)
            {
                byte[] request = ReadExactlyBlocking(socket, 12);
                if (request[7] != 2)
                    throw new InvalidOperationException(
                        $"Unexpected Modbus function {request[7]}");

                byte[] response =
                {
                    request[0], request[1], 0, 0, 0, 4, 1, 2, 1, 1
                };
                SendExactlyBlocking(socket, response);
            }
        }

        private static byte[] ReadExactlyBlocking(
            Socket socket,
            int count)
        {
            var bytes = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = socket.Receive(
                    bytes,
                    offset,
                    count - offset,
                    SocketFlags.None);
                if (read <= 0)
                    throw new InvalidOperationException(
                        "Client closed during sustained polling");
                offset += read;
            }
            return bytes;
        }

        private static void SendExactlyBlocking(
            Socket socket,
            byte[] bytes)
        {
            int offset = 0;
            while (offset < bytes.Length)
            {
                int sent = socket.Send(
                    bytes,
                    offset,
                    bytes.Length - offset,
                    SocketFlags.None);
                if (sent <= 0)
                    throw new InvalidOperationException(
                        "Client closed during sustained polling response");
                offset += sent;
            }
        }
    }
}
