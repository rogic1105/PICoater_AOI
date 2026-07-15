using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Services;
using IoBridge.Core;
using NUnit.Framework;

namespace AniloxRoll.Monitor.Integration.Tests
{
    [TestFixture]
    [NonParallelizable]
    public class IcpDasModbusTcpClientIntegrationTests
    {
        [Test]
        public async Task ReadTimeout_ObservesLateFailure_AndAllowsReconnect()
        {
            int unobservedCount = 0;
            EventHandler<UnobservedTaskExceptionEventArgs> onUnobserved = (sender, args) =>
            {
                Interlocked.Increment(ref unobservedCount);
                args.SetObserved();
            };

            var listener = new TcpListener(IPAddress.Loopback, 0);
            var client = new IcpDasModbusTcpClient { ReadWriteTimeoutMs = 50 };
            TaskScheduler.UnobservedTaskException += onUnobserved;
            listener.Start();

            try
            {
                int port = ((IPEndPoint)listener.LocalEndpoint).Port;
                Task<TcpClient> firstAccept = listener.AcceptTcpClientAsync();
                Assert.That(await client.ConnectAsync("127.0.0.1", port, 1000), Is.True);

                using (TcpClient serverSide = await firstAccept)
                {
                    Assert.ThrowsAsync<TimeoutException>(async () => await client.ReadDiStatuses());
                    Assert.That(client.IsConnected, Is.False);
                }

                Task<TcpClient> secondAccept = listener.AcceptTcpClientAsync();
                Assert.That(await client.ConnectAsync("127.0.0.1", port, 1000), Is.True);
                using (TcpClient serverSide = await secondAccept)
                {
                    Assert.That(client.IsConnected, Is.True);
                }

                client.Dispose();
                await Task.Delay(100);
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                Assert.That(unobservedCount, Is.Zero,
                    "Timed-out NetworkStream tasks must be observed instead of reaching the process-wide handler.");
            }
            finally
            {
                client.Dispose();
                listener.Stop();
                TaskScheduler.UnobservedTaskException -= onUnobserved;
            }
        }

        [Test]
        public async Task ControllerStartsBeforeServer_ReconnectsWhenServerAppears()
        {
            int port;
            var reservation = new TcpListener(IPAddress.Loopback, 0);
            reservation.Start();
            port = ((IPEndPoint)reservation.LocalEndpoint).Port;
            reservation.Stop();

            var listener = new TcpListener(IPAddress.Loopback, port);
            var transport = new IcpDasModbusTcpClient { ReadWriteTimeoutMs = 250 };
            var controller = new IoGrabController(transport)
            {
                PollIntervalMs = 20,
                ReconnectIntervalMs = 50
            };
            var connected = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            controller.OnConnectionChanged += value =>
            {
                if (value) connected.TrySetResult(true);
            };

            Task server = null;
            try
            {
                await controller.StartAsync("127.0.0.1", port);
                Assert.That(controller.IsConnected, Is.False,
                    "The controller must stay alive and disconnected while the module is powered off.");

                listener.Start();
                server = ServeModbusUntilDisconnected(await listener.AcceptTcpClientAsync());

                Task winner = await Task.WhenAny(connected.Task, Task.Delay(3000));
                Assert.That(winner, Is.SameAs(connected.Task),
                    "The background loop must accept a module that powers on after the application.");
                Assert.That(controller.IsConnected, Is.True);
                Assert.That(controller.CurrentState, Is.EqualTo(IoState.Idle));
            }
            finally
            {
                await controller.StopAsync();
                listener.Stop();
                if (server != null) await server;
                controller.Dispose();
            }
        }

        private static async Task ServeModbusUntilDisconnected(TcpClient client)
        {
            using (client)
            {
                NetworkStream stream = client.GetStream();
                try
                {
                    while (true)
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
                catch (IOException) { }
                catch (ObjectDisposedException) { }
            }
        }

        private static async Task<byte[]> ReadExactly(NetworkStream stream, int count)
        {
            var bytes = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = await stream.ReadAsync(bytes, offset, count - offset);
                if (read == 0) throw new IOException("Client disconnected");
                offset += read;
            }
            return bytes;
        }
    }
}
