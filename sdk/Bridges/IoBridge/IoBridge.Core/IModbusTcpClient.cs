using System;
using System.Threading.Tasks;

namespace IoBridge.Core
{
    /// <summary>
    /// Modbus TCP 客戶端介面，供 IoGrabController 注入測試用 mock。
    /// </summary>
    public interface IModbusTcpClient : IDisposable
    {
        int ReadWriteTimeoutMs { get; set; }
        bool IsConnected { get; }
        Task<bool> ConnectAsync(string ip, int port = 502, int timeoutMs = 5000);
        Task<bool[]> ReadDoStatuses();
        Task<bool[]> ReadDiStatuses();
        Task WriteDo(int index, bool value);
    }
}
