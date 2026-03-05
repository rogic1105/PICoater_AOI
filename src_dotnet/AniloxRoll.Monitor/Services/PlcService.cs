using System;
using AniloxRoll.Monitor.Core.Interop;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class PlcService : IDisposable
    {
        private IntPtr _plcHandle = IntPtr.Zero;

        public void InitializeMock()
        {
            if (_plcHandle != IntPtr.Zero)
            {
                return;
            }

            _plcHandle = NativeMethods.PICoaterAPI_CreateMockPlc();
            if (_plcHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create mock PLC handle.");
            }
        }

        public void Connect()
        {
            EnsureInitialized();
            int result = NativeMethods.PICoaterAPI_PlcConnect(_plcHandle);
            if (result != 0)
            {
                throw new InvalidOperationException($"Failed to connect PLC. Native error code: {result}.");
            }
        }

        public bool ReadBit(int address)
        {
            EnsureInitialized();
            int result = NativeMethods.PICoaterAPI_PlcReadBit(_plcHandle, address, out bool value);
            if (result != 0)
            {
                throw new InvalidOperationException(
                    $"Failed to read PLC bit at address {address}. Native error code: {result}.");
            }

            return value;
        }

        public void WriteBit(int address, bool value)
        {
            EnsureInitialized();
            int result = NativeMethods.PICoaterAPI_PlcWriteBit(_plcHandle, address, value);
            if (result != 0)
            {
                throw new InvalidOperationException(
                    $"Failed to write PLC bit at address {address}. Native error code: {result}.");
            }
        }

        public void Dispose()
        {
            if (_plcHandle != IntPtr.Zero)
            {
                NativeMethods.PICoaterAPI_DestroyPlc(_plcHandle);
                _plcHandle = IntPtr.Zero;
            }
        }

        private void EnsureInitialized()
        {
            if (_plcHandle == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(PlcService));
            }
        }
    }
}
