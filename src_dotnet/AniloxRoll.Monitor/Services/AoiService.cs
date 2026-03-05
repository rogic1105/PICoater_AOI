using System;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Interop;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class AoiService : IDisposable
    {
        private IntPtr _pipelineHandle = IntPtr.Zero;

        public void Initialize()
        {
            if (_pipelineHandle != IntPtr.Zero)
            {
                return;
            }

            _pipelineHandle = NativeMethods.PICoaterAPI_CreatePipeline();
            if (_pipelineHandle == IntPtr.Zero)
            {
                throw new InvalidOperationException("Failed to create AOI pipeline handle.");
            }
        }

        public void ProcessImage(
            int width,
            int height,
            IntPtr inputData,
            IntPtr backgroundOutput,
            IntPtr muraOutput,
            IntPtr ridgeOutput,
            IntPtr muraCurveMeanOutput,
            IntPtr muraCurveMaxOutput,
            float bgSigmaFactor,
            float ridgeSigma,
            float hessianMaxFactor,
            string ridgeMode,
            IntPtr stream)
        {
            EnsureInitialized();

            int result = NativeMethods.PICoaterAPI_ProcessPipeline(
                _pipelineHandle,
                width,
                height,
                inputData,
                backgroundOutput,
                muraOutput,
                ridgeOutput,
                muraCurveMeanOutput,
                muraCurveMaxOutput,
                bgSigmaFactor,
                ridgeSigma,
                hessianMaxFactor,
                ridgeMode,
                stream);

            if (result != 0)
            {
                throw new InvalidOperationException(GetLastError());
            }
        }

        public string GetLastError()
        {
            if (_pipelineHandle == IntPtr.Zero)
            {
                return "AOI pipeline is not initialized.";
            }

            IntPtr messagePtr = NativeMethods.PICoaterAPI_GetLastError(_pipelineHandle);
            return messagePtr == IntPtr.Zero
                ? "Unknown native error."
                : Marshal.PtrToStringAnsi(messagePtr) ?? "Unknown native error.";
        }

        public void Dispose()
        {
            if (_pipelineHandle != IntPtr.Zero)
            {
                NativeMethods.PICoaterAPI_DestroyPipeline(_pipelineHandle);
                _pipelineHandle = IntPtr.Zero;
            }
        }

        private void EnsureInitialized()
        {
            if (_pipelineHandle == IntPtr.Zero)
            {
                throw new ObjectDisposedException(nameof(AoiService));
            }
        }
    }
}
