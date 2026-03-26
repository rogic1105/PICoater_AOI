using System;
using System.Runtime.InteropServices;
using AniloxRoll.Monitor.Core.Interop;

namespace AniloxRoll.Monitor.Core.Services
{
    public sealed class AoiProcessRequest
    {
        public sealed class InputImage
        {
            public int Width { get; set; }
            public int Height { get; set; }
            public IntPtr Data { get; set; }
            public IntPtr Stream { get; set; }
        }

        public sealed class OutputBuffers
        {
            public int Width { get; set; }
            public int Height { get; set; }
            public IntPtr BackgroundData { get; set; }
            public IntPtr MuraData { get; set; }
            public IntPtr RidgeData { get; set; }
            public IntPtr MuraCurveMean { get; set; }
            public IntPtr MuraCurveMax { get; set; }
            public IntPtr MuraRowCurveMean { get; set; }
            public IntPtr MuraRowCurveMax { get; set; }
            public IntPtr Stream { get; set; }
        }

        public sealed class AlgorithmParams
        {
            public float BgSigmaFactor { get; set; }
            public float RidgeSigma { get; set; }
            public float HessianMaxFactor { get; set; }
            public string RidgeMode { get; set; } = "dark";
            public IntPtr PrecomputedColMean { get; set; } = IntPtr.Zero;
        }

        public InputImage Input { get; set; } = new InputImage();
        public OutputBuffers Output { get; set; } = new OutputBuffers();
        public AlgorithmParams Params { get; set; } = new AlgorithmParams();
    }

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

        public void ProcessImage(AoiProcessRequest request)
        {
            EnsureInitialized();

            if (request == null)
            {
                throw new ArgumentNullException(nameof(request));
            }

            if (request.Input.Width <= 0 || request.Input.Height <= 0)
            {
                throw new ArgumentException("Width and height must be positive.", nameof(request));
            }

            if (request.Input.Data == IntPtr.Zero)
            {
                throw new ArgumentException("InputData must not be null.", nameof(request));
            }

            var input = new AoiInputImageNative
            {
                Width = request.Input.Width,
                Height = request.Input.Height,
                Data = request.Input.Data,
                Stream = request.Input.Stream
            };

            var output = new AoiOutputBuffersNative
            {
                Width = request.Output.Width > 0 ? request.Output.Width : request.Input.Width,
                Height = request.Output.Height > 0 ? request.Output.Height : request.Input.Height,
                BackgroundData = request.Output.BackgroundData,
                MuraData = request.Output.MuraData,
                RidgeData = request.Output.RidgeData,
                MuraCurveMean = request.Output.MuraCurveMean,
                MuraCurveMax = request.Output.MuraCurveMax,
                MuraRowCurveMean = request.Output.MuraRowCurveMean,
                MuraRowCurveMax = request.Output.MuraRowCurveMax,
                Stream = request.Output.Stream != IntPtr.Zero ? request.Output.Stream : request.Input.Stream
            };

            IntPtr ridgeModePtr = IntPtr.Zero;
            try
            {
                if (!string.IsNullOrEmpty(request.Params.RidgeMode))
                {
                    ridgeModePtr = Marshal.StringToHGlobalAnsi(request.Params.RidgeMode);
                }

                var parameters = new AoiAlgorithmParamsNative
                {
                    BgSigmaFactor = request.Params.BgSigmaFactor,
                    RidgeSigma = request.Params.RidgeSigma,
                    HessianMaxFactor = request.Params.HessianMaxFactor,
                    RidgeMode = ridgeModePtr,
                    PrecomputedColMean = request.Params.PrecomputedColMean
                };

                int result = NativeMethods.PICoaterAPI_ProcessPipeline(
                    _pipelineHandle,
                    ref input,
                    ref parameters,
                    ref output);

                if (result != 0)
                {
                    throw new InvalidOperationException(GetLastError());
                }
            }
            finally
            {
                if (ridgeModePtr != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(ridgeModePtr);
                }
            }
        }

        /// <summary>
        /// 計算單幀影像的 column mean（去除離群值）。
        /// outColMean 必須是 host 端 float buffer，大小 = width。
        /// </summary>
        public void ComputeColumnMean(int width, int height, IntPtr inputData, float bgSigmaFactor, IntPtr outColMean)
        {
            EnsureInitialized();
            var input = new AoiInputImageNative
            {
                Width = width,
                Height = height,
                Data = inputData,
                Stream = IntPtr.Zero
            };
            int result = NativeMethods.PICoaterAPI_ComputeColumnMean(
                _pipelineHandle, ref input, bgSigmaFactor, outColMean);
            if (result != 0)
                throw new InvalidOperationException(GetLastError());
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
