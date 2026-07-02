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

            // 存檔縮圖（fused）。ResizeWidth/Height=0 或指標 Zero → native 跳過（純 live 幀）。
            public int ResizeWidth { get; set; }
            public int ResizeHeight { get; set; }
            public IntPtr ResizedRaw { get; set; }
            public IntPtr ResizedRidge { get; set; }
            public IntPtr ResizedMura { get; set; }
        }

        public sealed class AlgorithmParams
        {
            public float BgSigmaFactor { get; set; }
            public float RidgeSigma { get; set; }
            public float HessianMaxFactor { get; set; }
            public string RidgeMode { get; set; } = "vertical+horizontal";
            public IntPtr PrecomputedColMean { get; set; } = IntPtr.Zero;
        }

        public InputImage Input { get; set; } = new InputImage();
        public OutputBuffers Output { get; set; } = new OutputBuffers();
        public AlgorithmParams Params { get; set; } = new AlgorithmParams();
    }

    public sealed class AoiService : IDisposable
    {
        private IntPtr _pipelineHandle = IntPtr.Zero;

        /// <summary>使用的 pipeline 名（tanuki_pipeline 的食譜；4b 單一 API：run(name, json)）。</summary>
        private const string PipelineName = "find_stream_ridgeline";

        public void Initialize()
        {
            if (_pipelineHandle != IntPtr.Zero)
            {
                return;
            }

            // jsonOptions 可選方法（如 {"ridge_method":"gabor"}）；null = 預設 hessian
            _pipelineHandle = NativeMethods.TanukiPipeline_Create(PipelineName, null);
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
                Stream = request.Output.Stream != IntPtr.Zero ? request.Output.Stream : request.Input.Stream,
                ResizeWidth = request.Output.ResizeWidth,
                ResizeHeight = request.Output.ResizeHeight,
                ResizedRaw = request.Output.ResizedRaw,
                ResizedRidge = request.Output.ResizedRidge,
                ResizedMura = request.Output.ResizedMura
            };

            // 演算法參數組成 json（InvariantCulture：小數點一律 '.'，不受系統地區影響）
            string jsonParams = string.Format(
                System.Globalization.CultureInfo.InvariantCulture,
                "{{\"bg_sigma_factor\":{0},\"ridge_sigma\":{1},\"hessian_max_factor\":{2},\"ridge_mode\":\"{3}\"}}",
                request.Params.BgSigmaFactor,
                request.Params.RidgeSigma,
                request.Params.HessianMaxFactor,
                request.Params.RidgeMode ?? "vertical+horizontal");

            int result = NativeMethods.TanukiPipeline_Process(
                _pipelineHandle,
                ref input,
                jsonParams,
                request.Params.PrecomputedColMean,
                ref output);

            if (result != 0)
            {
                throw new InvalidOperationException(GetLastError());
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
            int result = NativeMethods.TanukiPipeline_ComputeColumnMean(
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

            IntPtr messagePtr = NativeMethods.TanukiPipeline_GetLastError(_pipelineHandle);
            return messagePtr == IntPtr.Zero
                ? "Unknown native error."
                : Marshal.PtrToStringAnsi(messagePtr) ?? "Unknown native error.";
        }

        public void Dispose()
        {
            if (_pipelineHandle != IntPtr.Zero)
            {
                NativeMethods.TanukiPipeline_Destroy(_pipelineHandle);
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
