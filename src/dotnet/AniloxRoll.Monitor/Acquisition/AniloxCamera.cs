using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using AOI.SDK.Core;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Camera
{
    /// <summary>
    /// 一台 Anilox 相機：MIL 取像/顯示/參數全部委派給 <see cref="MilCamera"/>（composition），
    /// 本類別只保留「非 MIL」邏輯 — picoater 檢測（GPU pipeline）、曲線事件、合圖裁切複製、縮圖存檔。
    /// 透過訂閱 <see cref="MilCamera.FrameReady"/> 在每幀 MIL 回呼執行緒上運作。
    /// </summary>
    public class AniloxCamera : IDisposable
    {
        // ==================== MIL（委派） ====================
        private MilCamera _mil;

        /// <summary>底層 MIL 封裝（合圖等需直接存取 MIL buffer 時使用）。</summary>
        public MIL_ID OwnerSystemId => _mil.OwnerSystemId;
        public MIL_ID MilDigitizer => _mil.MilDigitizer;
        public MIL_ID MilDisplay => _mil.MilDisplay;
        public MIL_ID MilSecondaryDisplay => _mil.MilSecondaryDisplay;

        // ==================== State（委派 MIL） ====================
        public bool IsLive => _mil.IsLive;
        public int CameraId { get; private set; }
        public bool IsConnected => _mil.IsConnected;
        public bool UserWantsGrab => _mil.UserWantsGrab;
        /// <summary>CLProtocol 初始化（含參數重套）已完成，可安全從硬體讀回參數。</summary>
        public bool IsHwParamsStable => _mil.IsHwParamsStable;

        public int FrameWidth  => _mil.FrameWidth;
        public int FrameHeight => _mil.FrameHeight;

        // ==================== Settings ====================
        public bool EnableImageProcessing { get; set; } = true;
        public bool EnableAutoCapture { get; set; } = false;
        public bool SaveOriginalBmp { get; set; } = false;
        public string CaptureRootPath { get; set; } = string.Empty;
        /// <summary>Grab 高度（0 = 用 DCF 預設）。必須在 Initialize() 之前設定（轉發給 _mil）。</summary>
        public int CameraGrabHeight { get; set; } = 0;

        /// <summary>
        /// 曝光時間（μs）。get/set 轉發給 _mil；Initialize() 前設定會在 _mil.Initialize() 時套用。
        /// live 調整請使用 SetExposureUs()。
        /// </summary>
        public double CameraExposureTimeUs
        {
            get => _mil.GetExposureUs();
            set => _mil.SetExposureUs(value);
        }

        public double HessianSigma { get; set; } = 85;
        public double HessianFixedMax { get; set; } = 1.0;
        public string RidgeMode { get; set; } = "vertical";
        /// <summary>Live 顯示方向："v" = vertical ridge, "h" = horizontal ridge。</summary>
        public string LiveDisplayDirection { get; set; } = "v";

        /// <summary>
        /// 預算背景 column mean（pinned host float*，size = frameWidth）。
        /// 非 IntPtr.Zero 時，pipeline 跳過每幀 column mean 計算，使用此固定背景。
        /// </summary>
        public IntPtr PrecomputedColMean { get; set; } = IntPtr.Zero;

        // ==================== Internal ====================
        private bool _isReleased = false;

        /// <summary>底層 MIL 封裝（供 LiveCameraManager 交給 MultiCameraMerger 工頭管理合圖）。</summary>
        public MilCamera Mil => _mil;

        /// <summary>
        /// 設定 Global merge 目標：委派給 _mil（MilCamera），合圖複製改在 MIL grab hook 內執行。
        /// </summary>
        internal void SetMergeTarget(MIL_ID buffer, int offsetX, int clipLeft, int clipWidth)
            => _mil.SetMergeTarget(buffer, offsetX, clipLeft, clipWidth);

        /// <summary>清除 Global merge 目標：委派給 _mil（MilCamera）。</summary>
        internal void ClearMergeTarget() => _mil.ClearMergeTarget();

        // ==================== 檢測記憶體（非 MIL） ====================
        private byte[] _hostInputBuffer = null;
        private byte[] _hostOutputBuffer = null;

        private readonly object _picoaterLock = new object();
        private readonly AoiService _aoiService = new AoiService();
        private NativeBufferPool _nativeBufferPool;

        private string _lastCaptureKey = string.Empty;

        /// <summary>
        /// 同 Line Rate 相機共用時間戳協調器。由 LiveCameraManager 注入。
        /// 為 null 時各台獨立使用 DateTime.Now。
        /// </summary>
        public CaptureTimestampCoordinator TimestampCoordinator { get; set; }

        // ==================== Save Format (resize + JPEG) ====================
        private int _saveResizeScale = 5;
        private int _saveJpgQuality  = 90;
        private IntPtr _rawResizeBuf  = IntPtr.Zero;
        private IntPtr _procResizeBuf = IntPtr.Zero;
        private int _resizeWidth  = 0;
        private int _resizeHeight = 0;

        /// <summary>存檔縮小倍率（1=不縮小，5=寬高各除以5）。必須在 Initialize() 之前設定。</summary>
        public int SaveResizeScale
        {
            get => _saveResizeScale;
            set => _saveResizeScale = value > 0 ? value : 1;
        }

        /// <summary>JPEG 存檔品質（1–100）。</summary>
        public int SaveJpgQuality
        {
            get => _saveJpgQuality;
            set => _saveJpgQuality = Math.Max(1, Math.Min(100, value));
        }

        // ==================== Resource Monitor ====================
        private readonly CameraFrameSaver _frameSaver = new CameraFrameSaver();

        /// <summary>最近一幀 GPU ProcessPipeline 耗時（ms）。</summary>
        public long LastGpuTimeMs { get; private set; }
        /// <summary>最近一幀存檔總大小（bytes，含 raw + proc + bin）。</summary>
        public long LastSaveBytesTotal => _frameSaver.LastSaveBytesTotal;
        /// <summary>本次 session 累計存檔大小（bytes）。</summary>
        public long SessionSaveBytes => _frameSaver.SessionSaveBytes;
        /// <summary>本次 session 累計存檔幀數。</summary>
        public long SessionFrameCount => _frameSaver.SessionFrameCount;

        // ==================== Telemetry（委派 MIL） ====================
        /// <summary>目前實際量測的 FPS（MdigInquire M_PROCESS_FRAME_RATE）。抓圖未啟動時回傳 0。</summary>
        public double CurrentFps => _mil.CurrentFps;

        // ==================== Events ====================
        public event Action<int, int, int, int> OnMouseDataChanged;
        public event Action<int> OnCameraClicked;

        /// <summary>每次 TrySaveCapture 成功存檔後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1)</summary>
        public event Action<int, string, float, float> OnInspectionResult;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        /// <summary>存檔完成回呼：傳入已儲存的檔案路徑陣列（供遠端複製佇列）。</summary>
        public Action<string[]> OnFilesSaved { get; set; }

        private float _lastMeanPeak = 0f;
        private float _lastMaxPeak  = 0f;

        // ==================== Constructor ====================
        public AniloxCamera(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            CameraId = id;
            EnableImageProcessing = enableImageProcessing;

            _mil = new MilCamera(systemId, id, devNum, dcfPath, panelHandle);

            // MIL 每幀回呼 → 本類別跑檢測/合圖/存檔
            _mil.FrameReady += OnMilFrameReady;
            // 滑鼠/點擊事件由 MIL 射出，原樣轉發給上層
            _mil.OnMouseDataChanged += (camId, x, y, p) => OnMouseDataChanged?.Invoke(camId, x, y, p);
            _mil.OnCameraClicked += camId => OnCameraClicked?.Invoke(camId);
        }

        // ==================== Initialize ====================
        public void Initialize()
        {
            _mil.CameraGrabHeight = CameraGrabHeight;
            _mil.Initialize();

            int w = _mil.FrameWidth;
            int h = _mil.FrameHeight;
            if (w <= 0 || h <= 0) return; // digitizer alloc 失敗

            // 檢測記憶體（非 MIL）
            _hostInputBuffer  = new byte[w * h];
            _hostOutputBuffer = new byte[w * h];

            _aoiService.Initialize();
            _nativeBufferPool = new NativeBufferPool(w, h, 1);
            AllocateResizeBuffers();
        }

        // ==================== Primary / Secondary Display（委派 MIL） ====================

        /// <summary>Detach / restore 主顯示（panelLiveCam 上的 MIL 顯示）。</summary>
        public void SetPrimaryDisplayVisible(bool visible) => _mil.SetPrimaryDisplayVisible(visible);

        public void SetSecondaryDisplay(IntPtr handle) => _mil.SetSecondaryDisplay(handle);

        /// <summary>查詢副顯示器（panelMainDisplay）的 zoom/pan 狀態。</summary>
        public bool TryGetSecondaryDisplayGeometry(out double zoomX, out double zoomY, out double panX, out double panY)
            => _mil.TryGetSecondaryDisplayGeometry(out zoomX, out zoomY, out panX, out panY);

        /// <summary>設定副顯示器的縮放與平移（用於自訂滾輪縮放）。</summary>
        public void SetSecondaryDisplayZoom(double zoom, double panX, double panY)
            => _mil.SetSecondaryDisplayZoom(zoom, panX, panY);

        /// <summary>重置副顯示器的縮放/平移為 fit-to-window。</summary>
        public void ResetSecondaryDisplayView() => _mil.ResetSecondaryDisplayView();

        // ==================== Exposure / Line Rate（委派 MIL） ====================

        /// <summary>設定曝光時間（μs）。</summary>
        public void SetExposureUs(double exposureUs) => _mil.SetExposureUs(exposureUs);

        /// <summary>回傳最後一次 SetExposureUs 的設定值（μs），不依賴硬體回讀。</summary>
        public double GetExposureUs() => _mil.GetExposureUs();

        /// <summary>從硬體讀回目前曝光時間（μs）。CLProtocol 未啟用時回傳 0。</summary>
        public double GetMeasuredExposureUs() => _mil.GetMeasuredExposureUs();

        /// <summary>透過 CLProtocol GenICam Feature 讀取 Line Rate（Hz）。CLProtocol 未啟用時回傳 0。</summary>
        public double GetLineRateHz() => _mil.GetLineRateHz();

        /// <summary>設定線掃速率（Hz）。CLProtocol 未就緒時僅記錄，待啟用後自動重套。</summary>
        public void SetLineRateHz(double hz) => _mil.SetLineRateHz(hz);

        // ==================== Grab Height ====================

        /// <summary>
        /// 變更 Grab 高度：MIL buffer 重分配由 _mil 處理，本類別重新分配檢測 host/pool/resize buffer。
        /// </summary>
        public void SetGrabHeight(int height)
        {
            if (height <= 0) return;

            // 1. 釋放檢測記憶體（MIL buffer 由 _mil.SetGrabHeight 處理）
            FreeInspectionBuffers();

            // 2. MIL 端重分配 grab/display buffer + rebind display + restart
            _mil.SetGrabHeight(height);
            CameraGrabHeight = _mil.CameraGrabHeight;

            // 3. 用 MIL 回報的實際尺寸重分配檢測記憶體
            int w = _mil.FrameWidth;
            int h = _mil.FrameHeight;
            if (w <= 0 || h <= 0) return;

            _hostInputBuffer  = new byte[w * h];
            _hostOutputBuffer = new byte[w * h];
            lock (_picoaterLock)
            {
                _nativeBufferPool = new NativeBufferPool(w, h, 1);
            }
            AllocateResizeBuffers();
        }

        /// <summary>釋放檢測記憶體（host + NativeBufferPool + resize pinned buffer）。MIL buffer 不在此處理。</summary>
        private void FreeInspectionBuffers()
        {
            FreeResizeBuffers();
            lock (_picoaterLock)
            {
                _nativeBufferPool?.Dispose();
                _nativeBufferPool = null;
            }
            _hostInputBuffer  = null;
            _hostOutputBuffer = null;
        }

        // ==================== Telemetry（委派 MIL） ====================

        /// <summary>透過 CLProtocol GenICam Feature 讀取相機本體溫度（°C）。未啟用時回傳 NaN。</summary>
        public double GetCameraTemperature() => _mil.GetCameraTemperature();

        /// <summary>取得擷取卡 FPGA 溫度（°C）。</summary>
        public double GetFpgaTemperature() => _mil.GetFpgaTemperature();

        /// <summary>取得板卡可用記憶體（MB）。</summary>
        public long GetMemoryFreeMB() => _mil.GetMemoryFreeMB();

        /// <summary>取得 PCIe 通道數。</summary>
        public int GetPcieNumberOfLanes() => _mil.GetPcieNumberOfLanes();

        /// <summary>取得 PCIe 速度字串（Gen1 / Gen2 / Gen3）。</summary>
        public string GetPcieSpeed() => _mil.GetPcieSpeed();

        /// <summary>DCF 設定的目標 FPS（M_SELECTED_FRAME_RATE）。</summary>
        public double GetSelectedFrameRate() => _mil.GetSelectedFrameRate();

        /// <summary>累計已處理的 Frame 數（M_PROCESS_FRAME_COUNT）。</summary>
        public long GetFrameCount() => _mil.GetFrameCount();

        /// <summary>Processing callback 遺漏的 Frame 數（M_PROCESS_FRAME_MISSED）。</summary>
        public long GetFrameMissed() => _mil.GetFrameMissed();

        /// <summary>硬體 Grab 層遺漏的 Frame 數（M_GRAB_FRAME_MISSED）。</summary>
        public long GetGrabFrameMissed() => _mil.GetGrabFrameMissed();

        /// <summary>掃描模式字串（"Line" 或 "Progressive"）。</summary>
        public string GetScanMode() => _mil.GetScanMode();

        // ==================== Grab Control（委派 MIL） ====================

        public void SetUserGrabIntent(bool enable) => _mil.SetUserGrabIntent(enable);

        public void ApplyGrabState() => _mil.ApplyGrabState();

        public bool CheckPresence() => _mil.CheckPresence();

        // ==================== MIL FrameReady 回呼（非 MIL 檢測/合圖/存檔） ====================

        /// <summary>
        /// 每幀 grab 完成由 _mil.FrameReady 觸發（MIL 回呼執行緒）。
        /// 一律跑 GPU 檢測以取得 Mura 曲線（供 CSV 判斷）；EnableImageProcessing 只控制「顯示」。
        /// </summary>
        private void OnMilFrameReady(MilCamera mil, MIL_ID modifiedBuffer)
        {
            if (_isReleased) return;
            if (modifiedBuffer == MIL.M_NULL) return;

            // 不管 EnableImageProcessing，一律執行 GPU 處理以取得 Mura 曲線（供 CSV 日誌判斷）
            bool processedByPicoater = TryApplyPicoaterRidge(modifiedBuffer);

            // EnableImageProcessing 控制「顯示」：勾選且處理成功才顯示處理結果，否則顯示原圖
            if (EnableImageProcessing && processedByPicoater)
                _mil.PutDisplayBytes(_hostOutputBuffer); // 已填好的處理結果寫入顯示 buffer
            else
                _mil.CopyToDisplay(modifiedBuffer);       // 顯示原圖

            // Global merge 複製（display buffer → 合併 buffer 裁切位置）已移至 MilCamera grab hook，
            // 在此 FrameReady handler 返回後由 _mil 執行（顯示 buffer 已更新完成）。

            TrySaveCapture(modifiedBuffer);
        }

        // ==================== Picoater Ridge Processing ====================

        /// <summary>
        /// 跑 GPU pipeline：MIL buffer → host → picoater → 填 _hostOutputBuffer + 觸發曲線事件。
        /// 顯示交由 OnMilFrameReady 用 _mil.PutDisplayBytes(_hostOutputBuffer) 處理（不在此寫 MIL buffer）。
        /// </summary>
        private bool TryApplyPicoaterRidge(MIL_ID srcBuffer)
        {
            if (srcBuffer == MIL.M_NULL) return false;
            int fw = _mil.FrameWidth;
            int fh = _mil.FrameHeight;
            if (fw <= 0 || fh <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;

            lock (_picoaterLock)
            {
                if (_nativeBufferPool == null) return false;

                IntPtr picoaterInputBuffer = _nativeBufferPool.InputBuffer;
                IntPtr picoaterRidgeBuffer = _nativeBufferPool.RidgeBuffer;
                if (picoaterInputBuffer == IntPtr.Zero || picoaterRidgeBuffer == IntPtr.Zero) return false;

                try
                {
                    _mil.GetFrameBytes(srcBuffer, _hostInputBuffer);
                    Marshal.Copy(_hostInputBuffer, 0, picoaterInputBuffer, _hostInputBuffer.Length);

                    IntPtr picoaterCurveMean    = _nativeBufferPool.CurveMeanBuffer;
                    IntPtr picoaterCurveMax     = _nativeBufferPool.CurveMaxBuffer;
                    IntPtr picoaterRowCurveMean = _nativeBufferPool.CurveRowMeanBuffer;
                    IntPtr picoaterRowCurveMax  = _nativeBufferPool.CurveRowMaxBuffer;

                    var swGpu = System.Diagnostics.Stopwatch.StartNew();
                    _aoiService.ProcessImage(new AoiProcessRequest
                    {
                        Input = new AoiProcessRequest.InputImage
                        {
                            Width  = fw,
                            Height = fh,
                            Data   = picoaterInputBuffer,
                            Stream = IntPtr.Zero
                        },
                        Output = new AoiProcessRequest.OutputBuffers
                        {
                            BackgroundData   = IntPtr.Zero,
                            MuraData         = _nativeBufferPool.MuraBuffer,
                            RidgeData        = picoaterRidgeBuffer,
                            MuraCurveMean    = picoaterCurveMean,
                            MuraCurveMax     = picoaterCurveMax,
                            MuraRowCurveMean = picoaterRowCurveMean,
                            MuraRowCurveMax  = picoaterRowCurveMax,
                            Stream           = IntPtr.Zero
                        },
                        Params = new AoiProcessRequest.AlgorithmParams
                        {
                            BgSigmaFactor  = 2.0f,
                            RidgeSigma     = (float)HessianSigma,
                            HessianMaxFactor = (float)HessianFixedMax,
                            RidgeMode      = "vertical+horizontal",  // 永遠計算雙方向，確保 V/H 皆可存檔
                            PrecomputedColMean = PrecomputedColMean
                        }
                    });
                    LastGpuTimeMs = swGpu.ElapsedMilliseconds;

                    // 從 Mura 曲線計算 peak（0-1 normalized），供 OnInspectionResult 使用
                    int curveLen = _nativeBufferPool.CurveBufferSize / sizeof(float);
                    if (curveLen > 0 && picoaterCurveMean != IntPtr.Zero && picoaterCurveMax != IntPtr.Zero)
                    {
                        float[] meanArr = new float[curveLen];
                        float[] maxArr  = new float[curveLen];
                        Marshal.Copy(picoaterCurveMean, meanArr, 0, curveLen);
                        Marshal.Copy(picoaterCurveMax,  maxArr,  0, curveLen);
                        float mp = 0f, xp = 0f;
                        for (int k = 0; k < curveLen; k++)
                        {
                            if (meanArr[k] > mp) mp = meanArr[k];
                            if (maxArr[k]  > xp) xp = maxArr[k];
                        }
                        _lastMeanPeak = mp / 255f;
                        _lastMaxPeak  = xp / 255f;

                        OnLiveCurveData?.Invoke(CameraId, meanArr, maxArr);

                        // Row curves (horizontal data)
                        int rowCurveLen = fh;
                        if (rowCurveLen > 0 && picoaterRowCurveMean != IntPtr.Zero && picoaterRowCurveMax != IntPtr.Zero)
                        {
                            float[] rowMeanArr = new float[rowCurveLen];
                            float[] rowMaxArr  = new float[rowCurveLen];
                            Marshal.Copy(picoaterRowCurveMean, rowMeanArr, 0, rowCurveLen);
                            Marshal.Copy(picoaterRowCurveMax,  rowMaxArr,  0, rowCurveLen);
                            OnLiveRowCurveData?.Invoke(CameraId, rowMeanArr, rowMaxArr);
                        }
                    }

                    // LiveDisplayDirection 控制顯示 V 或 H ridge → 填 _hostOutputBuffer（顯示由 OnMilFrameReady 做）
                    IntPtr displaySrc = (LiveDisplayDirection == "h")
                        ? _nativeBufferPool.MuraBuffer   // horizontal ridge
                        : picoaterRidgeBuffer;            // vertical ridge（預設）
                    Marshal.Copy(displaySrc, _hostOutputBuffer, 0, _hostOutputBuffer.Length);
                    return true;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] TryApplyPicoaterRidge failed: {ex.GetType().Name}: {ex.Message}");
                    return false;
                }
            }
        }

        // ==================== Background Column Mean ====================

        /// <summary>
        /// 計算當前 grab buffer 的 column mean（不跑完整 pipeline）。
        /// 結果寫入呼叫端提供的 float[] outColMean（長度 = FrameWidth）。
        /// 必須在相機已 allocate 且 grab 中呼叫。
        /// </summary>
        public bool TryComputeColumnMean(float[] outColMean)
        {
            int fw = _mil.FrameWidth;
            int fh = _mil.FrameHeight;
            if (fw <= 0 || fh <= 0) return false;
            if (_hostInputBuffer == null) return false;
            if (outColMean == null || outColMean.Length < fw) return false;

            lock (_picoaterLock)
            {
                if (_nativeBufferPool == null) return false;

                IntPtr inputBuffer = _nativeBufferPool.InputBuffer;
                if (inputBuffer == IntPtr.Zero) return false;

                try
                {
                    // 從最近 grab 到的原始影像（hook 中暫存）取資料
                    MIL_ID srcBuf = _mil.LastGrabBuffer;
                    if (srcBuf == MIL.M_NULL) return false;

                    _mil.GetFrameBytes(srcBuf, _hostInputBuffer);
                    Marshal.Copy(_hostInputBuffer, 0, inputBuffer, _hostInputBuffer.Length);

                    // host float buffer for result
                    IntPtr hColMean = Marshal.AllocHGlobal(fw * sizeof(float));
                    try
                    {
                        _aoiService.ComputeColumnMean(fw, fh, inputBuffer, 2.0f, hColMean);
                        Marshal.Copy(hColMean, outColMean, 0, fw);
                        return true;
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(hColMean);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] TryComputeColumnMean failed: {ex.Message}");
                    return false;
                }
            }
        }

        // ==================== Auto Capture ====================

        private void TrySaveCapture(MIL_ID sourceBuffer)
        {
            if (!EnableAutoCapture) return;
            if (sourceBuffer == MIL.M_NULL) return;
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) return;

            try
            {
                DateTime now = DateTime.Now;
                // 同 Line Rate 的相機共用時間戳，讓同一輪 grab 的檔名一致
                double lineRateHz = _mil.AppliedLineRateHz;
                if (TimestampCoordinator != null && lineRateHz > 0)
                    now = TimestampCoordinator.Coordinate((int)lineRateHz, now);

                string captureKey = now.ToString("yyyyMMdd_HHmmss.fff");
                if (string.Equals(_lastCaptureKey, captureKey, StringComparison.Ordinal)) return;

                // 提前更新，防止 Task.Run 延遲期間下一幀重複觸發存檔
                _lastCaptureKey = captureKey;

                string saveDir = Path.Combine(
                    CaptureRootPath,
                    now.ToString("yyyy"),
                    now.ToString("yyyyMM"),
                    now.ToString("yyyyMMdd"));

                string baseName = $"{now:yyyyMMdd_HHmmss.fff}-{CameraId}";

                int fw = _mil.FrameWidth;
                int fh = _mil.FrameHeight;

                byte[] rawBytes = null, procVBytes = null, procHBytes = null;
                float[] meanArr = null, maxArr = null;
                float[] rowMeanArr = null, rowMaxArr = null;
                int rw = _resizeWidth, rh = _resizeHeight;
                bool hasResizeData = false;

                // Pipeline 永遠跑 "vertical+horizontal"，因此 _ridgeBuffer=V, _muraBuffer=H，一律存 7 檔

                lock (_picoaterLock)
                {
                    if (_nativeBufferPool != null &&
                        _rawResizeBuf  != IntPtr.Zero &&
                        _procResizeBuf != IntPtr.Zero)
                    {
                        hasResizeData = true;
                        int pixels = rw * rh;

                        // GPU resize raw → _rawResizeBuf
                        NativeMethods.CoreCV_Resize_GPU(
                            _nativeBufferPool.InputBuffer, fw, fh,
                            _rawResizeBuf, rw, rh);
                        rawBytes = new byte[pixels];
                        Marshal.Copy(_rawResizeBuf, rawBytes, 0, pixels);

                        // _ridgeBuffer = vertical ridge → _proc_v.jpg
                        NativeMethods.CoreCV_Resize_GPU(
                            _nativeBufferPool.RidgeBuffer, fw, fh,
                            _procResizeBuf, rw, rh);
                        procVBytes = new byte[pixels];
                        Marshal.Copy(_procResizeBuf, procVBytes, 0, pixels);

                        // _muraBuffer = horizontal ridge → _proc_h.jpg
                        NativeMethods.CoreCV_Resize_GPU(
                            _nativeBufferPool.MuraBuffer, fw, fh,
                            _rawResizeBuf, rw, rh);
                        procHBytes = new byte[pixels];
                        Marshal.Copy(_rawResizeBuf, procHBytes, 0, pixels);

                        // Col curves（vertical ridge）
                        int curveLen = _nativeBufferPool.CurveBufferSize / sizeof(float);
                        if (curveLen > 0 &&
                            _nativeBufferPool.CurveMeanBuffer != IntPtr.Zero &&
                            _nativeBufferPool.CurveMaxBuffer  != IntPtr.Zero)
                        {
                            meanArr = new float[curveLen];
                            maxArr  = new float[curveLen];
                            Marshal.Copy(_nativeBufferPool.CurveMeanBuffer, meanArr, 0, curveLen);
                            Marshal.Copy(_nativeBufferPool.CurveMaxBuffer,  maxArr,  0, curveLen);
                        }

                        // Row curves（horizontal ridge）
                        int rowCurveLen = _nativeBufferPool.CurveRowBufferSize / sizeof(float);
                        if (rowCurveLen > 0 &&
                            _nativeBufferPool.CurveRowMeanBuffer != IntPtr.Zero &&
                            _nativeBufferPool.CurveRowMaxBuffer  != IntPtr.Zero)
                        {
                            rowMeanArr = new float[rowCurveLen];
                            rowMaxArr  = new float[rowCurveLen];
                            Marshal.Copy(_nativeBufferPool.CurveRowMeanBuffer, rowMeanArr, 0, rowCurveLen);
                            Marshal.Copy(_nativeBufferPool.CurveRowMaxBuffer,  rowMaxArr,  0, rowCurveLen);
                        }
                    }
                }

                // 快照目前峰值（callback 執行緒讀取，Task.Run 不回讀共享狀態）
                float meanPeak = _lastMeanPeak;
                float maxPeak  = _lastMaxPeak;
                int   camId    = CameraId;
                int   scale    = _saveResizeScale;
                int   quality  = _saveJpgQuality;
                bool  alsoBmp  = SaveOriginalBmp;
                int   origW    = fw;
                int   origH    = fh;

                if (hasResizeData)
                {
                    // 非壓縮模式：BMP 必須在 callback 同步匯出（sourceBuffer 會被 MIL 回收）
                    if (alsoBmp)
                    {
                        Directory.CreateDirectory(saveDir);
                        MIL.MbufExport(Path.Combine(saveDir, baseName + ".bmp"), MIL.M_BMP, sourceBuffer);
                    }

                    // 建立快照，背景執行緒存檔
                    var ctx = new CaptureContext
                    {
                        RawBytes = rawBytes, ProcVBytes = procVBytes, ProcHBytes = procHBytes,
                        MeanArr = meanArr, MaxArr = maxArr,
                        RowMeanArr = rowMeanArr, RowMaxArr = rowMaxArr,
                        ResizeWidth = rw, ResizeHeight = rh,
                        JpgQuality = quality, ScaleForHeader = scale,
                        SaveDir = saveDir, BaseName = baseName,
                        CameraId = camId, OrigWidth = origW, OrigHeight = origH,
                        MeanPeak = meanPeak, MaxPeak = maxPeak,
                        GpuTimeMs = LastGpuTimeMs,
                        OnResult = OnInspectionResult,
                        OnFilesSaved = OnFilesSaved
                    };

                    var saver = _frameSaver;
                    Task.Run(() =>
                    {
                        try { saver.SaveCapture(ctx); }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.WriteLine(
                                $"[CAM{camId}] TrySaveCapture(bg) failed: {ex.GetType().Name}: {ex.Message}");
                        }
                    });
                }
                else
                {
                    // Resize buffer 不可用時的 fallback（不應發生）
                    Directory.CreateDirectory(saveDir);
                    MIL.MbufExport(Path.Combine(saveDir, baseName + ".bmp"), MIL.M_BMP, sourceBuffer);
                    OnInspectionResult?.Invoke(CameraId, baseName, _lastMeanPeak, _lastMaxPeak);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] TrySaveCapture failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void AllocateResizeBuffers()
        {
            FreeResizeBuffers();
            int fw = _mil.FrameWidth;
            int fh = _mil.FrameHeight;
            if (fw <= 0 || fh <= 0 || _saveResizeScale <= 1) return;

            _resizeWidth  = fw / _saveResizeScale;
            _resizeHeight = fh / _saveResizeScale;
            if (_resizeWidth <= 0 || _resizeHeight <= 0) return;

            ulong sz = (ulong)(_resizeWidth * _resizeHeight);
            _rawResizeBuf  = NativeMethods.CoreCV_AllocPinned(sz);
            _procResizeBuf = NativeMethods.CoreCV_AllocPinned(sz);
        }

        private void FreeResizeBuffers()
        {
            if (_rawResizeBuf != IntPtr.Zero)
            {
                NativeMethods.CoreCV_FreePinned(_rawResizeBuf);
                _rawResizeBuf = IntPtr.Zero;
            }
            if (_procResizeBuf != IntPtr.Zero)
            {
                NativeMethods.CoreCV_FreePinned(_procResizeBuf);
                _procResizeBuf = IntPtr.Zero;
            }
        }

        // ==================== Dispose ====================

        public void Free() => Dispose();

        public void Dispose()
        {
            if (_isReleased)
            {
                _mil?.Dispose();
                return;
            }
            _isReleased = true;

            // MIL 資源（digitizer/display/buffers/hook/GCHandle）全由 _mil 釋放
            _mil?.Dispose();

            // 檢測記憶體（非 MIL）
            FreeResizeBuffers();
            lock (_picoaterLock)
            {
                _nativeBufferPool?.Dispose();
                _nativeBufferPool = null;
                _aoiService.Dispose();
            }
            _hostInputBuffer  = null;
            _hostOutputBuffer = null;
        }
    }
}
