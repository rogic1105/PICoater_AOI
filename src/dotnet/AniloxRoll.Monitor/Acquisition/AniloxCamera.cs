using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core;
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
        /// <summary>暫停存檔（不影響 grab/檢測/顯示，只跳過 TrySaveCapture）。
        /// 改參數期間由 LiveCameraManager 全相機設 true → 等全部恢復同步再設 false，避免存出「改參數重啟空檔」的不齊序列。</summary>
        public bool SuppressCapture { get; set; } = false;
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

        // 細線濾除（ridge_sigma）。實際值由 LiveCameraManager 從設定注入；此處預設＝唯一來源常數（勿寫誤導死值）。
        public double HessianSigma { get; set; } = InspectionEngineConfig.DefaultRidgeSigma;
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
        // 存檔縮圖目標（host pinned）。fused 一進多出：native 在檢測那次 ProcessImage 直接填這三塊。
        private IntPtr _rawResizeBuf  = IntPtr.Zero;   // 原圖縮圖
        private IntPtr _procResizeBuf = IntPtr.Zero;   // V 脊線縮圖
        private IntPtr _muraResizeBuf = IntPtr.Zero;   // H 脊線縮圖
        private int _resizeWidth  = 0;
        private int _resizeHeight = 0;
        // 本幀 fused 縮圖是否成功填好三塊 buffer（供 TrySaveCapture 判定；detection 失敗幀不得存舊縮圖）。
        private bool _lastFrameResized = false;
        private bool _fusedResizeLogged = false;   // 每台每次開程式只記一筆「fused 路徑已啟用」確認 log

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
        /// <summary>每次 TrySaveCapture 成功存檔後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1, maxCMean_0to1)</summary>
        public event Action<int, string, float, float, float, float, float> OnInspectionResult;

        /// <summary>ImageCanvas / 瀑布顯示路徑用：每幀提供「顯示 bytes(8-bit 灰階)+ 尺寸 + 本幀硬體 frame-start tick」(MIL 回呼執行緒)。
        /// bytes 是重用緩衝 → 訂閱者必須**同步消費**(組 bitmap 複製)、勿存 ref。只在有訂閱者時觸發。
        /// tick＝跨相機幀對齊的硬體鑰匙（同 Review 的 FrameTickIndex；瀑布即時用它聚類時間槽）。</summary>
        public event Action<int, byte[], int, int, long> OnDisplayFrame;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        /// <summary>存檔完成回呼：傳入已儲存的檔案路徑陣列（供遠端複製佇列）。</summary>
        public Action<string[]> OnFilesSaved { get; set; }
        public Action<int, string> OnCaptureSaveFailed { get; set; }

        private bool _flowLoggedDrainFrameDrop;
        private float _lastMeanPeak = 0f;
        private float _lastMaxPeak  = 0f;

        // ==================== Constructor ====================
        public AniloxCamera(MIL_ID systemId, int id, int devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            CameraId = id;
            EnableImageProcessing = enableImageProcessing;

            _mil = new MilCamera(systemId, id, devNum, dcfPath, panelHandle);

            // MIL 每幀回呼 → 本類別跑檢測/合圖/存檔
            _mil.FrameReady += OnMilFrameReady;
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

        // ==================== Exposure / Line Rate（委派 MIL） ====================

        /// <summary>設定曝光時間（μs）。</summary>
        public void SetExposureUs(double exposureUs) => _mil.SetExposureUs(exposureUs);

        /// <summary>回傳最後一次 SetExposureUs 的設定值（μs），不依賴硬體回讀。</summary>
        public double GetExposureUs() => _mil.GetExposureUs();

        /// <summary>從硬體讀回目前曝光時間（μs）。CLProtocol 未啟用時回傳 0。</summary>
        public double GetMeasuredExposureUs() => _mil.GetMeasuredExposureUs();

        /// <summary>目前套用的線掃率（Hz，設定值、不查硬體；便宜）。供算幀週期＝FrameHeight/lineRate。</summary>
        public double AppliedLineRateHz => _mil.AppliedLineRateHz;

        /// <summary>透過 CLProtocol GenICam Feature 讀取 Line Rate（Hz）。CLProtocol 未啟用時回傳 0。</summary>
        public double GetLineRateHz() => _mil.GetLineRateHz();

        /// <summary>相機/grabber 回報的線掃率上限（Hz）。CLProtocol 未就緒回 0。</summary>
        public double GetLineRateMaxHz() => _mil.GetLineRateMaxHz();

        /// <summary>相機/grabber 回報的 grab 高度上限（px）。只讀查詢；CLProtocol 未就緒 / 查不到回 0。</summary>
        public int GetGrabHeightMaxPx() => _mil.GetGrabHeightMaxPx();

        /// <summary>設定線掃速率（Hz）。CLProtocol 未就緒時僅記錄，待啟用後自動重套。</summary>
        public void SetLineRateHz(double hz) => _mil.SetLineRateHz(hz);

        // ==================== Grab Height ====================

        /// <summary>
        /// 變更 Grab 高度：MIL buffer 重分配由 _mil 處理，本類別重新分配檢測 host/pool/resize buffer。
        /// </summary>
        public void SetGrabHeight(int height)
        {
            if (height <= 0) return;

            // 根治高度變更 AccessViolation：原本「先 FreeInspectionBuffers（邊抓邊放）→ _mil 重啟後才重配 native」
            // 有兩個與 grab callback 競爭的窗。改：把「釋放舊 + 重配 native/host/resize」交給 _mil 的
            // onStoppedBeforeRestart 回呼 → 在「grab 已停（M_STOP 排乾 callback）、MIL buffer 已配新尺寸、尚未重啟」
            // 時做，**保證無 FrameReady 在跑** → 消除不一致窗。
            _mil.SetGrabHeight(height, onStoppedBeforeRestart: () =>
            {
                int w = _mil.FrameWidth;
                int h = _mil.FrameHeight;
                if (w <= 0 || h <= 0) return;

                FreeInspectionBuffers();   // 釋放舊 native/host/resize（此刻 grab 停著、無 callback）
                _hostInputBuffer  = new byte[w * h];
                _hostOutputBuffer = new byte[w * h];
                lock (_picoaterLock)
                {
                    _nativeBufferPool = new NativeBufferPool(w, h, 1);
                }
                AllocateResizeBuffers();
            });
            CameraGrabHeight = _mil.CameraGrabHeight;
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

        /// <summary>取得板卡總記憶體（MB）＝板載 on-board（硬體固定）。</summary>
        public long GetMemoryTotalMB() => _mil.GetMemoryTotalMB();

        /// <summary>所屬板（System）識別＝同板相機歸組用。</summary>
        public long OwnerSystemKey => _mil.OwnerSystemKey;

        /// <summary>是否已配 grab buffer（實際佔板載；拔線不釋放）。算板載安全高度用此而非 IsConnected。</summary>
        public bool HasGrabBuffers => _mil.HasGrabBuffers;

        /// <summary>grab 高度上限（px）。上層算好設入（純供 UI clamp 滑桿/顯示；CameraGrabHeight 設前已 clamp）。</summary>
        public int  EffectiveMaxGrabHeightPx { get => _mil.EffectiveMaxGrabHeightPx; set => _mil.EffectiveMaxGrabHeightPx = value; }

        /// <summary>診斷：log 相機 Height feature 的合法範圍 Min/Max/Increment。</summary>
        public void LogHeightFeatureInfo() => _mil.LogHeightFeatureInfo();

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

        /// <summary>分配後預先啟用 CLProtocol（背景）；grab 前完成，避免 grab 期間重套線掃掉幀。</summary>
        public void BeginCLProtocolInit() => _mil.BeginCLProtocolInit();

        /// <summary>斷線重連後重跑 CLProtocol（啟動時不在線、之後才連上 → 重新啟用才讀得到曝光/線掃參數）。</summary>
        public void RetryCLProtocolOnReconnect() => _mil.RetryCLProtocolOnReconnect();

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
            if (!mil.UserWantsGrab)
            {
                if (!_flowLoggedDrainFrameDrop)
                {
                    _flowLoggedDrainFrameDrop = true;
                    FlowTrace.Log($"drop drainedFrame after StopGrab cam{CameraId}");
                }
                return;
            }
            _flowLoggedDrainFrameDrop = false;

            // 不管 EnableImageProcessing，一律執行 GPU 處理以取得 Mura 曲線（供 CSV 日誌判斷）
            bool processedByPicoater = TryApplyPicoaterRidge(modifiedBuffer);

            // EnableImageProcessing 控制「顯示」：勾選且處理成功才顯示處理結果，否則顯示原圖
            bool showProcessed = EnableImageProcessing && processedByPicoater;
            if (showProcessed)
                _mil.PutDisplayBytes(_hostOutputBuffer); // 已填好的處理結果寫入顯示 buffer
            else
                _mil.CopyToDisplay(modifiedBuffer);       // 顯示原圖

            // ImageCanvas 顯示路徑：每幀把「顯示 bytes」交給訂閱者(同步組 bitmap)。MIL 模式無訂閱者→不觸發。
            var onDisp = OnDisplayFrame;
            if (onDisp != null)
            {
                byte[] disp = showProcessed ? _hostOutputBuffer : _hostInputBuffer;
                if (disp != null) onDisp(CameraId, disp, FrameWidth, FrameHeight, _mil.LastFrameStartTicks);
            }

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
            _lastFrameResized = false;   // 每幀先重置；成功跑完 fused 縮圖才設 true（detection 失敗幀不得存舊縮圖）
            if (srcBuffer == MIL.M_NULL) return false;
            int fw = _mil.FrameWidth;
            int fh = _mil.FrameHeight;
            if (fw <= 0 || fh <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;

            lock (_picoaterLock)
            {
                if (_nativeBufferPool == null) return false;
                // 高度變更瞬間：fw/fh（鎖外讀）可能與「鎖內已重配的 pool / host buffer」尺寸不一致 →
                // GPU/Marshal 越界 → AccessViolation。守門：尺寸超過任一 buffer 容量就跳過這幀（transient，不崩）。
                if ((ulong)fw * (ulong)fh > _nativeBufferPool.ImageBufferSize) return false;
                if (_hostInputBuffer == null || _hostInputBuffer.Length < fw * fh ||
                    _hostOutputBuffer == null || _hostOutputBuffer.Length < fw * fh) return false;

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

                    // fused 存檔縮圖：只在「這趟 grab 會存圖」時，要 native 在檢測同一次就把縮圖產出（免二次 H2D）。
                    // grab-level 決策（EnableAutoCapture 整趟固定）；SuppressCapture 期間 + 純 live grab 不縮。
                    // 去重/暫停跳過的少數幀會白縮一張（可忽略）；true→native 填 _rawResizeBuf/_procResizeBuf/_muraResizeBuf。
                    bool wantResize = EnableAutoCapture && !SuppressCapture
                        && !string.IsNullOrWhiteSpace(CaptureRootPath)
                        && _saveResizeScale > 1
                        && _resizeWidth > 0 && _resizeHeight > 0
                        && _rawResizeBuf != IntPtr.Zero && _procResizeBuf != IntPtr.Zero && _muraResizeBuf != IntPtr.Zero;

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
                            Stream           = IntPtr.Zero,
                            // fused 縮圖目標（wantResize=false 時全 0 → native 跳過）
                            ResizeWidth  = wantResize ? _resizeWidth  : 0,
                            ResizeHeight = wantResize ? _resizeHeight : 0,
                            ResizedRaw   = wantResize ? _rawResizeBuf  : IntPtr.Zero,
                            ResizedRidge = wantResize ? _procResizeBuf : IntPtr.Zero,
                            ResizedMura  = wantResize ? _muraResizeBuf : IntPtr.Zero
                        },
                        Params = new AoiProcessRequest.AlgorithmParams
                        {
                            BgSigmaFactor  = InspectionEngineConfig.PerFrameBgSigma,
                            RidgeSigma     = (float)HessianSigma,
                            HessianMaxFactor = (float)HessianFixedMax,
                            RidgeMode      = "vertical+horizontal",  // 永遠計算雙方向，確保 V/H 皆可存檔
                            PrecomputedColMean = PrecomputedColMean
                        }
                    });
                    LastGpuTimeMs = swGpu.ElapsedMilliseconds;
                    _lastFrameResized = wantResize;   // ProcessImage 成功；縮圖已填好（若有要）
                    if (wantResize && !_fusedResizeLogged)
                    {
                        _fusedResizeLogged = true;
                        System.Diagnostics.Trace.WriteLine(
                            $"[FusedResize] CAM{CameraId} 啟用：檢測+縮圖同次 device 產出 {fw}x{fh}→{_resizeWidth}x{_resizeHeight}"
                            + $"（免二次 H2D）ProcessMs(檢測+縮圖)={LastGpuTimeMs}");
                    }

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
                        _aoiService.ComputeColumnMean(fw, fh, inputBuffer, InspectionEngineConfig.DefaultBgSigma, hColMean);
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
            if (SuppressCapture) return;   // 改參數→恢復同步 窗口內暫停存檔（避免存不齊序列）
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

                byte[] rawBytes = null, procCBytes = null, procRBytes = null;
                float[] meanArr = null, maxArr = null;
                float[] rowMeanArr = null, rowMaxArr = null;
                int rw = _resizeWidth, rh = _resizeHeight;
                bool hasResizeData = false;

                // Pipeline 永遠跑 "vertical+horizontal"，因此 _ridgeBuffer=V, _muraBuffer=H，一律存 7 檔

                lock (_picoaterLock)
                {
                    // fused 一進多出：縮圖已由本幀檢測的 ProcessImage 就地產出（免二次 H2D），這裡只讀預縮好的 pinned buffer。
                    // 守門 `_lastFrameResized`：本幀 detection+縮圖確實成功才讀（detection 失敗幀 → 不存，避免存到上一幀的舊縮圖）。
                    if (_lastFrameResized &&
                        _rawResizeBuf  != IntPtr.Zero &&
                        _procResizeBuf != IntPtr.Zero &&
                        _muraResizeBuf != IntPtr.Zero &&
                        rw > 0 && rh > 0)
                    {
                        hasResizeData = true;
                        int pixels = rw * rh;

                        // raw 縮圖（native 從 device d_input 就地縮）
                        rawBytes = new byte[pixels];
                        Marshal.Copy(_rawResizeBuf, rawBytes, 0, pixels);

                        // Column ridge thumbnail.
                        procCBytes = new byte[pixels];
                        Marshal.Copy(_procResizeBuf, procCBytes, 0, pixels);

                        // Row ridge thumbnail.
                        procRBytes = new byte[pixels];
                        Marshal.Copy(_muraResizeBuf, procRBytes, 0, pixels);

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
                        RawBytes = rawBytes, ProcCBytes = procCBytes, ProcRBytes = procRBytes,
                        MeanC = meanArr, MaxC = maxArr,
                        MeanR = rowMeanArr, MaxR = rowMaxArr,
                        ResizeWidth = rw, ResizeHeight = rh,
                        JpgQuality = quality, ScaleForHeader = scale,
                        SaveDir = saveDir, BaseName = baseName,
                        CameraId = camId, OrigWidth = origW, OrigHeight = origH,
                        MeanPeak = meanPeak, MaxPeak = maxPeak,
                        GpuTimeMs = LastGpuTimeMs,
                        FrameStartTicks = _mil.LastFrameStartTicks,  // 本幀硬體 frame-start 戳（hook 早段已 latch）
                        OnResult = OnInspectionResult,
                        OnFilesSaved = OnFilesSaved
                    };

                    var saver = _frameSaver;
                    Task.Run(() =>
                    {
                        try { saver.SaveCapture(ctx); }
                        catch (Exception ex)
                        {
                            string error = ex.GetType().Name + ": " + ex.Message;
                            System.Diagnostics.Trace.WriteLine(
                                $"[CAM{camId}] TrySaveCapture(bg) failed: {error}");
                            OnCaptureSaveFailed?.Invoke(camId, error);
                        }
                    });
                }
                else
                {
                    // Resize buffer 不可用時的 fallback（不應發生）
                    Directory.CreateDirectory(saveDir);
                    MIL.MbufExport(Path.Combine(saveDir, baseName + ".bmp"), MIL.M_BMP, sourceBuffer);
                    OnInspectionResult?.Invoke(CameraId, baseName,
                        _lastMeanPeak, _lastMaxPeak, float.NaN, float.NaN, float.NaN);
                }
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] TrySaveCapture failed: {error}");
                OnCaptureSaveFailed?.Invoke(CameraId, error);
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
            _rawResizeBuf  = NativeMethods.TanukiCv_AllocPinned(sz);
            _procResizeBuf = NativeMethods.TanukiCv_AllocPinned(sz);
            _muraResizeBuf = NativeMethods.TanukiCv_AllocPinned(sz);
        }

        private void FreeResizeBuffers()
        {
            if (_rawResizeBuf != IntPtr.Zero)
            {
                NativeMethods.TanukiCv_FreePinned(_rawResizeBuf);
                _rawResizeBuf = IntPtr.Zero;
            }
            if (_procResizeBuf != IntPtr.Zero)
            {
                NativeMethods.TanukiCv_FreePinned(_procResizeBuf);
                _procResizeBuf = IntPtr.Zero;
            }
            if (_muraResizeBuf != IntPtr.Zero)
            {
                NativeMethods.TanukiCv_FreePinned(_muraResizeBuf);
                _muraResizeBuf = IntPtr.Zero;
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
