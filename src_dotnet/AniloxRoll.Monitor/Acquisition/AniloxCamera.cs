using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;
using AOI.SDK.Core;
using AOI.SDK.Utils;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Camera
{
    public class AniloxCamera : IDisposable
    {
        // ==================== MIL Resources ====================
        private MIL_ID _milDigitizer = MIL.M_NULL;
        private MIL_ID _milDisplay = MIL.M_NULL;
        private MIL_ID _milSecondaryDisplay = MIL.M_NULL;

        public MIL_ID MilDigitizer => _milDigitizer;
        public MIL_ID MilDisplay => _milDisplay;
        public MIL_ID MilSecondaryDisplay => _milSecondaryDisplay;

        private MIL_ID _milProcBuffer = MIL.M_NULL;
        private MIL_ID _ownerSystemId = MIL.M_NULL;
        private MIL_ID[] _milGrabBuffers = new MIL_ID[2];
        private MIL_ID _milDisplayBuffer = MIL.M_NULL;
        private MIL_ID _milLastGrabBuffer = MIL.M_NULL;  // hook 中暫存最近一幀原圖
        private MIL_INT _milGrabBufferListSize = 2;

        // ==================== State ====================
        public bool IsLive { get; private set; } = false;
        public int CameraId { get; private set; }
        public bool IsConnected { get; private set; } = false;
        public bool UserWantsGrab => _userWantsGrab;

        // ==================== Settings ====================
        public bool EnableImageProcessing { get; set; } = true;
        public bool EnableHessian { get; set; } = true;
        public bool EnableAutoCapture { get; set; } = false;
        public bool SaveOriginalBmp { get; set; } = false;
        public string CaptureRootPath { get; set; } = string.Empty;
        public int CameraGrabHeight { get; set; } = 0;

        /// <summary>
        /// 曝光時間（μs）。初始設定用，live 調整請使用 SetExposureUs()。
        /// </summary>
        public double CameraExposureTimeUs
        {
            get => _appliedExposureUs;
            set => _appliedExposureUs = value;
        }

        public double BinarizeThreshold { get; set; } = 128.0;
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

        // ==================== CLProtocol ====================
        /// <summary>CLProtocol（GenICam Camera Link）是否已成功啟用。</summary>
        private bool _clProtocolEnabled = false;
        private volatile bool _clProtocolInitStarted = false;
        /// <summary>最後一次 SetExposureUs 寫入的曝光值（μs）。不依賴硬體回讀。</summary>
        private double _appliedExposureUs = 0;
        /// <summary>最後一次 SetLineRateHz 寫入的線掃速率（Hz）。CLProtocol 就緒後重新套用。</summary>
        private double _appliedLineRateHz = 0;

        // ==================== Internal ====================
        private bool _userWantsGrab = false;
        private bool _isReleased = false;
        private bool _isSecondaryHooked = false;

        private MIL_INT _devNum;
        private string _dcfPath;
        private IntPtr _panelHandle;

        private int _frameWidth = 0;
        private int _frameHeight = 0;

        public int FrameWidth  => _frameWidth;
        public int FrameHeight => _frameHeight;

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

        [ThreadStatic] private static ImageCodecInfo _jpegCodec;
        private static ImageCodecInfo GetJpegEncoder()
        {
            if (_jpegCodec != null) return _jpegCodec;
            foreach (var c in ImageCodecInfo.GetImageEncoders())
                if (c.MimeType == "image/jpeg") { _jpegCodec = c; return c; }
            return null;
        }

        // ==================== FPS（來自 MdigInquire，同 MilCameraUnit）====================
        /// <summary>目前實際量測的 FPS（MdigInquire M_PROCESS_FRAME_RATE）。抓圖未啟動時回傳 0。</summary>
        public double CurrentFps
        {
            get
            {
                if (_milDigitizer == MIL.M_NULL) return 0;
                double fps = 0;
                MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_RATE, ref fps);
                return fps;
            }
        }

        // ==================== Secondary Display Geometry ====================

        /// <summary>
        /// 查詢副顯示器（panelMainDisplay）的 zoom/pan 狀態。
        /// MIL M_SCALE_DISPLAY + M_MOUSE_USE 會隨使用者滾輪操作改變。
        /// </summary>
        public bool TryGetSecondaryDisplayGeometry(
            out double zoomX, out double zoomY, out double panX, out double panY)
        {
            zoomX = zoomY = panX = panY = 0;
            if (_milSecondaryDisplay == MIL.M_NULL) return false;
            try
            {
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                return zoomX > 0 && zoomY > 0;
            }
            catch { return false; }
        }

        // ==================== Delegates / Events ====================
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseClickDelegate;
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        public event Action<int, int, int, int> OnMouseDataChanged;
        public event Action<int> OnCameraClicked;

        /// <summary>每次 TrySaveCapture 成功存檔後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, fileNameWithoutExt, meanPeak_0to1, maxPeak_0to1)</summary>
        public event Action<int, string, float, float> OnInspectionResult;

        /// <summary>每幀 GPU pipeline 完成後觸發（MIL 回呼執行緒）。
        /// 參數：(cameraId, curveMean_raw255, curveMax_raw255)</summary>
        public event Action<int, float[], float[]> OnLiveCurveData;
        public event Action<int, float[], float[]> OnLiveRowCurveData;

        private float _lastMeanPeak = 0f;
        private float _lastMaxPeak  = 0f;

        // ==================== Constructor ====================
        public AniloxCamera(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            _ownerSystemId = systemId;
            CameraId = id;
            _devNum = devNum;
            _dcfPath = dcfPath;
            _panelHandle = panelHandle;
            EnableImageProcessing = enableImageProcessing;

            _mouseStatusDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseStatusHandler);
            _mouseClickDelegate  = new MIL_DISP_HOOK_FUNCTION_PTR(MouseClickHandler);
            _processingDelegate  = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);
            _hUserData = GCHandle.Alloc(this);
        }

        // ==================== Initialize ====================
        public void Initialize()
        {
            if (_ownerSystemId == MIL.M_NULL) return;

            MIL.MdigAlloc(_ownerSystemId, _devNum, _dcfPath, MIL.M_DEFAULT, ref _milDigitizer);

            if (_milDigitizer != MIL.M_NULL)
            {
                // 先套用 Grab Height，再查詢實際尺寸以分配正確大小的 Buffer
                if (CameraGrabHeight > 0)
                    MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)CameraGrabHeight);

                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milDisplay);
                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milSecondaryDisplay);

                MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
                MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
                _frameWidth  = (int)sizeX;
                _frameHeight = (int)sizeY;

                _hostInputBuffer  = new byte[_frameWidth * _frameHeight];
                _hostOutputBuffer = new byte[_frameWidth * _frameHeight];

                _aoiService.Initialize();
                _nativeBufferPool = new NativeBufferPool(_frameWidth, _frameHeight, 1);
                AllocateResizeBuffers();

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                        MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                    MIL.MbufClear(_milGrabBuffers[i], 0);
                }

                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
                MIL.MbufClear(_milDisplayBuffer, 0);

                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_PROC, ref _milProcBuffer);
                MIL.MbufClear(_milProcBuffer, 0);

                MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
                MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_milDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_LEFT_BUTTON_DOWN, _mouseClickDelegate, (IntPtr)CameraId);

                // 初始曝光：此時 CLProtocol 尚未啟用，走 legacy MdigControl 路徑
                if (_appliedExposureUs > 0)
                    SetExposureUs(_appliedExposureUs);
            }
        }

        // ==================== Secondary Display ====================
        public void SetSecondaryDisplay(IntPtr handle)
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;

            if (handle == IntPtr.Zero)
            {
                if (_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    _isSecondaryHooked = false;
                }
                MIL.MdispSelectWindow(_milSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);
            }
            else
            {
                MIL.MdispSelectWindow(_milSecondaryDisplay, _milDisplayBuffer, handle);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                if (!_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                    _isSecondaryHooked = true;
                }
            }
        }

        // ==================== CLProtocol ====================

        /// <summary>
        /// 在背景執行緒啟動 CLProtocol 初始化，避免阻塞 Initialize()。
        /// MdigControl(M_GC_CLPROTOCOL, M_ENABLE) 需載入 CLProtocol DLL 並讀取相機 GenICam XML，
        /// 耗時較長，因此以 Task.Run 非同步執行，且必須在 MdigProcess 啟動後才呼叫。
        /// 設有 10 秒 Timeout：若硬體 hang 住，記錄警告並停用 CLProtocol，
        /// 避免背景 Task 永遠佔用 Thread Pool 且無任何回饋。
        /// </summary>
        private void StartCLProtocolAsync()
        {
            if (_clProtocolInitStarted) return;
            _clProtocolInitStarted = true;

            var initTask    = Task.Run((Action)TryEnableCLProtocol);
            var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10));

            // 任一完成後檢查：若 initTask 尚未完成，代表硬體逾時
            Task.WhenAny(initTask, timeoutTask).ContinueWith(_ =>
            {
                if (!initTask.IsCompleted)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] CLProtocol 初始化逾時（>10s）。" +
                        "CLProtocol 已停用，曝光/線掃速率維持 fallback 路徑。");
                    // _clProtocolEnabled 保持 false；initTask 繼續在背景等待硬體回應，
                    // 若最終完成，TryEnableCLProtocol 仍會套用設定（late init）。
                }
            });
        }

        private void TryEnableCLProtocol()
        {
            if (_milDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
                MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL, MIL.M_ENABLE);
                _clProtocolEnabled = true;

                // CLProtocol 就緒後重新套用曝光與線掃速率（改走 Feature API）
                if (!_isReleased)
                {
                    if (_appliedExposureUs > 0)
                        SetExposureUs(_appliedExposureUs);
                    if (_appliedLineRateHz > 0)
                        SetLineRateHz(_appliedLineRateHz);
                }
            }
            catch (Exception ex)
            {
                _clProtocolEnabled = false;
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] CLProtocol init failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ==================== Exposure Control ====================

        /// <summary>
        /// 設定曝光時間（μs）。
        /// CLProtocol 已啟用：MdigControlFeature("ExposureTime")，GenICam 單位直接為 μs。
        /// CLProtocol 未啟用：MdigControl(M_EXPOSURE_TIME)，MIL 單位為 ns，自動乘以 1000。
        /// 上限：clamp(floor(900000 / lineRateHz), 1, 10000)，與 UI CalcExpMax 公式一致。
        /// </summary>
        public void SetExposureUs(double exposureUs)
        {
            if (_milDigitizer == MIL.M_NULL || exposureUs <= 0) return;

            // 依 Line Rate 計算曝光上限（0.9 × 行週期），與 UI TrackBar 上限一致
            if (_appliedLineRateHz > 0)
            {
                double maxUs = Math.Max(1.0, Math.Min(10000.0, Math.Floor(900000.0 / _appliedLineRateHz)));
                if (exposureUs > maxUs)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetExposureUs: {exposureUs} μs 超出上限 {maxUs} μs（LR={_appliedLineRateHz} Hz），已夾緊。");
                    exposureUs = maxUs;
                }
            }

            if (_clProtocolEnabled)
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE,
                    "ExposureTime", MIL.M_TYPE_DOUBLE, ref exposureUs);
            else
                MIL.MdigControl(_milDigitizer, MIL.M_EXPOSURE_TIME, exposureUs * 1000.0);

            _appliedExposureUs = exposureUs;
        }

        /// <summary>回傳最後一次 SetExposureUs 的設定值（μs），不依賴硬體回讀。</summary>
        public double GetExposureUs() => _appliedExposureUs;

        /// <summary>
        /// 從硬體讀回目前曝光時間（μs）。
        /// CLProtocol 已啟用：MdigInquireFeature("ExposureTime")，直接為 μs。
        /// CLProtocol 未啟用：MdigInquire(M_EXPOSURE_TIME) 回傳 ns 後除以 1000；
        ///   Camera Link 無 CLProtocol 時硬體通常回傳 0。
        /// </summary>
        public double GetMeasuredExposureUs()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;

            if (_clProtocolEnabled)
            {
                double valUs = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE,
                    "ExposureTime", MIL.M_TYPE_DOUBLE, ref valUs);
                return valUs;
            }
            else
            {
                double valNs = 0;
                MIL.MdigInquire(_milDigitizer, MIL.M_EXPOSURE_TIME, ref valNs);
                return valNs > 0 ? valNs / 1000.0 : 0;
            }
        }

        // ==================== Grab Height ====================

        /// <summary>
        /// 變更 Grab 高度並重新分配所有 MIL 與 CUDA Pinned Buffer。
        /// 流程：停止抓圖 → 釋放舊 Buffer → 設定新高度 → 重新分配 → 重啟抓圖。
        /// 若分配失敗，自動 rollback 至原本高度；rollback 亦失敗則停用相機。
        /// </summary>
        public void SetGrabHeight(int height)
        {
            if (_milDigitizer == MIL.M_NULL || height <= 0) return;

            bool wasLive = IsLive;
            int  oldHeight = CameraGrabHeight;

            // 1. 停止抓圖（不修改 _userWantsGrab）
            if (wasLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            // 2–3. 釋放所有舊 Buffer
            FreeGrabBuffers();

            // 4–9. 分配新高度；失敗則 rollback 至原高度
            try
            {
                AllocateAndBind(height, wasLive);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] SetGrabHeight({height}) failed: {ex.GetType().Name}: {ex.Message}. Rolling back to {oldHeight}px.");
                FreeGrabBuffers(); // 清除可能的部分分配殘留
                try
                {
                    AllocateAndBind(oldHeight, wasLive);
                }
                catch (Exception rex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetGrabHeight rollback to {oldHeight}px also failed: {rex.GetType().Name}: {rex.Message}. Camera disabled.");
                    _userWantsGrab = false; // 防止 Timer 不斷重試已損壞的相機
                }
            }
        }

        /// <summary>釋放所有 MIL Grab/Display/Proc Buffer 與 CUDA Pinned Memory。</summary>
        private void FreeGrabBuffers()
        {
            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                if (_milGrabBuffers[i] != MIL.M_NULL)
                {
                    MIL.MbufFree(_milGrabBuffers[i]);
                    _milGrabBuffers[i] = MIL.M_NULL;
                }
            }
            if (_milDisplayBuffer != MIL.M_NULL) { MIL.MbufFree(_milDisplayBuffer); _milDisplayBuffer = MIL.M_NULL; }
            if (_milProcBuffer    != MIL.M_NULL) { MIL.MbufFree(_milProcBuffer);    _milProcBuffer    = MIL.M_NULL; }
            _milLastGrabBuffer = MIL.M_NULL;  // 不 free（它是 grab buffer 之一，已在上面釋放）

            FreeResizeBuffers();
            lock (_picoaterLock)
            {
                _nativeBufferPool?.Dispose();
                _nativeBufferPool = null;
            }
            _hostInputBuffer  = null;
            _hostOutputBuffer = null;
        }

        /// <summary>
        /// 設定指定高度，重新分配所有 Buffer 並重新綁定 Display。
        /// 呼叫前必須先呼叫 FreeGrabBuffers()。
        /// </summary>
        private void AllocateAndBind(int targetHeight, bool shouldRestart)
        {
            // 4. 設定新高度
            MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)targetHeight);
            CameraGrabHeight = targetHeight;

            // 5. 查詢實際尺寸（硬體可能夾緊至最近合法值）
            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            _frameWidth  = (int)sizeX;
            _frameHeight = (int)sizeY;

            // 6. 重新分配 CPU Buffer 與 NativeBufferPool + resize buffers
            _hostInputBuffer  = new byte[_frameWidth * _frameHeight];
            _hostOutputBuffer = new byte[_frameWidth * _frameHeight];
            lock (_picoaterLock)
            {
                _nativeBufferPool = new NativeBufferPool(_frameWidth, _frameHeight, 1);
            }
            AllocateResizeBuffers();

            // 7. 重新分配 MIL Buffer
            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                MIL.MbufClear(_milGrabBuffers[i], 0);
            }
            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            MIL.MbufClear(_milDisplayBuffer, 0);
            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_PROC, ref _milProcBuffer);
            MIL.MbufClear(_milProcBuffer, 0);

            // 8. 重新綁定主顯示視窗
            MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
            MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);

            // 9. 恢復抓圖
            if (shouldRestart && _userWantsGrab)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
        }

        // ==================== Line Rate（CLProtocol Feature API）====================

        /// <summary>透過 CLProtocol GenICam Feature 讀取 Line Rate（Hz）。CLProtocol 未啟用時回傳 0。</summary>
        public double GetLineRateHz()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE,
                    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch { return 0; }
        }

        /// <summary>
        /// 設定線掃速率（Hz）。CLProtocol 就緒時走 Feature API；尚未就緒時僅記錄，
        /// 待 TryEnableCLProtocol 完成後自動重新套用（同 SetExposureUs 的 _appliedExposureUs 機制）。
        /// </summary>
        public void SetLineRateHz(double hz)
        {
            if (hz <= 0) return;
            _appliedLineRateHz = hz;
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE,
                    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] SetLineRateHz({hz}) failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ==================== Temperature ====================

        /// <summary>透過 CLProtocol GenICam Feature 讀取相機本體溫度（°C）。未啟用時回傳 NaN。</summary>
        public double GetCameraTemperature()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE,
                    "DeviceTemperature", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] GetCameraTemperature failed: {ex.GetType().Name}: {ex.Message}");
                return double.NaN;
            }
        }

        /// <summary>取得擷取卡 FPGA 溫度（°C）。MsysInquire M_TEMPERATURE_FPGA。</summary>
        public double GetFpgaTemperature()
        {
            if (_ownerSystemId == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MsysInquire(_ownerSystemId, MIL.M_TEMPERATURE_FPGA, ref val);
                return val;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] GetFpgaTemperature failed: {ex.GetType().Name}: {ex.Message}");
                return double.NaN;
            }
        }

        // ==================== Hardware Telemetry ====================

        /// <summary>取得板卡可用記憶體（MB）。</summary>
        public long GetMemoryFreeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_FREE, ref val);
            return (long)val / (1024 * 1024);
        }

        /// <summary>取得板卡總記憶體（MB）。</summary>
        public long GetMemorySizeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_SIZE, ref val);
            return (long)val;
        }

        /// <summary>取得 PCIe 通道數。</summary>
        public int GetPcieNumberOfLanes()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_NUMBER_OF_LANES, ref val);
            return (int)val;
        }

        /// <summary>取得 PCIe 速度字串（Gen1 / Gen2 / Gen3）。</summary>
        public string GetPcieSpeed()
        {
            if (_ownerSystemId == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_SPEED, ref val);
            if (val == MIL.M_GEN1) return "Gen1";
            if (val == MIL.M_GEN2) return "Gen2";
            if (val == MIL.M_GEN3) return "Gen3";
            return $"0x{val:X}";
        }

        // ==================== Frame Statistics ====================

        /// <summary>DCF 設定的目標 FPS（MdigInquire M_SELECTED_FRAME_RATE）。</summary>
        public double GetSelectedFrameRate()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            double val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SELECTED_FRAME_RATE, ref val);
            return val;
        }

        /// <summary>累計已處理的 Frame 數（MdigInquire M_PROCESS_FRAME_COUNT）。</summary>
        public long GetFrameCount()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_COUNT, ref val);
            return (long)val;
        }

        /// <summary>Processing callback 遺漏的 Frame 數（MdigInquire M_PROCESS_FRAME_MISSED）。</summary>
        public long GetFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>硬體 Grab 層遺漏的 Frame 數（MdigInquire M_GRAB_FRAME_MISSED）。</summary>
        public long GetGrabFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_GRAB_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>掃描模式字串（"Line" 或 "Progressive"）。</summary>
        public string GetScanMode()
        {
            if (_milDigitizer == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SCAN_MODE, ref val);
            return (val == MIL.M_LINESCAN) ? "Line" : "Progressive";
        }

        // ==================== Grab Control ====================

        /// <summary>套用曝光設定（live 調整用，CLProtocol 自動選路）。Grab Height 請改用 SetGrabHeight()。</summary>
        public void ApplyAcquisitionSettings()
        {
            if (_appliedExposureUs > 0)
                SetExposureUs(_appliedExposureUs);
        }

        public void SetUserGrabIntent(bool enable)
        {
            _userWantsGrab = enable;
            ApplyGrabState();
        }

        /// <summary>
        /// 依 _userWantsGrab 與 IsLive 狀態決定啟動或停止 MdigProcess。
        /// 首次啟動後觸發 CLProtocol 背景初始化（同 MilCameraUnit 設計）。
        /// </summary>
        public void ApplyGrabState()
        {
            if (_isReleased || _milDigitizer == MIL.M_NULL) return;

            if (_userWantsGrab && !IsLive && CheckPresence())
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;

                // 所有 MIL/GPU 資源就緒後才在背景啟動 CLProtocol
                StartCLProtocolAsync();
            }
            else if (!_userWantsGrab && IsLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }
        }

        public bool CheckPresence()
        {
            // 先檢查 _isReleased：Dispose() 在第一行即設為 true，之後才釋放 MIL 資源。
            // 若不加此檢查，當 CameraStatusTimer_Tick 快照的相機物件恰好在 Dispose() 進行中，
            // MdigInquire 可能存取到已 MdigFree 的 digitizer 而導致 crash。
            if (_isReleased || _milDigitizer == MIL.M_NULL) { IsConnected = false; return false; }
            MIL_INT presence = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_CAMERA_PRESENT, ref presence);
            IsConnected = (presence == MIL.M_YES);
            return IsConnected;
        }

        // ==================== ProcessingFunction ====================

        private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (userPtr == IntPtr.Zero) return MIL.M_NULL;

            GCHandle hObj = GCHandle.FromIntPtr(userPtr);
            var cam = hObj.Target as AniloxCamera;
            if (cam == null || cam._isReleased) return MIL.M_NULL;

            MIL_ID modifiedBuffer = MIL.M_NULL;
            MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);
            cam._milLastGrabBuffer = modifiedBuffer;

            if (modifiedBuffer != MIL.M_NULL && cam._milProcBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                // 不管 EnableImageProcessing，一律執行 GPU 處理以取得 Mura 曲線（供 CSV 日誌判斷）
                bool processedByPicoater = cam.TryApplyPicoaterRidge(modifiedBuffer, cam._milProcBuffer);

                // EnableImageProcessing 控制「顯示」：勾選才顯示處理結果，否則顯示原圖
                if (cam.EnableImageProcessing && processedByPicoater)
                    MIL.MbufCopy(cam._milProcBuffer, cam._milDisplayBuffer);
                else
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);

                cam.TrySaveCapture(modifiedBuffer);
            }

            return MIL.M_NULL;
        }

        // ==================== Picoater Ridge Processing ====================

        private bool TryApplyPicoaterRidge(MIL_ID srcBuffer, MIL_ID dstBuffer)
        {
            if (srcBuffer == MIL.M_NULL || dstBuffer == MIL.M_NULL) return false;
            if (_frameWidth <= 0 || _frameHeight <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;

            lock (_picoaterLock)
            {
                if (_nativeBufferPool == null) return false;

                IntPtr picoaterInputBuffer = _nativeBufferPool.InputBuffer;
                IntPtr picoaterRidgeBuffer = _nativeBufferPool.RidgeBuffer;
                if (picoaterInputBuffer == IntPtr.Zero || picoaterRidgeBuffer == IntPtr.Zero) return false;

                try
                {
                    MIL.MbufGet2d(srcBuffer, 0, 0, _frameWidth, _frameHeight, _hostInputBuffer);
                    Marshal.Copy(_hostInputBuffer, 0, picoaterInputBuffer, _hostInputBuffer.Length);

                    IntPtr picoaterCurveMean    = _nativeBufferPool.CurveMeanBuffer;
                    IntPtr picoaterCurveMax     = _nativeBufferPool.CurveMaxBuffer;
                    IntPtr picoaterRowCurveMean = _nativeBufferPool.CurveRowMeanBuffer;
                    IntPtr picoaterRowCurveMax  = _nativeBufferPool.CurveRowMaxBuffer;

                    _aoiService.ProcessImage(new AoiProcessRequest
                    {
                        Input = new AoiProcessRequest.InputImage
                        {
                            Width  = _frameWidth,
                            Height = _frameHeight,
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
                        int rowCurveLen = _frameHeight;
                        if (rowCurveLen > 0 && picoaterRowCurveMean != IntPtr.Zero && picoaterRowCurveMax != IntPtr.Zero)
                        {
                            float[] rowMeanArr = new float[rowCurveLen];
                            float[] rowMaxArr  = new float[rowCurveLen];
                            Marshal.Copy(picoaterRowCurveMean, rowMeanArr, 0, rowCurveLen);
                            Marshal.Copy(picoaterRowCurveMax,  rowMaxArr,  0, rowCurveLen);
                            OnLiveRowCurveData?.Invoke(CameraId, rowMeanArr, rowMaxArr);
                        }
                    }

                    // LiveDisplayDirection 控制顯示 V 或 H ridge
                    IntPtr displaySrc = (LiveDisplayDirection == "h")
                        ? _nativeBufferPool.MuraBuffer   // horizontal ridge
                        : picoaterRidgeBuffer;            // vertical ridge（預設）
                    Marshal.Copy(displaySrc, _hostOutputBuffer, 0, _hostOutputBuffer.Length);
                    MIL.MbufPut2d(dstBuffer, 0, 0, _frameWidth, _frameHeight, _hostOutputBuffer);
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
            if (_frameWidth <= 0 || _frameHeight <= 0) return false;
            if (_hostInputBuffer == null) return false;
            if (outColMean == null || outColMean.Length < _frameWidth) return false;

            lock (_picoaterLock)
            {
                if (_nativeBufferPool == null) return false;

                IntPtr inputBuffer = _nativeBufferPool.InputBuffer;
                if (inputBuffer == IntPtr.Zero) return false;

                try
                {
                    // 從最近 grab 到的原始影像（hook 中暫存）取資料
                    MIL_ID srcBuf = _milLastGrabBuffer;
                    if (srcBuf == MIL.M_NULL) return false;

                    MIL.MbufGet2d(srcBuf, 0, 0, _frameWidth, _frameHeight, _hostInputBuffer);
                    Marshal.Copy(_hostInputBuffer, 0, inputBuffer, _hostInputBuffer.Length);

                    // host float buffer for result
                    IntPtr hColMean = Marshal.AllocHGlobal(_frameWidth * sizeof(float));
                    try
                    {
                        _aoiService.ComputeColumnMean(_frameWidth, _frameHeight, inputBuffer, 2.0f, hColMean);
                        Marshal.Copy(hColMean, outColMean, 0, _frameWidth);
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
                if (TimestampCoordinator != null && _appliedLineRateHz > 0)
                    now = TimestampCoordinator.Coordinate((int)_appliedLineRateHz, now);

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
                            _nativeBufferPool.InputBuffer, _frameWidth, _frameHeight,
                            _rawResizeBuf, rw, rh);
                        rawBytes = new byte[pixels];
                        Marshal.Copy(_rawResizeBuf, rawBytes, 0, pixels);

                        // _ridgeBuffer = vertical ridge → _proc_v.jpg
                        NativeMethods.CoreCV_Resize_GPU(
                            _nativeBufferPool.RidgeBuffer, _frameWidth, _frameHeight,
                            _procResizeBuf, rw, rh);
                        procVBytes = new byte[pixels];
                        Marshal.Copy(_procResizeBuf, procVBytes, 0, pixels);

                        // _muraBuffer = horizontal ridge → _proc_h.jpg
                        NativeMethods.CoreCV_Resize_GPU(
                            _nativeBufferPool.MuraBuffer, _frameWidth, _frameHeight,
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

                if (hasResizeData)
                {
                    // 非壓縮模式：BMP 必須在 callback 同步匯出（sourceBuffer 會被 MIL 回收）
                    if (alsoBmp)
                    {
                        Directory.CreateDirectory(saveDir);
                        MIL.MbufExport(Path.Combine(saveDir, baseName + ".bmp"), MIL.M_BMP, sourceBuffer);
                    }

                    // 其餘 I/O 移至背景執行緒，callback 立即返回，不阻塞連續抓圖
                    Task.Run(() =>
                    {
                        try
                        {
                            Directory.CreateDirectory(saveDir);
                            SaveJpegFromBytes(rawBytes, rw, rh,
                                Path.Combine(saveDir, baseName + "_raw.jpg"), quality);

                            if (procVBytes != null)
                                SaveJpegFromBytes(procVBytes, rw, rh,
                                    Path.Combine(saveDir, baseName + "_proc_v.jpg"), quality);

                            if (procHBytes != null)
                                SaveJpegFromBytes(procHBytes, rw, rh,
                                    Path.Combine(saveDir, baseName + "_proc_h.jpg"), quality);

                            if (meanArr != null)
                            {
                                SaveCurveBinFromArray(meanArr, scale,
                                    Path.Combine(saveDir, baseName + "_mean_v.bin"));
                                SaveCurveBinFromArray(maxArr,  scale,
                                    Path.Combine(saveDir, baseName + "_max_v.bin"));
                            }

                            if (rowMeanArr != null)
                            {
                                SaveCurveBinFromArray(rowMeanArr, scale,
                                    Path.Combine(saveDir, baseName + "_mean_h.bin"));
                                SaveCurveBinFromArray(rowMaxArr,  scale,
                                    Path.Combine(saveDir, baseName + "_max_h.bin"));
                            }

                            OnInspectionResult?.Invoke(camId, baseName, meanPeak, maxPeak);
                        }
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
            if (_frameWidth <= 0 || _frameHeight <= 0 || _saveResizeScale <= 1) return;

            _resizeWidth  = _frameWidth  / _saveResizeScale;
            _resizeHeight = _frameHeight / _saveResizeScale;
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

        /// <summary>
        /// 將 8-bit 灰階 byte[] 存成 JPEG（在 Task.Run 背景執行緒呼叫）。
        /// GDI+ JPEG encoder 需要 24bpp，透過 GCHandle pin + Graphics.DrawImage 轉換。
        /// </summary>
        private static void SaveJpegFromBytes(byte[] data, int w, int h, string path, int quality)
        {
            var gch = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                using (var bmp8  = ImageUtils.Create8bppBitmap(gch.AddrOfPinnedObject(), w, h))
                using (var bmp24 = new Bitmap(w, h, PixelFormat.Format24bppRgb))
                using (var g     = Graphics.FromImage(bmp24))
                {
                    g.DrawImage(bmp8, 0, 0, w, h);

                    var codec = GetJpegEncoder();
                    if (codec == null) { bmp24.Save(path); return; }

                    using (var ep = new EncoderParameters(1))
                    {
                        ep.Param[0] = new EncoderParameter(Encoder.Quality, (long)quality);
                        bmp24.Save(path, codec, ep);
                    }
                }
            }
            finally
            {
                gch.Free();
            }
        }

        /// <summary>
        /// 將 float[] 曲線資料寫成自描述 .bin 格式（在 Task.Run 背景執行緒呼叫）。
        /// Header: magic(4)"MCBF" + version(4)=1 + scale_factor(4f) + array_length(4) + float[]
        /// </summary>
        private static void SaveCurveBinFromArray(float[] arr, int scaleForHeader, string path)
        {
            if (arr == null || arr.Length == 0) return;
            using (var bw = new BinaryWriter(File.Open(path, FileMode.Create, FileAccess.Write)))
            {
                bw.Write(new byte[] { (byte)'M', (byte)'C', (byte)'B', (byte)'F' });
                bw.Write(1);                        // version
                bw.Write((float)scaleForHeader);    // scale_factor（JPEG 縮小倍率，供讀取時參考）
                bw.Write(arr.Length);               // array_length
                for (int i = 0; i < arr.Length; i++)
                    bw.Write(arr[i]);
            }
        }

        // ==================== Event Handlers ====================

        private MIL_INT MouseClickHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_isReleased) return MIL.M_NULL;
            OnCameraClicked?.Invoke(CameraId);
            return MIL.M_NULL;
        }

        private MIL_INT MouseStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_isReleased || _milDisplayBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

            int x = (int)posX;
            int y = (int)posY;
            int pixelValue = -1;

            MIL_INT sizeX = MIL.MbufInquire(_milDisplayBuffer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MbufInquire(_milDisplayBuffer, MIL.M_SIZE_Y, MIL.M_NULL);

            if (x >= 0 && x < sizeX && y >= 0 && y < sizeY)
            {
                byte[] data = new byte[1];
                MIL.MbufGet2d(_milDisplayBuffer, x, y, 1, 1, data);
                pixelValue = data[0];
            }

            OnMouseDataChanged?.Invoke(CameraId, x, y, pixelValue);
            return MIL.M_NULL;
        }

        // ==================== Dispose ====================

        public void Free() => Dispose();

        public void Dispose()
        {
            if (_isReleased)
            {
                if (_hUserData.IsAllocated) _hUserData.Free();
                return;
            }

            _isReleased = true;

            if (_milDigitizer != MIL.M_NULL)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;

                if (_milDisplay != MIL.M_NULL)
                {
                    MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_LEFT_BUTTON_DOWN + MIL.M_UNHOOK, _mouseClickDelegate, IntPtr.Zero);
                    MIL.MdispSelectWindow(_milDisplay, MIL.M_NULL, IntPtr.Zero);
                }

                if (_milSecondaryDisplay != MIL.M_NULL)
                {
                    if (_isSecondaryHooked)
                    {
                        MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                        _isSecondaryHooked = false;
                    }
                    MIL.MdispSelectWindow(_milSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);
                    MIL.MdispFree(_milSecondaryDisplay);
                    _milSecondaryDisplay = MIL.M_NULL;
                }

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    if (_milGrabBuffers[i] != MIL.M_NULL)
                    {
                        MIL.MbufFree(_milGrabBuffers[i]);
                        _milGrabBuffers[i] = MIL.M_NULL;
                    }
                }

                if (_milDisplayBuffer != MIL.M_NULL) { MIL.MbufFree(_milDisplayBuffer); _milDisplayBuffer = MIL.M_NULL; }
                if (_milProcBuffer    != MIL.M_NULL) { MIL.MbufFree(_milProcBuffer);    _milProcBuffer    = MIL.M_NULL; }

                FreeResizeBuffers();
                lock (_picoaterLock)
                {
                    _nativeBufferPool?.Dispose();
                    _nativeBufferPool = null;
                    _aoiService.Dispose();
                }

                _hostInputBuffer  = null;
                _hostOutputBuffer = null;

                if (_milDisplay != MIL.M_NULL) { MIL.MdispFree(_milDisplay); _milDisplay = MIL.M_NULL; }
                MIL.MdigFree(_milDigitizer);
                _milDigitizer = MIL.M_NULL;
            }

            if (_hUserData.IsAllocated) _hUserData.Free();
        }
    }
}
