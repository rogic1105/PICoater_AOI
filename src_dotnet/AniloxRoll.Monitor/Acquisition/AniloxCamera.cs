using System;
using System.Runtime.InteropServices;
using System.IO;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;
using AOI.SDK.Core;
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

        private string _lastCaptureSecondKey = string.Empty;

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

        // ==================== Delegates / Events ====================
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseClickDelegate;
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        public event Action<int, int, int, int> OnMouseDataChanged;
        public event Action<int> OnCameraClicked;

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
        /// </summary>
        private void StartCLProtocolAsync()
        {
            if (_clProtocolInitStarted) return;
            _clProtocolInitStarted = true;
            Task.Run(() => TryEnableCLProtocol());
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
            catch
            {
                _clProtocolEnabled = false;
            }
        }

        // ==================== Exposure Control ====================

        /// <summary>
        /// 設定曝光時間（μs）。
        /// CLProtocol 已啟用：MdigControlFeature("ExposureTime")，GenICam 單位直接為 μs。
        /// CLProtocol 未啟用：MdigControl(M_EXPOSURE_TIME)，MIL 單位為 ns，自動乘以 1000。
        /// </summary>
        public void SetExposureUs(double exposureUs)
        {
            if (_milDigitizer == MIL.M_NULL || exposureUs <= 0) return;

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
        /// </summary>
        public void SetGrabHeight(int height)
        {
            if (_milDigitizer == MIL.M_NULL || height <= 0) return;

            bool wasLive = IsLive;

            // 1. 停止抓圖（不修改 _userWantsGrab）
            if (wasLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            // 2. 釋放舊 MIL Buffer
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

            // 3. 釋放 CUDA Pinned Memory（NativeBufferPool）
            lock (_picoaterLock)
            {
                _nativeBufferPool?.Dispose();
                _nativeBufferPool = null;
            }
            _hostInputBuffer  = null;
            _hostOutputBuffer = null;

            // 4. 設定新高度
            MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)height);
            CameraGrabHeight = height;

            // 5. 查詢實際尺寸（硬體可能夾緊至最近合法值）
            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            _frameWidth  = (int)sizeX;
            _frameHeight = (int)sizeY;

            // 6. 重新分配 CPU Buffer 與 NativeBufferPool
            _hostInputBuffer  = new byte[_frameWidth * _frameHeight];
            _hostOutputBuffer = new byte[_frameWidth * _frameHeight];
            lock (_picoaterLock)
            {
                _nativeBufferPool = new NativeBufferPool(_frameWidth, _frameHeight, 1);
            }

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
            if (wasLive && _userWantsGrab)
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
            catch { }
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
            catch { return double.NaN; }
        }

        /// <summary>取得擷取卡 FPGA 溫度（°C）。MsysInquire M_TEMPERATURE_FPGA。</summary>
        public double GetFpgaTemperature()
        {
            if (_ownerSystemId == MIL.M_NULL) return double.NaN;
            double val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_TEMPERATURE_FPGA, ref val);
            return val;
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
            if (_milDigitizer == MIL.M_NULL) return;

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
            if (_milDigitizer == MIL.M_NULL) { IsConnected = false; return false; }
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

            if (modifiedBuffer != MIL.M_NULL && cam._milProcBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                if (!cam.EnableImageProcessing)
                {
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                    cam.TrySaveCapture(modifiedBuffer);
                    return MIL.M_NULL;
                }

                bool processedByPicoater = cam.TryApplyPicoaterRidge(modifiedBuffer, cam._milProcBuffer);

                if (processedByPicoater)
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
                            BackgroundData = IntPtr.Zero,
                            MuraData       = IntPtr.Zero,
                            RidgeData      = picoaterRidgeBuffer,
                            MuraCurveMean  = IntPtr.Zero,
                            MuraCurveMax   = IntPtr.Zero,
                            Stream         = IntPtr.Zero
                        },
                        Params = new AoiProcessRequest.AlgorithmParams
                        {
                            BgSigmaFactor  = 2.0f,
                            RidgeSigma     = (float)HessianSigma,
                            HessianMaxFactor = (float)HessianFixedMax,
                            RidgeMode      = "vertical"
                        }
                    });

                    Marshal.Copy(picoaterRidgeBuffer, _hostOutputBuffer, 0, _hostOutputBuffer.Length);
                    MIL.MbufPut2d(dstBuffer, 0, 0, _frameWidth, _frameHeight, _hostOutputBuffer);
                    return true;
                }
                catch { return false; }
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
                string secondKey = now.ToString("yyyyMMdd_HHmmss");
                if (string.Equals(_lastCaptureSecondKey, secondKey, StringComparison.Ordinal)) return;

                string saveDir = Path.Combine(
                    CaptureRootPath,
                    now.ToString("yyyy"),
                    now.ToString("yyyyMM"),
                    now.ToString("yyyyMMdd"));
                Directory.CreateDirectory(saveDir);

                string fileName = $"{now:yyyyMMdd_HHmmss}-{CameraId}.bmp";
                MIL.MbufExport(Path.Combine(saveDir, fileName), MIL.M_BMP, sourceBuffer);
                _lastCaptureSecondKey = secondKey;
            }
            catch { }
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
