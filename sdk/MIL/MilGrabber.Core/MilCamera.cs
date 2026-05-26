using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    /// <summary>
    /// 封裝一台相機的 MIL 取像/顯示操作（純 MIL 範圍）。
    /// 檢測 / 存檔 / 合圖 / 縮圖等「非 MIL」邏輯不在此 — 由上層訂閱 <see cref="FrameReady"/> 自行處理。
    /// namespace 用 MilGrabber（非 all-caps MIL），避免與 Matrox 的 MIL 型別在簡單名稱解析時衝突（CS0118）。
    /// 元件刻意保留 Mil 字樣：MIL 開發需插 USB dongle，獨立成 sdk/MIL 區並用 Mil 命名警示此依賴。
    /// </summary>
    public class MilCamera : IDisposable
    {
        // 同一張卡的多個 digitizer 並行呼叫 M_GC_CLPROTOCOL,M_ENABLE 會搶 MIL 內部鎖導致失敗，
        // 以 static lock 序列化，確保每台相機依序完成初始化。
        private static readonly object _clProtocolInitLock = new object();

        // ==================== MIL Resources ====================
        private MIL_ID _ownerSystemId = MIL.M_NULL;
        private MIL_ID _milDigitizer = MIL.M_NULL;
        private MIL_ID _milDisplay = MIL.M_NULL;
        private MIL_ID _milSecondaryDisplay = MIL.M_NULL;
        private MIL_ID[] _milGrabBuffers = new MIL_ID[2];
        private MIL_ID _milDisplayBuffer = MIL.M_NULL;
        private MIL_ID _milLastGrabBuffer = MIL.M_NULL;
        private MIL_INT _milGrabBufferListSize = 2;

        public MIL_ID OwnerSystemId => _ownerSystemId;
        public MIL_ID MilDigitizer => _milDigitizer;
        public MIL_ID MilDisplay => _milDisplay;
        public MIL_ID MilSecondaryDisplay => _milSecondaryDisplay;
        public MIL_ID MilDisplayBuffer => _milDisplayBuffer;
        /// <summary>hook 中暫存的最近一幀原圖 buffer（供上層在 FrameReady 外延遲取用）。</summary>
        public MIL_ID LastGrabBuffer => _milLastGrabBuffer;

        // ==================== Identity ====================
        public int CameraId { get; private set; }
        private MIL_INT _devNum;
        private string _dcfPath;
        private IntPtr _panelHandle;

        // ==================== State ====================
        public bool IsLive { get; private set; } = false;
        public bool IsConnected { get; private set; } = false;
        public bool UserWantsGrab => _userWantsGrab;
        private bool _userWantsGrab = false;
        private bool _isReleased = false;
        private bool _isSecondaryHooked = false;

        public int FrameWidth { get; private set; } = 0;
        public int FrameHeight { get; private set; } = 0;

        /// <summary>Grab 高度（0 = 用 DCF 預設）。必須在 Initialize() 之前設定。</summary>
        public int CameraGrabHeight { get; set; } = 0;

        // ==================== CLProtocol ====================
        private bool _clProtocolEnabled = false;
        private volatile bool _clProtocolInitStarted = false;
        private volatile bool _clProtocolInitDone = false;
        /// <summary>CLProtocol 初始化（含參數重套）已完成，可安全從硬體讀回參數。</summary>
        public bool IsHwParamsStable => !_clProtocolInitStarted || _clProtocolInitDone;
        /// <summary>CLProtocol（GenICam Camera Link）是否已成功啟用。</summary>
        public bool IsClProtocolEnabled => _clProtocolEnabled;

        private double _appliedExposureUs = 0;
        private double _appliedLineRateHz = 0;

        // ==================== Delegates / Events ====================
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseClickDelegate;
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        /// <summary>
        /// 每幀 grab 完成（MIL 回呼執行緒）。參數：(this, modifiedBuffer)。
        /// 上層在此跑檢測（GetFrameBytes → 處理 → PutDisplayBytes / CopyToDisplay）。
        /// 無訂閱者時 library 預設 MbufCopy(modifiedBuffer → displayBuffer) 顯示原圖。
        /// </summary>
        public event Action<MilCamera, MIL_ID> FrameReady;

        /// <summary>滑鼠移動（MIL 回呼）。參數：(cameraId, x, y, pixelValue)；pixelValue=-1 表示超出影像範圍。</summary>
        public event Action<int, int, int, int> OnMouseDataChanged;

        /// <summary>主顯示左鍵點擊（MIL 回呼）。參數：cameraId。</summary>
        public event Action<int> OnCameraClicked;

        // ==================== Constructor ====================
        public MilCamera(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle)
        {
            _ownerSystemId = systemId;
            CameraId = id;
            _devNum = devNum;
            _dcfPath = dcfPath;
            _panelHandle = panelHandle;

            _mouseStatusDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseStatusHandler);
            _mouseClickDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseClickHandler);
            _processingDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);
            _hUserData = GCHandle.Alloc(this);
        }

        // ==================== Initialize ====================
        public void Initialize()
        {
            if (_ownerSystemId == MIL.M_NULL) return;

            MIL.MdigAlloc(_ownerSystemId, _devNum, _dcfPath, MIL.M_DEFAULT, ref _milDigitizer);
            if (_milDigitizer == MIL.M_NULL) return;

            // 先套用 Grab Height，再查詢實際尺寸以分配正確大小的 Buffer
            if (CameraGrabHeight > 0)
                MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)CameraGrabHeight);

            MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milDisplay);
            MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milSecondaryDisplay);

            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            FrameWidth = (int)sizeX;
            FrameHeight = (int)sizeY;

            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                MIL.MbufClear(_milGrabBuffers[i], 0);
            }

            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            MIL.MbufClear(_milDisplayBuffer, 0);

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

        // ==================== Grab Control ====================

        public void SetUserGrabIntent(bool enable)
        {
            _userWantsGrab = enable;
            ApplyGrabState();
        }

        /// <summary>依 _userWantsGrab 與 IsLive 決定啟動/停止 MdigProcess；首次啟動後背景啟用 CLProtocol。</summary>
        public void ApplyGrabState()
        {
            if (_isReleased || _milDigitizer == MIL.M_NULL) return;

            if (_userWantsGrab && !IsLive && CheckPresence())
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
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
            // 先檢查 _isReleased：Dispose() 第一行即設 true，之後才釋放 MIL 資源，避免存取已 free 的 digitizer。
            if (_isReleased || _milDigitizer == MIL.M_NULL) { IsConnected = false; return false; }
            MIL_INT presence = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_CAMERA_PRESENT, ref presence);
            IsConnected = (presence == MIL.M_YES);
            return IsConnected;
        }

        // ==================== Processing Hook ====================

        private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (userPtr == IntPtr.Zero) return MIL.M_NULL;

            GCHandle hObj = GCHandle.FromIntPtr(userPtr);
            var cam = hObj.Target as MilCamera;
            if (cam == null || cam._isReleased) return MIL.M_NULL;

            MIL_ID modifiedBuffer = MIL.M_NULL;
            MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);
            cam._milLastGrabBuffer = modifiedBuffer;

            if (modifiedBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                var handler = cam.FrameReady;
                if (handler != null)
                    handler(cam, modifiedBuffer);                       // 上層檢測 + 自行決定顯示
                else
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer); // 預設顯示原圖
            }

            return MIL.M_NULL;
        }

        // ==================== Buffer Helpers（給上層檢測用） ====================

        /// <summary>把指定 MIL buffer 的影像複製到 host byte[]（長度需 ≥ FrameWidth*FrameHeight）。</summary>
        public void GetFrameBytes(MIL_ID buffer, byte[] dst)
        {
            if (buffer == MIL.M_NULL || dst == null) return;
            MIL.MbufGet2d(buffer, 0, 0, FrameWidth, FrameHeight, dst);
        }

        /// <summary>把 host byte[] 寫入顯示 buffer（顯示處理結果）。</summary>
        public void PutDisplayBytes(byte[] src)
        {
            if (_milDisplayBuffer == MIL.M_NULL || src == null) return;
            MIL.MbufPut2d(_milDisplayBuffer, 0, 0, FrameWidth, FrameHeight, src);
        }

        /// <summary>把指定 MIL buffer 複製到顯示 buffer（顯示原圖或上層處理後的 MIL buffer）。</summary>
        public void CopyToDisplay(MIL_ID src)
        {
            if (src == MIL.M_NULL || _milDisplayBuffer == MIL.M_NULL) return;
            MIL.MbufCopy(src, _milDisplayBuffer);
        }

        /// <summary>清空顯示 buffer（填黑）。停 grab 後 displayBuffer 殘留最後一幀，重新綁定顯示前清掉避免顯示殘影。</summary>
        public void ClearDisplay()
        {
            if (_milDisplayBuffer != MIL.M_NULL) MIL.MbufClear(_milDisplayBuffer, 0);
        }

        // ==================== CLProtocol ====================

        private void StartCLProtocolAsync()
        {
            if (_clProtocolInitStarted) return;
            _clProtocolInitStarted = true;

            var initTask = Task.Run((Action)TryEnableCLProtocol);
            var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10));

            Task.WhenAny(initTask, timeoutTask).ContinueWith(_ =>
            {
                if (!initTask.IsCompleted)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] CLProtocol 初始化逾時（>10s）。CLProtocol 已停用，曝光/線掃維持 fallback 路徑。");
                    _clProtocolInitDone = true;
                }
            });
        }

        private void TryEnableCLProtocol()
        {
            if (_milDigitizer == MIL.M_NULL) return;

            lock (_clProtocolInitLock)
            {
                if (_isReleased) return;
                try
                {
                    MIL_INT numDevIds = 0;
                    MIL.MdigInquire(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID_NUM, ref numDevIds);
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: {numDevIds} device ID(s) available.");

                    if (numDevIds > 0)
                    {
                        var devId = new System.Text.StringBuilder(512);
                        MIL.MdigInquire(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, devId);
                        string selectedId = devId.ToString();
                        System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: using device ID = \"{selectedId}\"");
                        MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, selectedId);
                    }
                    else
                    {
                        System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: no enumerated IDs, falling back to M_DEFAULT.");
                        MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
                    }

                    MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL, MIL.M_ENABLE);
                    _clProtocolEnabled = true;
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol enabled successfully.");

                    if (!_isReleased)
                    {
                        if (_appliedExposureUs > 0) SetExposureUs(_appliedExposureUs);
                        if (_appliedLineRateHz > 0) SetLineRateHz(_appliedLineRateHz);
                    }
                    _clProtocolInitDone = true;
                }
                catch (Exception ex)
                {
                    _clProtocolEnabled = false;
                    _clProtocolInitDone = true;
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol init failed: {ex.GetType().Name}: {ex.Message}");
                }
            }
        }

        // ==================== Exposure ====================

        /// <summary>
        /// 設定曝光（μs）。CLProtocol 啟用走 GenICam Feature（單位 μs），否則 legacy M_EXPOSURE_TIME（ns，自動 ×1000）。
        /// 上限：clamp(floor(900000 / lineRateHz), 1, 10000)。
        /// </summary>
        public void SetExposureUs(double exposureUs)
        {
            if (exposureUs <= 0) return;
            _appliedExposureUs = exposureUs;          // 先記住設定值（Initialize 前設也記得，待 Initialize() 套用）
            if (_milDigitizer == MIL.M_NULL) return;  // digitizer 未就緒：只記不套

            if (_appliedLineRateHz > 0)
            {
                double maxUs = Math.Max(1.0, Math.Min(10000.0, Math.Floor(900000.0 / _appliedLineRateHz)));
                if (exposureUs > maxUs)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetExposureUs: {exposureUs} μs 超出上限 {maxUs} μs（LR={_appliedLineRateHz} Hz），已夾緊。");
                    exposureUs = maxUs;
                    _appliedExposureUs = exposureUs;  // 夾緊後更新設定值
                }
            }

            if (_clProtocolEnabled)
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_DOUBLE, ref exposureUs);
            else
                MIL.MdigControl(_milDigitizer, MIL.M_EXPOSURE_TIME, exposureUs * 1000.0);
        }

        /// <summary>回傳最後一次 SetExposureUs 的設定值（μs），不依賴硬體回讀。</summary>
        public double GetExposureUs() => _appliedExposureUs;

        /// <summary>從硬體讀回曝光（μs）。CLProtocol 未啟用時回傳 0（避免相機預設值覆寫設定）。</summary>
        public double GetMeasuredExposureUs()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            if (_clProtocolEnabled)
            {
                double valUs = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_DOUBLE, ref valUs);
                return valUs;
            }
            return 0;
        }

        // ==================== Line Rate ====================

        /// <summary>最後一次 SetLineRateHz 設定的線掃速率（Hz）；不依賴硬體回讀，CLProtocol 未就緒也有值。
        /// 供時間戳協調等需要「設定值」而非「即時硬體值」的場景（GetLineRateHz 在 CLProtocol 未就緒時回 0）。</summary>
        public double AppliedLineRateHz => _appliedLineRateHz;

        public double GetLineRateHz()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch { return 0; }
        }

        /// <summary>查 grabber/相機的取樣頻率(line rate)上限(Hz)。需 CLProtocol 啟用；未就緒回 0。</summary>
        public double GetLineRateMaxHz()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            try
            {
                double max = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_MAX, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref max);
                return max;
            }
            catch { return 0; }
        }

        /// <summary>設定線掃速率（Hz）。CLProtocol 未就緒時僅記錄，待啟用後自動重套。</summary>
        public void SetLineRateHz(double hz)
        {
            if (hz <= 0) return;
            _appliedLineRateHz = hz;
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] SetLineRateHz({hz}) failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ==================== Grab Height ====================

        /// <summary>變更 Grab 高度並重新分配 MIL Buffer。失敗自動 rollback 至原高度。</summary>
        public void SetGrabHeight(int height)
        {
            if (_milDigitizer == MIL.M_NULL || height <= 0) return;

            bool wasLive = IsLive;
            int oldHeight = CameraGrabHeight;

            if (wasLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            FreeGrabBuffers();

            try
            {
                AllocateAndBind(height, wasLive);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] SetGrabHeight({height}) failed: {ex.GetType().Name}: {ex.Message}. Rolling back to {oldHeight}px.");
                FreeGrabBuffers();
                try
                {
                    AllocateAndBind(oldHeight, wasLive);
                }
                catch (Exception rex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetGrabHeight rollback to {oldHeight}px also failed: {rex.GetType().Name}: {rex.Message}. Camera disabled.");
                    _userWantsGrab = false;
                }
            }
        }

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
            _milLastGrabBuffer = MIL.M_NULL;
        }

        private void AllocateAndBind(int targetHeight, bool shouldRestart)
        {
            MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)targetHeight);
            CameraGrabHeight = targetHeight;

            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            FrameWidth = (int)sizeX;
            FrameHeight = (int)sizeY;

            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                MIL.MbufClear(_milGrabBuffers[i], 0);
            }
            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            MIL.MbufClear(_milDisplayBuffer, 0);

            MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
            MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);

            if (shouldRestart && _userWantsGrab)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
        }

        // ==================== Telemetry ====================

        /// <summary>目前實際量測 FPS（M_PROCESS_FRAME_RATE）。抓圖未啟動時回傳 0。</summary>
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

        /// <summary>相機本體溫度（°C，CLProtocol Feature DeviceTemperature）。未啟用回 NaN。</summary>
        public double GetCameraTemperature()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "DeviceTemperature", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch { return double.NaN; }
        }

        /// <summary>擷取卡 FPGA 溫度（°C，MsysInquire M_TEMPERATURE_FPGA）。</summary>
        public double GetFpgaTemperature()
        {
            if (_ownerSystemId == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MsysInquire(_ownerSystemId, MIL.M_TEMPERATURE_FPGA, ref val);
                return val;
            }
            catch { return double.NaN; }
        }

        /// <summary>板卡可用記憶體（MB）。</summary>
        public long GetMemoryFreeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_FREE, ref val);
            return (long)val / (1024 * 1024);
        }

        /// <summary>PCIe 通道數。</summary>
        public int GetPcieNumberOfLanes()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_NUMBER_OF_LANES, ref val);
            return (int)val;
        }

        /// <summary>PCIe 速度字串（Gen1 / Gen2 / Gen3）。</summary>
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

        /// <summary>DCF 設定的目標 FPS（M_SELECTED_FRAME_RATE）。</summary>
        public double GetSelectedFrameRate()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            double val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SELECTED_FRAME_RATE, ref val);
            return val;
        }

        /// <summary>累計已處理 Frame 數（M_PROCESS_FRAME_COUNT）。</summary>
        public long GetFrameCount()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_COUNT, ref val);
            return (long)val;
        }

        /// <summary>Processing callback 遺漏 Frame 數（M_PROCESS_FRAME_MISSED）。</summary>
        public long GetFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>硬體 Grab 層遺漏 Frame 數（M_GRAB_FRAME_MISSED）。</summary>
        public long GetGrabFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_GRAB_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>掃描模式（"Line" / "Progressive"）。</summary>
        public string GetScanMode()
        {
            if (_milDigitizer == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SCAN_MODE, ref val);
            return (val == MIL.M_LINESCAN) ? "Line" : "Progressive";
        }

        // ==================== Primary Display ====================

        /// <summary>Detach / restore 主顯示（visible=false → MdispSelectWindow(M_NULL)）。</summary>
        public void SetPrimaryDisplayVisible(bool visible)
        {
            if (_milDisplay == MIL.M_NULL) return;
            if (visible)
            {
                if (_panelHandle != IntPtr.Zero && _milDisplayBuffer != MIL.M_NULL)
                {
                    MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
                    MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                }
            }
            else
            {
                MIL.MdispSelectWindow(_milDisplay, MIL.M_NULL, IntPtr.Zero);
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

        /// <summary>查詢副顯示 zoom/pan 狀態（隨使用者滾輪改變）。</summary>
        public bool TryGetSecondaryDisplayGeometry(out double zoomX, out double zoomY, out double panX, out double panY)
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

        /// <summary>設定副顯示縮放/平移（M_UPDATE DISABLE/ENABLE 批次，避免閃爍）。</summary>
        public void SetSecondaryDisplayZoom(double zoom, double panX, double panY)
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;
            try
            {
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispZoom(_milSecondaryDisplay, zoom, zoom);
                MIL.MdispPan(_milSecondaryDisplay, panX, panY);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[MilCamera.SetSecondaryDisplayZoom] {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>重置副顯示縮放/平移為 fit-to-window。</summary>
        public void ResetSecondaryDisplayView()
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;
            try
            {
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[MilCamera.ResetSecondaryDisplayView] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ==================== Mouse Hooks（MIL 機制，射出座標事件給上層） ====================

        private MIL_INT MouseClickHandler(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (_isReleased) return MIL.M_NULL;
            OnCameraClicked?.Invoke(CameraId);
            return MIL.M_NULL;
        }

        private MIL_INT MouseStatusHandler(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (_isReleased || _milDisplayBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(eventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(eventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

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

            if (_milDigitizer != MIL.M_NULL && IsLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            if (_isSecondaryHooked && _milSecondaryDisplay != MIL.M_NULL)
            {
                MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                _isSecondaryHooked = false;
            }

            if (_milDisplay != MIL.M_NULL) MIL.MdispSelectWindow(_milDisplay, MIL.M_NULL, IntPtr.Zero);
            if (_milSecondaryDisplay != MIL.M_NULL) MIL.MdispSelectWindow(_milSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);

            FreeGrabBuffers();

            if (_milSecondaryDisplay != MIL.M_NULL) { MIL.MdispFree(_milSecondaryDisplay); _milSecondaryDisplay = MIL.M_NULL; }
            if (_milDisplay != MIL.M_NULL) { MIL.MdispFree(_milDisplay); _milDisplay = MIL.M_NULL; }
            if (_milDigitizer != MIL.M_NULL) { MIL.MdigFree(_milDigitizer); _milDigitizer = MIL.M_NULL; }

            if (_hUserData.IsAllocated) _hUserData.Free();
        }
    }

    /// <summary>
    /// 相機參數計算公式（純函式、無狀態、無 UI／MIL 依賴）— 跨專案共用的單一真相。
    /// 主程式與範例都呼叫這裡，避免同一條物理公式抄多份而分歧。
    /// </summary>
    public static class MilCameraParams
    {
        /// <summary>曝光時間(μs) × 線掃速率(Hz) ≈ 此固定常數（硬體限制）；線掃越高，曝光上限越低。</summary>
        public const int ExposureLineRateProduct = 900000;

        /// <summary>
        /// 依線掃速率算曝光動態上限(μs)：lineRateHz ≤ 0 → expMaxCap（無線掃資訊時用絕對上限）；
        /// 否則 clamp(ExposureLineRateProduct / lineRateHz, expMin, expMaxCap)。
        /// </summary>
        public static int CalcExposureMaxUs(int lineRateHz, int expMin, int expMaxCap)
        {
            if (lineRateHz <= 0) return expMaxCap;
            int v = ExposureLineRateProduct / lineRateHz;
            if (v < expMin) v = expMin;
            if (v > expMaxCap) v = expMaxCap;
            return v;
        }
    }
}
