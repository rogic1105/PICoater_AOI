using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;
using AOI.SDK.Core;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 封裝單台相機的所有 MIL 資源：Digitizer、Display、Grab Buffer、Processing Buffer。
    /// 透過 <see cref="Initialize"/> 分配資源，透過 <see cref="Free"/> 釋放。
    /// System 的生命週期由外部（CameraSession）管理，不在此類別內釋放。
    /// </summary>
    public class MilCameraUnit
    {
        public MIL_ID MilDigitizer = MIL.M_NULL;
        public MIL_ID MilDisplay = MIL.M_NULL;
        private MIL_ID _milProcBuffer = MIL.M_NULL;

        /// <summary>此相機所屬的 MIL System ID（由外部傳入，不在此管理生命週期）</summary>
        private MIL_ID _ownerSystemId = MIL.M_NULL;

        private MIL_ID[] _milGrabBuffers = new MIL_ID[2];
        private MIL_ID _milDisplayBuffer = MIL.M_NULL;
        private MIL_INT _milGrabBufferListSize = 2;

        /// <summary>目前是否正在連續抓圖（MdigProcess 已啟動）</summary>
        public bool IsLive { get; private set; } = false;

        /// <summary>此相機的識別 ID（對應 CameraConfig.Id）</summary>
        public int CameraId { get; private set; }

        /// <summary>最近一次 CheckPresence() 的結果</summary>
        public bool IsConnected { get; private set; } = false;

        /// <summary>使用者是否希望抓圖（由 SetUserGrabIntent 設定）</summary>
        public bool UserWantsGrab => _userWantsGrab;

        /// <summary>是否啟用 GPU 影像處理（true = 二值化，false = 直接顯示原圖）</summary>
        public bool EnableImageProcessing { get; set; } = true;

        public bool EnableHessian { get; set; } = true;

        /// <summary>GPU 二值化閾值（0–255）</summary>
        public double BinarizeThreshold { get; set; } = 128.0;

        public double HessianSigma { get; set; } = 85;
        public double HessianFixedMax { get; set; } = 1.0;

        private bool _userWantsGrab = false;
        private bool _isReleased = false;
        private MIL_INT _devNum;
        private string _dcfPath;
        private IntPtr _panelHandle;

        private int _frameWidth = 0;
        private int _frameHeight = 0;
        private byte[] _hostInputBuffer = null;
        private byte[] _hostOutputBuffer = null;
        private IntPtr _gpuInputBuffer = IntPtr.Zero;
        private IntPtr _gpuOutputBuffer = IntPtr.Zero;

        // ================= MdigInquire 屬性 =================

        /// <summary>
        /// 目前實際量測的 FPS（來自 MdigInquire M_PROCESS_FRAME_RATE）。
        /// 抓圖尚未啟動時回傳 0。
        /// </summary>
        public double CurrentFps
        {
            get
            {
                if (MilDigitizer == MIL.M_NULL) return 0;
                double fps = 0;
                MIL.MdigInquire(MilDigitizer, MIL.M_PROCESS_FRAME_RATE, ref fps);
                return fps;
            }
        }

        /// <summary>影像解析度字串，格式為 "寬×高"。Initialize 前回傳 "N/A"。</summary>
        public string Resolution => (_frameWidth > 0 && _frameHeight > 0)
            ? $"{_frameWidth}×{_frameHeight}" : "N/A";

        /// <summary>最後一次透過 SetExposureUs 寫入的曝光值（μs）</summary>
        private double _appliedExposureUs = 0;

        /// <summary>
        /// CLProtocol（GenICam Camera Link）是否已成功啟用。
        /// true = 曝光讀寫走 MdigControlFeature/MdigInquireFeature；
        /// false = fallback 至 MdigControl/MdigInquire（M_EXPOSURE_TIME，單位 ns）。
        /// </summary>
        private bool _clProtocolEnabled = false;

        // ================= CLProtocol 初始化 =================

        /// <summary>
        /// 在背景執行緒啟動 CLProtocol 初始化，避免阻塞 Initialize()。
        /// MdigControl(M_GC_CLPROTOCOL, M_ENABLE) 需載入 CLProtocol DLL 並讀取相機 GenICam XML，
        /// 耗時較長，因此以 Task.Run 非同步執行。
        /// 初始化完成後自動重新套用曝光設定值。
        /// </summary>
        private void StartCLProtocolAsync()
        {
            Task.Run(() => TryEnableCLProtocol());
        }

        /// <summary>
        /// 實際執行 CLProtocol 啟用流程（在背景執行緒中呼叫）。
        /// 成功後 <see cref="_clProtocolEnabled"/> 設為 true，並重新套用 _appliedExposureUs。
        /// 相機不支援 CLProtocol 時靜默失敗，維持 false 並 fallback 至傳統 MdigControl。
        /// </summary>
        private void TryEnableCLProtocol()
        {
            if (MilDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControl(MilDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
                MIL.MdigControl(MilDigitizer, MIL.M_GC_CLPROTOCOL, MIL.M_ENABLE);
                _clProtocolEnabled = true;

                // CLProtocol 就緒後，重新套用曝光值（改走 Feature API）
                if (_appliedExposureUs > 0 && !_isReleased)
                    SetExposureUs(_appliedExposureUs);
            }
            catch
            {
                _clProtocolEnabled = false;
            }
        }

        // ================= 曝光控制 =================

        /// <summary>
        /// 設定曝光時間（μs）。
        /// CLProtocol 已啟用時：MdigControlFeature("ExposureTime")，GenICam 單位為 μs，直接傳入。
        /// CLProtocol 未啟用時：MdigControl(M_EXPOSURE_TIME)，MIL 單位為 ns，自動乘以 1000。
        /// 無論哪種路徑，均記錄設定值至 _appliedExposureUs 供備援顯示。
        /// </summary>
        /// <param name="exposureUs">曝光時間（μs）</param>
        public void SetExposureUs(double exposureUs)
        {
            if (MilDigitizer == MIL.M_NULL || exposureUs <= 0) return;

            if (_clProtocolEnabled)
                MIL.MdigControlFeature(MilDigitizer, MIL.M_FEATURE_VALUE,
                    "ExposureTime", MIL.M_TYPE_DOUBLE, ref exposureUs);
            else
                MIL.MdigControl(MilDigitizer, MIL.M_EXPOSURE_TIME, exposureUs * 1000.0);

            _appliedExposureUs = exposureUs;
        }

        /// <summary>
        /// 回傳最後一次 SetExposureUs 寫入的設定值（μs）。
        /// 不依賴硬體回讀，永遠等於上次設定的數值。
        /// </summary>
        public double GetExposureUs()
        {
            return _appliedExposureUs;
        }

        /// <summary>
        /// 從硬體讀回目前曝光時間（μs）。
        /// CLProtocol 已啟用時：MdigInquireFeature("ExposureTime")，GenICam 單位為 μs，直接回傳。
        /// CLProtocol 未啟用時：MdigInquire(M_EXPOSURE_TIME)，回傳 ns 後除以 1000；硬體不支援時回傳 0。
        /// </summary>
        public double GetMeasuredExposureUs()
        {
            if (MilDigitizer == MIL.M_NULL) return 0;

            if (_clProtocolEnabled)
            {
                double valUs = 0;
                MIL.MdigInquireFeature(MilDigitizer, MIL.M_FEATURE_VALUE,
                    "ExposureTime", MIL.M_TYPE_DOUBLE, ref valUs);
                return valUs;
            }
            else
            {
                double valNs = 0;
                MIL.MdigInquire(MilDigitizer, MIL.M_EXPOSURE_TIME, ref valNs);
                return valNs > 0 ? valNs / 1000.0 : 0;
            }
        }

        // ================= MdigInquire 查詢方法 =================

        /// <summary>
        /// 取得 DCF 設定的目標 Frame Rate（fps）。
        /// 來自 MdigInquire M_SELECTED_FRAME_RATE。
        /// </summary>
        public double GetSelectedFrameRate()
        {
            if (MilDigitizer == MIL.M_NULL) return 0;
            double val = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_SELECTED_FRAME_RATE, ref val);
            return val;
        }

        /// <summary>
        /// 取得自 MdigProcess 啟動後累計已處理的 Frame 數。
        /// 來自 MdigInquire M_PROCESS_FRAME_COUNT。
        /// </summary>
        public long GetFrameCount()
        {
            if (MilDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_PROCESS_FRAME_COUNT, ref val);
            return (long)val;
        }

        /// <summary>
        /// 取得 Processing callback 來不及處理而遺漏的 Frame 數。
        /// 來自 MdigInquire M_PROCESS_FRAME_MISSED。
        /// </summary>
        public long GetFrameMissed()
        {
            if (MilDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_PROCESS_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>
        /// 取得硬體 Grab 層遺漏的 Frame 數（Buffer 不足導致）。
        /// 來自 MdigInquire M_GRAB_FRAME_MISSED。
        /// </summary>
        public long GetGrabFrameMissed()
        {
            if (MilDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_GRAB_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>
        /// 取得掃描模式字串。
        /// 來自 MdigInquire M_SCAN_MODE，回傳 "Line" 或 "Progressive"。
        /// </summary>
        public string GetScanMode()
        {
            if (MilDigitizer == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_SCAN_MODE, ref val);
            return (val == MIL.M_LINESCAN) ? "Line" : "Progressive";
        }

        // ================= MsysInquire 系統層級查詢 =================

        // ================= Grab 高度重新初始化 =================

        /// <summary>
        /// 變更 Grab 高度並重新分配所有 MIL 與 GPU 緩衝區。
        /// 直接呼叫 MdigControl(M_SOURCE_SIZE_Y)（參考 AniloxRoll.Monitor/AniloxCamera.cs）。
        /// 流程：停止抓圖 → 釋放舊 Buffer → 設定新高度 → 重新分配 → 重啟抓圖。
        /// </summary>
        /// <param name="height">新的影像高度（行數）</param>
        public void SetGrabHeight(int height)
        {
            if (MilDigitizer == MIL.M_NULL || height <= 0) return;

            bool wasLive = IsLive;

            // 1. 停止抓圖（不修改 _userWantsGrab）
            if (wasLive)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize,
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

            // 3. 釋放舊 GPU Buffer
            if (_gpuInputBuffer  != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU(_gpuInputBuffer);  _gpuInputBuffer  = IntPtr.Zero; }
            if (_gpuOutputBuffer != IntPtr.Zero) { CoreCVWrapper.CoreCV_FreeGPU(_gpuOutputBuffer); _gpuOutputBuffer = IntPtr.Zero; }
            _hostInputBuffer  = null;
            _hostOutputBuffer = null;

            // 4. 設定新高度（M_SOURCE_SIZE_Y 在 MIL .NET wrapper 中存在，參考 AniloxCamera.cs）
            MIL.MdigControl(MilDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)height);

            // 5. 重新查詢實際尺寸（硬體可能夾緊至最近合法值）
            MIL_INT sizeX = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            _frameWidth  = (int)sizeX;
            _frameHeight = (int)sizeY;

            // 6. 重新分配 CPU / GPU Buffer
            _hostInputBuffer  = new byte[_frameWidth * _frameHeight];
            _hostOutputBuffer = new byte[_frameWidth * _frameHeight];
            CoreCVWrapper.CoreCV_MallocGPU(out _gpuInputBuffer,  _frameWidth, _frameHeight);
            CoreCVWrapper.CoreCV_MallocGPU(out _gpuOutputBuffer, _frameWidth, _frameHeight);

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

            // 8. 重新綁定 Display Window
            MIL.MdispSelectWindow(MilDisplay, _milDisplayBuffer, _panelHandle);
            MIL.MdispControl(MilDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);

            // 9. 恢復抓圖
            if (wasLive && _userWantsGrab)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
        }

        // ================= Line Rate（CLProtocol Feature API）=================

        /// <summary>
        /// 透過 CLProtocol GenICam Feature "AcquisitionLineRate" 讀取目前 Line Rate（Hz）。
        /// CLProtocol 未啟用或相機不支援時回傳 0。
        /// </summary>
        public double GetLineRateHz()
        {
            if (!_clProtocolEnabled || MilDigitizer == MIL.M_NULL) return 0;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(MilDigitizer, MIL.M_FEATURE_VALUE,
                    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch
            {
                return 0;
            }
        }

        /// <summary>
        /// 透過 CLProtocol GenICam Feature "AcquisitionLineRate" 設定 Line Rate（Hz）。
        /// CLProtocol 未啟用或相機不支援時靜默失敗。
        /// </summary>
        /// <param name="hz">目標 Line Rate（Hz）</param>
        public void SetLineRateHz(double hz)
        {
            if (!_clProtocolEnabled || MilDigitizer == MIL.M_NULL || hz <= 0) return;
            try
            {
                MIL.MdigControlFeature(MilDigitizer, MIL.M_FEATURE_VALUE,
                    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
            }
            catch { }
        }

        /// <summary>
        /// 透過 CLProtocol GenICam Feature "DeviceTemperature" 讀取相機本體溫度（°C）。
        /// CLProtocol 未啟用或相機不支援此 Feature 時回傳 double.NaN。
        /// </summary>
        public double GetCameraTemperature()
        {
            if (!_clProtocolEnabled || MilDigitizer == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(MilDigitizer, MIL.M_FEATURE_VALUE,
                    "DeviceTemperature", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch
            {
                return double.NaN;
            }
        }

        /// <summary>
        /// 取得板卡 FPGA 溫度（°C）。
        /// 來自 MsysInquire M_TEMPERATURE_FPGA。
        /// 硬體不支援時回傳 double.NaN。
        /// </summary>
        public double GetFpgaTemperature()
        {
            if (_ownerSystemId == MIL.M_NULL) return double.NaN;
            double val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_TEMPERATURE_FPGA, ref val);
            return val;
        }

        /// <summary>
        /// 取得板卡可用記憶體（MB）。
        /// 來自 MsysInquire M_MEMORY_FREE（原始單位為 bytes，此方法已轉換為 MB）。
        /// System 為 M_NULL 時回傳 -1。
        /// </summary>
        public long GetMemoryFreeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_FREE, ref val);
            return (long)val / (1024 * 1024);
        }

        /// <summary>
        /// 取得板卡總記憶體大小（MB）。
        /// 來自 MsysInquire M_MEMORY_SIZE（MIL 已以 MB 為單位回傳）。
        /// </summary>
        public long GetMemorySizeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_SIZE, ref val);
            return (long)val;
        }

        /// <summary>
        /// 取得 PCIe 通道數。
        /// 來自 MsysInquire M_PCIE_NUMBER_OF_LANES。
        /// System 為 M_NULL 時回傳 -1。
        /// </summary>
        public int GetPcieNumberOfLanes()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_NUMBER_OF_LANES, ref val);
            return (int)val;
        }

        /// <summary>
        /// 取得 PCIe 傳輸速度字串。
        /// 來自 MsysInquire M_PCIE_SPEED，解碼為 "Gen1" / "Gen2" / "Gen3"。
        /// 無法識別時回傳十六進位原始值。
        /// </summary>
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

        // ================= Delegates =================

        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        /// <summary>滑鼠在 Display 上移動時觸發，參數：(camId, x, y, pixelValue)。pixelValue = -1 表示超出影像範圍。</summary>
        public event Action<int, int, int, int> OnMouseDataChanged;

        // ================= 建構子 =================

        /// <summary>
        /// 建立相機單元，注入所屬的 MIL System ID 及硬體參數。
        /// 資源尚未分配，需呼叫 <see cref="Initialize"/> 才能使用。
        /// </summary>
        /// <param name="systemId">已分配的 MIL System ID</param>
        /// <param name="id">相機識別 ID</param>
        /// <param name="devNum">Digitizer Device 編號（通常為 M_DEV0）</param>
        /// <param name="dcfPath">DCF 設定檔路徑</param>
        /// <param name="panelHandle">顯示用 Panel 的 HWND</param>
        /// <param name="enableImageProcessing">是否啟用 GPU 影像處理</param>
        public MilCameraUnit(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            _ownerSystemId = systemId;
            CameraId = id;
            _devNum = devNum;
            _dcfPath = dcfPath;
            _panelHandle = panelHandle;
            EnableImageProcessing = enableImageProcessing;

            _mouseStatusDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseStatusHandler);
            _processingDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);
            _hUserData = GCHandle.Alloc(this);
        }

        // ================= 初始化 =================

        /// <summary>
        /// 分配所有 MIL 資源：MdigAlloc、MdispAlloc、Grab Buffer、Display Buffer、Processing Buffer。
        /// 同時掛載 Mouse Hook 以提供即時座標與像素值回報。
        /// </summary>
        public void Initialize()
        {
            if (_ownerSystemId == MIL.M_NULL) return;

            MIL.MdigAlloc(_ownerSystemId, _devNum, _dcfPath, MIL.M_DEFAULT, ref MilDigitizer);

            if (MilDigitizer != MIL.M_NULL)
            {
                // 背景啟用 CLProtocol（載入 DLL + 讀取 GenICam XML，不阻塞 Initialize）
                StartCLProtocolAsync();

                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref MilDisplay);

                MIL_INT sizeX = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
                MIL_INT sizeY = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);

                _frameWidth = (int)sizeX;
                _frameHeight = (int)sizeY;
                _hostInputBuffer = new byte[_frameWidth * _frameHeight];
                _hostOutputBuffer = new byte[_frameWidth * _frameHeight];

                CoreCVWrapper.CoreCV_MallocGPU(out _gpuInputBuffer, _frameWidth, _frameHeight);
                CoreCVWrapper.CoreCV_MallocGPU(out _gpuOutputBuffer, _frameWidth, _frameHeight);

                // Grab Buffers：雙緩衝，供 MdigProcess 交替使用
                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                        MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                    MIL.MbufClear(_milGrabBuffers[i], 0);
                }

                // Display Buffer：顯示於 Panel 的緩衝區
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
                MIL.MbufClear(_milDisplayBuffer, 0);

                // Processing Buffer：GPU 處理結果的暫存區（不需要 M_DISP）
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_PROC, ref _milProcBuffer);
                MIL.MbufClear(_milProcBuffer, 0);

                MIL.MdispSelectWindow(MilDisplay, _milDisplayBuffer, _panelHandle);
                MIL.MdispControl(MilDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(MilDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(MilDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
            }
        }

        // ================= MdigProcess Callback =================

        /// <summary>
        /// MdigProcess 每抓到一張 Frame 時呼叫的 Callback。
        /// 依據 EnableImageProcessing 決定是否套用 GPU 二值化，
        /// 再將結果複製到 Display Buffer 以更新畫面。
        /// </summary>
        private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (userPtr == IntPtr.Zero) return MIL.M_NULL;

            GCHandle hObj = GCHandle.FromIntPtr(userPtr);
            var cam = hObj.Target as MilCameraUnit;
            if (cam == null || cam._isReleased) return MIL.M_NULL;

            MIL_ID modifiedBuffer = MIL.M_NULL;
            MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);

            if (modifiedBuffer != MIL.M_NULL && cam._milProcBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                if (!cam.EnableImageProcessing)
                {
                    // 影像處理關閉：直接複製原圖到顯示 Buffer
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                    return MIL.M_NULL;
                }

                bool processedByCoreCv = cam.TryApplyThresholdGpu(modifiedBuffer, cam._milProcBuffer);

                if (processedByCoreCv)
                    MIL.MbufCopy(cam._milProcBuffer, cam._milDisplayBuffer);
                else
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
            }

            return MIL.M_NULL;
        }

        // ================= 抓圖控制 =================

        /// <summary>
        /// 設定使用者的抓圖意圖並立即套用狀態。
        /// </summary>
        /// <param name="enable">true = 啟動抓圖，false = 停止抓圖</param>
        public void SetUserGrabIntent(bool enable)
        {
            _userWantsGrab = enable;
            ApplyGrabState();
        }

        /// <summary>
        /// 根據目前的 UserWantsGrab 與 IsLive 狀態，決定是否啟動或停止 MdigProcess。
        /// 啟動前會先確認相機已連線（CheckPresence）。
        /// </summary>
        public void ApplyGrabState()
        {
            if (MilDigitizer == MIL.M_NULL) return;

            if (_userWantsGrab && !IsLive && CheckPresence())
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
            else if (!_userWantsGrab && IsLive)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }
        }

        /// <summary>
        /// 透過 MdigInquire(M_CAMERA_PRESENT) 檢查相機是否連線，
        /// 同時更新 <see cref="IsConnected"/> 屬性。
        /// </summary>
        /// <returns>true = 相機存在且連線</returns>
        public bool CheckPresence()
        {
            if (MilDigitizer == MIL.M_NULL) { IsConnected = false; return false; }
            MIL_INT presence = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_CAMERA_PRESENT, ref presence);
            IsConnected = (presence == MIL.M_YES);
            return IsConnected;
        }

        // ================= 釋放資源 =================

        /// <summary>
        /// 釋放所有 MIL 資源（停止抓圖、釋放 Buffer、Display、Digitizer）。
        /// 注意：System 由外部（CameraSession）管理，此方法不釋放 System。
        /// </summary>
        public void Free()
        {
            _isReleased = true;

            if (MilDigitizer != MIL.M_NULL)
            {
                // 停止抓圖
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;

                if (MilDisplay != MIL.M_NULL)
                {
                    MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    MIL.MdispSelectWindow(MilDisplay, MIL.M_NULL, IntPtr.Zero);
                }

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    if (_milGrabBuffers[i] != MIL.M_NULL)
                    {
                        MIL.MbufFree(_milGrabBuffers[i]);
                        _milGrabBuffers[i] = MIL.M_NULL;
                    }
                }

                if (_milDisplayBuffer != MIL.M_NULL)
                {
                    MIL.MbufFree(_milDisplayBuffer);
                    _milDisplayBuffer = MIL.M_NULL;
                }
                if (_milProcBuffer != MIL.M_NULL)
                {
                    MIL.MbufFree(_milProcBuffer);
                    _milProcBuffer = MIL.M_NULL;
                }

                if (_gpuInputBuffer != IntPtr.Zero)
                {
                    CoreCVWrapper.CoreCV_FreeGPU(_gpuInputBuffer);
                    _gpuInputBuffer = IntPtr.Zero;
                }
                if (_gpuOutputBuffer != IntPtr.Zero)
                {
                    CoreCVWrapper.CoreCV_FreeGPU(_gpuOutputBuffer);
                    _gpuOutputBuffer = IntPtr.Zero;
                }
                _hostInputBuffer = null;
                _hostOutputBuffer = null;

                if (MilDisplay != MIL.M_NULL)
                {
                    MIL.MdispFree(MilDisplay);
                    MilDisplay = MIL.M_NULL;
                }
                MIL.MdigFree(MilDigitizer);
                MilDigitizer = MIL.M_NULL;
            }

            if (_hUserData.IsAllocated) _hUserData.Free();
        }

        // ================= GPU 影像處理 =================

        /// <summary>
        /// 使用 GPU 對 srcBuffer 進行二值化，結果寫入 dstBuffer。
        /// 流程：MbufGet2d → GPU Upload → CoreCV_Threshold_GPU → GPU Download → MbufPut2d。
        /// 任何步驟失敗均回傳 false，由 ProcessingFunction 自動 fallback 至原圖直接顯示。
        /// </summary>
        /// <param name="srcBuffer">來源 MIL Buffer（Grab 抓到的原圖）</param>
        /// <param name="dstBuffer">目標 MIL Buffer（處理後結果）</param>
        /// <returns>true = 處理成功；false = 任一步驟失敗</returns>
        private bool TryApplyThresholdGpu(MIL_ID srcBuffer, MIL_ID dstBuffer)
        {
            if (srcBuffer == MIL.M_NULL || dstBuffer == MIL.M_NULL) return false;
            if (_frameWidth <= 0 || _frameHeight <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;
            if (_gpuInputBuffer == IntPtr.Zero || _gpuOutputBuffer == IntPtr.Zero) return false;

            try
            {
                MIL.MbufGet2d(srcBuffer, 0, 0, _frameWidth, _frameHeight, _hostInputBuffer);

                GCHandle hIn = GCHandle.Alloc(_hostInputBuffer, GCHandleType.Pinned);
                GCHandle hOut = GCHandle.Alloc(_hostOutputBuffer, GCHandleType.Pinned);

                try
                {
                    int uploadResult = CoreCVWrapper.CoreCV_Upload(hIn.AddrOfPinnedObject(), _gpuInputBuffer, _frameWidth, _frameHeight);
                    if (uploadResult != 0) return false;

                    byte threshold = (byte)Math.Max(0, Math.Min(255, (int)BinarizeThreshold));
                    int thresholdResult = CoreCVWrapper.CoreCV_Threshold_GPU(_gpuInputBuffer, _frameWidth, _frameHeight, threshold, _gpuOutputBuffer);
                    if (thresholdResult != 0) return false;

                    int downloadResult = CoreCVWrapper.CoreCV_Download(_gpuOutputBuffer, hOut.AddrOfPinnedObject(), _frameWidth, _frameHeight);
                    if (downloadResult != 0) return false;
                }
                finally
                {
                    if (hIn.IsAllocated) hIn.Free();
                    if (hOut.IsAllocated) hOut.Free();
                }

                MIL.MbufPut2d(dstBuffer, 0, 0, _frameWidth, _frameHeight, _hostOutputBuffer);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // ================= 事件 Handler =================

        /// <summary>
        /// MdispHookFunction(M_MOUSE_MOVE) 的 Callback。
        /// 取得滑鼠在 Display Buffer 上的座標與像素值，
        /// 並透過 <see cref="OnMouseDataChanged"/> 事件通知外部。
        /// pixelValue 超出影像範圍時傳入 -1。
        /// </summary>
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
    }
}
