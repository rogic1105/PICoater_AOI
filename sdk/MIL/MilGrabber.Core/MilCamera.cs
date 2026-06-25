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
    public partial class MilCamera : IDisposable
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

        // ===== 階段二實驗（feat/grabheight-max-buffer）=====
        /// <summary>grab/display buffer 一次配「max 高度」，改高度只改 `M_SOURCE_SIZE_Y`、**不 free/realloc** →
        /// 測「能否避開 realloc→re-arm 累積（=改高度 stall 的根因路徑）」。預設 **false**＝現行安全 realloc 行為。
        /// ⚠ 未上機驗證：Matrox doc「line-scan 幀填滿整個 destination buffer 才完成」暗示 max-buffer 可能讓
        /// **幀變成 max 高度（壞）**。上機翻 true 測一次：幀高度正確＝max-buffer 可行；幀變 max 高度＝doc 疑慮
        /// 成立，改走 auto-allocate（MdigProcess bufarray=M_NULL）。詳見 docs/dev/grabheight-max-buffer-stage2.md。</summary>
        /// <summary>max-buffer 模式。**預設 false（已驗證不採用）**：grab 中拉高度真上限 ~12062（per-camera；板載 4 path
        /// 各自獨立到 ~12062，不是同板總和），且 7 台都配 max 高度 buffer 會撐爆 host 非分頁池。真正的解＝上層把高度一律
        /// cap 到 MaxGrabHeightPx（=12000，AcquisitionDefaults）。保留 flag 與 scaffold 供紀錄，預設走 buffer==source realloc。</summary>
        public static bool UseMaxHeightBuffers = false;
        /// <summary>max-buffer 配置高度（px，僅 UseMaxHeightBuffers=true 時用；目前預設 false 故未使用）。</summary>
        public static int MaxBufferHeightPx = 12000;
        private int _grabBufAllocH;                   // MIL grab/display buffer 實際配置的高度（max 或當前）

        // ===== grab buffer 高度上限（防撞板載 stall）=====
        /// <summary>本台實際生效的高度上限（px）。由上層 LiveCameraManager 算好設入（CameraGrabHeight 設前已 clamp）；
        /// 純供 UI clamp 滑桿/顯示。autoMax 計算刻意放上層，**不在 MIL Initialize 內查板載**（會 stall）。</summary>
        public int EffectiveMaxGrabHeightPx { get; set; }

        public MIL_ID OwnerSystemId => _ownerSystemId;
        public MIL_ID MilDigitizer => _milDigitizer;
        public MIL_ID MilDisplay => _milDisplay;
        public MIL_ID MilSecondaryDisplay => _milSecondaryDisplay;
        public MIL_ID MilDisplayBuffer => _milDisplayBuffer;
        /// <summary>hook 中暫存的最近一幀原圖 buffer（供上層在 FrameReady 外延遲取用）。</summary>
        public MIL_ID LastGrabBuffer => _milLastGrabBuffer;

        /// <summary>顯示時上下翻轉（線掃相機由下往上拍 → 顯示需反轉）。只影響「預設 grab hook 顯示」路徑
        /// （未訂閱 FrameReady 時）；訂閱 FrameReady 的上層自行決定顯示方向。</summary>
        public bool FlipVertical { get; set; }

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

        // ==================== Global Merge Target（即時合圖：把本台 display buffer 裁切後貼到合併 buffer） ====================
        // 由「多相機工頭」（MultiCameraMerger）或上層透過 SetMergeTarget/ClearMergeTarget 設定。
        // 設定後，每幀 grab hook 在 displayBuffer 更新完成後，會把裁切範圍貼到合併 buffer 的 xOffset 位置。
        private MIL_ID _mergedTargetBuffer = MIL.M_NULL;
        private int _mergedTargetOffsetX = 0;
        private int _mergedSrcClipLeft = 0;
        private int _mergedSrcClipWidth = 0;

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
        /// <param name="devNum">板內固定 device 位置（0-based 絕對值，對應 M_DEV0/1/2/3…）。
        /// 相機實體接線固定 → 直接寫絕對位置，**不加偏移**（不做 M_DEV0 + n），少槽卡（1/2 槽）只列實際在用的 channel。
        /// 這裡是「json device 號 → MdigAlloc 引數」的唯一轉換點：本機型 M_DEV0=0 故為 identity；
        /// 未來若遇 M_DEV0≠0 的擷取卡，只改這一行（caller 一律傳 json 固定值，不自行運算）。</param>
        public MilCamera(MIL_ID systemId, int id, int devNum, string dcfPath, IntPtr panelHandle)
        {
            _ownerSystemId = systemId;
            CameraId = id;
            _devNum = (MIL_INT)devNum;
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

            // 先套用 Grab Height，再查詢實際尺寸以分配正確大小的 Buffer。
            // 高度上限（autoMax / 防撞板載 clamp）由上層 LiveCameraManager 算好、CameraGrabHeight 設進來前已 clamp，
            // 此處不查板載/不算 autoMax —— 在 MIL 初始化序列中插入額外 MdigInquire 會讓 cam stall（實測）。
            if (CameraGrabHeight > 0)
                MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)CameraGrabHeight);

            MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milDisplay);
            MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _milSecondaryDisplay);
            if (_milDisplay == MIL.M_NULL)
                System.Diagnostics.Trace.TraceWarning($"[MilCamera CAM{CameraId}] MdispAlloc(primary) 失敗 → 主畫面 MIL 直繪不可用（SmartCanvas 路徑仍可）");

            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            FrameWidth = (int)sizeX;
            FrameHeight = (int)sizeY;

            // 階段二 flag：buffer 一次配 max 高度（之後改高度只改 M_SOURCE_SIZE_Y、不 realloc）。
            int bufH = UseMaxHeightBuffers ? System.Math.Max((int)sizeY, MaxBufferHeightPx) : (int)sizeY;
            _grabBufAllocH = bufH;
            // 診斷：確認 host grab buffer 真的一次配 max（bufH==MaxBufferHeightPx）而非 json 高度。
            // 註：板載 M_MEMORY 占用顯示的是 digitizer 的 source-size FIFO（隨 M_SOURCE_SIZE_Y 縮放），
            // **不是這裡的 host grab buffer**，故占用會隨高度變＝正常，不代表 max-buffer 沒生效。
            System.Diagnostics.Trace.WriteLine(
                $"[CAM{CameraId}] Initialize buffer：UseMaxHeightBuffers={UseMaxHeightBuffers} MaxBufferHeightPx={MaxBufferHeightPx} sizeY(json)={(int)sizeY} → bufH(host alloc)={bufH}");

            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, bufH, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                if (_milGrabBuffers[i] == MIL.M_NULL)
                    System.Diagnostics.Trace.TraceWarning($"[MilCamera CAM{CameraId}] MbufAlloc2d(grab[{i}]) 失敗 → 取像將失敗");
                else MIL.MbufClear(_milGrabBuffers[i], 0);
            }

            MIL.MbufAlloc2d(_ownerSystemId, sizeX, bufH, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            if (_milDisplayBuffer == MIL.M_NULL)
                System.Diagnostics.Trace.TraceWarning($"[MilCamera CAM{CameraId}] MbufAlloc2d(display buffer) 失敗 → MIL 直繪不可用");
            else MIL.MbufClear(_milDisplayBuffer, 0);

            // display/buffer 任一 M_NULL → 跳過 MdispSelectWindow（對 M_NULL 操作會 MIL 報錯）。
            // grab 仍進行，SmartCanvas 顯示路徑不靠 MIL display；只 MIL 直繪模式會黑畫面（已 log）。
            if (_milDisplay != MIL.M_NULL && _milDisplayBuffer != MIL.M_NULL)
            {
                MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
                MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_milDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                MIL.MdispHookFunction(_milDisplay, MIL.M_MOUSE_LEFT_BUTTON_DOWN, _mouseClickDelegate, (IntPtr)CameraId);
            }

            // 初始曝光：此時 CLProtocol 尚未啟用，走 legacy MdigControl 路徑
            if (_appliedExposureUs > 0)
                SetExposureUs(_appliedExposureUs);

            // 多相機相位量測：啟用 frame-start 硬體時戳 latch（診斷用）。
            EnableFrameStartTimestampLatch();
        }

        // ==================== Grab Control ====================

        public void SetUserGrabIntent(bool enable)
        {
            _userWantsGrab = enable;
            ApplyGrabState();
        }

        /// <summary>依 _userWantsGrab 與 IsLive 決定啟動/停止 MdigProcess。
        /// CLProtocol 不在此啟動 —— 改由分配相機後 <see cref="BeginCLProtocolInit"/> 預先在背景啟用，
        /// 避免第一次 grab 進行中才 enable + 重套線掃造成掉幀（cam1 最明顯）。</summary>
        public void ApplyGrabState()
        {
            if (_isReleased || _milDigitizer == MIL.M_NULL) return;

            if (_userWantsGrab && !IsLive && CheckPresence())
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
            else if (!_userWantsGrab && IsLive)
            {
                // 乾淨 drain（鏡像 SetGrabHeight）：M_STOP+M_WAIT 等佇列跑完 + M_GRAB_ABORT 立即中止 in-flight/佇列。
                // 裸 M_STOP 只取消佇列、不保證 in-flight 清乾淨 → re-grab 在「未乾淨」狀態 re-arm，兩台 M_START
                // 跨 frame 邊界時序不一 → 某台第一個完整幀晚一格（free-run 無 trigger 的量化效應）。
                // 乾淨 drain 後 re-arm 接近「第一次 grab」的乾淨狀態，兩台較易等到同一個完整 frame。
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP + MIL.M_WAIT, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                try { MIL.MdigControl(_milDigitizer, MIL.M_GRAB_ABORT, MIL.M_DEFAULT); } catch { }
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
            cam.CaptureFrameStartLatch(eventId);   // 多相機相位：讀本幀 frame-start 硬體時戳 + 記 phase log（診斷）

            if (modifiedBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                var handler = cam.FrameReady;
                if (handler != null)
                    handler(cam, modifiedBuffer);                       // 上層檢測 + 自行決定顯示
                else if (cam.FlipVertical)
                    MIL.MimFlip(modifiedBuffer, cam._milDisplayBuffer, MIL.M_FLIP_VERTICAL, MIL.M_DEFAULT); // 上下翻轉顯示
                else
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer); // 預設顯示原圖

                // 全域合圖：display buffer 更新完成後，把裁切範圍貼到合併 buffer 的對應位置。
                // 以 displayBuffer 為來源 → 合併圖反映「目前顯示的內容」（原圖或上層處理後）。
                cam.CopyDisplayToMergeTarget();
            }

            return MIL.M_NULL;
        }

        /// <summary>
        /// 若已設定 merge target，把本台 display buffer 的裁切範圍 MbufCopyClip 到合併 buffer 的 xOffset 位置。
        /// 在 grab hook 內（display buffer 更新後）呼叫。執行緒安全靠欄位讀取順序（buffer 最先清/最後設）。
        /// </summary>
        private void CopyDisplayToMergeTarget()
        {
            MIL_ID mergedBuf = _mergedTargetBuffer;
            MIL_ID dispBuf   = _milDisplayBuffer;
            if (mergedBuf == MIL.M_NULL || dispBuf == MIL.M_NULL) return;

            int clipLeft  = _mergedSrcClipLeft;
            int clipWidth = _mergedSrcClipWidth;
            int dstX      = _mergedTargetOffsetX + clipLeft;
            int fw = FrameWidth;
            int fh = FrameHeight;
            if (clipWidth <= 0 || clipLeft < 0 || clipLeft + clipWidth > fw) return;

            MIL_ID childBuf = MIL.M_NULL;
            MIL.MbufChild2d(dispBuf, clipLeft, 0, clipWidth, fh, ref childBuf);
            if (childBuf != MIL.M_NULL)
            {
                MIL.MbufCopyClip(childBuf, mergedBuf, dstX, 0);
                MIL.MbufFree(childBuf);
            }
        }

        /// <summary>
        /// 設定全域合圖目標：本台每幀把 display buffer 的 [srcLeft, srcLeft+srcWidth) 裁切範圍，
        /// 貼到合併 buffer 的 (xOffset + srcLeft, 0) 位置。
        /// buffer 最後設定，確保 grab hook 讀到完整狀態（thread safety）。
        /// </summary>
        public void SetMergeTarget(MIL_ID mergedBuffer, int xOffset, int srcLeft, int srcWidth)
        {
            _mergedTargetOffsetX = xOffset;
            _mergedSrcClipLeft   = srcLeft;
            _mergedSrcClipWidth  = srcWidth;
            _mergedTargetBuffer  = mergedBuffer; // buffer 最後設定（thread safety）
        }

        /// <summary>清除全域合圖目標：buffer 最先清除，立即停止 grab hook 內的合併複製（thread safety）。</summary>
        public void ClearMergeTarget()
        {
            _mergedTargetBuffer = MIL.M_NULL; // buffer 最先清除（thread safety）
            _mergedSrcClipLeft  = 0;
            _mergedSrcClipWidth = 0;
        }

        // ==================== Buffer Helpers / Display / Mouse Hooks → MilCamera.Display.cs ====================

        // ==================== CLProtocol（BeginCLProtocolInit / TryEnableCLProtocol）→ MilCamera.CLProtocol.cs ====================

        // ==================== Exposure / Line Rate / Grab Height → MilCamera.Params.cs（相機參數分區） ====================

        // ==================== Telemetry → MilCamera.Telemetry.cs（唯讀遙測 getter 分區） ====================

        // ==================== Primary / Secondary Display / Mouse Hooks → MilCamera.Display.cs ====================

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

}
