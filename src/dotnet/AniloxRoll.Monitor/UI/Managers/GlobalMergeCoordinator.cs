using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core;                     // PixelMmMapper（像素↔mm 唯一來源）
using AniloxRoll.Monitor.Core.Services;  // InspectionEngineConfig.MaxWidth

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 即時全域合圖的「秀」協調者 —— 從 LiveCameraManager 提取（2026-06-26 重構）。
    ///
    /// 合圖分工：
    ///  - 「拼」＝ <see cref="MultiCameraMerger"/> 工頭（純 MIL，sdk/MIL）：算佈局 + 合併 buffer + 每台 merge target。
    ///  - 「秀」＝ 本類別：擁有 MIL 合圖 display 的整個生命週期（alloc / select window / 33ms 防閃刷新 timer /
    ///    滑鼠 hook / free）+ 視野範圍查詢 + merged 分支的 zoom / pan / 1x / reset / pan-to-center。
    ///
    /// **狀態擁有權**：本類別獨佔 _merger（工頭）+ _mergedDisplay + 座標鏡像（_minStartMm…）+ timer + 滑鼠 hook。
    /// LiveCameraManager 不再持有任何 _merged* 欄位；ImageCanvas / Waterfall 路徑要佈局時讀 <see cref="Merger"/>。
    ///
    /// **與 LCM 的反向依賴**：穩定依賴（mainForm / panel / updatePixelInfo）走 ctor；會變動的值
    /// （screenMmPerPx / speed / lineRate / IsReleasing）走 Func 委派、避免反向參考整個 LCM；
    /// 視野中心選中相機變更走 <c>onViewCenterCamChanged</c> callback（UI selection event，由 LCM 更新選中狀態 + 重繪縮圖）。
    /// ImageCanvas / Waterfall 編排（mechanism 無關）留 LCM，本類別不知 ImageCanvas 存在。
    /// </summary>
    internal sealed class GlobalMergeCoordinator
    {
        // ── 穩定依賴（ctor 注入）──
        private readonly Form _mainForm;
        private readonly Panel _mainDisplayPanel;
        private readonly Action<string> _updatePixelInfo;
        private readonly Func<bool> _isReleasing;
        private readonly Func<double> _getScreenMmPerPx;
        private readonly Func<double> _getSpeedMPerMin;
        private readonly Func<double> _getLineRateHz0;
        private readonly Action<int> _onViewCenterCamChanged;

        // ── 擁有的合圖狀態 ──
        private MultiCameraMerger _merger;
        private MIL_ID _mergedDisplay = MIL.M_NULL;
        private double _minStartMm;   // 合併座標系原點（mm）
        private double _refOpsMm;     // 合併像素尺寸（mm/px）
        private int    _totalW;       // 合併 buffer 寬度（px）
        private int    _totalH;       // 合併 buffer 高度（px）
        private double[] _slotStartsMm;  // 各槽位起始 mm（含空缺）
        private double[] _slotEndsMm;    // 各槽位結束 mm（含空缺）
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseDelegate;
        private Timer _displayTimer;  // 定時刷新合圖 display（取代 MIL 自動刷新，避免多相機非同步閃爍）

        /// <summary>合圖是否啟用中。</summary>
        public bool IsActive { get; private set; }

        /// <summary>工頭（合圖佈局來源）；ImageCanvas / Waterfall 路徑讀 SlotStartsMm / RefOpsMm。未啟用為 null。</summary>
        public MultiCameraMerger Merger => _merger;

        /// <summary>是否已綁 MIL 合圖 display（ImageCanvas 模式為 false：合圖由 ImageDisplayView CPU 拼，不綁 MIL display）。</summary>
        public bool HasMilDisplay => _mergedDisplay != MIL.M_NULL;

        /// <summary>合併像素尺寸（mm/px），供 Waterfall 佈局同步。</summary>
        public double RefOpsMm => _refOpsMm;

        public GlobalMergeCoordinator(
            Form mainForm,
            Panel mainDisplayPanel,
            Action<string> updatePixelInfo,
            Func<bool> isReleasing,
            Func<double> getScreenMmPerPx,
            Func<double> getSpeedMPerMin,
            Func<double> getLineRateHz0,
            Action<int> onViewCenterCamChanged)
        {
            _mainForm = mainForm;
            _mainDisplayPanel = mainDisplayPanel;
            _updatePixelInfo = updatePixelInfo;
            _isReleasing = isReleasing;
            _getScreenMmPerPx = getScreenMmPerPx;
            _getSpeedMPerMin = getSpeedMPerMin;
            _getLineRateHz0 = getLineRateHz0;
            _onViewCenterCamChanged = onViewCenterCamChanged;
        }

        // ==================== Enable / Disable / Refresh ====================

        /// <summary>啟用全域合圖：建工頭算佈局 + 分配合併 buffer + 每台 merge target；showMilDisplay 時綁 MIL display。
        /// 失敗（工頭啟用失敗 / sysId 無效 / panel 已 dispose / display alloc 失敗）回 false 並自行清理。</summary>
        public bool Enable(IReadOnlyList<MilCamera> mils, MIL_ID sysId, double[] opsUm, double[] startPosMm, bool showMilDisplay)
        {
            if (IsActive) return false;
            if (mils == null || mils.Count == 0) return false;

            // 「拼」委派工頭：傳入底層 MilCamera 清單（空缺槽以 MaxWidth 作為標準寬度算全域範圍）
            _merger = new MultiCameraMerger(mils);
            if (!_merger.EnableMerge(opsUm, startPosMm, InspectionEngineConfig.MaxWidth))
            {
                _merger = null;
                return false;
            }
            if (sysId == MIL.M_NULL) { _merger.DisableMerge(); _merger = null; return false; }

            // 從工頭同步座標系參數（供滑鼠回呼 + overview 計算）
            SyncCoords();

            // panel 已 dispose（關閉/釋放期）→ 不碰 .Handle（會觸發 CreateHandle/ObjectDisposedException）
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) { _merger.DisableMerge(); _merger = null; return false; }

            // ImageCanvas 模式：合圖由 ImageDisplayView CPU 拼，不需 MIL 合圖 display（showMilDisplay=false）。
            // 關鍵：不把 MIL display 綁到 camLiveMain（否則 MIL display 的 M_MOUSE_USE 會攔截滾輪，
            // 疊在上面的 ImageCanvas 收不到 → 無法縮放）。MIL 直繪模式才走下面整套。
            if (showMilDisplay)
            {
                MIL.MdispAlloc(sysId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _mergedDisplay);
                // MdispAlloc 失敗(多 board/資源不足)→ M_NULL，後續 MdispControl/SelectWindow 對 M_NULL 會 MIL 報錯
                if (_mergedDisplay == MIL.M_NULL)
                {
                    Trace.TraceWarning("[GlobalMergeCoordinator.Enable] MdispAlloc 失敗（合圖 display）");
                    _merger.DisableMerge(); _merger = null; return false;
                }

                // 先關自動刷新「再」select window：避免 select 瞬間把 grab hook 尚未貼滿的合併 buffer
                // 顯示出來（半貼狀態 → 橫條殘影閃一下）。改由 33ms timer 手動刷新，確保上螢幕時已較完整。
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                // 改用定時器手動刷新（~30fps），確保每次顯示的是所有相機的最新合成結果
                _displayTimer = new Timer { Interval = 33 };
                _displayTimer.Tick += MergedDisplayTimer_Tick;
                _displayTimer.Start();

                // Hook 滑鼠移動 → 更新 lblPixelInfo
                _mouseDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MergedMouseStatusHandler);
                MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE, _mouseDelegate, IntPtr.Zero);
            }

            IsActive = true;
            return true;
        }

        /// <summary>停用全域合圖：釋放 MIL display（unhook + select-null + free，必須在工頭 MbufFree 合併 buffer 之前）+
        /// 工頭釋放合併 buffer + 清各相機 merge target。先翻 IsActive 旗標，讓背景 mouse hook / 即將停的 timer 早退。</summary>
        public void Disable()
        {
            if (!IsActive) return;
            IsActive = false;   // 先翻旗標（背景 mouse hook 看到 → MergedBuffer 已準備釋放、HandleMergedMouseData 也走 _isReleasing 守門）

            // 停止定時刷新
            if (_displayTimer != null)
            {
                _displayTimer.Stop();
                _displayTimer.Dispose();
                _displayTimer = null;
            }

            // Unhook 滑鼠 + 解除 display 綁定（必須在工頭 MbufFree 合併 buffer 之前）
            if (_mergedDisplay != MIL.M_NULL)
            {
                if (_mouseDelegate != null)
                    MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK,
                        _mouseDelegate, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispFree(_mergedDisplay);
                _mergedDisplay = MIL.M_NULL;
            }
            _mouseDelegate = null;

            // 「拆」委派工頭：清各相機 merge target + 釋放合併 buffer
            _merger?.DisableMerge();
            _merger = null;

            _slotStartsMm = null;
            _slotEndsMm   = null;
        }

        /// <summary>OPS/Start 變更時重算佈局（下一幀生效）。運算委派工頭；buffer 重分配時重綁 MIL display。
        /// 回傳工頭是否重新分配了合併 buffer。</summary>
        public bool RefreshLayout(double[] opsUm, double[] startPosMm)
        {
            if (!IsActive || _merger == null) return false;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return false; // 關閉期不碰 .Handle

            // 「拼」委派工頭（暫停合併 → 重算佈局 → 視需要重分配 buffer → 重設 merge target）
            // 回傳 true 表示合併 buffer 已重新分配，display 需重綁。
            // 註：本方法與 33ms timer Tick 同在 UI 執行緒，不會並發；故 reallocated 時的 select-null→reselect
            //     已足以避免 display 綁到已 free 的舊 buffer（不需 always-detach，保住使用者 zoom/pan）。
            bool reallocated = _merger.RefreshLayout(opsUm, startPosMm, InspectionEngineConfig.MaxWidth);

            if (reallocated && _mergedDisplay != MIL.M_NULL)
            {
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
            }

            SyncCoords();
            return reallocated;
        }

        /// <summary>從工頭同步座標系參數到本地鏡像欄位（值來源 = 工頭）。</summary>
        private void SyncCoords()
        {
            if (_merger == null) return;
            _minStartMm   = _merger.MinStartMm;
            _refOpsMm     = _merger.RefOpsMm;
            _totalW       = _merger.TotalW;
            _totalH       = _merger.TotalH;
            _slotStartsMm = _merger.SlotStartsMm;
            _slotEndsMm   = _merger.SlotEndsMm;
        }

        // ==================== Display Refresh / View Center ====================

        private void MergedDisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_mergedDisplay == MIL.M_NULL) return;
            try { MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_NOW); }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.DisplayTimer] {ex.GetType().Name}: {ex.Message}"); }
            UpdateSelectedCameraFromViewCenter();
        }

        private void UpdateSelectedCameraFromViewCenter()
        {
            if (_slotStartsMm == null) return;
            if (!TryGetViewRange(out double leftMm, out double rightMm)) return;
            double centerMm = (leftMm + rightMm) / 2.0;
            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < _slotStartsMm.Length; i++)
            {
                double dist = Math.Abs(centerMm - (_slotStartsMm[i] + _slotEndsMm[i]) / 2.0);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }
            // 視野中心最近的相機 → 回拋 LCM（UI selection event；去重 + 重繪縮圖由 LCM 負責）
            _onViewCenterCamChanged?.Invoke(bestIdx + 1);
        }

        /// <summary>把合圖 display pan 到指定相機（1-based）槽位中心。</summary>
        public void PanToCameraCenter(int camIdx)
        {
            if (_mergedDisplay == MIL.M_NULL || _slotStartsMm == null) return;
            int i = camIdx - 1;
            if (i < 0 || i >= _slotStartsMm.Length) return;
            try
            {
                double centerMm = (_slotStartsMm[i] + _slotEndsMm[i]) / 2.0;
                double centerPx = PixelMmMapper.MmToPixel(centerMm, _minStartMm, _refOpsMm);
                double zoomX = 0, panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                if (zoomX <= 0) return;
                double viewW  = _mainDisplayPanel.Width / zoomX;
                double newPanX = Math.Max(0, Math.Min(_totalW - viewW, centerPx - viewW / 2.0));
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispPan(_mergedDisplay, newPanX, panY);
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.PanToCenter] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ==================== View Reset / Zoom / 1x（merged 分支）====================

        /// <summary>重置合圖 display 縮放/平移為 fit-to-window。</summary>
        public void ResetView()
        {
            if (!IsActive || _mergedDisplay == MIL.M_NULL) return;
            try
            {
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.ResetView] {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>設合圖 display zoom 使實體倍率 = 1x（螢幕 1mm = 實際 1mm），以面板中心為基準。</summary>
        public void SetPhysical1x()
        {
            if (_mergedDisplay == MIL.M_NULL || _refOpsMm <= 0) return;
            double screenMmPerPx = _getScreenMmPerPx();
            if (screenMmPerPx <= 0) return;

            double zoom1x = PixelMmMapper.OneToOneZoom(_refOpsMm, screenMmPerPx);
            double cx = _mainDisplayPanel.Width / 2.0;
            double cy = _mainDisplayPanel.Height / 2.0;

            try
            {
                double curZoom = 0, curPanX = 0, curPanY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref curZoom);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref curPanX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref curPanY);
                if (curZoom <= 0) curZoom = 1.0;

                double imgCx = curPanX + cx / curZoom;
                double imgCy = curPanY + cy / curZoom;
                double newPanX = imgCx - cx / zoom1x;
                double newPanY = imgCy - cy / zoom1x;

                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispZoom(_mergedDisplay, zoom1x, zoom1x);
                MIL.MdispPan(_mergedDisplay, newPanX, newPanY);
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.SetPhysical1x] {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>滾輪縮放合圖 display（1.1x 步長），以面板中心為縮放基準點。</summary>
        public void ApplyZoom(int wheelDelta)
        {
            if (_mergedDisplay == MIL.M_NULL) return;
            double zoomX, panX, panY;
            try
            {
                zoomX = panX = panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.ApplyZoom.Inquire] {ex.GetType().Name}: {ex.Message}"); return; }
            if (zoomX <= 0) zoomX = 1.0;

            double factor = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
            double newZoom = zoomX * factor;
            if (newZoom < 0.05) newZoom = 0.05;
            if (newZoom > 32.0) newZoom = 32.0;

            double cx = _mainDisplayPanel.Width / 2.0;
            double cy = _mainDisplayPanel.Height / 2.0;
            double imgX = panX + cx / zoomX;
            double imgY = panY + cy / zoomX;
            double newPanX = imgX - cx / newZoom;
            double newPanY = imgY - cy / newZoom;

            try
            {
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispZoom(_mergedDisplay, newZoom, newZoom);
                MIL.MdispPan(_mergedDisplay, newPanX, newPanY);
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.ApplyZoom.Apply] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ==================== View Range（overview / 曲線聯動）====================

        /// <summary>合圖 display 的 X 視野範圍（mm），供 overview chart 聯動。</summary>
        public bool TryGetViewRange(out double leftMm, out double rightMm)
        {
            leftMm = rightMm = 0;
            if (!IsActive || _mergedDisplay == MIL.M_NULL) return false;
            try
            {
                double zoomX = 0, panX = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                if (zoomX <= 0) return false;

                double pixelLeft  = panX;
                double pixelRight = panX + _mainDisplayPanel.Width / zoomX;
                leftMm  = PixelMmMapper.PixelToMm(pixelLeft,  _minStartMm, _refOpsMm);
                rightMm = PixelMmMapper.PixelToMm(pixelRight, _minStartMm, _refOpsMm);
                return true;
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.TryGetViewRange] {ex.GetType().Name}: {ex.Message}"); return false; }
        }

        /// <summary>合圖 display 的 Y 視野範圍（pixel），供法向曲線圖聯動。</summary>
        public bool TryGetViewRangeY(out double topPixel, out double botPixel)
        {
            topPixel = botPixel = 0;
            if (!IsActive || _mergedDisplay == MIL.M_NULL) return false;
            try
            {
                double zoomY = 0, panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                if (zoomY <= 0) return false;

                topPixel = panY;
                botPixel = panY + _mainDisplayPanel.Height / zoomY;
                return true;
            }
            catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.TryGetViewRangeY] {ex.GetType().Name}: {ex.Message}"); return false; }
        }

        // ==================== Mouse Hook ====================

        private MIL_INT MergedMouseStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            MIL_ID mergedBuffer = _merger?.MergedBuffer ?? MIL.M_NULL;
            if (mergedBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

            int x = (int)posX;
            int y = (int)posY;
            int pixelValue = -1;

            if (x >= 0 && x < _totalW && y >= 0)
            {
                try
                {
                    byte[] data = new byte[1];
                    MIL.MbufGet2d(mergedBuffer, x, y, 1, 1, data);
                    pixelValue = data[0];
                }
                catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.MergedMouseStatus] {ex.GetType().Name}: {ex.Message}"); }
            }

            HandleMergedMouseData(x, y, pixelValue);
            return MIL.M_NULL;
        }

        private void HandleMergedMouseData(int x, int y, int pixelValue)
        {
            // MIL display hook 執行緒回 UI；關閉/釋放期 form 已 dispose → 守 guard 防 InvalidOperationException
            if (_isReleasing() || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => HandleMergedMouseData(x, y, pixelValue))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            string infoText;
            if (pixelValue == -1)
            {
                infoText = "即時影像 [全域合圖] | 游標超出影像範圍";
            }
            else
            {
                double physicalX = PixelMmMapper.PixelToMm(x, _minStartMm, _refOpsMm);

                double lineRateHz = _getLineRateHz0();
                double speedMPerMin = _getSpeedMPerMin();
                double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                    ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                double physicalY = y * rowPitchMm;

                // 合併 display zoom/pan → 視野範圍
                string rangeStr = "";
                string magStr = "-";
                if (TryGetViewRange(out double viewLeftMm, out double viewRightMm))
                {
                    rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                    double zoomX = 0;
                    try { MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX); }
                    catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.MergedMouseStatus.ZoomInquire] {ex.GetType().Name}: {ex.Message}"); }
                    if (zoomX > 0 && rowPitchMm > 0)
                    {
                        double panOffY = 0;
                        try { MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panOffY); }
                        catch (Exception ex) { Trace.TraceWarning($"[GlobalMergeCoordinator.MergedMouseStatus.PanInquire] {ex.GetType().Name}: {ex.Message}"); }
                        double viewTopMm = panOffY * rowPitchMm;
                        double viewBotMm = (panOffY + _mainDisplayPanel.Height / zoomX) * rowPitchMm;
                        rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                    }

                    double screenMmPerPx = _getScreenMmPerPx();
                    if (zoomX > 0 && screenMmPerPx > 0 && _refOpsMm > 0)
                    {
                        double physicalMag = PixelMmMapper.PhysicalMagnification(zoomX, screenMmPerPx, _refOpsMm);
                        magStr = $"{physicalMag:F2}x";
                    }
                }

                infoText = $"即時影像 [全域合圖] | " +
                           $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                           rangeStr +
                           $"座標: ({x}, {y}) | " +
                           $"亮度: {pixelValue} | " +
                           $"實體倍率:{magStr}";
            }

            _updatePixelInfo?.Invoke(infoText);
        }
    }
}
