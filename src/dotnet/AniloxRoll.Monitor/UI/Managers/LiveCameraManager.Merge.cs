using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
using TanukiCv.Controls; // LiveDisplayView（共用多相機監控顯示元件）
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（LOD GPU resize；P/Invoke 宣告唯一點）
    // partial：即時全域合圖（MIL 合併 display 佈局/刷新/滑鼠）→ 未來 LiveDisplayCoordinator

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // ==================== Global Merge ====================

        /// <summary>啟用即時全域合圖：工頭算佈局 + 分配合併 buffer + 設每台 merge target；本類別綁 display 顯示。</summary>
        public void EnableGlobalMerge(double[] opsUm, double[] startPosMm)
        {
            if (IsGlobalMergeActive || _cameras.Count == 0) return;

            // 「拼」委派工頭：傳入底層 MilCamera 清單（空缺槽以 MaxWidth 作為標準寬度算全域範圍）
            var mils = new List<MilCamera>(_cameras.Count);
            foreach (var cam in _cameras) mils.Add(cam.Mil);

            _merger = new MultiCameraMerger(mils);
            if (!_merger.EnableMerge(opsUm, startPosMm, InspectionEngineConfig.MaxWidth))
            {
                _merger = null;
                return;
            }

            MIL_ID sysId = _cameras[0].OwnerSystemId;
            if (sysId == MIL.M_NULL) { _merger.DisableMerge(); _merger = null; return; }

            // 解除所有相機的 secondary display，改用合併 display
            foreach (var cam in _cameras)
                cam.SetSecondaryDisplay(IntPtr.Zero);

            // 從工頭同步座標系參數（供滑鼠回呼 + overview 計算）
            SyncCoordsFromMerger();

            // panel 已 dispose（關閉/釋放期）→ 不碰 .Handle（會觸發 CreateHandle/ObjectDisposedException）
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) { _merger.DisableMerge(); _merger = null; return; }

            // SmartCanvas 模式：合圖由 LiveDisplayView CPU 拼，不需 MIL 合圖 display。
            // 關鍵：不把 MIL display 綁到 camLiveMain（否則 MIL display 的 M_MOUSE_USE 會攔截滾輪，
            // 疊在上面的 SmartCanvas 收不到 → 無法縮放）。MIL 直繪模式才走下面整套。
            if (!SmartCanvasMode)
            {
                MIL.MdispAlloc(sysId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref _mergedDisplay);
                // MdispAlloc 失敗(多 board/資源不足)→ M_NULL，後續 MdispControl/SelectWindow 對 M_NULL 會 MIL 報錯
                if (_mergedDisplay == MIL.M_NULL)
                {
                    System.Diagnostics.Trace.TraceWarning("[LiveCameraManager.EnableGlobalMerge] MdispAlloc 失敗（合圖 display）");
                    _merger.DisableMerge(); _merger = null; return;
                }

                // 先關自動刷新「再」select window：避免 select 瞬間把 grab hook 尚未貼滿的合併 buffer
                // 顯示出來（半貼狀態 → 橫條殘影閃一下）。改由 33ms timer 手動刷新，確保上螢幕時已較完整。
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                // 改用定時器手動刷新（~30fps），確保每次顯示的是所有相機的最新合成結果
                _mergedDisplayTimer = new Timer { Interval = 33 };
                _mergedDisplayTimer.Tick += MergedDisplayTimer_Tick;
                _mergedDisplayTimer.Start();

                // Hook 滑鼠移動 → 更新 lblPixelInfo
                _mergedMouseDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MergedMouseStatusHandler);
                MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE, _mergedMouseDelegate, IntPtr.Zero);
            }

            IsGlobalMergeActive = true;

            // SmartCanvas 合圖：用工頭佈局(各台 start/ops) CPU 拼（feedScale=1：主程式餵全解析度）
            if (SmartCanvasMode && _smartDisplay != null)
            {
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
                _smartDisplay.MergeAll = true;   // 全域＝合圖全部（含無畫面相機黑占位）
                _smartDisplay.SetMergeMode(true);
            }
        }

        /// <summary>從工頭同步座標系參數到本地鏡像欄位（值來源 = 工頭）。</summary>
        private void SyncCoordsFromMerger()
        {
            if (_merger == null) return;
            _mergedMinStartMm   = _merger.MinStartMm;
            _mergedRefOpsMm     = _merger.RefOpsMm;
            _mergedTotalW       = _merger.TotalW;
            _mergedTotalH       = _merger.TotalH;
            _mergedSlotStartsMm = _merger.SlotStartsMm;
            _mergedSlotEndsMm   = _merger.SlotEndsMm;
        }

        /// <summary>停用即時全域合圖：本類別釋放 display，工頭釋放合併 buffer + 清各相機 merge target。</summary>
        public void DisableGlobalMerge()
        {
            if (!IsGlobalMergeActive) return;

            // 停止定時刷新（顯示職責，留本類別）
            if (_mergedDisplayTimer != null)
            {
                _mergedDisplayTimer.Stop();
                _mergedDisplayTimer.Dispose();
                _mergedDisplayTimer = null;
            }

            // Unhook 滑鼠 + 解除 display 綁定（必須在工頭 MbufFree 合併 buffer 之前）
            if (_mergedDisplay != MIL.M_NULL)
            {
                if (_mergedMouseDelegate != null)
                    MIL.MdispHookFunction(_mergedDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK,
                        _mergedMouseDelegate, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispFree(_mergedDisplay);
                _mergedDisplay = MIL.M_NULL;
            }
            _mergedMouseDelegate = null;

            // 「拆」委派工頭：清各相機 merge target + 釋放合併 buffer
            _merger?.DisableMerge();
            _merger = null;

            _smartDisplay?.SetMergeMode(false); // SmartCanvas 回單相機

            IsGlobalMergeActive = false;
            _mergedSlotStartsMm = null;
            _mergedSlotEndsMm   = null;

            // 恢復使用者明確點選的相機 secondary display（_selectedMainCameraId 可能已被視野中心 timer 改寫）
            SwitchMainDisplay(_userSelectedMainCameraId);
        }

        /// <summary>OPS/Start 變更時，重新計算全域合圖佈局（下一幀生效）。運算委派工頭，顯示重綁留本類別。</summary>
        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm)
        {
            if (!IsGlobalMergeActive || _merger == null || _cameras.Count == 0) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return; // 關閉期不碰 .Handle

            // 「拼」委派工頭（暫停合併 → 重算佈局 → 視需要重分配 buffer → 重設 merge target）
            // 回傳 true 表示合併 buffer 已重新分配，display 需重綁。
            bool reallocated = _merger.RefreshLayout(opsUm, startPosMm, InspectionEngineConfig.MaxWidth);

            // 「秀」：buffer 重分配時，本類別重新 MdispSelectWindow 綁定新 buffer handle
            if (reallocated && _mergedDisplay != MIL.M_NULL)
            {
                MIL.MdispSelectWindow(_mergedDisplay, MIL.M_NULL, IntPtr.Zero);
                MIL.MdispSelectWindow(_mergedDisplay, _merger.MergedBuffer, _mainDisplayPanel.Handle);
                MIL.MdispControl(_mergedDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
            }

            // 從工頭同步座標系參數
            SyncCoordsFromMerger();

            // SmartCanvas 合圖佈局同步（feedScale=1：主程式餵全解析度顯示 bytes）
            if (SmartCanvasMode && _smartDisplay != null)
                _smartDisplay.SetLayout(startPosMm, opsUm, 1, RowPitchMm);
        }

        // ==================== Merged Display Refresh ====================

        private void MergedDisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_mergedDisplay == MIL.M_NULL) return;
            try { MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_NOW); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedDisplayTimer] {ex.GetType().Name}: {ex.Message}"); }
            UpdateSelectedCameraFromViewCenter();
        }

        private void PanMergedDisplayToCameraCenter(int camIdx)
        {
            if (_mergedDisplay == MIL.M_NULL || _mergedSlotStartsMm == null) return;
            int i = camIdx - 1;
            if (i < 0 || i >= _mergedSlotStartsMm.Length) return;
            try
            {
                double centerMm = (_mergedSlotStartsMm[i] + _mergedSlotEndsMm[i]) / 2.0;
                double centerPx = PixelMmMapper.MmToPixel(centerMm, _mergedMinStartMm, _mergedRefOpsMm);
                double zoomX = 0, panY = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                if (zoomX <= 0) return;
                double viewW  = _mainDisplayPanel.Width / zoomX;
                double newPanX = Math.Max(0, Math.Min(_mergedTotalW - viewW, centerPx - viewW / 2.0));
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_mergedDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispPan(_mergedDisplay, newPanX, panY);
                MIL.MdispControl(_mergedDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.PanToCenter] {ex.GetType().Name}: {ex.Message}"); }
        }

        private void UpdateSelectedCameraFromViewCenter()
        {
            if (_mergedSlotStartsMm == null) return;
            if (!TryGetMergedViewRange(out double leftMm, out double rightMm)) return;
            double centerMm = (leftMm + rightMm) / 2.0;
            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < _mergedSlotStartsMm.Length; i++)
            {
                double dist = Math.Abs(centerMm - (_mergedSlotStartsMm[i] + _mergedSlotEndsMm[i]) / 2.0);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }
            int newId = bestIdx + 1;
            if (newId == _selectedMainCameraId) return;
            _selectedMainCameraId = newId;
            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();
        }

        // ==================== Merged Display Mouse ====================

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

            if (x >= 0 && x < _mergedTotalW && y >= 0)
            {
                try
                {
                    byte[] data = new byte[1];
                    MIL.MbufGet2d(mergedBuffer, x, y, 1, 1, data);
                    pixelValue = data[0];
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus] {ex.GetType().Name}: {ex.Message}"); }
            }

            HandleMergedMouseData(x, y, pixelValue);
            return MIL.M_NULL;
        }

        private void HandleMergedMouseData(int x, int y, int pixelValue)
        {
            // MIL display hook 執行緒回 UI；關閉/釋放期 form 已 dispose → 守 guard 防 InvalidOperationException
            if (IsReleasing || _mainForm == null || _mainForm.IsDisposed || !_mainForm.IsHandleCreated) return;
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
                double physicalX = PixelMmMapper.PixelToMm(x, _mergedMinStartMm, _mergedRefOpsMm);

                var s = _inspectionSettings;
                double lineRateHz = (_cameraLineRateHz.Length > 0) ? _cameraLineRateHz[0] : 0;
                double speedMPerMin = s?.AniloxRollSpeedMPerMin ?? 0;
                double rowPitchMm = (speedMPerMin > 0 && lineRateHz > 0)
                    ? (speedMPerMin / 60.0 * 1000.0) / lineRateHz : 0;
                double physicalY = y * rowPitchMm;

                // 合併 display zoom/pan → 視野範圍
                string rangeStr = "";
                string magStr = "-";
                if (TryGetMergedViewRange(out double viewLeftMm, out double viewRightMm))
                {
                    rangeStr = $"X範圍:{viewLeftMm:F1}~{viewRightMm:F1} mm | ";

                    double zoomX = 0;
                    try { MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX); }
                    catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus.ZoomInquire] {ex.GetType().Name}: {ex.Message}"); }
                    if (zoomX > 0 && rowPitchMm > 0)
                    {
                        double panOffY = 0;
                        try { MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_Y, ref panOffY); }
                        catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.MergedMouseStatus.PanInquire] {ex.GetType().Name}: {ex.Message}"); }
                        double viewTopMm = panOffY * rowPitchMm;
                        double viewBotMm = (panOffY + _mainDisplayPanel.Height / zoomX) * rowPitchMm;
                        rangeStr += $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | ";
                    }

                    if (zoomX > 0 && _screenMmPerPx > 0 && _mergedRefOpsMm > 0)
                    {
                        double physicalMag = PixelMmMapper.PhysicalMagnification(zoomX, _screenMmPerPx, _mergedRefOpsMm);
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

            _updatePixelInfoCallback?.Invoke(infoText);
        }

        /// <summary>取得合併 display 的 X 視野範圍（mm），供 overview chart 聯動。</summary>
        public bool TryGetMergedViewRange(out double leftMm, out double rightMm)
        {
            leftMm = rightMm = 0;
            if (!IsGlobalMergeActive || _mergedDisplay == MIL.M_NULL) return false;
            try
            {
                double zoomX = 0, panX = 0;
                MIL.MdispInquire(_mergedDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_mergedDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                if (zoomX <= 0) return false;

                double pixelLeft  = panX;
                double pixelRight = panX + _mainDisplayPanel.Width / zoomX;
                leftMm  = PixelMmMapper.PixelToMm(pixelLeft,  _mergedMinStartMm, _mergedRefOpsMm);
                rightMm = PixelMmMapper.PixelToMm(pixelRight, _mergedMinStartMm, _mergedRefOpsMm);
                return true;
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.TryGetMergedViewRange] {ex.GetType().Name}: {ex.Message}"); return false; }
        }

        /// <summary>取得合併 display 的 Y 視野範圍（pixel），供法向曲線圖聯動。</summary>
        public bool TryGetMergedViewRangeY(out double topPixel, out double botPixel)
        {
            topPixel = botPixel = 0;
            if (!IsGlobalMergeActive || _mergedDisplay == MIL.M_NULL) return false;
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
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.TryGetMergedViewRangeY] {ex.GetType().Name}: {ex.Message}"); return false; }
        }
    }
}
