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
    // partial：顯示/畫布橋接（Display Switching + SmartCanvas + UI Helpers + 實體倍率 + 滾輪 zoom）→ 未來 LiveDisplayCoordinator

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // ==================== Display Switching ====================

        /// <summary>
        /// <summary>
        /// 重新套用目前選定相機的主顯示，用於 SetGrabHeight 後重新綁定畫面。
        /// </summary>
        public void RefreshMainDisplay()
        {
            SwitchMainDisplay(_selectedMainCameraId);
        }

        /// <summary>重置主顯示器（MIL secondary display）的縮放/平移為 fit-to-window。</summary>
        public void ResetMainDisplayView()
        {
            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                _globalMerge.ResetView();
                return;
            }
            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            cam?.ResetSecondaryDisplayView();
        }

        private void OnLivePanelPaint(object sender, PaintEventArgs e, int cameraIndex)
        {
            if (!(sender is Panel panel)) return;
            // SmartCanvas 模式選取框唯一來源 = sdk ThumbView（橘）；父 panel 只畫中性灰框（避免雙框）。
            // MIL 直繪模式（無 ThumbView）仍由此畫橘框。
            bool isSelected = cameraIndex == _selectedMainCameraId && !SmartCanvasMode;
            Color borderColor = isSelected ? Color.Orange : Color.FromArgb(60, 60, 60);
            int   borderWidth = isSelected ? 3 : 1;
            ControlPaint.DrawBorder(e.Graphics, panel.ClientRectangle,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid,
                borderColor, borderWidth, ButtonBorderStyle.Solid);
        }

        // ── 主畫面顯示模式（SmartCanvas / MilDirect / Waterfall 三選一互斥）──
        /// <summary>he_MainDisplay==Waterfall。</summary>
        private bool WaterfallMode => _inspectionSettings != null
            && _inspectionSettings.he_MainDisplay == AniloxRoll.Monitor.Core.Data.MainDisplayMode.Waterfall;

        /// <summary>依 he_MainDisplay 套用主畫面顯示模式（三選一互斥）。相機配置 / 開始抓取 / 設定即時變更都呼此
        /// → 切設定即生效、不必重啟程式。每條路徑各自冪等，互斥靠「先拆別的、再建自己」。</summary>
        public void ApplyMainDisplayMode()
        {
            if (WaterfallMode)
            {
                TeardownSmartDisplay();        // 露出底層 → 由 Waterfall 接管
                EnableWaterfallDisplay();
            }
            else
            {
                DisableWaterfallDisplay();
                if (SmartCanvasMode) EnsureSmartDisplay();
                else TeardownSmartDisplay();   // MilDirect：露出底層 MIL 直繪
            }
        }

        // ── Waterfall（瀑布圖）顯示路徑 ──
        private AniloxRoll.Monitor.UI.Widgets.WaterfallView _waterfallView;

        /// <summary>建 WaterfallView 接 camLiveMain，訂閱各相機每幀 bytes → 合成全幅 band 往下接。冪等。</summary>
        private void EnableWaterfallDisplay()
        {
            if (_waterfallView != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;
            int wfH = _inspectionSettings?.ImageView?.WaterfallTotalHeight ?? 30000;
            var wfMode = _inspectionSettings?.ImageView?.WaterfallFullMode ?? AniloxRoll.Monitor.Core.Data.WaterfallFullMode.Restart;
            // 槽數＝配置相機數（7，含未上線/掉線的黑布槽），非線上台數 _cameras.Count（會變 4）。
            // 對齊全域合圖「合圖全部」：用所有配置相機 start/寬，camId 直接對應槽位。
            int slotCount = _inspectionSettings?.GetCameraStartPositionMmArray()?.Length ?? _cameras.Count;
            _waterfallView = new AniloxRoll.Monitor.UI.Widgets.WaterfallView(_mainDisplayPanel, slotCount, wfH, wfMode, _screenMmPerPx);
            FeedWaterfallLayout(); // 無條件餵佈局：沒開全域合圖也要有 startMm，否則 PrepareBand 永遠 null → 全黑無畫面
            foreach (var cam in _cameras) cam.OnDisplayFrame += OnCameraWaterfallFrame;
        }

        /// <summary>餵瀑布圖合圖佈局：全域合圖開→用 merger 槽位（對齊 live 合圖）；沒開→退回設定的相機 start/ops（仍要有佈局才出畫面）。</summary>
        private void FeedWaterfallLayout()
        {
            if (_waterfallView == null) return;
            if (_globalMerge.IsActive && _globalMerge.Merger != null && _globalMerge.Merger.SlotStartsMm != null)
            {
                _waterfallView.SetLayout(_globalMerge.Merger.SlotStartsMm, null, _globalMerge.Merger.RefOpsMm);
                return;
            }
            if (_inspectionSettings == null) return;
            var startMm = _inspectionSettings.GetCameraStartPositionMmArray();
            var opsUm = _inspectionSettings.GetCameraOpsUmArray();
            double refOps = (opsUm != null && opsUm.Length > 0 && opsUm[0] > 0) ? opsUm[0] / 1000.0 : 0.024;
            _waterfallView.SetLayout(startMm, null, refOps);
        }

        /// <summary>切離 Waterfall → 解訂閱 + dispose（露出底層由 ApplyMainDisplayMode 接手別的模式）。</summary>
        private void DisableWaterfallDisplay()
        {
            if (_waterfallView == null) return;
            foreach (var cam in _cameras) cam.OnDisplayFrame -= OnCameraWaterfallFrame;
            _waterfallView.Dispose();
            _waterfallView = null;
        }

        private void OnCameraWaterfallFrame(int camId, byte[] bytes, int w, int h, long tick) => _waterfallView?.PushFrame(camId, bytes, w, h, tick);

        /// <summary>瀑布圖參數（總高 / 滿了行為）變更 → 重建以套新值（僅瀑布模式中）。</summary>
        public void RefreshWaterfallDisplay()
        {
            if (!WaterfallMode || _waterfallView == null) return;
            DisableWaterfallDisplay();
            EnableWaterfallDisplay();
        }

        // ── SmartCanvas 顯示路徑橋接 ──
        /// <summary>SmartCanvas 模式且尚未建立 → 在 camLiveMain 疊 SmartCanvas + 訂閱各相機每幀 bytes（冪等）。
        /// 在「相機配置」與「開始抓取」都呼叫 → 切設定後重開抓取即生效，不必重啟程式。</summary>
        private void EnsureSmartDisplay()
        {
            if (!SmartCanvasMode || _smartDisplay != null) return;
            if (_mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;
            _smartDisplay = new LiveDisplayView(_mainDisplayPanel, _cameraPanels, _screenMmPerPx);
            _smartDisplay.ThumbSelectedColor = Color.Orange;   // 沿用本產品選取色；選取框唯一來源=sdk ThumbView
            _smartDisplay.SelectRequested  += SmartSelectCamera;
            // 反向連動（合圖視野移動 → sdk 已自動高亮縮圖）：只同步 app 選中狀態，不走 SwitchMainDisplay（防重載/遞迴）
            _smartDisplay.SelectedCamChanged += camId => _selectedMainCameraId = camId;
            _smartDisplay.ViewRangeMmChanged += OnSmartViewRange;
            // SmartCanvas 模式下 MIL 滑鼠 hook 被覆蓋不觸發 → lblPixelInfo 改吃 LiveDisplayView 游標狀態（同源）
            _smartDisplay.CursorStatusChanged += OnSmartCursorStatus;
            _smartDisplay.SetSelected(_selectedMainCameraId);
            if (_globalMerge.IsActive && _globalMerge.Merger != null)
            {
                var merger = _globalMerge.Merger;
                var ops = new double[merger.SlotStartsMm?.Length ?? 0];
                for (int i = 0; i < ops.Length; i++) ops[i] = merger.RefOpsMm * 1000.0; // 均勻 ops（µm）
                _smartDisplay.SetLayout(merger.SlotStartsMm, ops, 1, RowPitchMm); // 主程式餵全解析度顯示 bytes → feedScale=1
            }
            _smartDisplay.MergeAll = _globalMerge.IsActive;     // 全域＝合圖全部（含無畫面相機黑占位）
            _smartDisplay.SetMergeMode(_globalMerge.IsActive);
            foreach (var cam in _cameras) cam.OnDisplayFrame += OnCameraDisplayFrame;
            if (_inspectionSettings != null) SetLodMode(_inspectionSettings.LiveLod); // 套目前 LOD 設定
        }

        /// <summary>套用動態 LOD 模式到 LiveDisplayView（he_LiveLod 變更 / 顯示建立時呼叫）。</summary>
        public void SetLodMode(LiveLodMode mode)
        {
            if (_smartDisplay == null) return;
            switch (mode)
            {
                case LiveLodMode.GPU: _lodBuffer.Arm(); _smartDisplay.EnableLod(_lodBuffer.Resize); break;
                case LiveLodMode.CPU: _smartDisplay.EnableLod(GrayResizeCpu.Resize); break;
                default:              _smartDisplay.DisableLod(); break;
            }
        }

        /// <summary>切回 MIL 模式（he_MainDisplay==MilDirect）→ 解訂閱 + dispose SmartCanvas，露出底層 MIL。</summary>
        private void TeardownSmartDisplay()
        {
            if (_smartDisplay == null) return;
            foreach (var cam in _cameras) cam.OnDisplayFrame -= OnCameraDisplayFrame;
            _smartDisplay.Dispose();
            _smartDisplay = null;
            _lodBuffer.Release();   // LOD pinned 釋放（GpuLodResizeBuffer 內鎖 + 旗標，等背景 provider 用完防 use-after-free）
        }

        private void OnCameraDisplayFrame(int camId, byte[] bytes, int w, int h, long tick) => _smartDisplay?.PushFrame(camId, bytes, w, h);
        private void SmartSelectCamera(int camId) => SwitchMainDisplay(camId);
        /// <summary>監控主畫面（LiveDisplayView）縮放/平移 → 把可見範圍轉給 form 連動 live 曲線圖
        /// （切向/overview 用 X 範圍、法向用 Y 範圍）。bin↔主畫面對齊。</summary>
        private void OnSmartViewRange(double leftMm, double rightMm, double topMm, double botMm)
            => OnLiveViewRange?.Invoke(leftMm, rightMm, topMm, botMm);

        /// <summary>SmartCanvas 模式：LiveDisplayView 游標狀態 → lblPixelInfo（mm 換算同源在 sdk，這裡只格式化）。
        /// 取代 MIL 滑鼠 hook（SmartCanvas 覆蓋 MIL display 後 hook 不觸發）。</summary>
        private void OnSmartCursorStatus(LiveDisplayView.CursorStatus s)
        {
            if (_updatePixelInfoCallback == null) return;
            string tag = IsGlobalMergeActive ? "全域合圖" : $"CAM {s.SelectedCamId}";
            _updatePixelInfoCallback.Invoke(
                $"即時影像 [{tag}] | " +
                $"位置:({s.CurMmX:F2}, {s.CurMmY:F2}) mm | " +
                $"X範圍:{s.ViewLeftMm:F1}~{s.ViewRightMm:F1} mm | " +
                $"Y範圍:{s.ViewTopMm:F1}~{s.ViewBotMm:F1} mm | " +
                $"座標: ({s.CursorX}, {s.CursorY}) | " +
                $"亮度: {s.Brightness} | " +
                $"實體倍率:{(s.PhysMag > 0 ? $"{s.PhysMag:F2}x" : "-")}");
        }

        /// <summary>監控主畫面可見範圍變更（leftX, rightX, topY, botY mm）→ form 訂閱、連動 live 曲線圖 zoom。</summary>
        public event Action<double, double, double, double> OnLiveViewRange;

        private void SwitchMainDisplay(int cameraIndex)
        {
            // 關閉/釋放期間 form 或 panel 可能已 dispose；存取 .Handle 會觸發 CreateHandle()
            // 而拋 ObjectDisposedException（FreeCameras→DisableGlobalMerge→此處的崩潰路徑）。
            if (_mainForm == null || _mainForm.IsDisposed
                || _mainDisplayPanel == null || _mainDisplayPanel.IsDisposed) return;

            if (_mainForm.InvokeRequired)
            {
                try { _mainForm.BeginInvoke(new Action(() => SwitchMainDisplay(cameraIndex))); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }

            _selectedMainCameraId = cameraIndex;
            _userSelectedMainCameraId = cameraIndex;
            _smartDisplay?.SetSelected(cameraIndex);

            foreach (var kvp in _liveParentPanels)
                kvp.Value.Invalidate();

            // Global merge 時主畫面由合併 display 控制，不切換單台；但 pan 到相機中心
            if (_globalMerge.IsActive)
            {
                _globalMerge.PanToCameraCenter(cameraIndex);
                return;
            }

            // SmartCanvas 模式：主畫面由 LiveDisplayView 顯示，不綁 MIL secondary display 到 camLiveMain
            // （MIL display 的 M_MOUSE_USE 會攔截滾輪，疊在上面的 SmartCanvas 無法縮放）。一律卸成 IntPtr.Zero。
            foreach (var cam in _cameras)
            {
                if (!SmartCanvasMode && cam.CameraId == cameraIndex)
                    cam.SetSecondaryDisplay(_mainDisplayPanel.Handle);
                else
                    cam.SetSecondaryDisplay(IntPtr.Zero);
            }
        }
        // ==================== UI Helpers ====================

        private void UpdateCameraStatus(string statusText, Color color)
        {
            foreach (var pair in _cameraStatusLabels)
            {
                pair.Value.Text      = $"{pair.Key}: {statusText}";
                pair.Value.ForeColor = color;
            }
        }

        private void UpdateSingleCameraStatus(int cameraIndex, string statusText, Color color)
        {
            if (_cameraStatusLabels.TryGetValue(cameraIndex, out var label))
            {
                label.Text      = $"{cameraIndex}: {statusText}";
                label.ForeColor = color;
            }
        }

        // ==================== Physical Magnification 1x ====================

        /// <summary>設定主顯示 zoom 使實體倍率 = 1x（螢幕 1mm = 實際 1mm）。</summary>
        public void SetPhysicalMagnification1x()
        {
            if (!IsLiveGrabbing || _screenMmPerPx <= 0) return;

            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                _globalMerge.SetPhysical1x();
                return;
            }

            int camIdx = _selectedMainCameraId - 1;
            var s = _inspectionSettings;
            double[] opsUmArr = s?.GetCameraOpsUmArray();
            if (opsUmArr == null || camIdx < 0 || camIdx >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[camIdx] / 1000.0;
            if (opsInMm <= 0) return;

            // physicalMag = zoom * screenMmPerPx / opsInMm = 1  →  zoom = opsInMm / screenMmPerPx
            double zoom1xCam = PixelMmMapper.OneToOneZoom(opsInMm, _screenMmPerPx);

            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;

            // 以面板中心為基準
            double cxCam = _mainDisplayPanel.Width / 2.0;
            double cyCam = _mainDisplayPanel.Height / 2.0;

            if (cam.TryGetSecondaryDisplayGeometry(out double curZoomCam, out _, out double curPanXCam, out double curPanYCam) && curZoomCam > 0)
            {
                double imgCx = curPanXCam + cxCam / curZoomCam;
                double imgCy = curPanYCam + cyCam / curZoomCam;
                double newPanX = imgCx - cxCam / zoom1xCam;
                double newPanY = imgCy - cyCam / zoom1xCam;
                cam.SetSecondaryDisplayZoom(zoom1xCam, newPanX, newPanY);
            }
            else
            {
                cam.SetSecondaryDisplayZoom(zoom1xCam, 0, 0);
            }
        }

        // ==================== Custom Wheel Zoom ====================

        internal void ApplyCustomZoom(int wheelDelta)
        {
            if (!IsLiveGrabbing) return;

            double zoomX, panX, panY;

            if (_globalMerge.IsActive && _globalMerge.HasMilDisplay)
            {
                // Global merge 模式：zoom/pan 合併 display（委派 coordinator）
                _globalMerge.ApplyZoom(wheelDelta);
                return;
            }

            var cam = _cameras.Find(c => c.CameraId == _selectedMainCameraId);
            if (cam == null) return;
            if (!cam.TryGetSecondaryDisplayGeometry(out zoomX, out _, out panX, out panY))
                return;

            double factor2 = wheelDelta > 0 ? 1.1 : (1.0 / 1.1);
            double newZoom2 = zoomX * factor2;
            if (newZoom2 < 0.05) newZoom2 = 0.05;
            if (newZoom2 > 32.0) newZoom2 = 32.0;

            // 以面板中心為縮放基準點
            double cx2 = _mainDisplayPanel.Width / 2.0;
            double cy2 = _mainDisplayPanel.Height / 2.0;
            double imgX2 = panX + cx2 / zoomX;
            double imgY2 = panY + cy2 / zoomX;
            double newPanX2 = imgX2 - cx2 / newZoom2;
            double newPanY2 = imgY2 - cy2 / newZoom2;

            cam.SetSecondaryDisplayZoom(newZoom2, newPanX2, newPanY2);
            OnAfterVerticalZoom?.Invoke();
        }

        /// <summary>攔截 camLiveMain 上的 WM_MOUSEWHEEL，用 1.1x 步長取代 MIL 預設的整數倍跳躍。</summary>
        private class WheelZoomFilter : IMessageFilter
        {
            private const int WM_MOUSEWHEEL = 0x020A;
            private readonly LiveCameraManager _mgr;

            public WheelZoomFilter(LiveCameraManager mgr) => _mgr = mgr;

            public bool PreFilterMessage(ref Message m)
            {
                if (m.Msg != WM_MOUSEWHEEL) return false;
                // SmartCanvas / Waterfall 模式：主畫面由各自的 SmartCanvas 自己處理滾輪 zoom（無 MIL 巨圖 display）。
                // filter 不可攔截，否則滾輪被吃掉 → SmartCanvas 收不到 → camLiveMain「縮不動」
                // （雙三擊是點擊事件、不走此 filter，故一直有反應）。此 filter 只服務 MIL 直繪合圖縮放。
                if (_mgr.SmartCanvasMode || _mgr.WaterfallMode) return false;
                if (!_mgr.IsLiveGrabbing) return false;

                var panel = _mgr._mainDisplayPanel;
                var screenPt = Cursor.Position;
                if (!panel.RectangleToScreen(panel.ClientRectangle).Contains(screenPt))
                    return false;

                int delta = (short)(m.WParam.ToInt64() >> 16);
                _mgr.ApplyCustomZoom(delta);
                return true; // 攔截訊息，不讓 MIL 處理
            }
        }
    }
}
