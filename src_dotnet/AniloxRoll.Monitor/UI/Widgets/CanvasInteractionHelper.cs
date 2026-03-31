using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using AOI.SDK.UI;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Widgets
{
    public class CanvasInteractionHelper
    {
        private readonly SmartCanvas _canvas;
        private readonly InspectionSettings _settings;
        private readonly ToolStripStatusLabel _statusLabel;
        private readonly MuraChartHelper _muraChartHelper;
        private readonly RowMuraChartHelper _rowMuraChartHelper;
        private readonly PictureBox[] _cameraPanels;
        private readonly ThumbnailGridPresenter _galleryManager;

        private int _currentCameraIndex = 0;
        private double _currentViewLeftMm = 0;
        private double _currentViewRightMm = 0;

        /// <summary>
        /// 顯示圖像相對於全解析度的縮放倍率。
        /// 新格式 JPEG 為 5（寬高各縮小5倍），舊格式 BMP 為 1。
        /// 影響 UpdateCanvasInfo 中 pixel→mm 的換算。
        /// </summary>
        private int _imageScaleFactor = 1;

        public void SetImageScaleFactor(int scale) => _imageScaleFactor = Math.Max(1, scale);

        /// <summary>螢幕每邏輯像素對應的實體 mm（用於計算實體倍率）。</summary>
        private double _screenMmPerPx = 0;

        public void SetScreenMmPerPixel(double mmPerPx) => _screenMmPerPx = mmPerPx;
        public double ScreenMmPerPixel => _screenMmPerPx;

        public string ImageInfoSuffix { get; set; } = "";

        private float _savedZoom = 1.0f;
        private PointF _savedPan = PointF.Empty;
        private bool _shouldRestoreView = false;

        // FitToScreen/SetView 會同步觸發 StatusChanged → UpdateCanvasInfo → UpdateViewRange，
        // 在程式碼主動呼叫 UpdateCanvas 期間需要壓制此路徑，
        // 讓 chart 只由呼叫端在 UpdateDataAndView 統一更新一次（避免閃爍與 range 錯誤）。
        private bool _suppressChartSync = false;

        // 世界座標存檔（mm），用於跨倍率的 view 還原
        private double _savedViewLeftMm = double.NaN;
        private double _savedViewRightMm = double.NaN;
        private double _savedYCenterFraction = 0.5;

        public CanvasInteractionHelper(
            SmartCanvas canvas,
            InspectionSettings settings,
            ToolStripStatusLabel statusLabel,
            MuraChartHelper muraChartHelper,
            RowMuraChartHelper rowMuraChartHelper,
            PictureBox[] cameraPanels,
            ThumbnailGridPresenter galleryManager)
        {
            _canvas = canvas;
            _settings = settings;
            _statusLabel = statusLabel;
            _muraChartHelper = muraChartHelper;
            _rowMuraChartHelper = rowMuraChartHelper;
            _cameraPanels = cameraPanels ?? Array.Empty<PictureBox>();
            _galleryManager = galleryManager;
        }

        /// <summary>回顧時的 CSV #CFG 快照；非 null 時優先使用其 Ops/Pos/閾值。</summary>
        public CsvConfigSnapshot ReviewConfig { get; set; }

        private double[] GetEffectiveOpsArray() =>
            ReviewConfig?.CamOps ?? _settings?.GetCameraOpsUmArray() ?? new double[7];

        private double[] GetEffectivePosArray() =>
            ReviewConfig?.CamPos ?? _settings?.GetCameraStartPositionMmArray() ?? new double[7];

        public void SetCurrentCameraIndex(int index) => _currentCameraIndex = index;

        /// <summary>在載入新圖前呼叫，以世界座標（mm）記住目前 viewport，支援跨倍率還原。</summary>
        public void SaveViewIfNeeded()
        {
            if (_canvas.Image == null) return; // 不重置 flag，保留先前的存檔

            // 嘗試以 mm 世界座標儲存（跨倍率安全）
            _savedViewLeftMm = double.NaN;
            if (_settings != null && _currentCameraIndex >= 0)
            {
                double[] opsUm  = GetEffectiveOpsArray();
                double[] startMm = GetEffectivePosArray();
                if (_currentCameraIndex < opsUm.Length)
                {
                    double opsInMm    = opsUm[_currentCameraIndex] / 1000.0;
                    double startPosMm = startMm[_currentCameraIndex];
                    float  zoom       = _canvas.Zoom;
                    PointF pan        = _canvas.PanOffset;

                    double pixelLeft  = (0              - pan.X) / zoom * _imageScaleFactor;
                    double pixelRight = (_canvas.Width  - pan.X) / zoom * _imageScaleFactor;
                    _savedViewLeftMm  = startPosMm + pixelLeft  * opsInMm;
                    _savedViewRightMm = startPosMm + pixelRight * opsInMm;

                    // Y：以圖片高度中心分率保存（0=頂, 0.5=中, 1=底）
                    double yCenterPx = (_canvas.Height / 2.0 - pan.Y) / zoom;
                    _savedYCenterFraction = _canvas.Image.Height > 0
                        ? yCenterPx / _canvas.Image.Height
                        : 0.5;
                }
            }

            // Pixel fallback（萬一 settings 不可用）
            _savedZoom = _canvas.Zoom;
            _savedPan  = _canvas.PanOffset;
            _shouldRestoreView = true;
        }

        /// <summary>釋放 Canvas 上的舊圖（不清除 thumbnail cache）。</summary>
        public void ClearCanvas()
        {
            if (_canvas.Image != null)
            {
                var old = _canvas.Image;
                _canvas.Image = null;
                old.Dispose();
            }
        }

        public void Invalidate() => _canvas?.Invalidate();

        /// <summary>顯示新圖，依 SaveViewIfNeeded 決定還原縮放或 FitToScreen。
        /// 若有世界座標存檔，以 mm 反算新倍率下的 zoom/pan（跨倍率連續）。
        /// FitToScreen/SetView 內部會觸發 StatusChanged → chart sync，
        /// 此處以 _suppressChartSync 壓制，讓呼叫端統一呼叫 UpdateDataAndView。</summary>
        public void UpdateCanvas(Bitmap newImage)
        {
            if (newImage == null) return;

            ClearCanvas();
            _canvas.Image = newImage;

            RestoreViewOrFitToScreen();
        }

        /// <summary>依 SaveViewIfNeeded 決定還原縮放或 FitToScreen。
        /// 用於 UpdateCanvas 內部，也供 stitched mode 外部呼叫（不經 ClearCanvas）。</summary>
        public void RestoreViewOrFitToScreen()
        {
            if (_canvas.Image == null) return;

            _suppressChartSync = true;
            try
            {
                if (_shouldRestoreView)
                {
                    _shouldRestoreView = false;

                    // 優先：以 mm 世界座標還原（支援 1x↔5x 等跨倍率跳轉）
                    if (!double.IsNaN(_savedViewLeftMm) && _settings != null && _currentCameraIndex >= 0)
                    {
                        double[] opsUm      = GetEffectiveOpsArray();
                        double[] startMmArr = GetEffectivePosArray();
                        if (_currentCameraIndex < opsUm.Length)
                        {
                            double opsInMm    = opsUm[_currentCameraIndex] / 1000.0;
                            double startPosMm = startMmArr[_currentCameraIndex];

                            double leftPx  = (_savedViewLeftMm  - startPosMm) / (opsInMm * _imageScaleFactor);
                            double rightPx = (_savedViewRightMm - startPosMm) / (opsInMm * _imageScaleFactor);
                            double widthPx = rightPx - leftPx;

                            if (widthPx > 0)
                            {
                                float zoom  = (float)(_canvas.Width / widthPx);
                                float panX  = (float)(-leftPx * zoom);
                                float yCenterPx = (float)(_savedYCenterFraction * _canvas.Image.Height);
                                float panY  = (float)(_canvas.Height / 2.0 - yCenterPx * zoom);
                                _canvas.SetView(zoom, new PointF(panX, panY));
                                return;
                            }
                        }
                    }

                    // Fallback：pixel 直接還原（同倍率，或 settings 不可用）
                    _canvas.SetView(_savedZoom, _savedPan);
                }
                else
                {
                    _canvas.FitToScreen();
                }
            }
            finally
            {
                _suppressChartSync = false;
            }
        }

        /// <summary>計算目前 canvas zoom/pan 對應的 mm 視野範圍，不觸發 chart 更新。</summary>
        public bool TryComputeCurrentViewRange(int cameraIndex, out double leftMm, out double rightMm)
        {
            leftMm = rightMm = 0;
            if (_settings == null || _canvas.Image == null) return false;

            double[] opsUmArray   = GetEffectiveOpsArray();
            double[] startMmArray = GetEffectivePosArray();
            if (cameraIndex < 0 || cameraIndex >= opsUmArray.Length) return false;

            float zoom = _canvas.Zoom;
            if (zoom <= 0) return false;

            double opsInMm    = opsUmArray[cameraIndex] / 1000.0;
            double startPosMm = startMmArray[cameraIndex];
            PointF pan        = _canvas.PanOffset;

            double pixelLeft  = (0             - pan.X) / zoom * _imageScaleFactor;
            double pixelRight = (_canvas.Width - pan.X) / zoom * _imageScaleFactor;

            leftMm  = startPosMm + pixelLeft  * opsInMm;
            rightMm = startPosMm + pixelRight * opsInMm;
            return true;
        }

        /// <summary>從目前 canvas 狀態更新 chartMura / chartMuraHorizontal 視野範圍。</summary>
        public void RefreshChartRange()
        {
            if (_canvas.Image == null) return;

            if (_muraChartHelper != null &&
                TryComputeCurrentViewRange(_currentCameraIndex, out double leftMm, out double rightMm))
            {
                _currentViewLeftMm  = leftMm;
                _currentViewRightMm = rightMm;
                _muraChartHelper.UpdateViewRange(leftMm, rightMm);
            }

            RefreshRowChartRange();
        }

        /// <summary>從目前 canvas 狀態單獨更新 chartMuraHorizontal Y 軸視野範圍。</summary>
        public void RefreshRowChartRange()
        {
            if (_rowMuraChartHelper == null || _canvas.Image == null) return;
            double rowPitch = _rowMuraChartHelper.RowPitchMm;
            if (rowPitch <= 0) return;

            float zoom = _canvas.Zoom;
            if (zoom <= 0) return;

            PointF pan = _canvas.PanOffset;
            double pixelTop = (0              - pan.Y) / zoom * _imageScaleFactor;
            double pixelBot = (_canvas.Height - pan.Y) / zoom * _imageScaleFactor;
            _rowMuraChartHelper.UpdateViewRange(pixelTop * rowPitch, pixelBot * rowPitch);
        }

        /// <summary>事件處理：canvas.StatusChanged → 更新 status bar 與 chart 視野範圍。</summary>
        public void UpdateCanvasInfo(CanvasInfo info)
        {
            if (_settings == null || _statusLabel == null) return;

            double[] cameraOpsUmArray = GetEffectiveOpsArray();
            double[] cameraStartPositionMmArray = GetEffectivePosArray();

            if (_currentCameraIndex < 0 || _currentCameraIndex >= cameraOpsUmArray.Length)
                return;

            double opsInUm = cameraOpsUmArray[_currentCameraIndex];
            double opsInMm = opsInUm / 1000.0;
            double startPosMm = cameraStartPositionMmArray[_currentCameraIndex];

            // 若圖像為縮小版（新格式 JPEG），需乘以縮放倍率還原成全解析度座標
            double physicalX = startPosMm + (info.ImageX * _imageScaleFactor * opsInMm);
            double rowPitchMm = _rowMuraChartHelper?.RowPitchMm ?? 0;
            double physicalY = info.ImageY * _imageScaleFactor * rowPitchMm;

            if (info.Zoom > 0)
            {
                double pixelLeft  = (0             - info.PanOffset.X) / info.Zoom * _imageScaleFactor;
                double pixelRight = (_canvas.Width - info.PanOffset.X) / info.Zoom * _imageScaleFactor;

                _currentViewLeftMm = startPosMm + (pixelLeft * opsInMm);
                _currentViewRightMm = startPosMm + (pixelRight * opsInMm);

                if (!_suppressChartSync)
                {
                    _muraChartHelper?.UpdateViewRange(_currentViewLeftMm, _currentViewRightMm);

                    // 法向（水平）Mura 曲線：canvas 垂直 viewport → chart Y 軸同步
                    // 傳入 canvas pixel mm（不 clamp），UpdateViewRange 內部反轉 + 補償
                    if (_rowMuraChartHelper != null)
                    {
                        double rowPitch = _rowMuraChartHelper.RowPitchMm;
                        if (rowPitch > 0)
                        {
                            double pixelTop = (0              - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                            double pixelBot = (_canvas.Height - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                            _rowMuraChartHelper.UpdateViewRange(pixelTop * rowPitch, pixelBot * rowPitch);
                        }
                    }
                }
            }

            // Y 軸視野範圍（mm）
            double viewTopMm = 0, viewBotMm = 0;
            if (info.Zoom > 0 && rowPitchMm > 0)
            {
                double pixelTop = (0              - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                double pixelBot = (_canvas.Height - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                viewTopMm = pixelTop * rowPitchMm;
                viewBotMm = pixelBot * rowPitchMm;
            }

            // 實體倍率：螢幕上 1mm = 實際 1mm 時為 1.0x
            string magStr = "-";
            if (info.Zoom > 0 && _screenMmPerPx > 0 && opsInMm > 0)
            {
                double physicalMag = (info.Zoom * _screenMmPerPx) / (_imageScaleFactor * opsInMm);
                magStr = $"{physicalMag:F2}x";
            }

            _statusLabel.Text =
                $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                $"X範圍:{_currentViewLeftMm:F1}~{_currentViewRightMm:F1} mm | " +
                $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | " +
                $"座標: ({info.ImageX}, {info.ImageY}) | " +
                $"亮度: {info.PixelColor.R} | " +
                $"實體倍率:{magStr}{ImageInfoSuffix}";
        }

        /// <summary>將 canvas 設為實體倍率 1x，滑鼠所指的影像位置移到 canvas 中央。</summary>
        public void SetPhysicalMagnification1x(Point mouseLocation)
        {
            if (_canvas.Image == null || _settings == null) return;

            double[] opsUmArr = GetEffectiveOpsArray();
            if (_currentCameraIndex < 0 || _currentCameraIndex >= opsUmArr.Length) return;

            double opsInMm = opsUmArr[_currentCameraIndex] / 1000.0;
            if (opsInMm <= 0 || _screenMmPerPx <= 0) return;

            float zoom1x = (float)(_imageScaleFactor * opsInMm / _screenMmPerPx);

            // 滑鼠下的影像座標
            float oldZoom = _canvas.Zoom;
            PointF oldPan = _canvas.PanOffset;
            float imgX = (mouseLocation.X - oldPan.X) / oldZoom;
            float imgY = (mouseLocation.Y - oldPan.Y) / oldZoom;

            // 該影像點放到 canvas 中央
            float cx = _canvas.Width / 2f;
            float cy = _canvas.Height / 2f;
            float newPanX = cx - imgX * zoom1x;
            float newPanY = cy - imgY * zoom1x;

            _canvas.SetView(zoom1x, new PointF(newPanX, newPanY));
        }

        /// <summary>事件處理：canvas.EdgeReached → 切換至相鄰相機。</summary>
        public void NavigateCamera(int direction)
        {
            int nextIndex = _currentCameraIndex + direction;
            if (nextIndex >= 0 && nextIndex < _cameraPanels.Length)
            {
                _galleryManager.Select(nextIndex);
            }
        }
    }
}
