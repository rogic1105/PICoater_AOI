using AniloxRoll.Monitor.Core.Data;
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

        private float _savedZoom = 1.0f;
        private PointF _savedPan = PointF.Empty;
        private bool _shouldRestoreView = false;

        // 世界座標存檔（mm），用於跨倍率的 view 還原
        private double _savedViewLeftMm = double.NaN;
        private double _savedViewRightMm = double.NaN;
        private double _savedYCenterFraction = 0.5;

        public CanvasInteractionHelper(
            SmartCanvas canvas,
            InspectionSettings settings,
            ToolStripStatusLabel statusLabel,
            MuraChartHelper muraChartHelper,
            PictureBox[] cameraPanels,
            ThumbnailGridPresenter galleryManager)
        {
            _canvas = canvas;
            _settings = settings;
            _statusLabel = statusLabel;
            _muraChartHelper = muraChartHelper;
            _cameraPanels = cameraPanels ?? Array.Empty<PictureBox>();
            _galleryManager = galleryManager;
        }

        public void SetCurrentCameraIndex(int index) => _currentCameraIndex = index;

        /// <summary>在載入新圖前呼叫，以世界座標（mm）記住目前 viewport，支援跨倍率還原。</summary>
        public void SaveViewIfNeeded()
        {
            if (_canvas.Image == null) { _shouldRestoreView = false; return; }

            // 嘗試以 mm 世界座標儲存（跨倍率安全）
            _savedViewLeftMm = double.NaN;
            if (_settings != null && _currentCameraIndex >= 0)
            {
                double[] opsUm  = _settings.GetCameraOpsUmArray();
                double[] startMm = _settings.GetCameraStartPositionMmArray();
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
        /// 若有世界座標存檔，以 mm 反算新倍率下的 zoom/pan（跨倍率連續）。</summary>
        public void UpdateCanvas(Bitmap newImage)
        {
            if (newImage == null) return;

            ClearCanvas();
            _canvas.Image = newImage;

            if (_shouldRestoreView)
            {
                _shouldRestoreView = false;

                // 優先：以 mm 世界座標還原（支援 1x↔5x 等跨倍率跳轉）
                if (!double.IsNaN(_savedViewLeftMm) && _settings != null && _currentCameraIndex >= 0)
                {
                    double[] opsUm   = _settings.GetCameraOpsUmArray();
                    double[] startMmArr = _settings.GetCameraStartPositionMmArray();
                    if (_currentCameraIndex < opsUm.Length)
                    {
                        double opsInMm    = opsUm[_currentCameraIndex] / 1000.0;
                        double startPosMm = startMmArr[_currentCameraIndex];

                        // mm → 新圖像素（使用當前 _imageScaleFactor，已在 SetImageScaleFactor 更新）
                        double leftPx  = (_savedViewLeftMm  - startPosMm) / (opsInMm * _imageScaleFactor);
                        double rightPx = (_savedViewRightMm - startPosMm) / (opsInMm * _imageScaleFactor);
                        double widthPx = rightPx - leftPx;

                        if (widthPx > 0)
                        {
                            float zoom = (float)(_canvas.Width / widthPx);
                            float panX = (float)(-leftPx * zoom);

                            // Y：從中心分率反算 panOffset
                            float yCenterPx = (float)(_savedYCenterFraction * newImage.Height);
                            float panY = (float)(_canvas.Height / 2.0 - yCenterPx * zoom);

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

        /// <summary>事件處理：canvas.StatusChanged → 更新 status bar 與 chart 視野範圍。</summary>
        public void UpdateCanvasInfo(CanvasInfo info)
        {
            if (_settings == null || _statusLabel == null) return;

            double[] cameraOpsUmArray = _settings.GetCameraOpsUmArray();
            double[] cameraStartPositionMmArray = _settings.GetCameraStartPositionMmArray();

            if (_currentCameraIndex < 0 || _currentCameraIndex >= cameraOpsUmArray.Length)
                return;

            double opsInUm = cameraOpsUmArray[_currentCameraIndex];
            double opsInMm = opsInUm / 1000.0;
            double startPosMm = cameraStartPositionMmArray[_currentCameraIndex];

            // 若圖像為縮小版（新格式 JPEG），需乘以縮放倍率還原成全解析度座標
            double physicalX = startPosMm + (info.ImageX * _imageScaleFactor * opsInMm);

            if (info.Zoom > 0)
            {
                double pixelLeft  = (0             - info.PanOffset.X) / info.Zoom * _imageScaleFactor;
                double pixelRight = (_canvas.Width - info.PanOffset.X) / info.Zoom * _imageScaleFactor;

                _currentViewLeftMm = startPosMm + (pixelLeft * opsInMm);
                _currentViewRightMm = startPosMm + (pixelRight * opsInMm);

                _muraChartHelper?.UpdateViewRange(_currentViewLeftMm, _currentViewRightMm);
            }

            _statusLabel.Text =
                $"位置:{physicalX:F2} mm | " +
                $"範圍:{_currentViewLeftMm:F1}~{_currentViewRightMm:F1} mm | " +
                $"座標: ({info.ImageX}, {info.ImageY}) | " +
                $"亮度: {info.PixelColor.R} | " +
                $"倍率:{info.Zoom:F2}x";
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
