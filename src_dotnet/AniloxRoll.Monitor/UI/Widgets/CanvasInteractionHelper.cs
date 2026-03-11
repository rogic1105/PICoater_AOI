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

        private float _savedZoom = 1.0f;
        private PointF _savedPan = PointF.Empty;
        private bool _shouldRestoreView = false;

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

        /// <summary>在載入新圖前呼叫，記住目前縮放位置以便還原。</summary>
        public void SaveViewIfNeeded()
        {
            if (_canvas.Image != null)
            {
                _savedZoom = _canvas.Zoom;
                _savedPan = _canvas.PanOffset;
                _shouldRestoreView = true;
            }
            else
            {
                _shouldRestoreView = false;
            }
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

        /// <summary>顯示新圖，依 SaveViewIfNeeded 決定還原縮放或 FitToScreen。</summary>
        public void UpdateCanvas(Bitmap newImage)
        {
            if (newImage == null) return;

            ClearCanvas();
            _canvas.Image = newImage;

            if (_shouldRestoreView)
            {
                _canvas.SetView(_savedZoom, _savedPan);
                _shouldRestoreView = false;
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

            double physicalX = startPosMm + (info.ImageX * opsInMm);

            if (info.Zoom > 0)
            {
                double pixelLeft = (0 - info.PanOffset.X) / info.Zoom;
                double pixelRight = (_canvas.Width - info.PanOffset.X) / info.Zoom;

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
