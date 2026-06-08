using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using TanukiCv.Controls;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
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
        private readonly ColumnCurveChartHelper _columnChartHelper;
        private readonly RowCurveChartHelper _rowChartHelper;
        private readonly PictureBox[] _cameraPanels;
        private readonly ThumbnailGridPresenter _galleryManager;

        private int _currentCameraIndex = 0;
        private double _currentViewLeftMm = 0;
        private double _currentViewRightMm = 0;

        /// <summary>全域/水平合圖模式：同步 chartReviewPatch X 軸視野。</summary>
        public ColumnCurveChartHelper OverviewChartHelper { get; set; }

        // 合圖模式座標覆寫：合圖 pixel 0 對應的 mm 起始位置與解析度
        private double? _mergedStartMm;
        private double? _mergedOpsUm;

        public void SetMergedCoordinates(double startMm, double opsUm)
        {
            _mergedStartMm = startMm;
            _mergedOpsUm = opsUm;
        }

        public void ClearMergedCoordinates()
        {
            _mergedStartMm = null;
            _mergedOpsUm = null;
        }

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

        private float _savedZoom = 0f;
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
            ColumnCurveChartHelper columnChartHelper,
            RowCurveChartHelper rowChartHelper,
            PictureBox[] cameraPanels,
            ThumbnailGridPresenter galleryManager)
        {
            _canvas = canvas;
            _settings = settings;
            _statusLabel = statusLabel;
            _columnChartHelper = columnChartHelper;
            _rowChartHelper = rowChartHelper;
            _cameraPanels = cameraPanels ?? Array.Empty<PictureBox>();
            _galleryManager = galleryManager;
        }

        /// <summary>回顧時的 CSV #CFG 快照；非 null 時優先使用其 Ops/Pos/閾值。</summary>
        public CsvConfigSnapshot ReviewConfig { get; set; }

        private double[] GetEffectiveOpsArray() =>
            ReviewConfig?.CamOps ?? _settings?.GetCameraOpsUmArray() ?? new double[7];

        private double[] GetEffectivePosArray() =>
            ReviewConfig?.CamPos ?? _settings?.GetCameraStartPositionMmArray() ?? new double[7];

        /// <summary>
        /// 取得目前模式的座標參數（mm/pixel 解析度與起始位置）。
        /// 合圖模式使用全域座標，否則依相機索引查陣列。
        /// </summary>
        private bool TryGetEffectiveCoordinates(out double opsInMm, out double startPosMm)
        {
            opsInMm = 0; startPosMm = 0;
            if (_mergedStartMm.HasValue && _mergedOpsUm.HasValue)
            {
                opsInMm    = _mergedOpsUm.Value / 1000.0;
                startPosMm = _mergedStartMm.Value;
                return true;
            }
            double[] opsUm   = GetEffectiveOpsArray();
            double[] startMm = GetEffectivePosArray();
            if (_currentCameraIndex >= 0 && _currentCameraIndex < opsUm.Length)
            {
                opsInMm    = opsUm[_currentCameraIndex] / 1000.0;
                startPosMm = startMm[_currentCameraIndex];
                return true;
            }
            return false;
        }

        public void SetCurrentCameraIndex(int index) => _currentCameraIndex = index;

        public void ClearSavedView() => _shouldRestoreView = false;

        /// <summary>在載入新圖前呼叫，以世界座標（mm）記住目前 viewport，支援跨倍率還原。
        /// Global 模式下不保留位置（導航後一律 FitToScreen）。</summary>
        public void SaveViewIfNeeded()
        {
            if (_canvas.Image == null) return; // 不重置 flag，保留先前的存檔

            // 嘗試以 mm 世界座標儲存（跨倍率安全）
            _savedViewLeftMm = double.NaN;
            if (_settings != null && _currentCameraIndex >= 0 &&
                TryGetEffectiveCoordinates(out double opsInMm, out double startPosMm))
            {
                float  zoom       = _canvas.Zoom;
                PointF pan        = _canvas.PanOffset;

                double pixelLeft  = (0              - pan.X) / zoom * _imageScaleFactor;
                double pixelRight = (_canvas.Width  - pan.X) / zoom * _imageScaleFactor;
                _savedViewLeftMm  = PixelMmMapper.PixelToMm(pixelLeft,  startPosMm, opsInMm);
                _savedViewRightMm = PixelMmMapper.PixelToMm(pixelRight, startPosMm, opsInMm);

                // Y：以圖片高度中心分率保存（0=頂, 0.5=中, 1=底）
                double yCenterPx = (_canvas.Height / 2.0 - pan.Y) / zoom;
                _savedYCenterFraction = _canvas.Image.Height > 0
                    ? yCenterPx / _canvas.Image.Height
                    : 0.5;
            }

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
                    bool restored = false;

                    // 優先：以 mm 世界座標還原（支援 1x↔5x 等跨倍率跳轉）
                    if (!double.IsNaN(_savedViewLeftMm) && _settings != null && _currentCameraIndex >= 0 &&
                        TryGetEffectiveCoordinates(out double opsInMm, out double startPosMm) &&
                        _canvas.Image != null)
                    {
                        double leftPx  = PixelMmMapper.MmToPixel(_savedViewLeftMm,  startPosMm, opsInMm * _imageScaleFactor);
                        double rightPx = PixelMmMapper.MmToPixel(_savedViewRightMm, startPosMm, opsInMm * _imageScaleFactor);
                        double widthPx = rightPx - leftPx;

                        if (widthPx > 0)
                        {
                            int imgW = _canvas.Image.Width;
                            int imgH = _canvas.Image.Height;
                            float zoom  = (float)(_canvas.Width / widthPx);
                            float panX  = (float)(-leftPx * zoom);
                            float yCenterPx = (float)(_savedYCenterFraction * imgH);
                            float panY  = (float)(_canvas.Height / 2.0 - yCenterPx * zoom);

                            // 安全檢查：與 FitToScreen zoom 比較
                            float fitZoom = Math.Min(
                                (float)_canvas.Width  / imgW,
                                (float)_canvas.Height / imgH);
                            float imgScreenW = imgW * zoom;
                            float imgScreenH = imgH * zoom;
                            bool zoomTooSmall  = fitZoom > 0 && zoom < fitZoom * 0.8f;
                            bool outOfBoundsX  = (panX + imgScreenW < 0) || (panX > _canvas.Width);
                            bool outOfBoundsY  = (panY + imgScreenH < 0) || (panY > _canvas.Height);

                            if (!zoomTooSmall && !outOfBoundsX && !outOfBoundsY)
                            {
                                _canvas.SetView(zoom, new PointF(panX, panY));
                                restored = true;
                            }
                        }
                    }

                    if (!restored && _savedZoom > 0)
                        _canvas.SetView(_savedZoom, _savedPan);
                    else if (!restored)
                        _canvas.FitToScreen();
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

            float zoom = _canvas.Zoom;
            if (zoom <= 0) return false;
            if (!TryGetEffectiveCoordinates(out double opsInMm, out double startPosMm)) return false;

            PointF pan = _canvas.PanOffset;

            double pixelLeft  = (0             - pan.X) / zoom * _imageScaleFactor;
            double pixelRight = (_canvas.Width - pan.X) / zoom * _imageScaleFactor;

            leftMm  = PixelMmMapper.PixelToMm(pixelLeft,  startPosMm, opsInMm);
            rightMm = PixelMmMapper.PixelToMm(pixelRight, startPosMm, opsInMm);
            return true;
        }

        /// <summary>從目前 canvas 狀態更新 chartMura / chartReviewHorizontal / chartReviewPatch 視野範圍。</summary>
        public void RefreshChartRange()
        {
            if (_canvas.Image == null) return;

            if (TryComputeCurrentViewRange(_currentCameraIndex, out double leftMm, out double rightMm))
            {
                _currentViewLeftMm  = leftMm;
                _currentViewRightMm = rightMm;
                _columnChartHelper?.UpdateViewRange(leftMm, rightMm);
                if (_settings?.StitchMode == StitchMode.Global)
                    OverviewChartHelper?.UpdateViewRange(leftMm, rightMm);
            }

            RefreshRowChartRange();
        }

        /// <summary>從目前 canvas 狀態單獨更新 chartReviewHorizontal Y 軸視野範圍。</summary>
        public void RefreshRowChartRange()
        {
            if (_rowChartHelper == null || _canvas.Image == null) return;
            double rowPitch = _rowChartHelper.RowPitchMm;
            if (rowPitch <= 0) return;

            float zoom = _canvas.Zoom;
            if (zoom <= 0) return;

            PointF pan = _canvas.PanOffset;
            double pixelTop = (0              - pan.Y) / zoom * _imageScaleFactor;
            double pixelBot = (_canvas.Height - pan.Y) / zoom * _imageScaleFactor;
            _rowChartHelper.UpdateViewRange(pixelTop * rowPitch, pixelBot * rowPitch);
        }

        /// <summary>事件處理：canvas.StatusChanged → 更新 status bar 與 chart 視野範圍。</summary>
        public void UpdateCanvasInfo(CanvasInfo info)
        {
            if (_settings == null || _statusLabel == null) return;
            if (!TryGetEffectiveCoordinates(out double opsInMm, out double startPosMm)) return;

            // 若圖像為縮小版（新格式 JPEG），需乘以縮放倍率還原成全解析度座標
            double physicalX = PixelMmMapper.PixelToMm(info.ImageX * _imageScaleFactor, startPosMm, opsInMm);
            double rowPitchMm = _rowChartHelper?.RowPitchMm ?? 0;
            double physicalY = info.ImageY * _imageScaleFactor * rowPitchMm;

            if (info.Zoom > 0)
            {
                double pixelLeft  = (0             - info.PanOffset.X) / info.Zoom * _imageScaleFactor;
                double pixelRight = (_canvas.Width - info.PanOffset.X) / info.Zoom * _imageScaleFactor;

                _currentViewLeftMm = PixelMmMapper.PixelToMm(pixelLeft, startPosMm, opsInMm);
                _currentViewRightMm = PixelMmMapper.PixelToMm(pixelRight, startPosMm, opsInMm);

                if (!_suppressChartSync)
                {
                    _columnChartHelper?.UpdateViewRange(_currentViewLeftMm, _currentViewRightMm);
                    if (_settings?.StitchMode == StitchMode.Global)
                        OverviewChartHelper?.UpdateViewRange(_currentViewLeftMm, _currentViewRightMm);

                    // 法向（水平）Mura 曲線：canvas 垂直 viewport → chart Y 軸同步
                    // 傳入 canvas pixel mm（不 clamp），UpdateViewRange 內部反轉 + 補償
                    if (_rowChartHelper != null)
                    {
                        double rowPitch = _rowChartHelper.RowPitchMm;
                        if (rowPitch > 0)
                        {
                            double pixelTop = (0              - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                            double pixelBot = (_canvas.Height - info.PanOffset.Y) / info.Zoom * _imageScaleFactor;
                            _rowChartHelper.UpdateViewRange(pixelTop * rowPitch, pixelBot * rowPitch);
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
                double physicalMag = PixelMmMapper.PhysicalMagnification(info.Zoom, _screenMmPerPx, _imageScaleFactor * opsInMm);
                magStr = $"{physicalMag:F2}x";
            }

            _statusLabel.Text =
                $"位置:({physicalX:F2}, {physicalY:F2}) mm | " +
                $"X範圍:{_currentViewLeftMm:F1}~{_currentViewRightMm:F1} mm | " +
                $"Y範圍:{viewTopMm:F1}~{viewBotMm:F1} mm | " +
                $"座標: ({info.ImageX}, {info.ImageY}) | " +
                $"亮度: {info.PixelColor.R} | " +
                $"實體倍率:{magStr}";

            // 游標位置（mm 數值）推給畫布 overlay：純置換像素座標，沿用 (數值, 數值) 格式、不加單位
            // （加 "位置:"/"mm" 會讓字串變長、超出游標失效小框 → 殘影）。亮度由 canvas 自繪。
            _canvas.SetCursorMm($"({physicalX:F2}, {physicalY:F2})");

            // 同步推給畫布 overlay（座標/亮度由 canvas 自繪；此處給四邊範圍 + 倍率）
            _canvas.SetRangeOverlay(
                magStr,
                $"{_currentViewLeftMm:F1}",  $"{_currentViewRightMm:F1}",
                $"{viewTopMm:F1}",           $"{viewBotMm:F1}");

            // 實體校正餵給 SmartCanvas（三擊實體 1:1 用，唯一來源；隨相機/scale/螢幕即時更新）
            _canvas.SetPhysicalCalibration(_imageScaleFactor * opsInMm, _screenMmPerPx);
        }

        // SetPhysicalMagnification1x 已移除：三擊實體 1:1 改由 SmartCanvas 內建（ZoomToOneToOne，
        // 校正由 UpdateCanvasInfo 餵 SetPhysicalCalibration）。手勢偵測單一來源收進 SmartCanvas。

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
