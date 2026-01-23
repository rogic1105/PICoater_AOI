using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AOI.SDK.UI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    public class FormInteractionHelper
    {
        private readonly Form _form;
        private readonly SmartCanvas _canvas;
        private readonly Button[] _buttonsToLock;
        private readonly List<Image> _thumbnailCache;
        private readonly AniloxRollPresenter _presenter;
        private readonly BatchInspectionService _inspectionService;
        private readonly ImageRepository _imageRepository;
        private readonly DateTimeNavigator _timeNavigator;
        private readonly ThumbnailGridPresenter _galleryManager;
        private readonly MuraChartHelper _muraChartHelper;
        private readonly InspectionSettings _settings;
        private readonly ToolStripStatusLabel _statusLabel;

        private int _currentCameraIndex = 0;

        // 單位: mm
        private double _currentViewLeftMm = 0;
        private double _currentViewRightMm = 0;

        private bool _isProcessedMode = false;
        private bool _isBusy = false;
        private float _savedZoom = 1.0f;
        private PointF _savedPan = PointF.Empty;
        private bool _shouldRestoreView = false;

        public FormInteractionHelper(
            Form form,
            SmartCanvas canvas,
            Button[] buttons,
            List<Image> cache,
            AniloxRollPresenter presenter,
            BatchInspectionService service,
            ImageRepository repo,
            DateTimeNavigator timeNav,
            ThumbnailGridPresenter galleryMgr,
            MuraChartHelper chartHelper,
            InspectionSettings settings,
            ToolStripStatusLabel statusLabel)
        {
            _form = form;
            _canvas = canvas;
            _buttonsToLock = buttons;
            _thumbnailCache = cache;
            _presenter = presenter;
            _inspectionService = service;
            _imageRepository = repo;
            _timeNavigator = timeNav;
            _galleryManager = galleryMgr;
            _muraChartHelper = chartHelper;
            _settings = settings;
            _statusLabel = statusLabel;
        }

        public void ApplySettingsToService()
        {
            if (_inspectionService == null || _settings == null) return;
            _inspectionService.UpdateAlgorithmParams(
                _settings.HessianMaxFactor,
                _settings.ErrorValueMean,
                _settings.ErrorValueMax
            );
        }

        public void HandleSettingsChanged()
        {
            if (_settings == null) return;
            _settings.SaveToSettings();
            ApplySettingsToService();
            if (_muraChartHelper != null)
            {
                _muraChartHelper.SetOps(_settings.Cam1_Ops);
            }
            if (_canvas != null) _canvas.Invalidate();
        }

        // [重點] 這裡計算視野並驅動 Chart
        public void UpdateCanvasInfo(AOI.SDK.UI.CanvasInfo info)
        {
            if (_settings == null || _statusLabel == null) return;

            double[] opsArray = _settings.GetOpsArray();
            double[] posArray = _settings.GetPosArray();

            if (_currentCameraIndex < 0 || _currentCameraIndex >= opsArray.Length)
                return;

            double opsInUm = opsArray[_currentCameraIndex];
            double opsInMm = opsInUm / 1000.0;
            double startPosMm = posArray[_currentCameraIndex];

            double physicalX = startPosMm + (info.ImageX * opsInMm);

            // 計算視野範圍 (mm)
            if (info.Zoom > 0)
            {
                double pixelLeft = (0 - info.PanOffset.X) / info.Zoom;
                double pixelRight = (_canvas.Width - info.PanOffset.X) / info.Zoom;

                _currentViewLeftMm = startPosMm + (pixelLeft * opsInMm);
                _currentViewRightMm = startPosMm + (pixelRight * opsInMm);

                // [更新 Chart] 設定 X 軸顯示範圍，這會讓 Chart 自動 Clip 掉範圍外的數據
                if (_muraChartHelper != null)
                {
                    _muraChartHelper.UpdateViewRange(_currentViewLeftMm, _currentViewRightMm);
                }
            }

            _statusLabel.Text =
                $"位置:{physicalX:F2} mm | " +
                $"範圍:{_currentViewLeftMm:F1}~{_currentViewRightMm:F1} mm | " +
                $"座標: ({info.ImageX}, {info.ImageY}) | " +
                $"亮度: {info.PixelColor.R} | " +
                $"倍率:{info.Zoom:F2}x";
        }

        public void NavigateCamera(int direction)
        {
            int nextIndex = _currentCameraIndex + direction;
            if (nextIndex >= 0 && nextIndex < 7)
            {
                _galleryManager.Select(nextIndex);
            }
        }

        public async Task LoadImages(bool enableProcess)
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

            _isProcessedMode = enableProcess;
            ClearOldImages();
            await _presenter.RunWorkflowAsync(enableProcess, _thumbnailCache);
        }

        public void SetUiLoadingState(bool isBusy)
        {
            _isBusy = isBusy;
            if (_form.InvokeRequired)
            {
                _form.Invoke(new Action<bool>(SetUiLoadingState), isBusy);
                return;
            }
            _form.Cursor = isBusy ? Cursors.WaitCursor : Cursors.Default;
            foreach (var btn in _buttonsToLock)
            {
                btn.Enabled = !isBusy;
            }
        }

        public void OnGallerySelectionChanged(int index)
        {
            if (index >= 0 && index < 7) _currentCameraIndex = index;

            if (_isBusy) return;

            try
            {
                AniloxRoll.Monitor.Core.Data.InspectionData data = _inspectionService.RunInspectionFullRes(index);

                if (data != null)
                {
                    UpdateCanvas(data.Image);

                    if (_muraChartHelper != null && _settings != null)
                    {
                        double[] posArray = _settings.GetPosArray();
                        double startPos = (index >= 0 && index < posArray.Length) ? posArray[index] : 0;

                        // 傳入 startPos 讓 X 軸座標正確
                        _muraChartHelper.UpdateData(data.MuraCurveMean, data.MuraCurveMax, startPos);
                    }
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(_form, "載入圖像失敗: " + ex.Message);
            }
        }

        public void SelectAndLoadFolder()
        {
            using (var fbd = new FolderBrowserDialog())
            {
                if (Directory.Exists(Properties.Settings.Default.LastDataPath))
                    fbd.SelectedPath = Properties.Settings.Default.LastDataPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    int lastCameraIndex = _galleryManager.SelectedIndex;
                    if (lastCameraIndex < 0) lastCameraIndex = 0;

                    _imageRepository.LoadDirectory(fbd.SelectedPath);
                    if (_imageRepository.FileCount == 0)
                    {
                        MessageBox.Show(_form, "該路徑下無符合格式的圖片！");
                        return;
                    }

                    Properties.Settings.Default.LastDataPath = fbd.SelectedPath;
                    Properties.Settings.Default.Save();

                    _timeNavigator.Initialize(Properties.Settings.Default.LastYear);

                    _galleryManager.Select(lastCameraIndex, triggerEvent: false);
                    _currentCameraIndex = lastCameraIndex;
                }
            }
        }

        public void ClearOldImages()
        {
            if (_canvas.Image != null)
            {
                var old = _canvas.Image;
                _canvas.Image = null;
                old.Dispose();
            }
            foreach (var img in _thumbnailCache) img.Dispose();
            _thumbnailCache.Clear();
            GC.Collect();
        }

        public void CleanupSystem()
        {
            ClearOldImages();
            _inspectionService?.Dispose();
        }

        private void UpdateCanvas(Bitmap newImage)
        {
            if (newImage == null) return;

            if (_canvas.Image != null)
            {
                var old = _canvas.Image;
                _canvas.Image = null;
                old.Dispose();
            }
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
    }
}