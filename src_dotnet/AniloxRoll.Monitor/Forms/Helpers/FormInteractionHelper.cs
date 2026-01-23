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

        // 狀態
        private bool _isProcessedMode = false;
        private bool _isBusy = false;
        private float _savedZoom = 1.0f;
        private PointF _savedPan = PointF.Empty;
        private bool _shouldRestoreView = false;

        // 建構子 (共 10 個參數)
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
            MuraChartHelper chartHelper) // 第 10 個參數
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
        }

        public async Task LoadImages(bool enableProcess)
        {
            // [新增] 1. 在切換模式前，記住當前的視圖狀態
            if (_canvas.Image != null)
            {
                _savedZoom = _canvas.Zoom;
                _savedPan = _canvas.PanOffset;
                _shouldRestoreView = true; // 設定旗標：下次 UpdateCanvas 時請還原
            }
            else
            {
                _shouldRestoreView = false;
            }

            _isProcessedMode = enableProcess;
            ClearOldImages();
            // 執行批次處理 (產生縮圖)
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
            if (_isBusy) return;

            try
            {
                AniloxRoll.Monitor.Core.Data.InspectionData data = _inspectionService.RunInspectionFullRes(index);

                if (data != null)
                {
                    // UpdateCanvas 內會處裡 Image
                    UpdateCanvas(data.Image);

                    // 更新圖表
                    if (_muraChartHelper != null)
                    {
                        _muraChartHelper.UpdateData(data.MuraCurveMean, data.MuraCurveMax);
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
                // 讀取上次路徑
                if (Directory.Exists(Properties.Settings.Default.LastDataPath))
                    fbd.SelectedPath = Properties.Settings.Default.LastDataPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    // 1. 記住當前相機 Index
                    int lastCameraIndex = _galleryManager.SelectedIndex;
                    if (lastCameraIndex < 0) lastCameraIndex = 0;

                    // 2. 載入資料
                    _imageRepository.LoadDirectory(fbd.SelectedPath);
                    if (_imageRepository.FileCount == 0)
                    {
                        MessageBox.Show(_form, "該路徑下無符合格式的圖片！");
                        return;
                    }

                    // 3. 儲存設定
                    Properties.Settings.Default.LastDataPath = fbd.SelectedPath;
                    Properties.Settings.Default.Save();

                    // 4. 初始化時間軸
                    _timeNavigator.Initialize(Properties.Settings.Default.LastYear);

                    // 5. 還原相機選取 (不觸發大圖載入，等待使用者點擊 ShowOriginal/Processed)
                    _galleryManager.Select(lastCameraIndex, triggerEvent: false);
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

            // 這裡 SmartCanvas 是單張圖片模式 (根據您之前的需求，多張模式暫緩)
            // 如果要用多張模式，這裡要改成 ClearItems + AddItem

            // 假設目前是單張模式:
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