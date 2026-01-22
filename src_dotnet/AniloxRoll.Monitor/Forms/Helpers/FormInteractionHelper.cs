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

        private bool _isProcessedMode = false;
        private bool _isBusy = false;

        // [新增] 為了處理資料夾載入，需要這三個元件的控制權
        private readonly ImageRepository _imageRepository;
        private readonly DateTimeNavigator _timeNavigator;
        private readonly ThumbnailGridPresenter _galleryManager;

        // 視圖狀態記憶
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
            // [新增] 透過建構子注入依賴
            ImageRepository repo,
            DateTimeNavigator timeNav,
            ThumbnailGridPresenter galleryMgr)
        {
            _form = form;
            _canvas = canvas;
            _buttonsToLock = buttons;
            _thumbnailCache = cache;
            _presenter = presenter;
            _inspectionService = service;

            // [新增] 指派
            _imageRepository = repo;
            _timeNavigator = timeNav;
            _galleryManager = galleryMgr;
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
                Bitmap bigImg = null;

                // 呼叫 Service 取得大圖 (自動處理翻轉)
                bigImg = _inspectionService.RunInspectionFullRes(index);

                UpdateCanvas(bigImg);
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
            if (_canvas.Image != null)
            {
                var old = _canvas.Image;
                _canvas.Image = null;
                old.Dispose();
            }
            _canvas.Image = newImage;

            // [新增] 2. 判斷是否需要還原視圖
            if (_shouldRestoreView)
            {
                // 如果是透過按鈕切換模式進來的，還原上一次的位置
                _canvas.SetView(_savedZoom, _savedPan);
                _shouldRestoreView = false; // 用完即丟，避免切換不同縮圖時也被還原
            }
            else
            {
                // 如果是直接點選縮圖進來的，就重置為全圖顯示
                _canvas.FitToScreen();
            }
        }
    }
}