using System;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AOI.SDK.UI;

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

        public FormInteractionHelper(
            Form form,
            SmartCanvas canvas,
            Button[] buttons,
            List<Image> cache,
            AniloxRollPresenter presenter,
            BatchInspectionService service)
        {
            _form = form;
            _canvas = canvas;
            _buttonsToLock = buttons;
            _thumbnailCache = cache;
            _presenter = presenter;
            _inspectionService = service;
        }

        public async Task LoadImages(bool enableProcess)
        {

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

                if (_isProcessedMode)
                {

                    bigImg = _inspectionService.RunInspectionFullRes(index);
                }
                else
                {
                    string path = _inspectionService.GetFilePath(index);
                    if (!string.IsNullOrEmpty(path))
                    {
                        bigImg = new Bitmap(path);
                    }
                }

                UpdateCanvas(bigImg);
            }
            catch (Exception ex)
            {
                MessageBox.Show(_form, "載入圖像失敗: " + ex.Message);
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
            _canvas.FitToScreen();
        }
    }
}