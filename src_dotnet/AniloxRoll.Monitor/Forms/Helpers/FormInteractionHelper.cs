using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AOI.SDK.UI;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    /// <summary>
    /// [View Helper] 負責封裝 AniloxRollForm 的 UI 互動邏輯。
    /// 將繁瑣的控制項操作 (如 Cursor 切換、Canvas 更新、按鈕鎖定) 移出 Form，
    /// 讓 Form 類別專注於事件綁定與生命週期管理。
    /// </summary>
    public class FormInteractionHelper
    {
        // 外部依賴 (View Components)
        private readonly Form _form;
        private readonly SmartCanvas _canvas;
        private readonly Button[] _buttonsToLock;
        private readonly List<Image> _thumbnailCache;

        // 業務邏輯依賴
        private readonly AniloxRollPresenter _presenter;
        private readonly BatchInspectionService _inspectionService;

        // 內部狀態
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

        // =========================================================
        // Public API (供 Form 呼叫)
        // =========================================================

        public async Task LoadImages(bool enableProcess)
        {
            _isProcessedMode = enableProcess;
            ClearOldImages();
            await _presenter.RunWorkflowAsync(enableProcess, _thumbnailCache);
        }

        public void SetUiLoadingState(bool isBusy)
        {
            _isBusy = isBusy;

            // 必須在 UI Thread 操作
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
            // 若正在批次運算中，禁止切換大圖，避免共用 Buffer 衝突
            if (_isBusy) return;

            try
            {
                Bitmap bigImg = null;

                // 根據模式決定顯示原圖或檢測圖
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
                MessageBox.Show(_form, "載入大圖失敗: " + ex.Message);
            }
        }

        public void CleanupSystem()
        {
            // 1. 清除 UI 圖片
            ClearOldImages();

            // 2. 釋放 Service (C++ Pinned Memory)
            _inspectionService?.Dispose();
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