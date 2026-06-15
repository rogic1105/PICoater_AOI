using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using TanukiCv.Controls;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.UI.State;

namespace AniloxRoll.Monitor.UI.Widgets
{
    public class FormInteractionContext
    {
        public Form Form { get; set; }
        public SmartCanvas Canvas { get; set; }
        public Button[] ButtonsToLock { get; set; }
        public List<Image> ThumbnailCache { get; set; }
        public AniloxRollPresenter Presenter { get; set; }
        public BatchInspectionService InspectionService { get; set; }
        public ImageRepository ImageRepository { get; set; }
        public DateTimeNavigator TimeNavigator { get; set; }
        public ThumbnailGridPresenter GalleryManager { get; set; }
        public ColumnCurveChartHelper ColumnChartHelper { get; set; }
        public InspectionSettings Settings { get; set; }
        public ToolStripStatusLabel StatusLabel { get; set; }
        public PictureBox[] CameraPanels { get; set; }
        public RowCurveChartHelper RowChartHelper { get; set; }
    }

    public class FormInteractionHelper
    {
        private readonly Form _form;
        private readonly Button[] _buttonsToLock;
        private readonly List<Image> _thumbnailCache;
        private readonly AniloxRollPresenter _presenter;
        private readonly BatchInspectionService _inspectionService;
        private readonly ImageRepository _imageRepository;
        private readonly DateTimeNavigator _timeNavigator;
        private readonly ThumbnailGridPresenter _galleryManager;
        private readonly ColumnCurveChartHelper _columnChartHelper;
        private readonly RowCurveChartHelper _rowChartHelper;
        private readonly InspectionSettings _settings;
        private readonly CanvasInteractionHelper _canvasHelper;

        /// <summary>
        /// 回顧資料夾的 CSV #CFG 快照。設定後，chart/canvas 優先使用 CFG 的 Ops/Pos/閾值。
        /// </summary>
        public CsvConfigSnapshot ReviewConfig
        {
            get => _canvasHelper.ReviewConfig;
            set => _canvasHelper.ReviewConfig = value;
        }

        private bool _isProcessedMode = false;
        private bool _isBusy = false;

        public FormInteractionHelper(FormInteractionContext context)
        {
            if (context == null) throw new ArgumentNullException(nameof(context));

            _form = context.Form;
            _buttonsToLock = context.ButtonsToLock;
            _thumbnailCache = context.ThumbnailCache;
            _presenter = context.Presenter;
            _inspectionService = context.InspectionService;
            _imageRepository = context.ImageRepository;
            _timeNavigator = context.TimeNavigator;
            _galleryManager = context.GalleryManager;
            _columnChartHelper = context.ColumnChartHelper;
            _rowChartHelper = context.RowChartHelper;
            _settings = context.Settings;

            _canvasHelper = new CanvasInteractionHelper(
                context.Canvas,
                context.Settings,
                context.StatusLabel,
                context.ColumnChartHelper,
                context.RowChartHelper,
                context.CameraPanels,
                context.GalleryManager);
        }

        // ── Canvas 事件代理 ──────────────────────────────────────────────
        public void UpdateCanvasInfo(CanvasInfo info) => _canvasHelper.UpdateCanvasInfo(info);
        public void NavigateCamera(int direction) => _canvasHelper.NavigateCamera(direction);
        public void SetCanvasScaleAndCamera(int scaleFactor, int cameraIndex)
        {
            _canvasHelper.SetImageScaleFactor(scaleFactor);
            _canvasHelper.SetCurrentCameraIndex(cameraIndex);
        }
        public bool TryComputeCurrentViewRange(int cameraIndex, out double leftMm, out double rightMm)
            => _canvasHelper.TryComputeCurrentViewRange(cameraIndex, out leftMm, out rightMm);
        public void RefreshChartRange() => _canvasHelper.RefreshChartRange();
        public void RefreshRowChartRange() => _canvasHelper.RefreshRowChartRange();
        public void SaveCanvasView() => _canvasHelper.SaveViewIfNeeded();
        public void RestoreCanvasViewOrFit() => _canvasHelper.RestoreViewOrFitToScreen();
        public void ClearCanvasView() => _canvasHelper.ClearSavedView();
        public void SetScreenMmPerPixel(double mmPerPx) => _canvasHelper.SetScreenMmPerPixel(mmPerPx);
        public double ScreenMmPerPixel => _canvasHelper.ScreenMmPerPixel;
        public double RowPitchMm => _rowChartHelper?.RowPitchMm ?? 0;

        /// <summary>設定全域/水平合圖模式：chartReviewVertical 與 canvas 座標聯動。</summary>
        public void SetMergedMode(ColumnCurveChartHelper overviewHelper, double startMm, double opsUm)
        {
            _canvasHelper.OverviewChartHelper = overviewHelper;
            _canvasHelper.SetMergedCoordinates(startMm, opsUm);
        }

        /// <summary>離開合圖模式：清除 chartReviewVertical 聯動與座標覆寫。</summary>
        public void ClearMergedMode()
        {
            _canvasHelper.OverviewChartHelper = null;
            _canvasHelper.ClearMergedCoordinates();
        }

        // ── 設定 ─────────────────────────────────────────────────────────
        public void ApplySettingsToService()
        {
            if (_inspectionService == null || _settings == null) return;
            // Pipeline 用 V 閾值（pipeline 主處理方向，閾值在 UI/chart/DO 判斷用，pipeline 不依賴）
            _inspectionService.UpdateAlgorithmParams(
                _settings.HessianMaxFactorV,
                _settings.ErrorValueMeanV,
                _settings.ErrorValueMaxV,
                InspectionRecipe.RidgeDirectionToNative(_settings.RidgeDir)
            );
        }

        public void SetRidgeDirection(string dir) => _inspectionService?.SetRidgeDirection(dir);

        /// <summary>
        /// 套用 setting 變更的副作用。
        /// L2：save disk 已由 SettingsHub 統一處理，這裡只做「應用到下游服務」。
        /// </summary>
        public void HandleSettingsChanged()
        {
            if (_settings == null) return;
            ApplySettingsToService();
            _columnChartHelper?.SetOps(_settings.Cam1_Ops);
            _canvasHelper.Invalidate();
        }

        // ── 工作流程 ──────────────────────────────────────────────────────
        public async Task LoadImages(bool enableProcess)
        {
            _canvasHelper.SaveViewIfNeeded();
            _isProcessedMode = enableProcess;
            ClearOldImages();
            await _presenter.RunWorkflowAsync(enableProcess, _thumbnailCache);
        }

        public void SetUiLoadingState(bool isBusy)
        {
            _isBusy = isBusy;
            if (_form == null || _form.IsDisposed || !_form.IsHandleCreated) return;
            if (_form.InvokeRequired)
            {
                try { _form.Invoke(new Action<bool>(SetUiLoadingState), isBusy); }
                catch (InvalidOperationException) { /* ObjectDisposedException 亦繼承自此 */ }
                return;
            }
            _form.Cursor = isBusy ? Cursors.WaitCursor : Cursors.Default;
            foreach (var btn in _buttonsToLock)
            {
                btn.Enabled = !isBusy;
            }
        }

        // ── Gallery 選取 ──────────────────────────────────────────────────
        // ── 時間導航 ──────────────────────────────────────────────────────
        public void NavigateToDateTime(DateTime dt) => _timeNavigator.NavigateTo(dt);

        public void LoadDirectoryAndInitNavigator(string path)
        {
            _imageRepository.LoadDirectory(path);
            if (_imageRepository.FileCount > 0)
                _timeNavigator.Initialize(UserSessionState.LastYear);
        }

        // ── 資料夾選擇 ────────────────────────────────────────────────────
        public void SelectAndLoadFolder()
        {
            using (var fbd = new FolderBrowserDialog())
            {
                string preferredPath = UserSessionState.LastDataPath;
                if (!Directory.Exists(preferredPath)) preferredPath = _settings?.CaptureRootPath;
                if (string.IsNullOrEmpty(preferredPath) || !Directory.Exists(preferredPath))
                    preferredPath = @"D:\Anilox\Captures";
                if (Directory.Exists(preferredPath))
                    fbd.SelectedPath = preferredPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    int lastCameraIndex = _galleryManager.SelectedIndex;
                    if (lastCameraIndex < 0) lastCameraIndex = 0;

                    // 路徑修正：使用者選的目錄無 yyyy 子目錄但底下有 Captures\yyyy → 自動往下走
                    // (使用者誤選 D:\Anilox 時自動轉成 D:\Anilox\Captures)
                    string selectedPath = fbd.SelectedPath;
                    if (!HasYearSubdir(selectedPath))
                    {
                        string capturesSub = Path.Combine(selectedPath, "Captures");
                        if (HasYearSubdir(capturesSub)) selectedPath = capturesSub;
                    }

                    UserSessionState.SetLastDataPath(selectedPath);
                    UserSessionState.Save();

                    _imageRepository.LoadDirectory(selectedPath);
                    if (_imageRepository.FileCount == 0)
                    {
                        MessageBox.Show(_form, "該路徑下無符合格式的圖片！");
                        return;
                    }

                    _timeNavigator.Initialize(UserSessionState.LastYear);

                    _galleryManager.Select(lastCameraIndex, triggerEvent: false);
                    _canvasHelper.SetCurrentCameraIndex(lastCameraIndex);
                }
            }
        }

        /// <summary>判斷指定目錄下是否有「4 位數字」的子目錄（即 yyyy）。</summary>
        private static bool HasYearSubdir(string path)
        {
            if (string.IsNullOrEmpty(path) || !Directory.Exists(path)) return false;
            try
            {
                foreach (var d in Directory.GetDirectories(path))
                {
                    string name = Path.GetFileName(d);
                    if (name.Length == 4 && int.TryParse(name, out _)) return true;
                }
            }
            catch { /* 權限或 IO 失敗忽略 */ }
            return false;
        }

        // ── 資源清理 ──────────────────────────────────────────────────────
        public void ClearOldImages()
        {
            _canvasHelper.ClearCanvas();
            // 先清 PictureBox 引用，再 Dispose Bitmap。
            // 若順序相反，await Task.Run 期間 UI 執行緒空出，
            // Windows Paint 事件會嘗試繪製已 Dispose 的 Bitmap，拋 ArgumentException。
            _galleryManager.ClearImages();
            foreach (var img in _thumbnailCache)
            {
                try { img.Dispose(); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[FormInteractionHelper] Bitmap.Dispose failed: {ex.GetType().Name}: {ex.Message}");
                }
            }
            _thumbnailCache.Clear();
            // blocking:false — 讓 GC 在 background 執行，不阻塞 UI 執行緒
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Optimized, blocking: false);
        }

    }
}
