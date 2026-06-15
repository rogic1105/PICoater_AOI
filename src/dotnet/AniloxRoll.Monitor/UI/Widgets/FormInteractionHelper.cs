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

        // Wave2 2b：回顧 CFG 快照 + 螢幕 mm/px facade 自存（原寄放 CanvasInteractionHelper，2b-ii 已砍枯幹）。
        private CsvConfigSnapshot _reviewConfig;
        private double _screenMmPerPx;

        /// <summary>
        /// 回顧資料夾的 CSV #CFG 快照。設定後，回顧曲線/座標優先使用 CFG 的 Ops/Pos/閾值
        /// （RSC 取座標、PushFrames 餵 LiveDisplayView 用）。
        /// </summary>
        public CsvConfigSnapshot ReviewConfig
        {
            get => _reviewConfig;
            set => _reviewConfig = value;
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
            // Wave2 2b-ii：CanvasInteractionHelper（回顧顯示路徑）已整棵砍——
            //   顯示/互動/座標 overlay/視野連動全由 sdk LiveDisplayView 承接（ReviewDisplayManager 包）。
            //   原本一堆 canvas 事件代理（UpdateCanvasInfo/NavigateCamera/TryComputeCurrentViewRange/
            //   RefreshChartRange/SaveCanvasView/SetMergedMode...）都是死碼，隨枯幹移除。
        }

        // ── 螢幕校正（餵 PushFrames + Background 的實體倍率換算）────────────
        public void SetScreenMmPerPixel(double mmPerPx) => _screenMmPerPx = mmPerPx;
        public double ScreenMmPerPixel => _screenMmPerPx;
        public double RowPitchMm => _rowChartHelper?.RowPitchMm ?? 0;

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
        }

        // ── 工作流程 ──────────────────────────────────────────────────────
        public async Task LoadImages(bool enableProcess)
        {
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
