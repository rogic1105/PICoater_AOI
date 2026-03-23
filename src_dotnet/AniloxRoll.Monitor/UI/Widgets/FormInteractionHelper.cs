using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AOI.SDK.UI;
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
        public MuraChartHelper MuraChartHelper { get; set; }
        public InspectionSettings Settings { get; set; }
        public ToolStripStatusLabel StatusLabel { get; set; }
        public PictureBox[] CameraPanels { get; set; }
        public Label ImageFormatLabel { get; set; }
        public Label ImageScaleLabel { get; set; }
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
        private readonly MuraChartHelper _muraChartHelper;
        private readonly InspectionSettings _settings;
        private readonly CanvasInteractionHelper _canvasHelper;
        private readonly Label _imageFormatLabel;
        private readonly Label _imageScaleLabel;

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
            _muraChartHelper = context.MuraChartHelper;
            _settings = context.Settings;
            _imageFormatLabel = context.ImageFormatLabel;
            _imageScaleLabel = context.ImageScaleLabel;

            _canvasHelper = new CanvasInteractionHelper(
                context.Canvas,
                context.Settings,
                context.StatusLabel,
                context.MuraChartHelper,
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

        // ── 設定 ─────────────────────────────────────────────────────────
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
            ConfigManager.SaveInspectionSettings(_settings);
            ApplySettingsToService();
            _muraChartHelper?.SetOps(_settings.Cam1_Ops);
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

        public void RefreshCurrentCanvasResult()
        {
            OnGallerySelectionChanged(_galleryManager?.SelectedIndex ?? 0);
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

        // ── Gallery 選取 ──────────────────────────────────────────────────
        public void OnGallerySelectionChanged(int index)
        {
            _canvasHelper.SetCurrentCameraIndex(index);

            if (_isBusy) return;

            var swTotal = Stopwatch.StartNew();

            try
            {
                var sw = Stopwatch.StartNew();
                InspectionData data = _inspectionService.RunInspectionFullRes(index);
                long fullResMs = sw.ElapsedMilliseconds;

                long canvasMs = 0, chartMs = 0;

                if (data != null)
                {
                    _canvasHelper.SetImageScaleFactor(data.ScaleFactor);

                    if (_imageFormatLabel != null)
                        _imageFormatLabel.Text = data.IsCompressedJpeg ? "壓縮 JPEG" : "原始 BMP";
                    if (_imageScaleLabel != null)
                        _imageScaleLabel.Text = $"縮放: {data.ScaleFactor}x";

                    sw.Restart();
                    _canvasHelper.UpdateCanvas(data.Image);
                    canvasMs = sw.ElapsedMilliseconds;

                    if (_muraChartHelper != null && _settings != null)
                    {
                        sw.Restart();
                        double[] cameraStartPositionMmArray = _settings.GetCameraStartPositionMmArray();
                        double startPos = (index >= 0 && index < cameraStartPositionMmArray.Length)
                            ? cameraStartPositionMmArray[index] : 0;

                        // FitToScreen/SetView 同步更新 Zoom/PanOffset，
                        // UpdateDataAndView 在 SuspendUpdates/ResumeUpdates 內一次完成資料+縮放，
                        // 且 RefreshThresholds 已從 UpdateDataAndView 移除（threshold 未改變不需重算），
                        // 故整個 chart 只 redraw 一次，不閃爍。
                        _canvasHelper.TryComputeCurrentViewRange(index, out double leftMm, out double rightMm);
                        _muraChartHelper.UpdateDataAndView(data.MuraCurveMean, data.MuraCurveMax,
                            startPos, leftMm, rightMm);
                        chartMs = sw.ElapsedMilliseconds;
                    }
                }
                else
                {
                    if (_imageFormatLabel != null) _imageFormatLabel.Text = "";
                    if (_imageScaleLabel != null)  _imageScaleLabel.Text  = "";
                }

                Console.WriteLine(
                    $"[OnSelect] Cam{index + 1} | " +
                    $"FullRes={fullResMs,5}ms | Canvas={canvasMs,4}ms | Chart={chartMs,4}ms | " +
                    $"Total={swTotal.ElapsedMilliseconds,5}ms");
            }
            catch (Exception ex)
            {
                MessageBox.Show(_form, "載入圖像失敗: " + ex.Message);
            }
        }

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
                if (!Directory.Exists(preferredPath)) preferredPath = @"D:\AniloxCaptures";
                if (Directory.Exists(preferredPath))
                    fbd.SelectedPath = preferredPath;

                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    int lastCameraIndex = _galleryManager.SelectedIndex;
                    if (lastCameraIndex < 0) lastCameraIndex = 0;

                    UserSessionState.SetLastDataPath(fbd.SelectedPath);
                    UserSessionState.Save();

                    _imageRepository.LoadDirectory(fbd.SelectedPath);
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

        public void CleanupSystem()
        {
            ClearOldImages();
            _inspectionService?.Dispose();
        }
    }
}
