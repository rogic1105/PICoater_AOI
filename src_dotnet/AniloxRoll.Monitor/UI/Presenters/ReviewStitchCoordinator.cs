using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using AOI.SDK.UI;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// Context 物件：傳遞 UI 控制項與服務參考給 <see cref="ReviewStitchCoordinator"/>。
    /// </summary>
    public class ReviewStitchContext
    {
        public SmartCanvas Canvas { get; set; }
        public Chart ChartOverview { get; set; }
        public Chart ChartMuraVertical { get; set; }
        public Chart ChartMuraHorizontal { get; set; }
        public FormInteractionHelper InteractionHelper { get; set; }
        public ColumnCurveChartHelper ColumnChartHelper { get; set; }
        public RowCurveChartHelper RowChartHelper { get; set; }
        public ColumnCurveChartHelper OverviewHelper { get; set; }
        public ThumbnailGridPresenter GalleryManager { get; set; }

        public BatchInspectionService InspectionService { get; set; }
        public ImageRepository ImageRepository { get; set; }
        public DataStatisticsPresenter DataStatsPresenter { get; set; }

        public InspectionSettings Settings { get; set; }
        public DateTimeNavigator DateTimeNavigator { get; set; }
        public int CameraCount { get; set; }
    }

    /// <summary>
    /// Review tab 的 Stitch 模式管理：LoadGrabStitchedViewAsync、合圖、overview chart 聯動。
    /// 持有 _stitchedImages、_globalMergedImage、_periodMergedImage 等生命週期。
    /// </summary>
    public class ReviewStitchCoordinator
    {
        private readonly ReviewStitchContext _ctx;

        // ── State ──
        private Bitmap[] _stitchedImages;
        private float[][] _stitchedCurveMean;
        private float[][] _stitchedCurveMax;
        private float[][] _stitchedRowCurveMean;
        private float[][] _stitchedRowCurveMax;
        private CsvConfigSnapshot _currentGrabConfig;
        private Bitmap _globalMergedImage;
        private Bitmap _periodMergedImage;

        // ── Public State ──
        public bool IsStitchMode => _stitchedImages != null;
        public bool IsGlobalMerged => _globalMergedImage != null;
        public bool IsPeriodMerged => _periodMergedImage != null;
        public CsvConfigSnapshot CurrentGrabConfig => _currentGrabConfig;

        /// <summary>上一次 Review 頁面的處理模式旗標。</summary>
        public bool LastReviewProcessedMode { get; set; }

        /// <summary>目前的 ridge 方向（"v" 或 "h"）。</summary>
        public string ActiveRidgeDirection { get; set; } = "v";

        public ReviewStitchCoordinator(ReviewStitchContext ctx)
        {
            _ctx = ctx;
        }

        /// <summary>延遲注入 DataStatsPresenter（初始化順序：coordinator 先於 presenter 建立）。</summary>
        public void SetDataStatsPresenter(DataStatisticsPresenter presenter)
        {
            _ctx.DataStatsPresenter = presenter;
        }

        /// <summary>
        /// 載入 GrabId 的拼接影像（使用上次的 processed 模式）。
        /// </summary>
        public Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo)
            => LoadGrabStitchedViewAsync(grabId, hintFrom, hintTo, LastReviewProcessedMode);

        /// <summary>
        /// 載入 GrabId 的拼接影像。背景執行拼接後更新 UI。
        /// </summary>
        public async Task LoadGrabStitchedViewAsync(string grabId, DateTime hintFrom, DateTime hintTo,
            bool enableProcess)
        {
            string root = !string.IsNullOrWhiteSpace(UI.State.UserSessionState.LastDataPath)
                          ? UI.State.UserSessionState.LastDataPath : _ctx.DataStatsPresenter.StatsDataRootPath;
            if (string.IsNullOrWhiteSpace(root)) return;

            _ctx.InteractionHelper.SetUiLoadingState(true);
            LastReviewProcessedMode = enableProcess;
            _ctx.Settings.EnableReviewEnhance = enableProcess;
            var swTotal = Stopwatch.StartNew();
            try
            {
                long csvMs = 0, stitchMs = 0;
                int totalImgCount = 0;
                string ridgeDir = ActiveRidgeDirection;
                int camCount = _ctx.CameraCount;
                float[][] newCurveMean    = new float[camCount][];
                float[][] newCurveMax     = new float[camCount][];
                float[][] newRowCurveMean = new float[camCount][];
                float[][] newRowCurveMax  = new float[camCount][];
                CsvConfigSnapshot grabCfg = null;
                var inspSvc = _ctx.InspectionService;
                var newImages = await Task.Run(() =>
                {
                    var swCsv = Stopwatch.StartNew();
                    var grouped = InspectionStatisticsService.LoadImagePathsForGrabId(
                        root, grabId, hintFrom, hintTo);
                    grabCfg = InspectionStatisticsService.LoadConfigForGrabId(
                        root, grabId, hintFrom, hintTo);
                    csvMs = swCsv.ElapsedMilliseconds;
                    foreach (var kv in grouped) totalImgCount += kv.Value.Count;

                    var swStitch = Stopwatch.StartNew();
                    int scale = InspectionEngineConfig.DefaultSaveResizeScale;
                    var imgs = new Bitmap[camCount];
                    for (int i = 0; i < camCount; i++)
                    {
                        int camId = i + 1;
                        if (grouped.TryGetValue(camId, out var paths) && paths.Count > 0)
                        {
                            try
                            {
                                bool isBmp = paths[0].EndsWith(".bmp", StringComparison.OrdinalIgnoreCase);

                                if (enableProcess && isBmp && inspSvc != null)
                                {
                                    string capturedDir = ridgeDir;
                                    Func<string, Bitmap> procLoader = (p) =>
                                    {
                                        var bmp = inspSvc.ProcessBmpAtScale(p, scale,
                                            out float[] m, out float[] x);
                                        if (capturedDir == "h" && bmp != null)
                                        {
                                            string baseName = System.IO.Path.Combine(
                                                System.IO.Path.GetDirectoryName(p),
                                                System.IO.Path.GetFileNameWithoutExtension(p));
                                            string procH = baseName + "_proc_h.jpg";
                                            if (System.IO.File.Exists(procH))
                                            {
                                                bmp.Dispose();
                                                byte[] bytes = System.IO.File.ReadAllBytes(procH);
                                                using (var ms = new System.IO.MemoryStream(bytes))
                                                    return new Bitmap(ms);
                                            }
                                        }
                                        return bmp;
                                    };
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, procLoader,
                                        ridgeDirection: ridgeDir);
                                }
                                else
                                {
                                    Func<string, Bitmap> bmpLoader = inspSvc != null
                                        ? (Func<string, Bitmap>)(p => inspSvc.LoadBmpAtScale(p, scale))
                                        : null;
                                    imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, bmpLoader,
                                        useProcessed: enableProcess, ridgeDirection: ridgeDir);
                                }
                                CurveMergeHelper.MergeCurves(paths, out newCurveMean[i], out newCurveMax[i]);
                                CurveMergeHelper.MergeRowCurves(paths, out newRowCurveMean[i], out newRowCurveMax[i]);
                            }
                            catch (Exception ex)
                            {
                                Trace.WriteLine(
                                    $"[StitchView] CAM{camId}: {ex.GetType().Name}: {ex.Message}");
                            }
                        }
                    }
                    stitchMs = swStitch.ElapsedMilliseconds;
                    return imgs;
                });

                ClearStitchedMode();
                _stitchedImages       = newImages;
                _stitchedCurveMean    = newCurveMean;
                _stitchedCurveMax     = newCurveMax;
                _stitchedRowCurveMean = newRowCurveMean;
                _stitchedRowCurveMax  = newRowCurveMax;
                _currentGrabConfig = grabCfg;
                _ctx.InteractionHelper.ReviewConfig = grabCfg;
                _ctx.DataStatsPresenter?.SetReviewGroupBoxes(true);

                double[] opsArr = null, posArr = null;
                if (_ctx.Settings.StitchMode == StitchMode.Global)
                {
                    opsArr = grabCfg?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
                    posArr = grabCfg?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
                    _globalMergedImage = GrabImageStitcher.MergeHorizontal(
                        _stitchedImages, opsArr, posArr, InspectionEngineConfig.DefaultSaveResizeScale);
                }

                _ctx.GalleryManager.SetImages(_stitchedImages);

                if (_globalMergedImage != null)
                {
                    ShowMergedImageInCanvas(_globalMergedImage, opsArr, posArr);
                    // Global 模式：7 台 row curves 重疊合併
                    UpdateGlobalRowChart();
                }
                else
                {
                    ShowStitchedCameraInCanvas(_ctx.GalleryManager.SelectedIndex);
                }
                UpdateStitchedOverviewChart();

                Trace.WriteLine($"[StitchView] {grabId} proc={enableProcess} | CSV={csvMs}ms | Stitch={stitchMs}ms | Total={swTotal.ElapsedMilliseconds}ms");

                // Resource log
                int loadedCams = 0, finalW = 0, finalH = 0;
                for (int i = 0; i < newImages.Length; i++)
                {
                    if (newImages[i] != null)
                    {
                        loadedCams++;
                        if (finalW == 0) { finalW = newImages[i].Width; finalH = newImages[i].Height; }
                    }
                }
                string mode;
                if (_globalMergedImage != null)
                {
                    mode = "Global";
                    finalW = _globalMergedImage.Width;
                    finalH = _globalMergedImage.Height;
                }
                else
                {
                    mode = (totalImgCount > loadedCams) ? "Stitch" : "Single";
                }
                Core.Camera.CameraFrameSaver.AppendReviewResourceLog(mode, loadedCams, totalImgCount,
                    finalW, finalH, swTotal.ElapsedMilliseconds);
            }
            finally
            {
                _ctx.InteractionHelper.SetUiLoadingState(false);
            }
        }

        /// <summary>將合併圖顯示在 canvasMain 並啟用 chartOverview 聯動。</summary>
        public void ShowMergedImageInCanvas(Bitmap mergedImage, double[] opsArr, double[] posArr)
        {
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;
            _ctx.InteractionHelper.SetCanvasScaleAndCamera(scale, 0);
            EnableMergedOverviewSync(opsArr, posArr);
            _ctx.Canvas.Image = mergedImage;
            _ctx.InteractionHelper.RestoreCanvasViewOrFit();
        }

        public void EnableMergedOverviewSync(double[] opsArr, double[] posArr)
        {
            double globalMinMm = double.MaxValue;
            double refOpsUm = opsArr[0];
            for (int i = 0; i < opsArr.Length && i < _ctx.CameraCount; i++)
                if (posArr[i] < globalMinMm) globalMinMm = posArr[i];
            if (globalMinMm == double.MaxValue) globalMinMm = 0;

            _ctx.InteractionHelper.SetMergedMode(_ctx.OverviewHelper, globalMinMm, refOpsUm);
            if (_ctx.ChartOverview.ChartAreas.Count > 0)
                _ctx.ChartOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = true;
        }

        /// <summary>離開合圖模式：清除座標覆寫、停用互動 zoom。
        /// 不重設 ScaleView（ZoomReset）：避免 await 期間 message pump 渲染出全範圍閃爍，
        /// 由後續 UpdateDataAndView 原子性地取代資料與 zoom。</summary>
        public void DisableMergedOverviewSync()
        {
            _ctx.InteractionHelper.ClearMergedMode();
            if (_ctx.ChartOverview.ChartAreas.Count > 0)
            {
                _ctx.ChartOverview.ChartAreas[0].AxisX.ScaleView.Zoomable = false;
            }
        }

        public void ClearStitchedMode()
        {
            DisableMergedOverviewSync();
            if (_globalMergedImage != null)
            {
                if (_ctx.Canvas.Image == _globalMergedImage) _ctx.Canvas.Image = null;
                Widgets.BitmapPool.Return(_globalMergedImage);
                _globalMergedImage = null;
            }
            if (_periodMergedImage != null)
            {
                if (_ctx.Canvas.Image == _periodMergedImage) _ctx.Canvas.Image = null;
                Widgets.BitmapPool.Return(_periodMergedImage);
                _periodMergedImage = null;
            }
            if (_stitchedImages == null) return;
            _ctx.Canvas.Image = null;
            _ctx.GalleryManager.ClearImages();
            foreach (var bmp in _stitchedImages) Widgets.BitmapPool.Return(bmp);
            _stitchedImages = null;
            _stitchedCurveMean    = null;
            _stitchedCurveMax     = null;
            _stitchedRowCurveMean = null;
            _stitchedRowCurveMax  = null;
            _currentGrabConfig = null;
            _ctx.ColumnChartHelper?.SetOps(_ctx.Settings.Cam1_Ops);
            _ctx.ColumnChartHelper?.SetThresholds(_ctx.Settings.ErrorValueMean, _ctx.Settings.ErrorValueMax);
            _ctx.DataStatsPresenter?.SetReviewGroupBoxes(false);
        }

        /// <summary>Global 模式：7 台 row curves 重疊合併後更新法向曲線圖。</summary>
        private void UpdateGlobalRowChart()
        {
            if (_ctx.RowChartHelper == null || _stitchedRowCurveMean == null) return;
            CurveMergeHelper.MergeRowCurvesOverlap(
                _stitchedRowCurveMean, _stitchedRowCurveMax,
                _ctx.CameraCount, out float[] mergedMean, out float[] mergedMax);
            if (mergedMean != null)
            {
                _ctx.RowChartHelper.UpdateData(mergedMean, mergedMax);
                _ctx.InteractionHelper.RefreshRowChartRange();
            }
        }

        /// <summary>
        /// 合圖路徑：用 _stitchedCurveMean/Max 更新 chart1 全覽圖。
        /// </summary>
        public void UpdateStitchedOverviewChart()
        {
            if (_stitchedCurveMean == null) return;

            double[] opsArr, posArr;
            float errMean, errMax;
            if (_currentGrabConfig != null)
            {
                opsArr  = _currentGrabConfig.CamOps;
                posArr  = _currentGrabConfig.CamPos;
                errMean = _currentGrabConfig.ErrorValueMean;
                errMax  = _currentGrabConfig.ErrorValueMax;
            }
            else
            {
                opsArr  = _ctx.Settings.GetCameraOpsUmArray();
                posArr  = _ctx.Settings.GetCameraStartPositionMmArray();
                errMean = _ctx.Settings.ErrorValueMean;
                errMax  = _ctx.Settings.ErrorValueMax;
            }

            CurveMergeHelper.UpdateOverviewChart(_stitchedCurveMean, _stitchedCurveMax,
                opsArr, posArr, errMean, errMax,
                _ctx.OverviewHelper, _ctx.CameraCount, _ctx.Settings.StitchMode,
                ViewRangeProvider);
        }

        /// <summary>顯示單台相機拼接影像，並更新對應的 mura chart。</summary>
        public void ShowStitchedCameraInCanvas(int idx)
        {
            if (_stitchedImages == null) return;
            var bmp = (idx >= 0 && idx < _stitchedImages.Length) ? _stitchedImages[idx] : null;

            _ctx.InteractionHelper.SetCanvasScaleAndCamera(
                InspectionEngineConfig.DefaultSaveResizeScale, idx);

            _ctx.Canvas.Image = bmp;
            if (bmp != null) _ctx.InteractionHelper.RestoreCanvasViewOrFit();

            if (_ctx.Settings.StitchMode == StitchMode.Global)
            {
                // Global 模式：切向曲線圖清空（單台資料無意義），法向曲線圖正常載入
                _ctx.ChartMuraVertical.Series["Mean"].Points.Clear();
                _ctx.ChartMuraVertical.Series["Max"].Points.Clear();
            }
            else if (_ctx.ColumnChartHelper != null && _ctx.Settings != null)
            {
                // Vertical 模式：正常載入切向曲線
                float[] mean = (_stitchedCurveMean != null && idx >= 0 && idx < _stitchedCurveMean.Length)
                    ? _stitchedCurveMean[idx] : null;
                float[] max = (_stitchedCurveMax != null && idx >= 0 && idx < _stitchedCurveMax.Length)
                    ? _stitchedCurveMax[idx] : null;

                double[] posArr;
                if (_currentGrabConfig != null)
                {
                    double opsUm = (idx >= 0 && idx < _currentGrabConfig.CamOps.Length)
                        ? _currentGrabConfig.CamOps[idx] : _ctx.Settings.Cam1_Ops;
                    _ctx.ColumnChartHelper.SetOps(opsUm);
                    _ctx.ColumnChartHelper.SetThresholds(
                        _currentGrabConfig.ErrorValueMean, _currentGrabConfig.ErrorValueMax);
                    posArr = _currentGrabConfig.CamPos;
                }
                else
                {
                    posArr = _ctx.Settings.GetCameraStartPositionMmArray();
                }

                double startPos = (idx >= 0 && idx < posArr.Length) ? posArr[idx] : 0;
                _ctx.InteractionHelper.TryComputeCurrentViewRange(idx, out double leftMm, out double rightMm);
                _ctx.ColumnChartHelper.UpdateDataAndView(mean, max, startPos, leftMm, rightMm);
            }

            // 法向曲線圖
            if (_ctx.RowChartHelper != null)
            {
                if (_ctx.Settings.StitchMode == StitchMode.Global)
                {
                    // Global 模式：7 台 row curves 重疊合併
                    UpdateGlobalRowChart();
                }
                else
                {
                    // Vertical 模式：單台 row curves
                    float[] rowMean = (_stitchedRowCurveMean != null && idx >= 0 && idx < _stitchedRowCurveMean.Length)
                        ? _stitchedRowCurveMean[idx] : null;
                    float[] rowMax = (_stitchedRowCurveMax != null && idx >= 0 && idx < _stitchedRowCurveMax.Length)
                        ? _stitchedRowCurveMax[idx] : null;
                    if (rowMean != null)
                    {
                        _ctx.RowChartHelper.UpdateData(rowMean, rowMax);
                        _ctx.InteractionHelper.RefreshRowChartRange();
                    }
                }
            }
        }

        /// <summary>
        /// 原圖路徑（非 Stitch）：合併全域圖（Period 切換用）。
        /// </summary>
        public void ApplyGlobalMergeIfNeeded()
        {
            if (_ctx.Settings.StitchMode != StitchMode.Global) return;

            var cfg = _ctx.InteractionHelper.ReviewConfig;
            double[] opsArr = cfg?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] posArr = cfg?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;

            var filesMap = _ctx.ImageRepository.GetImages(
                _ctx.DateTimeNavigator.GetCurrentYear(),
                _ctx.DateTimeNavigator.GetCurrentMonth(),
                _ctx.DateTimeNavigator.GetCurrentDay(),
                _ctx.DateTimeNavigator.GetCurrentHour(),
                _ctx.DateTimeNavigator.GetCurrentMin(),
                _ctx.DateTimeNavigator.GetCurrentSec());
            if (filesMap == null || filesMap.Count == 0) return;

            Func<string, Bitmap> bmpLoader = _ctx.InspectionService != null
                ? (Func<string, Bitmap>)(p => _ctx.InspectionService.LoadBmpAtScale(p, scale))
                : null;

            int camCount = _ctx.CameraCount;
            var camImages = new Bitmap[camCount];
            for (int i = 0; i < camCount; i++)
            {
                if (filesMap.TryGetValue(i + 1, out string path))
                {
                    try
                    {
                        camImages[i] = GrabImageStitcher.LoadCameraImage(path, scale, bmpLoader,
                            useProcessed: LastReviewProcessedMode, ridgeDirection: ActiveRidgeDirection);
                    }
                    catch (Exception ex)
                    {
                        Trace.WriteLine($"[GlobalMerge] CAM{i + 1}: {ex.GetType().Name}: {ex.Message}");
                    }
                }
            }

            _periodMergedImage = GrabImageStitcher.MergeHorizontal(camImages, opsArr, posArr, scale);

            foreach (var img in camImages) img?.Dispose();

            if (_periodMergedImage != null)
                ShowMergedImageInCanvas(_periodMergedImage, opsArr, posArr);

            // Global 模式：切向曲線圖清空（單台資料無意義）
            _ctx.ChartMuraVertical.Series["Mean"].Points.Clear();
            _ctx.ChartMuraVertical.Series["Max"].Points.Clear();
        }

        /// <summary>
        /// 原圖路徑：從當前 Repository 時間點讀取 .bin 曲線更新 chartOverview 全覽圖。
        /// </summary>
        public void UpdateOverviewChartFromRepository()
        {
            if (_ctx.OverviewHelper == null || _stitchedImages != null) return;

            var images = _ctx.ImageRepository.GetImages(
                _ctx.DateTimeNavigator.GetCurrentYear(),
                _ctx.DateTimeNavigator.GetCurrentMonth(),
                _ctx.DateTimeNavigator.GetCurrentDay(),
                _ctx.DateTimeNavigator.GetCurrentHour(),
                _ctx.DateTimeNavigator.GetCurrentMin(),
                _ctx.DateTimeNavigator.GetCurrentSec());

            if (images == null || images.Count == 0)
            {
                _ctx.ChartOverview.Series["Mean"].Points.Clear();
                _ctx.ChartOverview.Series["Max"].Points.Clear();
                if (_ctx.ChartOverview.ChartAreas.Count > 0)
                    _ctx.ChartOverview.ChartAreas[0].AxisX.ScaleView.ZoomReset();
                return;
            }

            int camCount = _ctx.CameraCount;
            var curveMean = new float[camCount][];
            var curveMax  = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                if (!images.TryGetValue(i + 1, out string path)) continue;
                string basePath = CurveMergeHelper.GetCurveBasePath(path);
                curveMean[i] = InspectionEngine.LoadCurveBin(basePath + "_mean_v.bin")
                            ?? InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                curveMax[i]  = InspectionEngine.LoadCurveBin(basePath + "_max_v.bin")
                            ?? InspectionEngine.LoadCurveBin(basePath + "_max.bin");
            }

            var reviewCfg = _ctx.InteractionHelper?.ReviewConfig;
            if (reviewCfg != null)
            {
                CurveMergeHelper.UpdateOverviewChart(curveMean, curveMax,
                    reviewCfg.CamOps, reviewCfg.CamPos, reviewCfg.ErrorValueMean, reviewCfg.ErrorValueMax,
                    _ctx.OverviewHelper, camCount, _ctx.Settings.StitchMode, ViewRangeProvider);
            }
            else
            {
                CurveMergeHelper.UpdateOverviewChart(curveMean, curveMax,
                    _ctx.Settings.GetCameraOpsUmArray(), _ctx.Settings.GetCameraStartPositionMmArray(),
                    _ctx.Settings.ErrorValueMean, _ctx.Settings.ErrorValueMax,
                    _ctx.OverviewHelper, camCount, _ctx.Settings.StitchMode, ViewRangeProvider);
            }
        }

        private double ViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            if (_ctx.InteractionHelper == null) return defaultValue;
            if (!_ctx.InteractionHelper.TryComputeCurrentViewRange(cameraIndex, out double left, out double right))
                return defaultValue;
            return isLeft ? left : right;
        }
    }
}
