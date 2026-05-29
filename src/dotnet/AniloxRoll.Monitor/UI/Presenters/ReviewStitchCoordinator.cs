using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using TanukiCv.Controls;
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
        public Chart ChartReviewPatch { get; set; }
        public Chart ChartReviewVertical { get; set; }
        public Chart ChartReviewHorizontal { get; set; }
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

        /// <summary>
        /// UpdateStitchedOverviewChart 完成後觸發，傳遞與 chartReviewPatch 相同的曲線資料，
        /// 供外部（AniloxRollForm）同步 chartDataPatch。
        /// 參數：(mean[][], max[][], opsUm[], startPosMm[], errMean, errMax)
        /// </summary>
        public event Action<float[][], float[][], double[], double[], float, float> StitchedCurveUpdated;

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
            // L2 SSoT：setting 由 caller 透過 SettingsHub 設置，coordinator 不再 bypass Hub 直接寫 memory。
            // caller 路徑：PropertyGrid → OnSettingChanged → ApplyReviewEnhance → ReloadCurrentStitchedView；
            //              chart click → Hub.Set(hd_EnableReviewEnhance) → 同上。
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
                                imgs[i] = GrabImageStitcher.StitchCamera(paths, scale, null,
                                    useProcessed: enableProcess, ridgeDirection: ridgeDir);
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
                    ShowStitchedCameraInCanvas(_ctx.GalleryManager.SelectedIndex, resetView: false);
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

        /// <summary>將合併圖顯示在 camReviewMain 並啟用 chartReviewPatch 聯動。</summary>
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
            if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
                _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.Zoomable = true;
        }

        /// <summary>Form 關閉時控制項已 disposed → 這幾條 cleanup 的 UI 操作應 no-op：
        /// Form 自身會清控制項與資源，且關程式時 fire-and-forget 的 StitchMode 切換 async
        /// 可能續跑碰到已 disposed 的 chartReviewPatch/canvas → NullReferenceException。</summary>
        private bool UiDisposed =>
            (_ctx?.ChartReviewPatch?.IsDisposed ?? true) || (_ctx?.Canvas?.IsDisposed ?? true);

        /// <summary>離開合圖模式：清除座標覆寫、停用互動 zoom。
        /// 不重設 ScaleView（ZoomReset）：避免 await 期間 message pump 渲染出全範圍閃爍，
        /// 由後續 UpdateDataAndView 原子性地取代資料與 zoom。</summary>
        public void DisableMergedOverviewSync()
        {
            if (UiDisposed) return;
            _ctx.InteractionHelper.ClearMergedMode();
            if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
            {
                _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.Zoomable = false;
            }
        }

        /// <summary>切到 Vertical 時清掉殘留的 Global/Period 合圖 bitmap（保留 _stitchedImages 與曲線）。</summary>
        public void DisposeGlobalMergedImage()
        {
            if (UiDisposed) return;
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
        }

        public void ClearStitchedMode()
        {
            if (UiDisposed) return;
            DisposeGlobalMergedImage();
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
            _ctx.ColumnChartHelper?.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            _ctx.DataStatsPresenter?.SetReviewGroupBoxes(false);
        }

        /// <summary>Global 模式：7 台 row curves 重疊合併後更新法向曲線圖。
        /// row chart 是水平 (row) 曲線 → 用 (HM_V_capture / HM_H_current) ratio rescale，
        /// 讓 PropertyGrid 改水平正規值時 H 曲線坡度立即變化。
        /// 公式：bin baked-in 的縮放是 HM_V_capture（native 只用單一 HM=V），
        /// view-time 想要的目標是 HM_H_current，所以 ratio = HM_V_capture / HM_H_current。</summary>
        private void UpdateGlobalRowChart()
        {
            if (_ctx.RowChartHelper == null || _stitchedRowCurveMean == null) return;
            CurveMergeHelper.MergeRowCurvesOverlap(
                _stitchedRowCurveMean, _stitchedRowCurveMax,
                _ctx.CameraCount, out float[] mergedMean, out float[] mergedMax);
            if (mergedMean != null)
            {
                float captureHmV = _currentGrabConfig?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
                HessianRescaleHelper.RescaleInPlace1D(mergedMean, captureHmV, _ctx.Settings.HessianMaxFactorH);
                HessianRescaleHelper.RescaleInPlace1D(mergedMax,  captureHmV, _ctx.Settings.HessianMaxFactorH);
                _ctx.RowChartHelper.UpdateData(mergedMean, mergedMax);
                _ctx.InteractionHelper.RefreshRowChartRange();
            }
        }

        /// <summary>
        /// 合圖路徑：用 _stitchedCurveMean/Max 更新 chart1 全覽圖。
        /// 套用 view-time 正規值 rescale + 當前閾值：
        ///   - 曲線：(bin/255) × (HM_capture / HM_current) ← 改 PropertyGrid 正規值會立即反映坡度
        ///   - 閾值線：用 _ctx.Settings 的當前 ErrorValueMeanV/MaxV ← 改 PropertyGrid 閾值會立即移動門檻線
        ///   - OPS/Pos：用該 grab 的 #CFG 快照（與資料一起 baked-in，不可後驗調整）
        /// </summary>
        public void UpdateStitchedOverviewChart()
        {
            if (_stitchedCurveMean == null) return;

            double[] opsArr, posArr;
            float captureHm;
            if (_currentGrabConfig != null)
            {
                opsArr    = _currentGrabConfig.CamOps;
                posArr    = _currentGrabConfig.CamPos;
                captureHm = _currentGrabConfig.HessianMaxFactorV;
            }
            else
            {
                opsArr    = _ctx.Settings.GetCameraOpsUmArray();
                posArr    = _ctx.Settings.GetCameraStartPositionMmArray();
                captureHm = _ctx.Settings.HessianMaxFactorV;
            }
            // 閾值固定用當前 Settings（view-time 可調），不再從 _currentGrabConfig 取
            float errMean = _ctx.Settings.ErrorValueMeanV;
            float errMax  = _ctx.Settings.ErrorValueMaxV;

            // chartReviewPatch 是垂直 (column) 曲線 → 用 V 的 capture/current ratio
            var displayMean = HessianRescaleHelper.CloneAndRescale2D(_stitchedCurveMean, captureHm, _ctx.Settings.HessianMaxFactorV);
            var displayMax  = HessianRescaleHelper.CloneAndRescale2D(_stitchedCurveMax,  captureHm, _ctx.Settings.HessianMaxFactorV);

            CurveMergeHelper.UpdateOverviewChart(displayMean, displayMax,
                opsArr, posArr, errMean, errMax,
                _ctx.OverviewHelper, _ctx.CameraCount, _ctx.Settings.StitchMode,
                ViewRangeProvider);

            StitchedCurveUpdated?.Invoke(displayMean, displayMax, opsArr, posArr, errMean, errMax);
        }


        /// <summary>顯示單台相機拼接影像，並更新對應的 mura chart。
        /// resetView=true（camReview 切換相機）：Vertical 模式強制 fit to screen。
        /// resetView=false（強化方向重載）：尊重呼叫端 SaveCanvasView 的視野存檔。</summary>
        public void ShowStitchedCameraInCanvas(int idx, bool resetView = true)
        {
            if (_stitchedImages == null) return;
            var bmp = (idx >= 0 && idx < _stitchedImages.Length) ? _stitchedImages[idx] : null;

            _ctx.InteractionHelper.SetCanvasScaleAndCamera(
                InspectionEngineConfig.DefaultSaveResizeScale, idx);

            _ctx.Canvas.Image = bmp;
            if (bmp != null)
            {
                if (resetView && _ctx.Settings.StitchMode == StitchMode.Vertical)
                    _ctx.InteractionHelper.ClearCanvasView();
                _ctx.InteractionHelper.RestoreCanvasViewOrFit();
            }

            UpdatePerCameraCharts(idx);
        }

        /// <summary>
        /// 更新單台相機的 chartReviewVertical（V）+ chartReviewHorizontal（H）。
        /// 套用 view-time 正規值 rescale：
        ///   - V 曲線：(bin/255) × (HM_V_capture / HM_V_current) → 改 PropertyGrid 垂直正規值生效
        ///   - H 曲線：(bin/255) × (HM_V_capture / HM_H_current) → 改 PropertyGrid 水平正規值生效
        /// 閾值線用當前 Settings（view-time tunable）。
        /// </summary>
        public void UpdatePerCameraCharts(int idx)
        {
            if (_stitchedImages == null) return;

            // 切向 (Column / V)
            if (_ctx.Settings.StitchMode == StitchMode.Global)
            {
                // Global 模式：單台切向資料無意義，清空
                if (_ctx.ChartReviewVertical != null)
                {
                    _ctx.ChartReviewVertical.Series["Mean"].Points.Clear();
                    _ctx.ChartReviewVertical.Series["Max"].Points.Clear();
                }
            }
            else if (_ctx.ColumnChartHelper != null && _ctx.Settings != null)
            {
                float[] mean = (_stitchedCurveMean != null && idx >= 0 && idx < _stitchedCurveMean.Length)
                    ? _stitchedCurveMean[idx] : null;
                float[] max = (_stitchedCurveMax != null && idx >= 0 && idx < _stitchedCurveMax.Length)
                    ? _stitchedCurveMax[idx] : null;

                double[] posArr;
                float captureHmV;
                if (_currentGrabConfig != null)
                {
                    double opsUm = (idx >= 0 && idx < _currentGrabConfig.CamOps.Length)
                        ? _currentGrabConfig.CamOps[idx] : _ctx.Settings.Cam1_Ops;
                    _ctx.ColumnChartHelper.SetOps(opsUm);
                    posArr = _currentGrabConfig.CamPos;
                    captureHmV = _currentGrabConfig.HessianMaxFactorV;
                }
                else
                {
                    posArr = _ctx.Settings.GetCameraStartPositionMmArray();
                    captureHmV = _ctx.Settings.HessianMaxFactorV;
                }
                // 閾值固定用當前 Settings（view-time tunable）
                _ctx.ColumnChartHelper.SetThresholds(
                    _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);

                var displayMean = HessianRescaleHelper.CloneAndRescale1D(mean, captureHmV, _ctx.Settings.HessianMaxFactorV);
                var displayMax  = HessianRescaleHelper.CloneAndRescale1D(max,  captureHmV, _ctx.Settings.HessianMaxFactorV);

                double startPos = (idx >= 0 && idx < posArr.Length) ? posArr[idx] : 0;
                _ctx.InteractionHelper.TryComputeCurrentViewRange(idx, out double leftMm, out double rightMm);
                _ctx.ColumnChartHelper.UpdateDataAndView(displayMean, displayMax, startPos, leftMm, rightMm);
            }

            // 法向 (Row / H)
            if (_ctx.RowChartHelper != null)
            {
                if (_ctx.Settings.StitchMode == StitchMode.Global)
                {
                    UpdateGlobalRowChart();
                }
                else
                {
                    float[] rowMean = (_stitchedRowCurveMean != null && idx >= 0 && idx < _stitchedRowCurveMean.Length)
                        ? _stitchedRowCurveMean[idx] : null;
                    float[] rowMax = (_stitchedRowCurveMax != null && idx >= 0 && idx < _stitchedRowCurveMax.Length)
                        ? _stitchedRowCurveMax[idx] : null;
                    if (rowMean != null)
                    {
                        float captureHmV = _currentGrabConfig?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
                        var displayMean = HessianRescaleHelper.CloneAndRescale1D(rowMean, captureHmV, _ctx.Settings.HessianMaxFactorH);
                        var displayMax  = HessianRescaleHelper.CloneAndRescale1D(rowMax,  captureHmV, _ctx.Settings.HessianMaxFactorH);
                        _ctx.RowChartHelper.UpdateData(displayMean, displayMax);
                        _ctx.InteractionHelper.RefreshRowChartRange();
                    }
                }
            }
        }

        /// <summary>
        /// 由 PropertyGrid 變更觸發：重畫當前選中相機的 V/H per-camera charts。
        /// 不重設 canvas view（避免使用者 zoom/pan 被打斷）。
        /// </summary>
        public void RefreshCurrentCameraChartsForSettingsChange()
        {
            if (_stitchedImages == null) return;
            int idx = _ctx.GalleryManager.SelectedIndex;
            if (idx < 0) idx = 0;
            UpdatePerCameraCharts(idx);
        }

        /// <summary>
        /// Stitch 模式（cbReviewId 已載入）切換到 Global：
        /// 直接用記憶體中的 _stitchedImages 合圖，不重新讀碟。
        /// </summary>
        public void MergeAndShowFromStitchedImages()
        {
            if (_stitchedImages == null) return;
            var cfg = _ctx.InteractionHelper.ReviewConfig;
            double[] opsArr = cfg?.CamOps ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] posArr = cfg?.CamPos ?? _ctx.Settings.GetCameraStartPositionMmArray();
            int scale = InspectionEngineConfig.DefaultSaveResizeScale;

            _globalMergedImage?.Dispose();
            _globalMergedImage = GrabImageStitcher.MergeHorizontal(_stitchedImages, opsArr, posArr, scale);
            if (_globalMergedImage != null)
                ShowMergedImageInCanvas(_globalMergedImage, opsArr, posArr);

            _ctx.ChartReviewVertical.Series["Mean"].Points.Clear();
            _ctx.ChartReviewVertical.Series["Max"].Points.Clear();
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
            _ctx.ChartReviewVertical.Series["Mean"].Points.Clear();
            _ctx.ChartReviewVertical.Series["Max"].Points.Clear();
        }

        /// <summary>
        /// 原圖路徑：從當前 Repository 時間點讀取 .bin 曲線更新 chartReviewPatch 全覽圖。
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
                _ctx.ChartReviewPatch.Series["Mean"].Points.Clear();
                _ctx.ChartReviewPatch.Series["Max"].Points.Clear();
                if (_ctx.ChartReviewPatch.ChartAreas.Count > 0)
                    _ctx.ChartReviewPatch.ChartAreas[0].AxisX.ScaleView.ZoomReset();
                return;
            }

            int camCount = _ctx.CameraCount;
            var curveMean = new float[camCount][];
            var curveMax  = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                if (!images.TryGetValue(i + 1, out string path)) continue;
                string basePath = CurveMergeHelper.GetCurveBasePath(path);
                curveMean[i] = InspectionEngine.LoadCurveBin(basePath + CaptureFileNaming.MeanV)
                            ?? InspectionEngine.LoadCurveBin(basePath + CaptureFileNaming.MeanVLegacy);
                curveMax[i]  = InspectionEngine.LoadCurveBin(basePath + CaptureFileNaming.MaxV)
                            ?? InspectionEngine.LoadCurveBin(basePath + CaptureFileNaming.MaxVLegacy);
            }

            var reviewCfg = _ctx.InteractionHelper?.ReviewConfig;
            if (reviewCfg != null)
            {
                CurveMergeHelper.UpdateOverviewChart(curveMean, curveMax,
                    reviewCfg.CamOps, reviewCfg.CamPos, reviewCfg.ErrorValueMeanV, reviewCfg.ErrorValueMaxV,
                    _ctx.OverviewHelper, camCount, _ctx.Settings.StitchMode, ViewRangeProvider);
            }
            else
            {
                CurveMergeHelper.UpdateOverviewChart(curveMean, curveMax,
                    _ctx.Settings.GetCameraOpsUmArray(), _ctx.Settings.GetCameraStartPositionMmArray(),
                    _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV,
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
