using System;
using System.Collections.Generic;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// Data tab「Mura 空間分布圖」（chartDataPatch / chartDataColumn）的繪圖職責 ——
    /// 從 DataStatisticsPresenter 提取（2026-06-30）。只管「這張缺陷分布圖怎麼畫」：
    /// 範圍/時間模式 aggregate 平均、單片模式用單一 grab .bin + #CFG（與 chartReviewColumn 對齊、套 view-time 正規值 rescale）、
    /// 設定變更重畫閾值線、從回顧同步資料、清空。
    ///
    /// 不擁有「現在哪個模式 / 有哪些 grab / 資料根目錄」這些核心狀態：透過注入的 Func 即時讀
    /// （DataStatisticsPresenter 仍是這些狀態的單一真相）。對外 public 門面（RefreshForSettingsChange /
    /// SyncFromReview）由 DataStatisticsPresenter 轉發，外部呼叫端零修改。
    /// </summary>
    public sealed class MuraProfileChartPresenter
    {
        private readonly DataStatisticsContext _ctx;
        private readonly Func<System.Windows.Forms.GroupBox> _getActiveStatMode;
        private readonly Func<List<GrabIdInfo>> _getGrabIdInfos;
        private readonly Func<string> _getStatsRoot;

        private ColumnCurveChartHelper _muraProfileHelper;

        public MuraProfileChartPresenter(
            DataStatisticsContext ctx,
            Func<System.Windows.Forms.GroupBox> getActiveStatMode,
            Func<List<GrabIdInfo>> getGrabIdInfos,
            Func<string> getStatsRoot)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getActiveStatMode = getActiveStatMode ?? throw new ArgumentNullException(nameof(getActiveStatMode));
            _getGrabIdInfos = getGrabIdInfos ?? throw new ArgumentNullException(nameof(getGrabIdInfos));
            _getStatsRoot = getStatsRoot ?? throw new ArgumentNullException(nameof(getStatsRoot));
        }

        public void Init()
        {
            if (_ctx.ChartDataPatch == null) return;
            _muraProfileHelper = new ColumnCurveChartHelper(_ctx.ChartDataPatch);
        }

        public void Update(IList<GrabIdInfo> grabIds)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;

            // 單片模式（GrpDataSingleSheet）：永遠用 cbDataId.SelectedIndex 對應 grab，不依賴 caller 傳入的 grabIds。
            // 原因：listViewGrabDetail 點選時 _suppressRangeOnSingleSheetSync=true 跳過範圍 cb 同步，
            // 但 caller 仍會用舊 cbDataIdStart/End 範圍呼這函式 → 若用 grabIds[0] 會顯示舊範圍的第一筆而非剛點的 grab。
            // view-time 正規值 rescale（HM_capture / HM_current）讓改 PropertyGrid 正規值時曲線坡度立即變化。
            var grabIdInfos = _getGrabIdInfos();
            if (_getActiveStatMode() == _ctx.GrpDataSingleSheet)
            {
                int singleIdx = _ctx.CbDataGrabId.SelectedIndex;
                if (singleIdx >= 0 && singleIdx < grabIdInfos.Count)
                    UpdateForSingleGrab(grabIdInfos[singleIdx]);
                else
                    Clear();
                return;
            }

            if (grabIds == null || grabIds.Count == 0)
            {
                Clear();
                return;
            }
            // 範圍/時間模式：aggregate 多 grab 平均，當作歷史快照不做 view-time rescale

            // ── 範圍/時間模式：舊 aggregate 邏輯 ──
            var (meanDict, maxDict) = InspectionStatisticsService.LoadAvgMuraProfile(
                _getStatsRoot(), grabIds);
            if (meanDict.Count == 0)
            {
                Clear();
                return;
            }
            int camCount = _ctx.CameraCount;
            var allMean = new float[camCount][];
            var allMax  = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                meanDict.TryGetValue(i + 1, out allMean[i]);
                maxDict.TryGetValue(i + 1, out allMax[i]);
            }
            CurveMergeHelper.UpdateOverviewChart(
                allMean, allMax,
                _ctx.Settings.GetCameraOpsUmArray(),
                _ctx.Settings.GetCameraStartPositionMmArray(),
                _ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV,
                _muraProfileHelper, camCount,
                StitchMode.Vertical, null);
        }

        /// <summary>
        /// 用單一 grab 的 .bin（MergeCurves 合多 capture）+ 該 grab 的 CSV #CFG OPS/Pos
        /// 更新 chartDataColumn，與 chartReviewColumn 完全對齊。不依賴 camReviewMain 是否載入。
        /// 套用 view-time 正規值 rescale：display = (bin/255) × (HM_capture / HM_current)；
        /// 改 PropertyGrid 正規值會立刻反映在曲線坡度上。
        /// </summary>
        private void UpdateForSingleGrab(GrabIdInfo info)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;
            string statsRoot = _getStatsRoot();
            if (string.IsNullOrWhiteSpace(statsRoot)) return;

            var grabCfg = InspectionStatisticsService.LoadConfigForGrabId(
                statsRoot, info.GrabId, info.Earliest, info.Latest);
            var grouped = InspectionStatisticsService.LoadImagePathsForGrabId(
                statsRoot, info.GrabId, info.Earliest, info.Latest);

            int camCount = _ctx.CameraCount;
            var allMean = new float[camCount][];
            var allMax  = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                int camId = i + 1;
                if (grouped.TryGetValue(camId, out var paths) && paths.Count > 0)
                    CurveMergeHelper.MergeCurves(paths, out allMean[i], out allMax[i]);
            }

            // view-time 正規值 rescale：chartDataColumn 是欄曲線，用 V 的 capture/current ratio
            float captureHm = grabCfg?.HessianMaxFactorV ?? _ctx.Settings.HessianMaxFactorV;
            HessianRescaleHelper.RescaleInPlace2D(allMean, allMax, captureHm, _ctx.Settings.HessianMaxFactorV);

            double[] ops = grabCfg?.CamOps  ?? _ctx.Settings.GetCameraOpsUmArray();
            double[] pos = grabCfg?.CamPos  ?? _ctx.Settings.GetCameraStartPositionMmArray();
            float errMean = _ctx.Settings.ErrorValueMeanV;  // view-time 閾值用當前 Settings
            float errMax  = _ctx.Settings.ErrorValueMaxV;

            CurveMergeHelper.UpdateOverviewChart(
                allMean, allMax, ops, pos, errMean, errMax,
                _muraProfileHelper, camCount,
                _ctx.Settings.StitchMode, null);
        }

        /// <summary>
        /// 由 PropertyGrid 變更觸發：刷新 chartDataColumn 的閾值線 + view-time 正規值 rescale。
        /// 不重做 RefreshStats（避免重算統計）；只重畫 chart。
        /// </summary>
        public void RefreshForSettingsChange()
        {
            if (_muraProfileHelper == null) return;
            _muraProfileHelper.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            // 單片模式才需要按 HM 重算曲線坡度；aggregate 模式維持快照
            var grabIdInfos = _getGrabIdInfos();
            if (_getActiveStatMode() == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < grabIdInfos.Count)
            {
                UpdateForSingleGrab(grabIdInfos[_ctx.CbDataGrabId.SelectedIndex]);
            }
        }

        /// <summary>
        /// SingleSheet 模式：直接使用 Review tab 已載入的曲線資料（已套 view-time HM rescale），
        /// 確保 chartDataColumn 與 chartReviewColumn 完全一致（相同 OPS/Pos 與顯示值）。
        /// </summary>
        public void SyncFromReview(float[][] mean, float[][] max,
            double[] ops, double[] pos, float errMean, float errMax)
        {
            if (_muraProfileHelper == null) return;
            CurveMergeHelper.UpdateOverviewChart(mean, max, ops, pos, errMean, errMax,
                _muraProfileHelper, _ctx.CameraCount, StitchMode.Vertical, null);
        }

        public void Clear()
        {
            if (_ctx.ChartDataPatch == null) return;
            foreach (var s in _ctx.ChartDataPatch.Series)
                s.Points.Clear();
        }
    }
}
