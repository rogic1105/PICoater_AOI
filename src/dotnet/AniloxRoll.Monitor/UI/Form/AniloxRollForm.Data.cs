using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using LightBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Binders;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>AniloxRollForm 檢測數據 Tab 相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- 檢測數據 Tab ---
        // ==========================================

        private void ApplyReviewViewRangeToCharts(
            double leftMm, double rightMm, double topMm, double bottomMm,
            bool prepared)
        {
            _reviewViewLeftMm = leftMm;
            _reviewViewRightMm = rightMm;
            _reviewViewTopMm = topMm;
            _reviewViewBotMm = bottomMm;

            var swSync = System.Diagnostics.Stopwatch.StartNew();
            if (prepared)
            {
                _reviewOverviewHelper?.UpdateViewRangeImmediate(leftMm, rightMm);
                _dataStatsPresenter?.SetPreparedReviewViewRange(leftMm, rightMm, topMm, bottomMm);
            }
            else
            {
                _reviewOverviewHelper?.UpdateViewRange(leftMm, rightMm);
                _dataStatsPresenter?.SetReviewViewRange(leftMm, rightMm, topMm, bottomMm);
            }
            long ovMs = swSync.ElapsedMilliseconds;
            if (prepared)
                _reviewRowSync?.SetPreparedViewRange(topMm, bottomMm);
            else
                _reviewRowSync?.SetViewRange(topMm, bottomMm);
            long rowMs = swSync.ElapsedMilliseconds - ovMs;

            _reviewSyncCount++;
            _reviewSyncOvMax = Math.Max(_reviewSyncOvMax, ovMs);
            _reviewSyncRowMax = Math.Max(_reviewSyncRowMax, rowMs);
            long totalMs = swSync.ElapsedMilliseconds;
            if (totalMs > 25)
                Trace.WriteLine($"[ReviewSync] SLOW ov={ovMs}ms row={rowMs}ms");
            if (_reviewSyncCount >= 120)
            {
                Trace.WriteLine($"[ReviewSync] 120 events: ovMax={_reviewSyncOvMax}ms rowMax={_reviewSyncRowMax}ms");
                _reviewSyncCount = 0;
                _reviewSyncOvMax = 0;
                _reviewSyncRowMax = 0;
            }
        }

        private ImageViewRange? ComputeReviewFitViewRange(
            int[] widths, int[] heights, double[] opsUm, double[] positionsMm,
            bool isGlobal, double rowPitchMm, double trimHeadMm, double trimTailMm)
        {
            if (_reviewDisplayManager == null) return null;
            if (!_reviewDisplayManager.TryComputeFitViewRange(
                widths, heights, opsUm, positionsMm, isGlobal,
                InspectionEngineConfig.DefaultSaveResizeScale,
                rowPitchMm, ShouldFlipDisplayVertical(),
                trimHeadMm, trimTailMm, out ImageViewRange range))
                return null;
            return range;
        }

        private void BeginReviewPrefitProbe(string grabId)
        {
            _reviewPrefitGeneration++;
            _reviewPrefitGrabId = grabId ?? "";
            _reviewPrefitStartTick = Environment.TickCount;
        }

        private void LogReviewPrefitApplied()
        {
            if (_reviewPrefitGeneration <= 0) return;
            FlowTrace.Dvt(
                $"RV prefitApply {_reviewPrefitGrabId} after={PrefitElapsedMs()}ms " +
                $"visible={tabMain.SelectedTab == tabPageReview} " +
                $"col={DescribeReviewAxis(chartReviewColumn, isRow: false)} " +
                $"row={DescribeReviewAxis(chartReviewRow, isRow: true)}");
        }

        private void LogReviewPrefitPaint(bool isRow)
        {
            int generation = _reviewPrefitGeneration;
            if (generation <= 0) return;
            if (isRow)
            {
                if (_reviewPrefitRowPaintGeneration == generation) return;
                _reviewPrefitRowPaintGeneration = generation;
            }
            else
            {
                if (_reviewPrefitColumnPaintGeneration == generation) return;
                _reviewPrefitColumnPaintGeneration = generation;
            }

            var chart = isRow ? chartReviewRow : chartReviewColumn;
            FlowTrace.Dvt(
                $"RV prefitPaint {_reviewPrefitGrabId} chart={(isRow ? "row" : "col")} " +
                $"after={PrefitElapsedMs()}ms {DescribeReviewAxis(chart, isRow)}");
        }

        private void LogReviewMainRange(double leftMm, double rightMm, double topMm, double bottomMm)
        {
            string state = $"viewX={leftMm:F2}~{rightMm:F2} viewY={topMm:F2}~{bottomMm:F2}";
            if (string.Equals(_reviewLastMainRangeState, state, StringComparison.Ordinal)) return;
            _reviewLastMainRangeState = state;
            FlowTrace.Dvt($"RV mainRange {CurrentReviewRangeGrabId()} {state}");
        }

        private void LogReviewChartPaint(bool isRow)
        {
            LogReviewPrefitPaint(isRow);

            var chart = isRow ? chartReviewRow : chartReviewColumn;
            string state = DescribeReviewAxis(chart, isRow);
            string previous = isRow ? _reviewLastRowRangeState : _reviewLastColumnRangeState;
            if (string.Equals(previous, state, StringComparison.Ordinal)) return;
            if (isRow)
                _reviewLastRowRangeState = state;
            else
                _reviewLastColumnRangeState = state;
            FlowTrace.Dvt(
                $"RV chartRange {CurrentReviewRangeGrabId()} chart={(isRow ? "row" : "col")} {state}");
        }

        private string CurrentReviewRangeGrabId()
            => string.IsNullOrWhiteSpace(_reviewPrefitGrabId) ? "-" : _reviewPrefitGrabId;

        private int PrefitElapsedMs()
            => unchecked(Environment.TickCount - _reviewPrefitStartTick);

        private static string DescribeReviewAxis(
            System.Windows.Forms.DataVisualization.Charting.Chart chart, bool isRow)
        {
            if (chart == null || chart.IsDisposed || chart.ChartAreas.Count == 0)
                return "unavailable";
            var axis = isRow
                ? chart.ChartAreas[0].AxisY
                : chart.ChartAreas[0].AxisX;
            return $"axis={axis.Minimum:F2}~{axis.Maximum:F2}/view={axis.ScaleView.ViewMinimum:F2}~{axis.ScaleView.ViewMaximum:F2}";
        }

        /// <summary>檢測報表 Y 軸 setting（gb 模式 / gc/gd/ge 各週期 YMax）→ 套到 Data 統計 charts。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入。）</summary>
        private void HandleChartScaleSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.gb_ChartScaleMode))
                _dataStatsPresenter.ApplyChartScaleFromSettings();
            else if (name == nameof(InspectionSettings.gc_YearlyYMax))
                _dataStatsPresenter.ApplyChartScaleForChart("Yearly");
            else if (name == nameof(InspectionSettings.gd_MonthlyYMax))
                _dataStatsPresenter.ApplyChartScaleForChart("Monthly");
            else if (name == nameof(InspectionSettings.ge_DailyYMax))
                _dataStatsPresenter.ApplyChartScaleForChart("Daily");
        }

        /// <summary>檢測參數變更 → 重畫 chartDataColumn 曲線 + 重算統計。
        /// **只在真正影響 Data 曲線/Pass-Fail 的參數才跑**（正規值 V/H、檢出方向、V/H 平均/最大閾值）；
        /// 其餘設定（IO/光源/儲存/DO_MURA 暫停/主畫面顯示/LOD/合圖…）不動 Data 曲線，避免無關設定
        /// 觸發 chartDataColumn reload+重綁造成「閃一下再復原」。細線濾除(de_RidgeSigma)是 capture-time、
        /// 不改已存 .bin，故不列入。（Wave3：Data 副作用從 OnSettingChanged 共用區搬入本 feature handler。）</summary>
        private void HandleDataStatsSettingsChanged(string name)
        {
            switch (name)
            {
                case nameof(InspectionSettings.dc_HessianMaxFactorV):
                case nameof(InspectionSettings.dd_HessianMaxFactorH):
                case nameof(InspectionSettings.eb_RidgeDir):
                case nameof(InspectionSettings.eca_ColumnCurveMode):
                case nameof(InspectionSettings.ec_ErrorValueMeanV):
                case nameof(InspectionSettings.ed_ErrorValueMaxV):
                case nameof(InspectionSettings.ee_ErrorValueMeanH):
                case nameof(InspectionSettings.ef_ErrorValueMaxH):
                    _dataStatsPresenter?.RefreshMuraProfileForSettingsChange(name); // 記憶體內重算，保留圖表座標
                    ScheduleStatsRefresh();                                      // debounce 重算 Pass/Fail 統計 + 明細
                    break;
            }
        }

        private void SetupDataTab()
        {
            _dataStatsPresenter = new DataStatisticsPresenter(new DataStatisticsContext
            {
                CbGrabIdStart = cbDataIdStart, CbGrabIdEnd = cbDataIdEnd,
                CbDataGrabId = cbDataId, CbReviewGrabId = cbReviewId,
                BtnSelectDataFolder = btnDataSelectFolder, BtnShowFail = btnDataShowFail,
                GroupBoxGrabIdRange = groupBoxGrabIdRange, GrpDataSingleSheet = grpDataSingleSheet,
                GrpReviewGrabNav = grpReviewGrabNav, GrpReviewTimePeriod = grpReviewTimePeriod,
                GrabDetailList = new GrabDetailListBinder(listViewGrabDetail, CameraCount),
                PanelStatCams = new[] { camData1, camData2, camData3,
                                        camData4, camData5, camData6, camData7 },
                PanelStatRow = camDataRow,
                ChartDataPatch = chartDataColumn,
                ChartDataRow = chartDataRow,
                ChartDataYieldYearly = chartDataYieldYearly, ChartDataYieldMonthly = chartDataYieldMonthly, ChartDataYieldDaily = chartDataYieldDaily,
                CbChartYear = cbDataYieldYear, CbChartMonth = cbDataYieldMonth, CbChartDay = cbDataYieldDay,
                LblChartNavYear = lblChartNavYear, LblChartNavMonth = lblChartNavMonth, LblChartNavDay = lblChartNavDay,
                Settings = _settings, CameraCount = CameraCount,
                ReviewViewRangeProvider = () => double.IsNaN(_reviewViewLeftMm)
                    ? null
                    : new[] { _reviewViewLeftMm, _reviewViewRightMm, _reviewViewTopMm, _reviewViewBotMm },
                ReviewFitViewRangeProvider = ComputeReviewFitViewRange,
            });

            // 年/月/日 label 做成「看起來可點」的浮雕小晶片（Fixed3D 外框 + 手指游標）；點擊行為由 navigator 接
            foreach (var lbl in new[] { lblChartNavYear, lblChartNavMonth, lblChartNavDay })
            {
                lbl.BorderStyle = BorderStyle.Fixed3D;
                lbl.Cursor = Cursors.Hand;
                lbl.Padding = new Padding(6, 2, 6, 2);
                lbl.TextAlign = ContentAlignment.MiddleCenter;
                lbl.AccessibleName = lbl.Name;
            }

            _dataStatsPresenter.Initialize();

            // 延遲注入：_stitchCoordinator 在 InitUiLayer 初始化時 _dataStatsPresenter 尚未建立
            _stitchCoordinator.SetDataStatsPresenter(_dataStatsPresenter);
            _dataStatsPresenter.SingleGrabCurvePresented +=
                _stitchCoordinator.CacheDataCurveSnapshot;

            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）——僅用於升序排列的 ComboBox
            foreach (var cb in new[] { cbDataYieldYear, cbDataYieldMonth, cbDataYieldDay })
                _wheelInterceptors.Add(new ComboBoxWheelReverser(cb));

            // 跨 Tab 事件
            _dataStatsPresenter.GrabIdSelectedFromData += OnDataGrabIdSelected;
            _dataStatsPresenter.GrabIdSelectedFromReview += OnReviewGrabIdSelected;
            _dataStatsPresenter.PeriodComboManualChanged += OnPeriodComboChanged;
            _dataStatsPresenter.DataFolderSelected += OnDataFolderSelected;

            // Data → Review tab 切換：pending selection 才載圖；既有內容一律補一次可見重繪。
            tabMain.SelectedIndexChanged += async (s, e) =>
            {
                if (_suppressTabIntent) return;
                if (tabMain.SelectedTab == tabPageData)
                {
                    try
                    {
                        await _dataStatsPresenter.EnsureReportGrabIdCombosAsync();
                    }
                    catch (Exception ex)
                    {
                        Trace.WriteLine($"[tabMain -> Data] {ex}");
                    }
                    return;
                }
                if (tabMain.SelectedTab != tabPageReview) return;

                if (_reviewDirty && _hasPendingDataReviewSelection)
                {
                    var pending = _pendingDataReviewSelection;
                    int idx = pending.idx;
                    if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count ||
                        !string.Equals(_dataStatsPresenter.GrabIdInfos[idx].GrabId, pending.grabId, StringComparison.Ordinal))
                    {
                        idx = _dataStatsPresenter.GrabIdInfos.FindIndex(
                            candidate => string.Equals(candidate.GrabId, pending.grabId, StringComparison.Ordinal));
                    }
                    if (idx >= 0 && idx < _dataStatsPresenter.GrabIdInfos.Count)
                    {
                        var info = _dataStatsPresenter.GrabIdInfos[idx];
                        try
                        {
                            using (_dataStatsPresenter.GrabIdCrossGuard.Enter())
                            using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                            {
                                cbReviewId.SelectedIndex = idx;
                                _dateTimeNavigator.SetPeriodToCombo(info.Earliest);
                            }
                            _presenter.UpdatePeriodNavigationState();
                            _dataStatsPresenter.UpdateGrabIdNavState();
                            _dataStatsPresenter.SetReviewGroupBoxes(true);
                            _reviewDirty = false;
                            _hasPendingDataReviewSelection = false;
                            FlowTrace.Log($"DT review sync apply {info.GrabId}");
                            await LoadGrabStitchedViewGuardRowRangeAsync(
                                info.GrabId, info.Earliest, info.Latest,
                                _stitchCoordinator.LastReviewProcessedMode,
                                ReviewContentLoadMode.ReuseSharedCurves);
                            // 2b-ii：fit 由 ImageDisplayView 首幀自動 fit 承接
                        }
                        catch (Exception ex)
                        {
                            _reviewDirty = true;
                            Trace.WriteLine($"[tabMain → Review] {ex}");
                        }
                    }
                }

                // SelectedIndexChanged 發生時子控制項的可見 layout 尚未保證完成；延後一個 UI message，
                // 再以目前尺寸補 LOD tile + paint。只補顯示，不重讀檔或重設視野。
                if (IsHandleCreated && !IsDisposed && !Disposing)
                {
                    try
                    {
                        BeginInvoke(new Action(() =>
                        {
                            if (tabMain.SelectedTab == tabPageReview)
                                _reviewDisplayManager?.RefreshVisible();
                        }));
                    }
                    catch (InvalidOperationException) { /* Form 正在關閉 */ }
                }
            };
        }

        private (string grabId, int idx) _pendingDataReviewSelection;
        private bool _hasPendingDataReviewSelection;

        private void OnDataGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                _pendingDataReviewSelection = (grabId, idx);
                _hasPendingDataReviewSelection = true;
                _reviewDirty = true;
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataGrabIdSelected] {ex}"); }
        }

        // 快速滾序號的載入 debounce（250ms）：滾動中只送 latest-only 曲線；
        // 停頓才同步日期/時間、載入最後一張圖片、儲存 session、同步 Data tab。
        private Timer _reviewLoadDebounce;
        private (string grabId, DateTime earliest, DateTime latest, int idx, int sequence, int direction) _pendingReviewLoad;
        private int _reviewSelectionSeq;
        private int _lastReviewSelectionIndex = -1;

        private void OnReviewGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                int selectionSeq = ++_reviewSelectionSeq;
                int direction = _lastReviewSelectionIndex < 0
                    ? 0
                    : Math.Sign(idx - _lastReviewSelectionIndex);
                _lastReviewSelectionIndex = idx;
                // Keep the serialized thumbnail lane alive: while it is busy, intermediate
                // selections collapse into its latest pending request. Only the debounced
                // full-resolution load must be invalidated immediately.
                _stitchCoordinator.InvalidateSettledImageLoad();

                // 分層載入：曲線（輕）即時跟滾動 → 使用者快速掃 chart 找異常；影像（重）settle 才載。
                _ = _stitchCoordinator.LoadGrabCurvesOnlyAsync(grabId, earliest, latest);
                _ = _stitchCoordinator.LoadGrabThumbnailAsync(grabId, earliest, latest);

                _pendingReviewLoad = (
                    grabId, earliest, latest, idx, selectionSeq, direction);
                if (_reviewLoadDebounce == null)
                {
                    _reviewLoadDebounce = new Timer { Interval = 250 };
                    _reviewLoadDebounce.Tick += async (s2, e2) =>
                    {
                        _reviewLoadDebounce.Stop();
                        var p = _pendingReviewLoad;
                        // ComboBox.Items.Contains/SelectedItem 會線性搜尋當日時間清單；
                        // 大量序號下不可逐格執行，只在 latest selection settle 後同步一次。
                        using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                            _dateTimeNavigator.SetPeriodToCombo(p.earliest);
                        _presenter.UpdatePeriodNavigationState();
                        // session 只在使用者停下後落盤一次。
                        _dateTimeNavigator.SaveCurrentSelection();
                        try
                        {
                            await LoadGrabStitchedViewGuardRowRangeAsync(p.grabId, p.earliest, p.latest);
                            if (IsDisposed || Disposing ||
                                p.sequence != _reviewSelectionSeq) return;
                            _reviewDirty = false;
                            _stitchCoordinator.BeginAdjacentPrefetch(
                                _dataStatsPresenter.GrabIdInfos,
                                p.idx,
                                p.direction);
                        }
                        catch (Exception ex) { Trace.WriteLine($"[ReviewLoadDebounce] {ex}"); }
                        // Data tab 同步排在影像之後：統計全重算（掃目錄+CSV 解析+Mura 圖 bin IO）
                        // 是 UI 執行緒重活——settle 才做一次，且使用者先看到影像。
                        // （原在每格序號 inline 跑、註解自稱「便宜」→ 快撥 18 格＝18 輪重算
                        //   ＝UiStall 5.7s＋曲線快路全餓死，2026-07-10 log 定罪）
                        if (!IsDisposed && !Disposing &&
                            p.sequence == _reviewSelectionSeq)
                            SyncDataTabFromReviewSettled(p.idx);
                    };
                }
                _reviewLoadDebounce.Stop();
                _reviewLoadDebounce.Start();   // 重壓計時：每次選取重新等 250ms，停下才載入最終選取
            }
            catch (Exception ex) { Trace.WriteLine($"[OnReviewGrabIdSelected] {ex}"); }
        }

        /// <summary>Review 序號 settle 後同步 Data tab（統計/Mura 圖重算的唯一觸發點——不得回到逐格 inline）。</summary>
        private void SyncDataTabFromReviewSettled(int idx)
        {
            try
            {
                if (IsDisposed || Disposing) return;
                if (_dataStatsPresenter.GrabIdCrossGuard.IsSet) return;
                if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
                var info = _dataStatsPresenter.GrabIdInfos[idx];
                _dataStatsPresenter.SyncDataGrabIdFromReview(idx, info);
            }
            catch (Exception ex) { Trace.WriteLine($"[SyncDataTabFromReviewSettled] {ex}"); }
        }

        private async void OnDataFolderSelected(string path)
        {
            try
            {
                // 同步 Review tab：先載入 ImageRepository + Navigator，再走共用 reset + 主畫面載入。
                UserSessionState.SetLastDataPath(path);
                UserSessionState.Save();
                _reviewBusyUi?.SetBusy(true);
                try
                {
                    await _reviewFolderCoordinator.LoadDirectoryAndInitNavigatorAsync(path);
                }
                finally
                {
                    _reviewBusyUi?.SetBusy(false);
                }
                _presenter.UpdatePeriodNavigationState();
                // DataPresenter 已透過 LoadDataFolder 同步 _grabIdInfos，skip SyncFromReviewFolder
                await ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: true);
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataFolderSelected] {ex}"); }
        }
    }
}
