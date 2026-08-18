using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Binders;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Services;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>Data tab 所有 UI 控制項的參照。</summary>
    public class DataStatisticsContext
    {
        // --- 序號範圍 ---
        public ComboBox CbGrabIdStart { get; set; }
        public ComboBox CbGrabIdEnd { get; set; }

        // --- 單片 / Review 序號 ---
        public ComboBox CbDataGrabId { get; set; }
        public ComboBox CbReviewGrabId { get; set; }

        // --- 序號導航按鈕 ---

        // --- 資料夾 / 篩選 ---
        public Button BtnSelectDataFolder { get; set; }
        public Button BtnShowFail { get; set; }

        // --- GroupBox ---
        public GroupBox GroupBoxGrabIdRange { get; set; }
        public GroupBox GrpDataSingleSheet { get; set; }
        public GroupBox GrpReviewGrabNav { get; set; }
        public GroupBox GrpReviewTimePeriod { get; set; }

        // --- 年/月/日 期間 label（可點 → 範圍序號取該期間；active 綠色高亮）---
        public Label LblChartNavYear { get; set; }
        public Label LblChartNavMonth { get; set; }
        public Label LblChartNavDay { get; set; }

        // --- 統計 ---
        public GrabDetailListBinder GrabDetailList { get; set; }
        public Panel[] PanelStatCams { get; set; }
        public Panel PanelStatRow { get; set; }
        public Chart ChartDataPatch { get; set; }
        public Chart ChartDataRow { get; set; }

        // --- 趨勢圖 ---
        public Chart ChartDataYieldYearly { get; set; }
        public Chart ChartDataYieldMonthly { get; set; }
        public Chart ChartDataYieldDaily { get; set; }

        // --- 趨勢圖導航 ---
        public ComboBox CbChartYear { get; set; }
        public ComboBox CbChartMonth { get; set; }
        public ComboBox CbChartDay { get; set; }

        // --- 設定 ---
        public InspectionSettings Settings { get; set; }

        public Func<double[]> ReviewViewRangeProvider { get; set; }
        internal Func<int[], int[], double[], double[], bool, double, double, double, ImageViewRange?>
            ReviewFitViewRangeProvider { get; set; }

        public int CameraCount { get; set; } = 7;
    }

    /// <summary>
    /// 管理 tabPageData 的所有邏輯：統計、ComboBox 級聯、序號導航、趨勢圖、Detail ListView。
    /// 跨 Tab 同步透過事件通知 Form。
    /// </summary>
    public class DataStatisticsPresenter : IDisposable
    {
        private readonly DataStatisticsContext _ctx;
        private readonly InspectionStatsPresenter _statsPresenter;

        // --- 狀態 ---
        private string _statsDataRootPath = string.Empty;
        private SortedSet<DateTime> _statAvailableTimes = new SortedSet<DateTime>();
        private List<GrabIdInfo> _grabIdInfos = new List<GrabIdInfo>();
        private List<GrabIdInfo> _rangeGrabIdInfos = new List<GrabIdInfo>();
        private List<GrabDetail> _currentDetails = new List<GrabDetail>();
        private readonly ReportCurveVerdictIndex _curveVerdictIndex =
            new ReportCurveVerdictIndex();
        private readonly ReportCurveVerdictPresenter _curveVerdictPresenter;
        private bool _showFailOnly;
        private bool _preserveDetailListDuringSelection;
        private DataRangePreviewCoordinator _rangePreview;
        private readonly ReportSingleGrabSelectionCoordinator _singleGrabSelection;
        private Dictionary<string, float> _captureHmVByGrabId =
            new Dictionary<string, float>(StringComparer.Ordinal);
        private Dictionary<string, CsvConfigSnapshot> _captureConfigByGrabId =
            new Dictionary<string, CsvConfigSnapshot>(StringComparer.Ordinal);
        private readonly ReportCurveVerdictIndexCoordinator _curveVerdictIndexCoordinator;
        private bool _reportGrabIdCombosPopulated;
        private Task _reportGrabIdComboLoadTask;

        // --- 圖表導航 ---

        // --- Guards ---
        internal EventGuard StatComboGuard => _dateGrabIdNavigator.StatComboGuard;
        internal EventGuard GrabIdNavGuard => _dateGrabIdNavigator.GrabIdNavGuard;
        internal EventGuard GrabIdCrossGuard => _dateGrabIdNavigator.GrabIdCrossGuard;
        // listViewGrabDetail commit 時設 true：OnSingleSheetComboChanged 跳過範圍 cb 同步，
        // 保留使用者目前的 cbDataIdStart/End 不變。


        // --- Mura Profile Chart（繪圖職責已提取到 MuraProfileChartPresenter）---
        private MuraProfileChartPresenter _muraChart;
        private YieldPeriodChartPresenter _yieldPeriodCharts;
        private DataDateGrabIdNavigator _dateGrabIdNavigator;

        // --- 常數 ---
        private static readonly Color _activeGrpFill = Color.FromArgb(220, 248, 225);
        private static readonly Color _activeGrpBorder = Color.FromArgb(0, 140, 60);

        // --- 事件 ---
        /// <summary>Data tab 序號選取 → 通知 Form 記錄待同步至 Review 的最後一筆。
        /// (grabId, earliest, latest, selectedIndex)</summary>
        public event Action<string, DateTime, DateTime, int> GrabIdSelectedFromData;

        /// <summary>Review tab 序號選取 → 通知 Form 載入拼接圖。
        /// (grabId, earliest, latest, selectedIndex)</summary>
        public event Action<string, DateTime, DateTime, int> GrabIdSelectedFromReview;

        /// <summary>通知 Form period combo 手動變更。</summary>
        public event Action PeriodComboManualChanged;

        /// <summary>報表單序號曲線完成後，提供同一份原始曲線給回顧重用。</summary>
        internal event Action<string, string, SingleGrabCurveData> SingleGrabCurvePresented;

        public string StatsDataRootPath => _statsDataRootPath;
        public SortedSet<DateTime> StatAvailableTimes => _statAvailableTimes;
        public List<GrabIdInfo> GrabIdInfos => _grabIdInfos;

        public DataStatisticsPresenter(DataStatisticsContext ctx)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _statsPresenter = new InspectionStatsPresenter(ctx.PanelStatCams, ctx.PanelStatRow);
            _curveVerdictPresenter = new ReportCurveVerdictPresenter(
                _curveVerdictIndex,
                CreateThresholdContext,
                () => CsvConfigSnapshot.FromSettings(_ctx.Settings),
                () => _ctx.CameraCount,
                () => FlowTrace.DvtEnabled,
                FlowTrace.Log);
            _curveVerdictIndexCoordinator = new ReportCurveVerdictIndexCoordinator(
                _curveVerdictIndex,
                () => _statsDataRootPath,
                CreateThresholdContext,
                TryPostCurveVerdictAction,
                RefreshCurvePeakVerdictViews,
                () => _ctx.GrabDetailList.RefreshAll(),
                FlowTrace.Log,
                FlowTrace.Dvt);
            _singleGrabSelection = new ReportSingleGrabSelectionCoordinator(
                () => Convert.ToString(_ctx.CbDataGrabId.SelectedItem),
                RefreshSelectedGrab,
                FlowTrace.Log);
            _dateGrabIdNavigator = new DataDateGrabIdNavigator(_ctx,
                () => _grabIdInfos,
                () => _rangeGrabIdInfos,
                ScheduleRangeRefresh,
                ScheduleSelectedGrabRefresh,
                (grabId, earliest, latest, idx) => GrabIdSelectedFromData?.Invoke(grabId, earliest, latest, idx),
                (grabId, earliest, latest, idx) => GrabIdSelectedFromReview?.Invoke(grabId, earliest, latest, idx),
                SetGroupBoxActive, SetChipActive);
            _ctx.GrabDetailList.IsSelectionActive =
                () => _dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet;
        }

        // ══════════════════════════════════════════════════════════════
        // 初始化
        // ══════════════════════════════════════════════════════════════

        public void Initialize()
        {
            _statsPresenter.Initialize();

            _statsDataRootPath = _ctx.Settings?.CaptureRootPath ?? string.Empty;

            _ctx.BtnSelectDataFolder.Click += BtnSelectDataFolder_Click;
            _ctx.BtnShowFail.Click += BtnShowFail_Click;
            _dateGrabIdNavigator.WireEvents();
            _ctx.GrabDetailList.RowCommitted += OnGrabDetailRowCommitted;
            _ctx.GrabDetailList.Initialize();
            _muraChart = new MuraProfileChartPresenter(_ctx,
                () => _dateGrabIdNavigator.ActiveStatMode, () => _grabIdInfos, () => _statsDataRootPath);
            _muraChart.SingleGrabCurvePresented += OnSingleGrabCurvePresented;
            _muraChart.Init();
            _yieldPeriodCharts = new YieldPeriodChartPresenter(
                _ctx,
                () => _statAvailableTimes,
                () => _grabIdInfos,
                () => _curveVerdictIndex.Details);
            _yieldPeriodCharts.Init();

            _rangePreview = new DataRangePreviewCoordinator(
                ClearRangePreviewPresentation,
                () => RefreshStats(updateRangeCurve: false),
                ApplyRangeListPreview,
                ApplyRangeCurvePreviewAsync,
                FlowTrace.Log);
            FlowTrace.Log(
                $"DT range policy listMs={DataRangePreviewCoordinator.ListPreviewIntervalMs} " +
                $"curveMs={DataRangePreviewCoordinator.CurvePreviewIntervalMs} " +
                $"settleMs={DataRangePreviewCoordinator.SettleIntervalMs} curveMode=monotonic " +
                $"curveSamples={DataRangePreviewCoordinator.CurveSampleLimit} " +
                $"curveCacheEntries={InspectionMuraProfileRepository.RangeCurveCacheEntryCapacity} " +
                $"curveCacheMB={InspectionMuraProfileRepository.RangeCurveCacheByteCapacityMb}");

            _ctx.GrpReviewTimePeriod.Click += (s, e) => PeriodComboManualChanged?.Invoke();

            // Data tab：點選 GroupBox 標題切換 active stat 模式（與 GrpReviewGrabNav.Click 相同模式）
        }

        // ══════════════════════════════════════════════════════════════
        // 資料夾選擇
        // ══════════════════════════════════════════════════════════════

        /// <summary>通知 Form 執行資料夾選擇的完整流程（含 Review tab 同步）。</summary>
        public event Action<string> DataFolderSelected;

        private void BtnSelectDataFolder_Click(object sender, EventArgs e)
        {
            FlowTrace.Log("ui:【讀取資料】鈕（Data）");   // intent 行（孤兒判讀規則）
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.Description = "選擇 Captures 根目錄";
                dlg.SelectedPath = string.IsNullOrWhiteSpace(_statsDataRootPath)
                    ? (_ctx.Settings?.CaptureRootPath ?? string.Empty)
                    : _statsDataRootPath;
                dlg.ShowNewFolderButton = false;

                if (dlg.ShowDialog() == DialogResult.OK)
                {
                    LoadDataFolder(dlg.SelectedPath);
                    DataFolderSelected?.Invoke(dlg.SelectedPath);
                }
            }
        }

        /// <summary>載入指定資料夾的統計資料，填充所有 ComboBox。</summary>
        public void LoadDataFolder(string path)
        {
            string selectedPath = path;
            path = CaptureStoragePaths.ResolveSelectedDataRoot(
                selectedPath,
                _ctx.Settings?.CaptureRootPath);
            if (!string.Equals(path, selectedPath, StringComparison.OrdinalIgnoreCase))
                FlowTrace.Log($"DT data root upgraded from={selectedPath} to={path}");

            CancelRangePreview();
            _muraChart?.ResetSingleGrabCache();
            ResetSingleGrabDetailIndex();
            _statsDataRootPath = path;
            LoadStatisticsSnapshot(path);

            PopulateAllGrabIdCombos(selectDataGrabId: false);
            _reportGrabIdCombosPopulated = true;

            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);

            // 預設單片模式（與 Review tab btnReviewSelectFolder 一致）— 顯示最新一筆 grab（cbDataId descending [0]）。
            // 起始序號 cbDataIdStart 預設「最舊一筆」（descending 清單末筆）、結束序號最新 → 切到範圍模式即涵蓋全部。
            // 單片分支用 cbDataId 算 stats，不靠 start/end，故 start=最舊不影響單片顯示。
            _dateGrabIdNavigator.SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            SelectLatestInSingleSheetMode();
            RefreshStats();
        }

        /// <summary>從 Review tab 選擇資料夾後同步載入序號清單。</summary>
        public void PrepareReviewFolderCatalog(string path, IList<GrabIdInfo> catalogGrabIds)
        {
            CancelRangePreview();
            _muraChart?.ResetSingleGrabCache();
            ResetSingleGrabDetailIndex();
            _statsDataRootPath = path;
            _grabIdInfos = catalogGrabIds == null
                ? new List<GrabIdInfo>()
                : catalogGrabIds.ToList();
            // Period charts are report data. Building their 30,000-entry sorted
            // set here delays Review first paint and duplicates the CSV snapshot.
            _statAvailableTimes = new SortedSet<DateTime>();
            _curveVerdictIndex.ReplaceDetails(
                path,
                new Dictionary<string, GrabDetail>(StringComparer.Ordinal),
                CreateThresholdContext());
            _captureHmVByGrabId = new Dictionary<string, float>(StringComparer.Ordinal);
            _captureConfigByGrabId = new Dictionary<string, CsvConfigSnapshot>(StringComparer.Ordinal);
            RefreshRangeGrabIdInfos();

            _dateGrabIdNavigator.PopulateReviewGrabIdCombo();
            _reportGrabIdCombosPopulated = false;
            _reportGrabIdComboLoadTask = null;
            FlowTrace.Log($"RV catalog ready grabs={_grabIdInfos.Count} source=image-index");
        }

        public async Task CompleteReviewFolderStatisticsAsync(string path)
        {
            string selectedGrabId = Convert.ToString(_ctx.CbReviewGrabId.SelectedItem);
            var watch = Stopwatch.StartNew();
            ThresholdContext threshold = CreateThresholdContext();
            InspectionStatisticsSnapshot snapshot = await Task.Run(
                () => InspectionStatisticsService.LoadSnapshot(path, threshold));
            ApplyStatisticsSnapshot(path, threshold, snapshot, watch.ElapsedMilliseconds);

            bool catalogMatches = ReviewComboMatchesStatistics();
            if (!catalogMatches)
            {
                _dateGrabIdNavigator.PopulateReviewGrabIdCombo();
                if (!string.IsNullOrEmpty(selectedGrabId))
                {
                    int selectedIndex = _ctx.CbReviewGrabId.Items.IndexOf(selectedGrabId);
                    if (selectedIndex >= 0)
                        _ctx.CbReviewGrabId.SelectedIndex = selectedIndex;
                }
            }

            _reportGrabIdCombosPopulated = false;
            _reportGrabIdComboLoadTask = null;
            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);
            FlowTrace.Log(
                $"RV statistics ready grabs={_grabIdInfos.Count} catalogMatch={catalogMatches}");
        }

        private bool ReviewComboMatchesStatistics()
        {
            if (_ctx.CbReviewGrabId.Items.Count != _grabIdInfos.Count)
                return false;

            for (int i = 0; i < _grabIdInfos.Count; i++)
            {
                if (!string.Equals(
                    Convert.ToString(_ctx.CbReviewGrabId.Items[i]),
                    _grabIdInfos[i].GrabId,
                    StringComparison.Ordinal))
                    return false;
            }

            return true;
        }

        public async Task EnsureReportGrabIdCombosAsync()
        {
            if (_reportGrabIdCombosPopulated) return;

            Task loadTask = _reportGrabIdComboLoadTask;
            if (loadTask == null)
            {
                loadTask = PopulateReportGrabIdCombosCoreAsync();
                _reportGrabIdComboLoadTask = loadTask;
            }

            try
            {
                await loadTask;
            }
            catch
            {
                if (ReferenceEquals(_reportGrabIdComboLoadTask, loadTask))
                    _reportGrabIdComboLoadTask = null;
                throw;
            }
        }

        private async Task PopulateReportGrabIdCombosCoreAsync()
        {
            var watch = Stopwatch.StartNew();
            await _dateGrabIdNavigator.PopulateReportGrabIdCombosAsync();
            _dateGrabIdNavigator.SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            SelectLatestInSingleSheetMode();
            RefreshStats();
            _reportGrabIdCombosPopulated = true;
            FlowTrace.Log($"DT deferred controls ready count={_rangeGrabIdInfos.Count} ms={watch.ElapsedMilliseconds}");
        }

        private void LoadStatisticsSnapshot(string path)
        {
            var watch = Stopwatch.StartNew();
            var threshold = CreateThresholdContext();
            InspectionStatisticsSnapshot snapshot =
                InspectionStatisticsService.LoadSnapshot(path, threshold);
            ApplyStatisticsSnapshot(path, threshold, snapshot, watch.ElapsedMilliseconds);
        }

        private void ApplyStatisticsSnapshot(
            string path,
            ThresholdContext threshold,
            InspectionStatisticsSnapshot snapshot,
            long elapsedMilliseconds)
        {
            _statAvailableTimes = snapshot.AvailableTimes;
            _grabIdInfos = snapshot.GrabIdsDescending;
            _curveVerdictIndex.ReplaceDetails(
                path, snapshot.DetailsByGrabId, threshold);
            _captureHmVByGrabId = snapshot.CaptureHmVByGrabId;
            _captureConfigByGrabId = snapshot.ConfigByGrabId;
            RefreshRangeGrabIdInfos();

            StartColumnCurvePeakIndexBuild(resetExisting: true);

            FlowTrace.Log(
                $"DT stats snapshot csv={snapshot.CsvFileCount} " +
                $"records={snapshot.RecordCount} grabs={_grabIdInfos.Count} " +
                $"ms={elapsedMilliseconds}");
        }

        // ══════════════════════════════════════════════════════════════
        // 序號 ComboBox
        // ══════════════════════════════════════════════════════════════

        public void SyncDataGrabIdFromReview(int idx, GrabIdInfo info) =>
            _dateGrabIdNavigator.SyncDataGrabIdFromReview(idx, info);

        public void UpdateGrabIdNavState() => _dateGrabIdNavigator.UpdateGrabIdNavState();

        public void SyncGrabIdFromTime(DateTime current) => _dateGrabIdNavigator.SyncGrabIdFromTime(current);

        private string GetDetailListStartGrabId()
        {
            if (_ctx.CbGrabIdStart.SelectedItem != null)
                return _ctx.CbGrabIdStart.SelectedItem.ToString();
            if (_ctx.CbDataGrabId.SelectedItem != null)
                return _ctx.CbDataGrabId.SelectedItem.ToString();
            return _grabIdInfos.Count > 0 ? _grabIdInfos[0].GrabId : string.Empty;
        }

        private string GetDetailListEndGrabId()
        {
            if (_ctx.CbGrabIdEnd.SelectedItem != null)
                return _ctx.CbGrabIdEnd.SelectedItem.ToString();
            if (_ctx.CbDataGrabId.SelectedItem != null)
                return _ctx.CbDataGrabId.SelectedItem.ToString();
            return _grabIdInfos.Count > 0 ? _grabIdInfos[0].GrabId : string.Empty;
        }

        public void RefreshStats(bool updateRangeCurve = true)
        {
            if (updateRangeCurve) CancelRangePreview();
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            bool indexWasCurrent = IsSingleGrabDetailIndexCurrent();
            EnsureSingleGrabDetailIndex();
            bool verdictsChanged = ApplyCurrentCurveVerdictsIfNeeded();
            if (_showFailOnly && (!indexWasCurrent || verdictsChanged))
            {
                string preferredGrabId = Convert.ToString(_ctx.CbDataGrabId.SelectedItem);
                RefreshRangeGrabIdInfos();
                int rangeOptions = _dateGrabIdNavigator.RefreshFilteredGrabIdCombos(preferredGrabId);
                if (rangeOptions == 0 &&
                    (_dateGrabIdNavigator.ActiveStatMode == _ctx.GroupBoxGrabIdRange ||
                     _dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet))
                {
                    ClearRangePresentation();
                    return;
                }
            }

            // SingleSheet mode：用 cbDataId.SelectedIndex 算單 grab stats（start=end=該 grab）。
            // 不靠 cbDataIdStart/End 範圍；cbDataId 變更不連動範圍 cb（範圍獨立），
            // 故 listViewGrabDetail 點選後 stats 仍對齊到剛點的單 grab。
            if (_dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet
                && TryGetSelectedDataGrabInfo(out _))
            {
                if (!_preserveDetailListDuringSelection)
                {
                    var swList = Stopwatch.StartNew();
                    string listStart = GetDetailListStartGrabId();
                    string listEnd = GetDetailListEndGrabId();
                    _currentDetails = GetIndexedDetailsForSelectedRange();
                    ApplyFailFilter();
                    FlowTrace.Log($"DT list reload range={listStart}~{listEnd} rows={_currentDetails.Count} ms={swList.ElapsedMilliseconds} source=index");
                }
                RefreshSelectedGrab();
                return;
            }

            if (TryGetSelectedRange(out List<GrabIdInfo> rangeInfos))
            {
                _statsPresenter.UpdateRowResult(null);
                _muraChart?.ClearRow();
                string startGrabId = _ctx.CbGrabIdStart.Text;
                string endGrabId = _ctx.CbGrabIdEnd.Text;

                if (!_preserveDetailListDuringSelection)
                {
                    var swList = Stopwatch.StartNew();
                    _currentDetails = GetIndexedDetailsForSelectedRange(rangeInfos);
                    ApplyFailFilter();
                    _statsPresenter.Update(
                        InspectionStatisticsService.ComputeStatsFromDetails(_currentDetails));
                    FlowTrace.Log($"DT list reload range={startGrabId}~{endGrabId} rows={_currentDetails.Count} ms={swList.ElapsedMilliseconds} source=index");
                    GrabIdRangeSource source =
                        _dateGrabIdNavigator.ActiveRangeSource;
                    if (source == GrabIdRangeSource.Year ||
                        source == GrabIdRangeSource.Month ||
                        source == GrabIdRangeSource.Day)
                    {
                        FlowTrace.Log(
                            $"DT period list source={source} " +
                            $"range={startGrabId}~{endGrabId} " +
                            $"expected={rangeInfos.Count} indexed={_currentDetails.Count} " +
                            $"visible={_ctx.GrabDetailList.VisibleCount} " +
                            $"ms={swList.ElapsedMilliseconds}");
                    }
                }
                else
                    _statsPresenter.Update(
                        InspectionStatisticsService.ComputeStatsFromDetails(_currentDetails));

                if (updateRangeCurve)
                {
                    // Range archives can be cold and large. Always use the cancellable
                    // latest-only background path; never scan them from the UI thread.
                    _rangePreview?.Start();
                }
                return;
            }

        }

        private void ScheduleRangeRefresh()
        {
            CancelSelectedGrabRefresh();
            if (_rangePreview == null)
            {
                RefreshStats();
                return;
            }
            _rangePreview.Start();
        }

        private void CancelSelectedGrabRefresh()
        {
            _singleGrabSelection.Cancel();
        }

        private void ClearRangePreviewPresentation()
        {
            // Row data belongs to a single grab. Never leave it visible while a range is moving.
            _muraChart?.ClearRow();
            _statsPresenter.UpdateRowResult(null);
        }

        private async System.Threading.Tasks.Task ApplyRangeCurvePreviewAsync(
            int generation, Func<int> getLatestGeneration,
            CancellationToken cancellationToken)
        {
            if (!TryGetSelectedRange(out List<GrabIdInfo> rangeInfos))
                return;

            await _muraChart.UpdateRangePreviewAsync(
                rangeInfos, generation, getLatestGeneration, cancellationToken);
        }

        private bool TryGetSelectedRange(out List<GrabIdInfo> rangeInfos)
        {
            if (_dateGrabIdNavigator.ActiveStatMode != _ctx.GroupBoxGrabIdRange)
            {
                rangeInfos = null;
                return false;
            }
            return _dateGrabIdNavigator.TryGetSelectedRange(out rangeInfos);
        }

        private void CancelRangePreview()
        {
            _rangePreview?.Cancel();
        }

        public void Dispose()
        {
            _curveVerdictIndexCoordinator.Dispose();
            _singleGrabSelection.Dispose();
            _rangePreview?.Dispose();
            _rangePreview = null;
            if (_muraChart != null)
            {
                _muraChart.SingleGrabCurvePresented -= OnSingleGrabCurvePresented;
                _muraChart.Dispose();
            }
            _muraChart = null;
            _ctx.GrabDetailList.RowCommitted -= OnGrabDetailRowCommitted;
            _ctx.GrabDetailList.Dispose();
        }

        private void ScheduleSelectedGrabRefresh()
        {
            _singleGrabSelection.Schedule();
        }

        private void OnSingleGrabCurvePresented(
            string root, string grabId, SingleGrabCurveData data)
        {
            ApplyMergedCurveVerdicts(grabId, data);
            SingleGrabCurvePresented?.Invoke(root, grabId, data);
        }

        private void ApplyMergedCurveVerdicts(string grabId, SingleGrabCurveData data)
        {
            if (string.IsNullOrWhiteSpace(grabId) || data == null) return;

            GrabDetail detail = _currentDetails.FirstOrDefault(item =>
                string.Equals(item.GrabId, grabId, StringComparison.Ordinal));
            if (detail == null)
            {
                EnsureSingleGrabDetailIndex();
                _curveVerdictIndex.Details.TryGetValue(grabId, out detail);
            }
            if (detail == null) return;

            if (!_curveVerdictPresenter.ApplyVisibleCurves(grabId, data, detail))
                return;

            _ctx.GrabDetailList.Refresh(grabId);
            if (string.Equals(Convert.ToString(_ctx.CbDataGrabId.SelectedItem),
                grabId, StringComparison.Ordinal))
            {
                _statsPresenter.Update(BuildSingleGrabStats(detail));
                _statsPresenter.UpdateRowResult(detail.RowResult);
            }
        }

        /// <summary>單片序號快路：List 範圍內容不變，只更新該筆統計、Mura curve 與反白。</summary>
        private void RefreshSelectedGrab()
        {
            CancelSelectedGrabRefresh();
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;
            if (!TryGetSelectedDataGrabInfo(out GrabIdInfo grab)) return;

            var sw = Stopwatch.StartNew();
            var detail = _currentDetails.FirstOrDefault(item => item.GrabId == grab.GrabId);
            if (detail == null)
            {
                EnsureSingleGrabDetailIndex();
                _curveVerdictIndex.Details.TryGetValue(grab.GrabId, out detail);
            }
            bool cacheHit = detail != null;
            Dictionary<int, CameraStats> stats;
            if (cacheHit)
            {
                stats = BuildSingleGrabStats(detail);
            }
            else
            {
                var threshold = CreateThresholdContext();
                stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, grab.GrabId, grab.GrabId, threshold);
            }

            _statsPresenter.Update(stats);
            _statsPresenter.UpdateRowResult(detail?.RowResult);
            _ctx.GrabDetailList.Highlight(grab.GrabId);
            _muraChart.Update(null);
            FlowTrace.Log($"DT selected {grab.GrabId} stats={(cacheHit ? "cache" : "scan")} list=keep ms={sw.ElapsedMilliseconds}");
        }

        private bool TryGetSelectedDataGrabInfo(out GrabIdInfo info)
        {
            info = null;
            string grabId = Convert.ToString(_ctx.CbDataGrabId.SelectedItem);
            if (string.IsNullOrWhiteSpace(grabId)) return false;
            info = _grabIdInfos.FirstOrDefault(candidate =>
                string.Equals(candidate.GrabId, grabId, StringComparison.Ordinal));
            return info != null;
        }

        private void ResetSingleGrabDetailIndex()
        {
            _curveVerdictIndexCoordinator.Cancel();
            _curveVerdictIndex.Reset();
        }

        private bool IsSingleGrabDetailIndexCurrent()
        {
            return _curveVerdictIndex.IsCurrent(_statsDataRootPath);
        }

        private bool ApplyCurrentCurveVerdictsIfNeeded()
        {
            return _curveVerdictPresenter.ApplyCurrentIfNeeded(
                Convert.ToString(_ctx.CbDataGrabId.SelectedItem));
        }

        private void EnsureSingleGrabDetailIndex()
        {
            if (IsSingleGrabDetailIndexCurrent())
                return;

            var sw = Stopwatch.StartNew();
            var details = new Dictionary<string, GrabDetail>(StringComparer.Ordinal);
            if (_grabIdInfos.Count > 0)
            {
                var loadThreshold = CreateThresholdContext();
                List<GrabDetail> loadedDetails = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath,
                    _grabIdInfos[_grabIdInfos.Count - 1].GrabId,
                    _grabIdInfos[0].GrabId,
                    loadThreshold);
                foreach (GrabDetail item in loadedDetails)
                    details[item.GrabId] = item;
            }

            ThresholdContext threshold = CreateThresholdContext();
            _curveVerdictIndex.ReplaceDetails(
                _statsDataRootPath, details, threshold);
            _curveVerdictPresenter.Project(CreateThresholdContext());
            StartColumnCurvePeakIndexBuild(resetExisting: false);
            FlowTrace.Log($"DT stats index rows={_curveVerdictIndex.Details.Count} ms={sw.ElapsedMilliseconds}");
        }

        private void StartColumnCurvePeakIndexBuild(bool resetExisting)
        {
            _curveVerdictIndexCoordinator.Start(
                resetExisting,
                _statsDataRootPath,
                _grabIdInfos,
                _captureHmVByGrabId,
                _captureConfigByGrabId,
                _ctx.CameraCount);
        }

        private bool TryPostCurveVerdictAction(Action action)
        {
            if (action == null || _ctx.CbDataGrabId.IsDisposed ||
                !_ctx.CbDataGrabId.IsHandleCreated)
                return false;
            try
            {
                _ctx.CbDataGrabId.BeginInvoke(action);
                return true;
            }
            catch (InvalidOperationException)
            {
                // The form can close between the handle check and BeginInvoke.
                return false;
            }
        }

        private void RefreshCurvePeakVerdictViews()
        {
            _ctx.GrabDetailList.RefreshAll();
            RefreshStats(updateRangeCurve: false);
            RefreshPeriodCharts();
            _curveVerdictPresenter.AuditSelected(
                Convert.ToString(_ctx.CbDataGrabId.SelectedItem), "index");
        }

        private bool ApplyRangeListPreview(int generation)
        {
            if (_preserveDetailListDuringSelection
                || !IsSingleGrabDetailIndexCurrent())
                return false;

            var sw = Stopwatch.StartNew();
            if (!TryGetSelectedRange(out List<GrabIdInfo> rangeInfos))
                return false;
            _currentDetails = GetIndexedDetailsForSelectedRange(rangeInfos);
            ApplyFailFilter();
            _statsPresenter.Update(
                InspectionStatisticsService.ComputeStatsFromDetails(_currentDetails));

            string start = _ctx.CbGrabIdStart.Text;
            string end = _ctx.CbGrabIdEnd.Text;
            FlowTrace.Log($"DT range list preview gen={generation} range={start}~{end} rows={_currentDetails.Count} ms={sw.ElapsedMilliseconds} source=index");
            return true;
        }

        private List<GrabDetail> GetIndexedDetailsForSelectedRange()
        {
            return _dateGrabIdNavigator.TryGetSelectedRange(out List<GrabIdInfo> rangeInfos)
                ? GetIndexedDetailsForSelectedRange(rangeInfos)
                : new List<GrabDetail>();
        }

        private List<GrabDetail> GetIndexedDetailsForSelectedRange(List<GrabIdInfo> rangeInfos)
        {
            var details = new List<GrabDetail>(rangeInfos.Count);
            foreach (GrabIdInfo info in rangeInfos)
            {
                if (_curveVerdictIndex.Details.TryGetValue(info.GrabId, out GrabDetail detail))
                    details.Add(detail);
            }
            return details;
        }

        private static Dictionary<int, CameraStats> BuildSingleGrabStats(GrabDetail detail)
        {
            var stats = new Dictionary<int, CameraStats>();
            for (int i = 0; i < 7; i++)
            {
                var camera = new CameraStats { CamId = i + 1 };
                bool? failed = detail.CamResult[i];
                if (failed.HasValue)
                {
                    if (failed.Value) camera.Fail = 1;
                    else camera.Pass = 1;
                }
                stats[camera.CamId] = camera;
            }
            return stats;
        }

        private ThresholdContext CreateThresholdContext() => new ThresholdContext(
            _ctx.Settings.HessianMaxFactorV,
            _ctx.Settings.ErrorValueMeanV,
            _ctx.Settings.ErrorValueMaxV,
            _ctx.Settings.HessianMaxFactorH,
            _ctx.Settings.ErrorValueMeanH,
            _ctx.Settings.ErrorValueMaxH,
            _ctx.Settings.ColumnCurveMode,
            _ctx.Settings.RidgeDir);

        // ══════════════════════════════════════════════════════════════
        // Detail ListView
        // ══════════════════════════════════════════════════════════════


        private void OnGrabDetailRowCommitted(object sender, GrabDetailRowCommittedEventArgs e)
        {
            if (StatComboGuard.IsSet) return;
            string grabId = e.GrabId;
            _curveVerdictPresenter.Audit(grabId, "click");

            // Toggle：第二次點同 row + 已是 SingleSheet → 切回 GroupBoxGrabIdRange（範圍模式，stats 用 cbDataIdStart/End）
            if (e.IsRepeated && _dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet)
            {
                FlowTrace.Log($"ui:【明細列表】同列再點 {grabId} → 回範圍模式");
                ExecuteWithDetailListRedrawSuspended(() =>
                {
                    _ctx.GrabDetailList.ClearSelection();
                    _muraChart?.Clear();          // 先清圖，避免同列二次點選時殘留上一筆 CURVE
                    _dateGrabIdNavigator.SetActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
                    RefreshStats();
                });
                return;
            }
            FlowTrace.Log($"ui:【明細列表】→ {grabId}");

            int idx = _ctx.CbDataGrabId.Items.IndexOf(grabId);
            if (idx < 0) return;
            if (_ctx.CbDataGrabId.SelectedIndex == idx)
            {
                // SelectedIndex 沒變 → 不會觸發 OnSingleSheetComboChanged，但仍需確保 active 模式為單片
                ExecuteWithDetailListRedrawSuspended(() =>
                {
                    if (_dateGrabIdNavigator.ActiveStatMode != _ctx.GrpDataSingleSheet)
                        _dateGrabIdNavigator.SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
                });
                return;
            }
            ExecuteWithDetailListRedrawSuspended(
                () => _dateGrabIdNavigator.CommitDataGrabIdFromDetailList(grabId));
        }

        private void ExecuteWithDetailListRedrawSuspended(Action action)
        {
            if (action == null) return;
            bool previous = _preserveDetailListDuringSelection;
            _preserveDetailListDuringSelection = true;
            try
            {
                _ctx.GrabDetailList.ExecuteWithRedrawSuspended(action);
            }
            finally { _preserveDetailListDuringSelection = previous; }
        }

        // ══════════════════════════════════════════════════════════════
        // Mura 空間分布曲線圖（拼接式，與 chartLiveColumn 相同格式）
        // ══════════════════════════════════════════════════════════════

        // Mura 分布圖繪圖職責已提取到 MuraProfileChartPresenter（2026-06-30）；以下為對外 public 門面轉發。

        /// <summary>由 PropertyGrid 變更觸發：刷新 chartDataColumn 閾值線 + view-time 正規值 rescale（不重算統計）。</summary>
        public void RefreshMuraProfileForSettingsChange(string settingName)
            => _muraChart?.RefreshForSettingsChange(settingName);

        /// <summary>SingleSheet 模式：用 Review tab 已載入曲線資料更新 chartDataColumn，與 chartReviewColumn 一致。</summary>
        public void SyncMuraProfileFromReview(float[][] mean, float[][] max,
            double[] ops, double[] pos, float errMean, float errMax)
            => _muraChart?.SyncFromReview(mean, max, ops, pos, errMean, errMax);

        public void SetReviewViewRange(double leftMm, double rightMm, double topMm, double bottomMm)
            => _muraChart?.SetReviewViewRange(leftMm, rightMm, topMm, bottomMm);

        public void SetPreparedReviewViewRange(double leftMm, double rightMm, double topMm, double bottomMm)
            => _muraChart?.SetPreparedReviewViewRange(leftMm, rightMm, topMm, bottomMm);

        // ══════════════════════════════════════════════════════════════
        // 異常篩選
        // ══════════════════════════════════════════════════════════════

        private async void BtnShowFail_Click(object sender, EventArgs e)
        {
            _ctx.BtnShowFail.Enabled = false;
            try
            {
                string preferredGrabId =
                    Convert.ToString(_ctx.CbDataGrabId.SelectedItem);
                _showFailOnly = !_showFailOnly;
                _ctx.BtnShowFail.Text =
                    _showFailOnly ? "○ 顯示全部" : "△ 顯示異常";
                _ctx.BtnShowFail.BackColor = _showFailOnly
                    ? Color.FromArgb(255, 235, 238)
                    : SystemColors.Control;

                EnsureSingleGrabDetailIndex();
                RefreshRangeGrabIdInfos();
                int rangeOptions = await _dateGrabIdNavigator
                    .RefreshFilteredGrabIdCombosAsync(preferredGrabId);
                int dataOptions = _ctx.CbDataGrabId.Items.Count;
                _currentDetails = GetIndexedDetailsForSelectedRange();
                ApplyFailFilter();
                string selectedGrabId = dataOptions == 0
                    ? "empty"
                    : Convert.ToString(_ctx.CbDataGrabId.SelectedItem);

                string range = rangeOptions == 0
                    ? "empty"
                    : $"{_ctx.CbGrabIdStart.Text}~{_ctx.CbGrabIdEnd.Text}";
                FlowTrace.Log(
                    $"ui:【篩選異常】→ {(_showFailOnly ? "只顯示異常" : "顯示全部")} " +
                    $"dataOptions={dataOptions} rangeOptions={rangeOptions} " +
                    $"selected={selectedGrabId} range={range}");

                if (rangeOptions == 0)
                {
                    ClearRangePresentation();
                    return;
                }
                if (_dateGrabIdNavigator.ActiveStatMode ==
                    _ctx.GroupBoxGrabIdRange)
                    RefreshStats();
                else if (_dateGrabIdNavigator.ActiveStatMode ==
                    _ctx.GrpDataSingleSheet)
                    RefreshSelectedGrab();
                _ctx.GrabDetailList.Highlight(selectedGrabId);
            }
            finally
            {
                _ctx.BtnShowFail.Enabled = true;
            }
        }

        private void ApplyFailFilter()
        {
            var toShow = _showFailOnly
                ? _currentDetails.Where(IsFailedDetail).ToList()
                : _currentDetails;
            _ctx.GrabDetailList.SetItems(toShow);
        }

        private void RefreshRangeGrabIdInfos()
        {
            _rangeGrabIdInfos = _showFailOnly
                ? SelectFailRangeInfos(_grabIdInfos, _curveVerdictIndex.Details)
                : _grabIdInfos;
        }

        internal static List<GrabIdInfo> SelectFailRangeInfos(
            IList<GrabIdInfo> allInfos,
            IDictionary<string, GrabDetail> detailsByGrabId)
        {
            var result = new List<GrabIdInfo>();
            if (allInfos == null || detailsByGrabId == null) return result;
            foreach (GrabIdInfo info in allInfos)
            {
                if (info != null && detailsByGrabId.TryGetValue(info.GrabId, out GrabDetail detail)
                    && IsFailedDetail(detail))
                    result.Add(info);
            }
            return result;
        }

        private static bool IsFailedDetail(GrabDetail detail) =>
            detail != null && (detail.CamResult.Any(result => result == true) || detail.RowResult == true);

        private void ClearRangePresentation()
        {
            _currentDetails = new List<GrabDetail>();
            _ctx.GrabDetailList.SetItems(_currentDetails);
            _statsPresenter.Update(new Dictionary<int, CameraStats>());
            _statsPresenter.UpdateRowResult(null);
            _muraChart?.Clear();
        }

        // ══════════════════════════════════════════════════════════════
        // 趨勢圖（年 / 月 / 日）
        // ══════════════════════════════════════════════════════════════

        public void ApplyChartScaleFromSettings() => _yieldPeriodCharts?.ApplyChartScaleFromSettings();

        public void ApplyChartScaleForChart(string chartName) =>
            _yieldPeriodCharts?.ApplyChartScaleForChart(chartName);

        public void PopulateChartNavigators() => _yieldPeriodCharts?.PopulateChartNavigators();

        public void PopulateChartNavigators(DateTime? hintDate) =>
            _yieldPeriodCharts?.PopulateChartNavigators(hintDate);

        /// <summary>由 PropertyGrid 設定變更觸發，重新整理 chartDataYieldYearly/Monthly/Daily，讓 Settings 立刻套用 Pass/Fail。</summary>
        public void RefreshPeriodCharts()
        {
            EnsureSingleGrabDetailIndex();
            ApplyCurrentCurveVerdictsIfNeeded();
            _yieldPeriodCharts?.RefreshPeriodCharts();
        }
        public void PopulateAllGrabIdCombos(bool selectDataGrabId = false) =>
            _dateGrabIdNavigator.PopulateAllGrabIdCombos(selectDataGrabId);
        public Task PopulateAllGrabIdCombosAsync(bool selectDataGrabId = false) =>
            _dateGrabIdNavigator.PopulateAllGrabIdCombosAsync(selectDataGrabId);
        public void SetReviewGroupBoxes(bool grabNavActive)
        {
            SetGroupBoxActive(_ctx.GrpReviewGrabNav, grabNavActive);
            SetGroupBoxActive(_ctx.GrpReviewTimePeriod, !grabNavActive);
        }

        /// <summary>讀取資料後預設切到單片模式，並保留範圍模式預設為全資料範圍。
        /// btnDataSelectFolder / btnReviewSelectFolder 共用的最後一步。</summary>
        public void SelectLatestInSingleSheetMode()
        {
            if (_grabIdInfos.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    if (_ctx.CbGrabIdStart.Items.Count > 0)
                    {
                        _ctx.CbGrabIdStart.SelectedIndex = _ctx.CbGrabIdStart.Items.Count - 1; // 最舊
                        _ctx.CbGrabIdEnd.SelectedIndex = 0;                                  // 最新
                    }
                    if (_ctx.CbDataGrabId.Items.Count > 0)
                        _ctx.CbDataGrabId.SelectedIndex = 0;                      // 單片顯示最新
                }
            }

            _dateGrabIdNavigator.SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
        }

        /// <summary>年/月/日 期間 label chip 的 active 高亮：綠底綠字（與 groupBox active 同色）／恢復預設。</summary>
        private static void SetChipActive(Label lbl, bool active)
        {
            if (lbl == null) return;
            lbl.BackColor = active ? _activeGrpFill : System.Drawing.SystemColors.Control;
            lbl.ForeColor = active ? _activeGrpBorder : System.Drawing.SystemColors.ControlText;
        }

        private static void SetGroupBoxActive(GroupBox box, bool active)
        {
            if (active)
            {
                box.Paint -= ActiveGroupBox_Paint;
                box.Paint += ActiveGroupBox_Paint;
            }
            else
            {
                box.Paint -= ActiveGroupBox_Paint;
            }
            box.Invalidate();
        }

        private static void ActiveGroupBox_Paint(object sender, PaintEventArgs e)
        {
            var g = e.Graphics;
            var box = (GroupBox)sender;
            int textH = (int)g.MeasureString(box.Text, box.Font).Height;
            int midY = textH / 2;

            using (var brush = new SolidBrush(_activeGrpFill))
                g.FillRectangle(brush, 0, midY, box.Width, box.Height - midY);
            using (var pen = new Pen(_activeGrpBorder, 1.5f))
                g.DrawRectangle(pen, 0, midY, box.Width - 1, box.Height - midY - 1);

            var textSize = g.MeasureString(box.Text, box.Font);
            using (var bgBrush = new SolidBrush(_activeGrpFill))
                g.FillRectangle(bgBrush, 6, 0, textSize.Width + 2, textH);
            using (var textBrush = new SolidBrush(_activeGrpBorder))
                g.DrawString(box.Text, box.Font, textBrush, 8, 0);
        }
    }
}
