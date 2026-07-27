using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Threading;
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
        internal Func<int[], int[], double[], double[], bool, double, ImageViewRange?>
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
        private Dictionary<string, GrabDetail> _singleGrabDetailIndex =
            new Dictionary<string, GrabDetail>(StringComparer.Ordinal);
        private string _singleGrabDetailIndexRoot = string.Empty;
        private float _singleGrabDetailIndexHmV;
        private float _singleGrabDetailIndexErrMean;
        private float _singleGrabDetailIndexErrMax;
        private float _singleGrabDetailIndexHmH;
        private float _singleGrabDetailIndexRowErrMean;
        private float _singleGrabDetailIndexRowErrMax;
        private bool _singleGrabDetailIndexReady;
        private bool _showFailOnly;
        private bool _preserveDetailListDuringSelection;
        private DataRangePreviewCoordinator _rangePreview;
        private const int SingleGrabPreviewIntervalMs = 33;
        private System.Windows.Forms.Timer _singleGrabPreviewTimer;
        private int _singleGrabPreviewRequests;

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
            _singleGrabPreviewTimer = new System.Windows.Forms.Timer
            {
                Interval = SingleGrabPreviewIntervalMs
            };
            _singleGrabPreviewTimer.Tick += SingleGrabPreviewTimer_Tick;
            _muraChart = new MuraProfileChartPresenter(_ctx,
                () => _dateGrabIdNavigator.ActiveStatMode, () => _grabIdInfos, () => _statsDataRootPath);
            _muraChart.SingleGrabCurvePresented += OnSingleGrabCurvePresented;
            _muraChart.Init();
            _yieldPeriodCharts = new YieldPeriodChartPresenter(
                _ctx,
                () => _statAvailableTimes,
                () => _grabIdInfos,
                () => _singleGrabDetailIndex);
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
        public void SyncFromReviewFolder(string path)
        {
            CancelRangePreview();
            _muraChart?.ResetSingleGrabCache();
            ResetSingleGrabDetailIndex();
            _statsDataRootPath = path;
            LoadStatisticsSnapshot(path);

            PopulateAllGrabIdCombos();

            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);
            SelectLatestInSingleSheetMode();
            RefreshStats();
        }

        private void LoadStatisticsSnapshot(string path)
        {
            var watch = Stopwatch.StartNew();
            var threshold = CreateThresholdContext();
            InspectionStatisticsSnapshot snapshot =
                InspectionStatisticsService.LoadSnapshot(path, threshold);

            _statAvailableTimes = snapshot.AvailableTimes;
            _grabIdInfos = snapshot.GrabIdsDescending;
            _singleGrabDetailIndex.Clear();
            foreach (var entry in snapshot.DetailsByGrabId)
                _singleGrabDetailIndex[entry.Key] = entry.Value;
            _singleGrabDetailIndexRoot = path;
            _singleGrabDetailIndexHmV = threshold.CurrentHmV;
            _singleGrabDetailIndexErrMean = threshold.CurrentErrMean;
            _singleGrabDetailIndexErrMax = threshold.CurrentErrMax;
            _singleGrabDetailIndexHmH = threshold.CurrentHmH;
            _singleGrabDetailIndexRowErrMean = threshold.CurrentRowErrMean;
            _singleGrabDetailIndexRowErrMax = threshold.CurrentRowErrMax;
            _singleGrabDetailIndexReady = true;
            RefreshRangeGrabIdInfos();

            FlowTrace.Log(
                $"DT stats snapshot csv={snapshot.CsvFileCount} " +
                $"records={snapshot.RecordCount} grabs={_grabIdInfos.Count} " +
                $"ms={watch.ElapsedMilliseconds}");
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
            if (_showFailOnly && !indexWasCurrent)
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
                }
                else
                    _statsPresenter.Update(
                        InspectionStatisticsService.ComputeStatsFromDetails(_currentDetails));

                if (updateRangeCurve)
                    _muraChart.Update(rangeInfos, rangeInfos);
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
            _singleGrabPreviewTimer?.Stop();
            _singleGrabPreviewRequests = 0;
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
            _rangePreview?.Dispose();
            _rangePreview = null;
            if (_singleGrabPreviewTimer != null)
            {
                _singleGrabPreviewTimer.Stop();
                _singleGrabPreviewTimer.Tick -= SingleGrabPreviewTimer_Tick;
                _singleGrabPreviewTimer.Dispose();
                _singleGrabPreviewTimer = null;
            }
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
            if (_singleGrabPreviewTimer == null)
            {
                RefreshSelectedGrab();
                return;
            }

            _singleGrabPreviewRequests++;
            if (!_singleGrabPreviewTimer.Enabled)
                _singleGrabPreviewTimer.Start();
        }

        private void SingleGrabPreviewTimer_Tick(object sender, EventArgs e)
        {
            _singleGrabPreviewTimer.Stop();
            int requestCount = _singleGrabPreviewRequests;
            _singleGrabPreviewRequests = 0;

            if (!TryGetSelectedDataGrabInfo(out GrabIdInfo selected))
                return;

            FlowTrace.Log($"ui:【報表序號】→ {selected.GrabId}");
            if (requestCount > 1)
            {
                FlowTrace.Log(
                    $"DT selected coalesced {selected.GrabId} " +
                    $"skipped={requestCount - 1} intervalMs={SingleGrabPreviewIntervalMs}");
            }
            RefreshSelectedGrab();
        }

        private void OnSingleGrabCurvePresented(
            string root, string grabId, SingleGrabCurveData data)
        {
            SingleGrabCurvePresented?.Invoke(root, grabId, data);
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
                _singleGrabDetailIndex.TryGetValue(grab.GrabId, out detail);
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
            _singleGrabDetailIndex.Clear();
            _singleGrabDetailIndexRoot = string.Empty;
            _singleGrabDetailIndexReady = false;
        }

        private bool IsSingleGrabDetailIndexCurrent()
        {
            return _singleGrabDetailIndexReady
                && string.Equals(_singleGrabDetailIndexRoot, _statsDataRootPath, StringComparison.OrdinalIgnoreCase)
                && _singleGrabDetailIndexHmV == _ctx.Settings.HessianMaxFactorV
                && _singleGrabDetailIndexErrMean == _ctx.Settings.ErrorValueMeanV
                && _singleGrabDetailIndexErrMax == _ctx.Settings.ErrorValueMaxV
                && _singleGrabDetailIndexHmH == _ctx.Settings.HessianMaxFactorH
                && _singleGrabDetailIndexRowErrMean == _ctx.Settings.ErrorValueMeanH
                && _singleGrabDetailIndexRowErrMax == _ctx.Settings.ErrorValueMaxH;
        }

        private void EnsureSingleGrabDetailIndex()
        {
            float hmV = _ctx.Settings.HessianMaxFactorV;
            float errMean = _ctx.Settings.ErrorValueMeanV;
            float errMax = _ctx.Settings.ErrorValueMaxV;
            if (IsSingleGrabDetailIndexCurrent())
                return;

            var sw = Stopwatch.StartNew();
            _singleGrabDetailIndex.Clear();
            if (_grabIdInfos.Count > 0)
            {
                var threshold = CreateThresholdContext();
                List<GrabDetail> details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath,
                    _grabIdInfos[_grabIdInfos.Count - 1].GrabId,
                    _grabIdInfos[0].GrabId,
                    threshold);
                foreach (GrabDetail item in details)
                    _singleGrabDetailIndex[item.GrabId] = item;
            }

            _singleGrabDetailIndexRoot = _statsDataRootPath;
            _singleGrabDetailIndexHmV = hmV;
            _singleGrabDetailIndexErrMean = errMean;
            _singleGrabDetailIndexErrMax = errMax;
            _singleGrabDetailIndexHmH = _ctx.Settings.HessianMaxFactorH;
            _singleGrabDetailIndexRowErrMean = _ctx.Settings.ErrorValueMeanH;
            _singleGrabDetailIndexRowErrMax = _ctx.Settings.ErrorValueMaxH;
            _singleGrabDetailIndexReady = true;
            FlowTrace.Log($"DT stats index rows={_singleGrabDetailIndex.Count} ms={sw.ElapsedMilliseconds}");
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
                if (_singleGrabDetailIndex.TryGetValue(info.GrabId, out GrabDetail detail))
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
            _ctx.Settings.ErrorValueMaxH);

        // ══════════════════════════════════════════════════════════════
        // Detail ListView
        // ══════════════════════════════════════════════════════════════


        private void OnGrabDetailRowCommitted(object sender, GrabDetailRowCommittedEventArgs e)
        {
            if (StatComboGuard.IsSet) return;
            string grabId = e.GrabId;

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
        public void RefreshMuraProfileForSettingsChange() => _muraChart?.RefreshForSettingsChange();

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

        private void BtnShowFail_Click(object sender, EventArgs e)
        {
            string preferredGrabId = Convert.ToString(_ctx.CbDataGrabId.SelectedItem);
            _showFailOnly = !_showFailOnly;
            _ctx.BtnShowFail.Text = _showFailOnly ? "○ 顯示全部" : "△ 顯示異常";
            _ctx.BtnShowFail.BackColor = _showFailOnly
                ? Color.FromArgb(255, 235, 238)
                : SystemColors.Control;

            EnsureSingleGrabDetailIndex();
            RefreshRangeGrabIdInfos();
            int rangeOptions = _dateGrabIdNavigator.RefreshFilteredGrabIdCombos(preferredGrabId);
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
            if (_dateGrabIdNavigator.ActiveStatMode == _ctx.GroupBoxGrabIdRange)
                RefreshStats();
            else if (_dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet)
                RefreshSelectedGrab();
            _ctx.GrabDetailList.Highlight(selectedGrabId);
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
                ? SelectFailRangeInfos(_grabIdInfos, _singleGrabDetailIndex)
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
            _yieldPeriodCharts?.RefreshPeriodCharts();
        }
        public void PopulateAllGrabIdCombos(bool selectDataGrabId = false) =>
            _dateGrabIdNavigator.PopulateAllGrabIdCombos(selectDataGrabId);
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
