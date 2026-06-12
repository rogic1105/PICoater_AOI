using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>Data tab 所有 UI 控制項的參照。</summary>
    public class DataStatisticsContext
    {
        // --- 時間範圍 ---
        public ComboBox CbStartDate { get; set; }
        public ComboBox CbStartTime { get; set; }
        public ComboBox CbEndDate { get; set; }
        public ComboBox CbEndTime { get; set; }

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
        public GroupBox GroupBoxTimeRange { get; set; }
        public GroupBox GrpReviewGrabNav { get; set; }
        public GroupBox GrpReviewTimePeriod { get; set; }

        // --- 統計 ---
        public ListView ListViewGrabDetail { get; set; }
        public Panel[] PanelStatCams { get; set; }
        public Chart ChartDataPatch { get; set; }

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

        public int CameraCount { get; set; } = 7;
    }

    /// <summary>
    /// 管理 tabPageData 的所有邏輯：統計、ComboBox 級聯、序號導航、趨勢圖、Detail ListView。
    /// 跨 Tab 同步透過事件通知 Form。
    /// </summary>
    public class DataStatisticsPresenter
    {
        private readonly DataStatisticsContext _ctx;
        private readonly InspectionStatsPresenter _statsPresenter;

        // --- 狀態 ---
        private string _statsDataRootPath = string.Empty;
        private SortedSet<DateTime> _statAvailableTimes = new SortedSet<DateTime>();
        private List<GrabIdInfo> _grabIdInfos = new List<GrabIdInfo>();
        private List<GrabDetail> _currentDetails = new List<GrabDetail>();
        private bool _showFailOnly;
        private GroupBox _activeStatMode;

        // --- 圖表導航 ---
        private List<int> _chartYears = new List<int>();
        private List<int> _chartMonths = new List<int>();
        private List<int> _chartDays = new List<int>();

        // --- Guards ---
        internal readonly EventGuard StatComboGuard = new EventGuard();
        internal readonly EventGuard GrabIdNavGuard = new EventGuard();
        internal readonly EventGuard GrabIdCrossGuard = new EventGuard();
        private readonly EventGuard _chartNavGuard = new EventGuard();
        // listViewGrabDetail commit 時設 true：OnSingleSheetComboChanged 跳過範圍 cb 同步，
        // 保留使用者目前的 cbDataIdStart/End + cbDataDateStart/Time + cbDataDateEnd/Time 不變。
        private bool _suppressRangeOnSingleSheetSync;

        private int _lastChartToggleTick;

        // --- Mura Profile Chart ---
        private ColumnCurveChartHelper _muraProfileHelper;

        // --- 常數 ---
        private static readonly Color _detailPass = Color.FromArgb(232, 245, 233);
        private static readonly Color _detailFail = Color.FromArgb(255, 235, 238);
        private static readonly Color _activeGrpFill = Color.FromArgb(220, 248, 225);
        private static readonly Color _activeGrpBorder = Color.FromArgb(0, 140, 60);

        // --- 事件 ---
        /// <summary>Data tab 序號選取 → 通知 Form 載入拼接圖。
        /// (grabId, earliest, latest, selectedIndex)</summary>
        public event Action<string, DateTime, DateTime, int> GrabIdSelectedFromData;

        /// <summary>Review tab 序號選取 → 通知 Form 載入拼接圖。
        /// (grabId, earliest, latest, selectedIndex)</summary>
        public event Action<string, DateTime, DateTime, int> GrabIdSelectedFromReview;

        /// <summary>通知 Form period combo 手動變更。</summary>
        public event Action PeriodComboManualChanged;

        public string StatsDataRootPath => _statsDataRootPath;
        public SortedSet<DateTime> StatAvailableTimes => _statAvailableTimes;
        public List<GrabIdInfo> GrabIdInfos => _grabIdInfos;

        public DataStatisticsPresenter(DataStatisticsContext ctx)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _statsPresenter = new InspectionStatsPresenter(ctx.PanelStatCams);
            _activeStatMode = ctx.GroupBoxGrabIdRange;
        }

        // ══════════════════════════════════════════════════════════════
        // 初始化
        // ══════════════════════════════════════════════════════════════

        public void Initialize()
        {
            _statsPresenter.Initialize();

            DateTime today = DateTime.Today;
            PopulateStatDateCombos(today.AddDays(-7), today);

            _statsDataRootPath = _ctx.Settings?.CaptureRootPath ?? string.Empty;

            _ctx.BtnSelectDataFolder.Click += BtnSelectDataFolder_Click;
            _ctx.BtnShowFail.Click += BtnShowFail_Click;
            WireStatDateCombos();
            InitGrabDetailListView();
            InitMuraProfileChart();
            InitPeriodCharts();
            _ctx.CbChartYear.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) OnChartYearIndexChanged(); };
            _ctx.CbChartMonth.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) OnChartMonthIndexChanged(); };
            _ctx.CbChartDay.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) OnChartDayIndexChanged(); };

            _ctx.CbGrabIdStart.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: true);
            _ctx.CbGrabIdEnd.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: false);
            _ctx.CbDataGrabId.SelectedIndexChanged += (s, e) => OnSingleSheetComboChanged();
            _ctx.CbReviewGrabId.SelectedIndexChanged += (s, e) => OnReviewGrabIdChanged();
            _ctx.GrpReviewGrabNav.Click += (s, e) => OnReviewGrabIdChanged();
            _ctx.GrpReviewTimePeriod.Click += (s, e) => PeriodComboManualChanged?.Invoke();

            // Data tab：點選 GroupBox 標題切換 active stat 模式（與 GrpReviewGrabNav.Click 相同模式）
            _ctx.GrpDataSingleSheet.Click   += (s, e) => SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            _ctx.GroupBoxGrabIdRange.Click  += (s, e) => SwitchActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
            _ctx.GroupBoxTimeRange.Click    += (s, e) => SwitchActiveStatGroupBox(_ctx.GroupBoxTimeRange);
        }

        // ══════════════════════════════════════════════════════════════
        // 資料夾選擇
        // ══════════════════════════════════════════════════════════════

        /// <summary>通知 Form 執行資料夾選擇的完整流程（含 Review tab 同步）。</summary>
        public event Action<string> DataFolderSelected;

        private void BtnSelectDataFolder_Click(object sender, EventArgs e)
        {
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.Description = "選擇 AniloxCaptures 根目錄";
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
            _statsDataRootPath = path;
            _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(path);
            _grabIdInfos = InspectionStatisticsService.LoadGrabIdInfosDescending(path);

            PopulateAllGrabIdCombos(selectDataGrabId: false);

            if (_statAvailableTimes.Count > 0)
                PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max);

            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);

            // 預設單片模式（與 Review tab btnReviewSelectFolder 一致）— 最新一筆 grab（descending [0]）。
            // 對齊 cbDataIdStart=End=0 → RefreshStats 的單片分支取得單 grab 範圍。
            SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            if (_grabIdInfos.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = 0;
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;
                    _ctx.CbDataGrabId.SelectedIndex = 0;
                }
            }
            RefreshStats();
        }

        /// <summary>從 Review tab 選擇資料夾後同步載入序號清單。</summary>
        public void SyncFromReviewFolder(string path)
        {
            _statsDataRootPath = path;
            _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(path);
            _grabIdInfos = InspectionStatisticsService.LoadGrabIdInfosDescending(path);

            PopulateAllGrabIdCombos();

            if (_statAvailableTimes.Count > 0)
                PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max);

            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);
            RefreshStats();
        }

        // ══════════════════════════════════════════════════════════════
        // 日期/時間 ComboBox
        // ══════════════════════════════════════════════════════════════

        private void PopulateStatDateCombos(DateTime start, DateTime end)
        {
            // 包 StatComboGuard 抑制 cbDataDateStart/Time + cbDataDateEnd/Time 的 SelectedIndexChanged：
            // 避免程式化填充觸發 OnStartComboChanged → SetActiveStatGroupBox(TimeRange) +
            // RefreshStats 級聯（4 次 RefreshStats 用大時間範圍掃 CSV 灌大量資料進 listView，
            // 隨後 LoadDataFolder 才設回 SingleSheet 收回 — listViewGrabDetail「瞬間爆量再縮回」）。
            using (StatComboGuard.Enter())
            {
                var dates = GetAvailableDateStrings();
                string startDateStr = start.ToString("yyyy-MM-dd");
                string endDateStr = end.ToString("yyyy-MM-dd");
                string startTimeStr = start.ToString("HH:mm:ss");
                string endTimeStr = end.ToString("HH:mm:ss");

                _ctx.CbStartDate.Items.Clear();
                _ctx.CbStartDate.Items.AddRange(dates.ToArray());
                int si = dates.IndexOf(startDateStr);
                _ctx.CbStartDate.SelectedIndex = si >= 0 ? si : (dates.Count > 0 ? dates.Count - 1 : -1);
                RefreshStatTimeCombo(_ctx.CbStartDate, _ctx.CbStartTime, startTimeStr);

                _ctx.CbEndDate.Items.Clear();
                _ctx.CbEndDate.Items.AddRange(dates.ToArray());
                int ei = dates.IndexOf(endDateStr);
                _ctx.CbEndDate.SelectedIndex = ei >= 0 ? ei : (dates.Count > 0 ? 0 : -1);
                RefreshStatTimeCombo(_ctx.CbEndDate, _ctx.CbEndTime, endTimeStr);
            }
        }

        private void RefreshStatTimeCombo(ComboBox dateCb, ComboBox timeCb, string preferred)
        {
            var times = GetAvailableTimeStrings(dateCb.Text);
            timeCb.Items.Clear();
            timeCb.Items.AddRange(times.ToArray());
            if (times.Count == 0) return;
            int idx = times.IndexOf(preferred);
            timeCb.SelectedIndex = idx >= 0 ? idx : (times.Count > 0 ? 0 : -1);
        }

        // ══════════════════════════════════════════════════════════════
        // Cascading ComboBox 邏輯
        // ══════════════════════════════════════════════════════════════

        private void WireStatDateCombos()
        {
            _ctx.CbStartDate.SelectedIndexChanged += (s, e) => OnStartComboChanged(1);
            _ctx.CbStartTime.SelectedIndexChanged += (s, e) => OnStartComboChanged(2);
            _ctx.CbEndDate.SelectedIndexChanged += (s, e) => OnEndComboChanged(1);
            _ctx.CbEndTime.SelectedIndexChanged += (s, e) => OnEndComboChanged(2);
        }

        private void OnStartComboChanged(int fromLevel)
        {
            if (StatComboGuard.IsSet) return;
            SetActiveStatGroupBox(_ctx.GroupBoxTimeRange);
            if (_statAvailableTimes.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(_ctx.CbStartDate, _ctx.CbStartTime, _ctx.CbStartTime.Text);
                    ClampEndToStart();
                }
            }
            RefreshStats();
        }

        private void OnEndComboChanged(int fromLevel)
        {
            if (StatComboGuard.IsSet) return;
            SetActiveStatGroupBox(_ctx.GroupBoxTimeRange);
            if (_statAvailableTimes.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(_ctx.CbEndDate, _ctx.CbEndTime, _ctx.CbEndTime.Text);
                    ClampStartToEnd();
                }
            }
            RefreshStats();
        }

        /// <summary>
        /// 程式化把日期時間 combo 對齊到指定 DateTime（**不會觸發 OnStart/EndComboChanged 切換到 TimeRange 模式**）。
        /// 呼叫端必須已包在 `StatComboGuard.Enter()` 內 — H4：guard 邊界很重要，新增 caller 時務必確認。
        /// </summary>
        private void SetCombosToDateTime(bool isStart, DateTime dt)
        {
            string dateStr = dt.ToString("yyyy-MM-dd");
            string timeStr = dt.ToString("HH:mm:ss");
            if (isStart)
            {
                if (_ctx.CbStartDate.Items.Contains(dateStr)) _ctx.CbStartDate.SelectedItem = dateStr;
                else _ctx.CbStartDate.Text = dateStr;
                RefreshStatTimeCombo(_ctx.CbStartDate, _ctx.CbStartTime, timeStr);
            }
            else
            {
                if (_ctx.CbEndDate.Items.Contains(dateStr)) _ctx.CbEndDate.SelectedItem = dateStr;
                else _ctx.CbEndDate.Text = dateStr;
                RefreshStatTimeCombo(_ctx.CbEndDate, _ctx.CbEndTime, timeStr);
            }
        }

        private void ClampEndToStart()
        {
            if (!TryBuildDateTimeFromCombos(_ctx.CbStartDate, _ctx.CbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(_ctx.CbEndDate, _ctx.CbEndTime, out DateTime end)) return;
            if (start <= end) return;
            var view = _statAvailableTimes.GetViewBetween(start, DateTime.MaxValue);
            DateTime newEnd = view.Count > 0 ? view.Min : _statAvailableTimes.Max;
            SetCombosToDateTime(false, newEnd);
        }

        private void ClampStartToEnd()
        {
            if (!TryBuildDateTimeFromCombos(_ctx.CbStartDate, _ctx.CbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(_ctx.CbEndDate, _ctx.CbEndTime, out DateTime end)) return;
            if (start <= end) return;
            var view = _statAvailableTimes.GetViewBetween(DateTime.MinValue, end);
            DateTime newStart = view.Count > 0 ? view.Max : _statAvailableTimes.Min;
            SetCombosToDateTime(true, newStart);
        }

        // ══════════════════════════════════════════════════════════════
        // 序號 ComboBox
        // ══════════════════════════════════════════════════════════════

        private void OnGrabIdComboChanged(bool isStart)
        {
            if (StatComboGuard.IsSet || _grabIdInfos.Count == 0) return;
            SetActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);

            int idx1 = _ctx.CbGrabIdStart.SelectedIndex;
            int idx2 = _ctx.CbGrabIdEnd.SelectedIndex;
            if (idx1 < 0 || idx2 < 0) return;

            using (StatComboGuard.Enter())
            {
                if (isStart && idx1 < idx2)
                    _ctx.CbGrabIdEnd.SelectedIndex = idx1;
                else if (!isStart && idx2 > idx1)
                    _ctx.CbGrabIdStart.SelectedIndex = idx2;

                var startInfo = _grabIdInfos[_ctx.CbGrabIdStart.SelectedIndex];
                var endInfo = _grabIdInfos[_ctx.CbGrabIdEnd.SelectedIndex];
                SetCombosToDateTime(true, startInfo.Earliest);
                SetCombosToDateTime(false, endInfo.Latest);
            }

            RefreshStats();
        }

        private void OnSingleSheetComboChanged()
        {
            UpdateDataGrabIdNavState();
            if (StatComboGuard.IsSet || _grabIdInfos.Count == 0) return;
            if (GrabIdCrossGuard.IsSet) return;

            SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            int idx = _ctx.CbDataGrabId.SelectedIndex;
            if (idx < 0) return;

            if (!_suppressRangeOnSingleSheetSync)
            {
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = idx;
                    _ctx.CbGrabIdEnd.SelectedIndex = idx;
                    var info = _grabIdInfos[idx];
                    SetCombosToDateTime(true, info.Earliest);
                    SetCombosToDateTime(false, info.Latest);
                }
            }

            RefreshStats();

            // 跨 Tab 同步：通知 Form
            if (!GrabIdCrossGuard.IsSet && _ctx.CbReviewGrabId.Items.Count > 0
                && idx < _ctx.CbReviewGrabId.Items.Count)
            {
                var info = _grabIdInfos[idx];
                GrabIdSelectedFromData?.Invoke(info.GrabId, info.Earliest, info.Latest, idx);
            }
        }


        private void OnReviewGrabIdChanged()
        {
            UpdateGrabIdNavState();
            if (GrabIdNavGuard.IsSet) return;
            if (GrabIdCrossGuard.IsSet) return;
            if (_grabIdInfos.Count == 0) return;
            int idx = _ctx.CbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= _grabIdInfos.Count) return;

            var info = _grabIdInfos[idx];
            GrabIdSelectedFromReview?.Invoke(info.GrabId, info.Earliest, info.Latest, idx);
        }

        /// <summary>由 Form 呼叫：Review→Data 同步完成後設定 combo + 統計。</summary>
        public void SyncDataGrabIdFromReview(int idx, GrabIdInfo info)
        {
            using (GrabIdCrossGuard.Enter())
            {
                _ctx.CbDataGrabId.SelectedIndex = idx;
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = idx;
                    _ctx.CbGrabIdEnd.SelectedIndex = idx;
                    SetCombosToDateTime(true, info.Earliest);
                    SetCombosToDateTime(false, info.Latest);
                }
                RefreshStats();
                SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            }
        }

        // ══════════════════════════════════════════════════════════════
        // 序號導航
        // ══════════════════════════════════════════════════════════════

        private void StepReviewGrabId(int delta)
        {
            if (_grabIdInfos.Count == 0) return;
            int next = _ctx.CbReviewGrabId.SelectedIndex + delta;
            if (next >= 0 && next < _ctx.CbReviewGrabId.Items.Count)
                _ctx.CbReviewGrabId.SelectedIndex = next;
        }

        private void StepDataGrabId(int delta)
        {
            if (_grabIdInfos.Count == 0) return;
            int next = _ctx.CbDataGrabId.SelectedIndex + delta;
            if (next >= 0 && next < _ctx.CbDataGrabId.Items.Count)
                _ctx.CbDataGrabId.SelectedIndex = next;
        }

        public void UpdateGrabIdNavState()
        {
            int idx = _ctx.CbReviewGrabId.SelectedIndex;
            int count = _ctx.CbReviewGrabId.Items.Count;
            UpdateDataGrabIdNavState();
        }

        private void UpdateDataGrabIdNavState()
        {
            int idx = _ctx.CbDataGrabId.SelectedIndex;
            int count = _ctx.CbDataGrabId.Items.Count;
        }

        /// <summary>時間 ComboBox 變更時，同步 cbReviewId 到包含該時間的序號。</summary>
        public void SyncGrabIdFromTime(DateTime current)
        {
            if (GrabIdNavGuard.IsSet || _grabIdInfos.Count == 0) return;

            int bestIdx = -1;
            long bestDiff = long.MaxValue;
            for (int i = 0; i < _grabIdInfos.Count; i++)
            {
                var info = _grabIdInfos[i];
                if (current >= info.Earliest && current <= info.Latest)
                {
                    bestIdx = i;
                    break;
                }
                long diff = Math.Abs(current.Ticks - info.Earliest.Ticks);
                if (diff < bestDiff)
                {
                    bestDiff = diff;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0 && bestIdx < _ctx.CbReviewGrabId.Items.Count
                && bestIdx != _ctx.CbReviewGrabId.SelectedIndex)
            {
                using (GrabIdNavGuard.Enter())
                    _ctx.CbReviewGrabId.SelectedIndex = bestIdx;
            }
        }

        // ══════════════════════════════════════════════════════════════
        // 統計
        // ══════════════════════════════════════════════════════════════

        public void RefreshStats()
        {
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            // view-time threshold context：以當前 Settings 的閾值 + 垂直正規值即時重算 Pass/Fail，
            // 不再用 CSV 內的 MaxExceed/MeanExceed（那是 capture-time baked-in）。
            var ctx = new ThresholdContext(
                _ctx.Settings.HessianMaxFactorV,
                _ctx.Settings.ErrorValueMeanV,
                _ctx.Settings.ErrorValueMaxV);

            // SingleSheet mode：用 cbDataId.SelectedIndex 算單 grab stats（start=end=該 grab）。
            // 不靠 cbDataIdStart/End 範圍，這樣 listViewGrabDetail 點選後（_suppressRangeOnSingleSheetSync=true
            // 跳過 range cb 同步）stats 仍對齊到剛點的單 grab。
            if (_activeStatMode == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < _grabIdInfos.Count)
            {
                var grab = _grabIdInfos[_ctx.CbDataGrabId.SelectedIndex];
                var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, grab.GrabId, grab.GrabId, ctx);
                var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath, grab.GrabId, grab.GrabId, ctx);
                _statsPresenter.Update(stats);
                _currentDetails = details;
                ApplyFailFilter();
                UpdateMuraProfileChart(null);  // SingleSheet branch 內自己查 cbDataId 取 grab
                return;
            }

            if (_activeStatMode != _ctx.GroupBoxTimeRange
                && _ctx.CbGrabIdStart.SelectedIndex >= 0 && _ctx.CbGrabIdEnd.SelectedIndex >= 0
                && _grabIdInfos.Count > 0)
            {
                var startInfo = _grabIdInfos[_ctx.CbGrabIdStart.SelectedIndex];
                var endInfo = _grabIdInfos[_ctx.CbGrabIdEnd.SelectedIndex];

                var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startInfo.GrabId, endInfo.GrabId, ctx);
                var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath, startInfo.GrabId, endInfo.GrabId, ctx);

                _statsPresenter.Update(stats);
                _currentDetails = details;
                ApplyFailFilter();

                int si = _ctx.CbGrabIdStart.SelectedIndex;
                int ei = _ctx.CbGrabIdEnd.SelectedIndex;
                int lo = Math.Min(si, ei); int hi = Math.Max(si, ei);
                var rangeInfos = _grabIdInfos.GetRange(lo, hi - lo + 1);
                UpdateMuraProfileChart(EvenSample(rangeInfos, 50));
                return;
            }

            if (!TryParseStatDateTime(out DateTime start, out DateTime end)) return;

            var grabInfosInRange = _grabIdInfos
                .Where(g => g.Earliest <= end && g.Latest >= start).ToList();

            if (grabInfosInRange.Count > 0)
            {
                string startId = grabInfosInRange.OrderBy(g => g.GrabId, StringComparer.Ordinal).First().GrabId;
                string endId = grabInfosInRange.OrderBy(g => g.GrabId, StringComparer.Ordinal).Last().GrabId;

                var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startId, endId, ctx);
                var details = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                    _statsDataRootPath, startId, endId, ctx);

                _statsPresenter.Update(stats);
                _currentDetails = details;
                UpdateMuraProfileChart(EvenSample(grabInfosInRange, 10));
            }
            else
            {
                var statsTime = InspectionStatisticsService.Compute(_statsDataRootPath, start, end, ctx);
                _statsPresenter.Update(statsTime);
                _currentDetails = new List<GrabDetail>();
                ClearMuraProfileChart();
            }
            ApplyFailFilter();
        }

        private bool TryParseStatDateTime(out DateTime start, out DateTime end)
        {
            start = end = DateTime.MinValue;
            if (!TryBuildDateTimeFromCombos(_ctx.CbStartDate, _ctx.CbStartTime, out start)) return false;
            if (!TryBuildDateTimeFromCombos(_ctx.CbEndDate, _ctx.CbEndTime, out end)) return false;
            if (end.Millisecond == 0) end = end.AddMilliseconds(999);
            return start <= end;
        }

        private static bool TryBuildDateTimeFromCombos(ComboBox dateCb, ComboBox timeCb, out DateTime result)
        {
            result = DateTime.MinValue;
            string dateText = dateCb.Text ?? "";
            string timeText = timeCb.Text ?? "";
            string combined = dateText + " " + timeText;
            if (DateTime.TryParseExact(combined, new[] { "yyyy-MM-dd HH:mm:ss.fff", "yyyy-MM-dd HH:mm:ss" },
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.None, out result))
                return true;
            return false;
        }

        // ══════════════════════════════════════════════════════════════
        // Detail ListView
        // ══════════════════════════════════════════════════════════════

        private void InitGrabDetailListView()
        {
            var lv = _ctx.ListViewGrabDetail;
            lv.View = View.Details;
            lv.FullRowSelect = true;
            lv.GridLines = true;
            lv.Columns.Clear();
            lv.Items.Clear();

            lv.Columns.Add("料件序號", -1, HorizontalAlignment.Center);
            for (int i = 1; i <= _ctx.CameraCount; i++)
                lv.Columns.Add($"{i}", -1, HorizontalAlignment.Center);
            FitListViewColumnsProportional(lv);

            // 點選明細列表的列 → MouseDown 時 ListView 預設視覺先反白（顯示被選中），
            // MouseUp 才 commit 切到該序號（與 cbDataId 變更流程共用 OnSingleSheetComboChanged）。
            // commit 時包 _suppressRangeOnSingleSheetSync 跳過範圍 cb 同步，
            // 保留 cbDataIdStart/End + cbDataDateStart/Time + cbDataDateEnd/Time 不變。
            lv.MouseUp += OnGrabDetailRowCommitted;
        }

        private string _lastListViewSelectedGrabId;

        private void OnGrabDetailRowCommitted(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left) return;
            if (StatComboGuard.IsSet) return;
            var lv = _ctx.ListViewGrabDetail;
            if (lv.SelectedItems.Count == 0) return;
            string grabId = lv.SelectedItems[0].Text;
            if (string.IsNullOrEmpty(grabId)) return;

            // Toggle：第二次點同 row + 已是 SingleSheet → 切回 GroupBoxGrabIdRange（範圍模式，stats 用 cbDataIdStart/End）
            if (grabId == _lastListViewSelectedGrabId && _activeStatMode == _ctx.GrpDataSingleSheet)
            {
                _lastListViewSelectedGrabId = null;
                lv.SelectedIndices.Clear();  // 清掉反白，視覺回到「無選中」
                SwitchActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
                RefreshStats();
                return;
            }
            _lastListViewSelectedGrabId = grabId;

            int idx = _ctx.CbDataGrabId.Items.IndexOf(grabId);
            if (idx < 0) return;
            if (_ctx.CbDataGrabId.SelectedIndex == idx)
            {
                // SelectedIndex 沒變 → 不會觸發 OnSingleSheetComboChanged，但仍需確保 active 模式為單片
                if (_activeStatMode != _ctx.GrpDataSingleSheet)
                {
                    SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
                    RefreshStats();  // 切 mode 後 stats + chartDataVertical 對齊單片
                }
                return;
            }
            _suppressRangeOnSingleSheetSync = true;
            try { _ctx.CbDataGrabId.SelectedIndex = idx; } // → OnSingleSheetComboChanged 接手（內含 RefreshStats）
            finally { _suppressRangeOnSingleSheetSync = false; }
        }

        private void UpdateGrabDetailListView(List<GrabDetail> details)
        {
            var lv = _ctx.ListViewGrabDetail;
            // 改用 MouseUp 訂閱後，Items.Clear/Add 不會觸發 commit 路徑，
            // 不需 unsubscribe/resubscribe；BeginUpdate/EndUpdate 包住批量重填，
            // SelectedIndices.Clear() 避免殘留高亮。
            lv.BeginUpdate();
            lv.Items.Clear();
            lv.SelectedIndices.Clear();

            foreach (var d in details)
            {
                var item = new ListViewItem(d.GrabId);
                bool rowHasFail = false;

                for (int i = 0; i < _ctx.CameraCount; i++)
                {
                    if (d.CamResult[i] == null)
                        item.SubItems.Add("—");
                    else if (d.CamResult[i] == false)
                        item.SubItems.Add("○");
                    else
                    {
                        item.SubItems.Add("×");
                        rowHasFail = true;
                    }
                }

                item.BackColor = rowHasFail ? _detailFail : _detailPass;
                lv.Items.Add(item);
            }

            lv.EndUpdate();
            lv.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
        }

        public static void FitListViewColumnsProportional(ListView lv, bool useContent = false)
        {
            if (lv.Columns.Count == 0) return;
            int available = lv.ClientSize.Width - SystemInformation.VerticalScrollBarWidth;
            if (available <= 0) return;

            using (var g = lv.CreateGraphics())
            {
                var weights = new float[lv.Columns.Count];
                float totalWeight = 0;
                for (int i = 0; i < lv.Columns.Count; i++)
                {
                    float w = g.MeasureString(lv.Columns[i].Text + "WW", lv.Font).Width;
                    if (useContent)
                    {
                        foreach (ListViewItem item in lv.Items)
                        {
                            string text = i == 0 ? item.Text
                                        : i < item.SubItems.Count ? item.SubItems[i].Text : "";
                            float cw = g.MeasureString(text + "WW", lv.Font).Width;
                            if (cw > w) w = cw;
                        }
                    }
                    weights[i] = w;
                    totalWeight += w;
                }
                if (totalWeight <= 0) return;

                int assigned = 0;
                for (int i = 0; i < lv.Columns.Count; i++)
                {
                    int colW = (i < lv.Columns.Count - 1)
                        ? (int)(available * weights[i] / totalWeight)
                        : available - assigned;
                    lv.Columns[i].Width = Math.Max(20, colW);
                    assigned += lv.Columns[i].Width;
                }
            }
        }

        // ══════════════════════════════════════════════════════════════
        // Mura 空間分布曲線圖（拼接式，與 chartLiveVertical 相同格式）
        // ══════════════════════════════════════════════════════════════

        private void InitMuraProfileChart()
        {
            if (_ctx.ChartDataPatch == null) return;
            _muraProfileHelper = new ColumnCurveChartHelper(_ctx.ChartDataPatch);
        }

        private void UpdateMuraProfileChart(IList<GrabIdInfo> grabIds)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;

            // 單片模式（GrpDataSingleSheet）：永遠用 cbDataId.SelectedIndex 對應 grab，不依賴 caller 傳入的 grabIds。
            // 原因：listViewGrabDetail 點選時 _suppressRangeOnSingleSheetSync=true 跳過範圍 cb 同步，
            // 但 caller 仍會用舊 cbDataIdStart/End 範圍呼這函式 → 若用 grabIds[0] 會顯示舊範圍的第一筆而非剛點的 grab。
            // view-time 正規值 rescale（HM_capture / HM_current）讓改 PropertyGrid 正規值時曲線坡度立即變化。
            if (_activeStatMode == _ctx.GrpDataSingleSheet)
            {
                int singleIdx = _ctx.CbDataGrabId.SelectedIndex;
                if (singleIdx >= 0 && singleIdx < _grabIdInfos.Count)
                    UpdateMuraProfileForSingleGrab(_grabIdInfos[singleIdx]);
                else
                    ClearMuraProfileChart();
                return;
            }

            if (grabIds == null || grabIds.Count == 0)
            {
                ClearMuraProfileChart();
                return;
            }
            // 範圍/時間模式：aggregate 多 grab 平均，當作歷史快照不做 view-time rescale

            // ── 範圍/時間模式：舊 aggregate 邏輯 ──
            var (meanDict, maxDict) = InspectionStatisticsService.LoadAvgMuraProfile(
                _statsDataRootPath, grabIds);
            if (meanDict.Count == 0)
            {
                ClearMuraProfileChart();
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
        /// 更新 chartDataVertical，與 chartReviewVertical 完全對齊。不依賴 camReviewMain 是否載入。
        /// 套用 view-time 正規值 rescale：display = (bin/255) × (HM_capture / HM_current)；
        /// 改 PropertyGrid 正規值會立刻反映在曲線坡度上。
        /// </summary>
        private void UpdateMuraProfileForSingleGrab(GrabIdInfo info)
        {
            if (_muraProfileHelper == null || _ctx.Settings == null) return;
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            var grabCfg = InspectionStatisticsService.LoadConfigForGrabId(
                _statsDataRootPath, info.GrabId, info.Earliest, info.Latest);
            var grouped = InspectionStatisticsService.LoadImagePathsForGrabId(
                _statsDataRootPath, info.GrabId, info.Earliest, info.Latest);

            int camCount = _ctx.CameraCount;
            var allMean = new float[camCount][];
            var allMax  = new float[camCount][];
            for (int i = 0; i < camCount; i++)
            {
                int camId = i + 1;
                if (grouped.TryGetValue(camId, out var paths) && paths.Count > 0)
                    CurveMergeHelper.MergeCurves(paths, out allMean[i], out allMax[i]);
            }

            // view-time 正規值 rescale：chartDataVertical 是垂直曲線，用 V 的 capture/current ratio
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
        /// 由 PropertyGrid 變更觸發：刷新 chartDataVertical 的閾值線 + view-time 正規值 rescale。
        /// 不重做 RefreshStats（避免重算統計）；只重畫 chart。
        /// </summary>
        public void RefreshMuraProfileForSettingsChange()
        {
            if (_muraProfileHelper == null) return;
            _muraProfileHelper.SetThresholds(_ctx.Settings.ErrorValueMeanV, _ctx.Settings.ErrorValueMaxV);
            // 單片模式才需要按 HM 重算曲線坡度；aggregate 模式維持快照
            if (_activeStatMode == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < _grabIdInfos.Count)
            {
                UpdateMuraProfileForSingleGrab(_grabIdInfos[_ctx.CbDataGrabId.SelectedIndex]);
            }
        }

        /// <summary>
        /// SingleSheet 模式：直接使用 Review tab 已載入的曲線資料（已套 view-time HM rescale），
        /// 確保 chartDataVertical 與 chartReviewVertical 完全一致（相同 OPS/Pos 與顯示值）。
        /// </summary>
        public void SyncMuraProfileFromReview(float[][] mean, float[][] max,
            double[] ops, double[] pos, float errMean, float errMax)
        {
            if (_muraProfileHelper == null) return;
            CurveMergeHelper.UpdateOverviewChart(mean, max, ops, pos, errMean, errMax,
                _muraProfileHelper, _ctx.CameraCount, StitchMode.Vertical, null);
        }

        private void ClearMuraProfileChart()
        {
            if (_ctx.ChartDataPatch == null) return;
            foreach (var s in _ctx.ChartDataPatch.Series)
                s.Points.Clear();
        }

        private static List<T> EvenSample<T>(IList<T> list, int maxCount)
        {
            if (list.Count <= maxCount) return new List<T>(list);
            var result = new List<T>(maxCount);
            double step = (list.Count - 1.0) / (maxCount - 1);
            for (int i = 0; i < maxCount; i++)
                result.Add(list[(int)Math.Round(i * step)]);
            return result;
        }

        // ══════════════════════════════════════════════════════════════
        // 異常篩選
        // ══════════════════════════════════════════════════════════════

        private void BtnShowFail_Click(object sender, EventArgs e)
        {
            _showFailOnly = !_showFailOnly;
            _ctx.BtnShowFail.Text = _showFailOnly ? "○ 顯示全部" : "△ 顯示異常";
            _ctx.BtnShowFail.BackColor = _showFailOnly
                ? Color.FromArgb(255, 235, 238)
                : SystemColors.Control;
            ApplyFailFilter();
        }

        private void ApplyFailFilter()
        {
            var toShow = _showFailOnly
                ? _currentDetails.Where(d => d.CamResult.Any(r => r == true)).ToList()
                : _currentDetails;
            UpdateGrabDetailListView(toShow);
        }

        // ══════════════════════════════════════════════════════════════
        // 趨勢圖（年 / 月 / 日）
        // ══════════════════════════════════════════════════════════════

        private void InitPeriodCharts()
        {
            var cs = _ctx.Settings.Chart;
            InitOneChart(_ctx.ChartDataYieldYearly, yDefault: cs.YearlyYMax, xCount: 12, xStart: 1);
            InitOneChart(_ctx.ChartDataYieldMonthly, yDefault: cs.MonthlyYMax, xCount: 31, xStart: 1);
            InitOneChart(_ctx.ChartDataYieldDaily, yDefault: cs.DailyYMax, xCount: 24, xStart: 0);

            if (cs.ScaleMode == ChartScaleMode.Auto)
            {
                var empty = new List<PeriodStats>();
                ApplyAutoScale(_ctx.ChartDataYieldYearly, empty);
                ApplyAutoScale(_ctx.ChartDataYieldMonthly, empty);
                ApplyAutoScale(_ctx.ChartDataYieldDaily, empty);
            }

            _ctx.ChartDataYieldYearly.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldMonthly.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldDaily.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldYearly.MouseClick += PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldMonthly.MouseClick += PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldDaily.MouseClick += PeriodChart_ToggleAutoScale;
        }

        private void PeriodChart_ToggleAutoScale(object sender, MouseEventArgs e)
        {
            int now = Environment.TickCount;
            if (now - _lastChartToggleTick < 500) return;
            _lastChartToggleTick = now;

            var chart = (Chart)sender;
            if (chart.ChartAreas.Count == 0) return;

            bool isAuto = "auto".Equals(chart.Tag);

            if (isAuto)
            {
                int fixedMax = chart == _ctx.ChartDataYieldYearly ? _ctx.Settings.Chart.YearlyYMax
                             : chart == _ctx.ChartDataYieldMonthly ? _ctx.Settings.Chart.MonthlyYMax
                             : _ctx.Settings.Chart.DailyYMax;
                ApplyFixedScale(chart, fixedMax);
            }
            else
            {
                var data = new List<PeriodStats>();
                var sPass = chart.Series["合格"];
                var sFail = chart.Series["異常"];
                for (int i = 0; i < sPass.Points.Count; i++)
                    data.Add(new PeriodStats { Label = "", Pass = (int)sPass.Points[i].YValues[0], Fail = (int)sFail.Points[i].YValues[0] });
                ApplyAutoScale(chart, data);
            }
        }

        private static void InitOneChart(Chart chart, int xLabelAngle = 0, int yDefault = 10,
            int xCount = 0, int xStart = 1)
        {
            chart.ChartAreas.Clear();
            chart.Series.Clear();
            chart.Legends.Clear();
            chart.Titles.Clear();

            var area = new ChartArea("Main");
            area.AxisX.MajorGrid.Enabled = true;
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);
            area.AxisX.MajorGrid.LineDashStyle = ChartDashStyle.Dot;
            area.AxisX.MajorTickMark.Enabled = true;
            area.AxisX.MajorTickMark.LineColor = Color.FromArgb(120, 120, 120);
            area.AxisX.IsMarginVisible = false;
            area.AxisX.Interval = 1;
            area.AxisX.LabelStyle.Angle = xLabelAngle;
            area.AxisX.IsLabelAutoFit = false;
            area.AxisX.LabelStyle.Font = new Font("Arial", 10f);
            area.AxisY.LineColor = Color.Transparent;
            area.AxisY.MajorGrid.Enabled = false;
            area.AxisY.MajorTickMark.Enabled = false;
            area.AxisY.MinorTickMark.Enabled = false;
            area.AxisY.LabelStyle.Enabled = false;
            area.AxisY.Minimum = 0;
            area.AxisY.Interval = yDefault / 5.0;
            area.AxisY.Maximum = yDefault;
            area.AxisY.MajorGrid.Enabled = true;
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);
            area.AxisY.MajorGrid.LineDashStyle = ChartDashStyle.Dot;
            area.AxisY2.Enabled = AxisEnabled.True;
            area.AxisY2.MajorGrid.Enabled = false;
            area.AxisY2.MajorTickMark.Enabled = true;
            area.AxisY2.MajorTickMark.LineColor = Color.FromArgb(120, 120, 120);
            area.AxisY2.LabelStyle.Font = new Font("Arial", 5f);
            area.AxisY2.Minimum = 0;
            area.AxisY2.Maximum = yDefault;
            area.AxisY2.Interval = yDefault / 5.0;
            area.AxisY2.LabelStyle.Interval = yDefault;
            area.InnerPlotPosition.Auto = false;
            area.InnerPlotPosition.X = 0f;
            area.InnerPlotPosition.Y = 12f;
            area.InnerPlotPosition.Width = 93f;
            area.InnerPlotPosition.Height = 66f;
            chart.ChartAreas.Add(area);

            var legend = new Legend("L");
            legend.IsDockedInsideChartArea = true;
            legend.DockedToChartArea = "Main";
            legend.Docking = Docking.Top;
            legend.Alignment = StringAlignment.Far;
            legend.Font = new Font("Arial", 6.5f);
            legend.BackColor = Color.Transparent;
            legend.BorderColor = Color.Transparent;
            chart.Legends.Add(legend);

            var sPass = new Series("合格");
            sPass.ChartType = SeriesChartType.StackedColumn;
            sPass.Color = Color.FromArgb(102, 187, 106);
            sPass.ChartArea = "Main";
            sPass.Legend = "L";
            sPass.YAxisType = AxisType.Secondary;
            chart.Series.Add(sPass);

            var sFail = new Series("異常");
            sFail.ChartType = SeriesChartType.StackedColumn;
            sFail.Color = Color.FromArgb(239, 83, 80);
            sFail.ChartArea = "Main";
            sFail.Legend = "L";
            sFail.YAxisType = AxisType.Secondary;
            chart.Series.Add(sFail);

            if (xCount > 0)
            {
                for (int i = 0; i < xCount; i++)
                {
                    sPass.Points.AddXY((xStart + i).ToString(), 0);
                    sFail.Points.AddXY((xStart + i).ToString(), 0);
                }
            }
        }

        private void FillPeriodChart(Chart chart, List<PeriodStats> data)
        {
            var sPass = chart.Series["合格"];
            var sFail = chart.Series["異常"];
            sPass.Points.Clear();
            sFail.Points.Clear();
            foreach (var p in data)
            {
                sPass.Points.AddXY(p.Label, p.Pass);
                sFail.Points.AddXY(p.Label, p.Fail);
            }

            if (_ctx.Settings.Chart.ScaleMode == ChartScaleMode.Auto || "auto".Equals(chart.Tag))
                ApplyAutoScale(chart, data);
        }

        private static void ApplyAutoScale(Chart chart, List<PeriodStats> data)
        {
            chart.Tag = "auto";
            int maxTotal = 0;
            foreach (var p in data)
                maxTotal = Math.Max(maxTotal, p.Pass + p.Fail);
            int niceMax = Math.Max(5, (int)(Math.Ceiling(maxTotal / 5.0) * 5));
            SetChartYRange(chart, niceMax * 1.05, niceMax / 5.0, niceMax);
        }

        private static void ApplyFixedScale(Chart chart, int fixedMax)
        {
            chart.Tag = null;
            SetChartYRange(chart, fixedMax, fixedMax / 5.0, fixedMax);
        }

        private static void SetChartYRange(Chart chart, double yMax, double yStep, double labelInterval)
        {
            var area = chart.ChartAreas["Main"];
            area.AxisY.Maximum = yMax;
            area.AxisY.Interval = yStep;
            area.AxisY.MajorGrid.Interval = yStep;
            area.AxisY2.Maximum = yMax;
            area.AxisY2.Interval = yStep;
            area.AxisY2.MajorGrid.Interval = yStep;
            area.AxisY2.LabelStyle.Interval = labelInterval;
        }

        public void ApplyChartScaleFromSettings()
        {
            if (_ctx.Settings.Chart.ScaleMode == ChartScaleMode.Fixed)
            {
                ApplyFixedScale(_ctx.ChartDataYieldYearly, _ctx.Settings.Chart.YearlyYMax);
                ApplyFixedScale(_ctx.ChartDataYieldMonthly, _ctx.Settings.Chart.MonthlyYMax);
                ApplyFixedScale(_ctx.ChartDataYieldDaily, _ctx.Settings.Chart.DailyYMax);
            }
            else
            {
                foreach (var chart in new[] { _ctx.ChartDataYieldYearly, _ctx.ChartDataYieldMonthly, _ctx.ChartDataYieldDaily })
                {
                    if (chart.ChartAreas.Count == 0) continue;
                    var sPass = chart.Series["合格"];
                    var sFail = chart.Series["異常"];
                    var data = new List<PeriodStats>();
                    for (int i = 0; i < sPass.Points.Count; i++)
                        data.Add(new PeriodStats { Label = "", Pass = (int)sPass.Points[i].YValues[0], Fail = (int)sFail.Points[i].YValues[0] });
                    ApplyAutoScale(chart, data);
                }
            }
        }

        public void ApplyFixedScaleForChart(string chartName, int fixedMax)
        {
            var chart = chartName == "Yearly" ? _ctx.ChartDataYieldYearly
                      : chartName == "Monthly" ? _ctx.ChartDataYieldMonthly
                      : _ctx.ChartDataYieldDaily;
            ApplyFixedScale(chart, fixedMax);
        }

        // ══════════════════════════════════════════════════════════════
        // 圖表導航列（◄ 年/月/日 ►）
        // ══════════════════════════════════════════════════════════════

        private void RefillChartComboBox(ComboBox cb, List<int> values, int preferred = -1)
        {
            using (_chartNavGuard.Enter())
            {
                cb.Items.Clear();
                foreach (var v in values) cb.Items.Add(v.ToString());
                if (preferred >= 0)
                {
                    int idx = values.IndexOf(preferred);
                    cb.SelectedIndex = idx >= 0 ? idx : (values.Count > 0 ? values.Count - 1 : -1);
                }
                else
                {
                    cb.SelectedIndex = values.Count > 0 ? values.Count - 1 : -1;
                }
            }
        }

        public void PopulateChartNavigators() => PopulateChartNavigators(null);

        public void PopulateChartNavigators(DateTime? hintDate)
        {
            _chartYears = GetAvailableYears();
            RefillChartComboBox(_ctx.CbChartYear, _chartYears, hintDate?.Year ?? -1);
            OnChartYearIndexChanged(hintDate);
        }

        private void OnChartYearIndexChanged() => OnChartYearIndexChanged(null);
        private void OnChartYearIndexChanged(DateTime? hint)
        {
            int idx = _ctx.CbChartYear.SelectedIndex;
            bool ok = idx >= 0 && idx < _chartYears.Count;

            _chartMonths = ok ? GetAvailableMonths(_chartYears[idx]) : new List<int>();
            RefillChartComboBox(_ctx.CbChartMonth, _chartMonths, hint?.Month ?? -1);

            if (!ok) return;
            int year = _chartYears[idx];
            var ctx = BuildThresholdContext();
            FillPeriodChart(_ctx.ChartDataYieldYearly,
                InspectionStatisticsService.ComputeGroupedByMonthOfYear(_statsDataRootPath,
                    new DateTime(year, 1, 1), new DateTime(year, 12, 31, 23, 59, 59), ctx));

            OnChartMonthIndexChanged(hint);
        }

        private ThresholdContext BuildThresholdContext() => new ThresholdContext(
            _ctx.Settings.HessianMaxFactorV,
            _ctx.Settings.ErrorValueMeanV,
            _ctx.Settings.ErrorValueMaxV);

        /// <summary>由 PropertyGrid 變更觸發：重畫 chartDataYieldYearly/Monthly/Daily（以當前 Settings 重算 Pass/Fail）。</summary>
        public void RefreshPeriodCharts()
        {
            if (_ctx.CbChartYear?.SelectedIndex >= 0) OnChartYearIndexChanged();
        }

        private void OnChartMonthIndexChanged() => OnChartMonthIndexChanged(null);
        private void OnChartMonthIndexChanged(DateTime? hint)
        {
            int idx = _ctx.CbChartMonth.SelectedIndex;
            int yIdx = _ctx.CbChartYear.SelectedIndex;
            bool ok = idx >= 0 && idx < _chartMonths.Count && yIdx >= 0;

            _chartDays = ok ? GetAvailableDays(_chartYears[yIdx], _chartMonths[idx]) : new List<int>();
            RefillChartComboBox(_ctx.CbChartDay, _chartDays, hint?.Day ?? -1);

            if (!ok) return;
            int year = _chartYears[yIdx];
            int month = _chartMonths[idx];
            int lastDay = DateTime.DaysInMonth(year, month);
            FillPeriodChart(_ctx.ChartDataYieldMonthly,
                InspectionStatisticsService.ComputeGroupedByDayOfMonth(_statsDataRootPath,
                    new DateTime(year, month, 1), new DateTime(year, month, lastDay, 23, 59, 59),
                    BuildThresholdContext()));

            OnChartDayIndexChanged();
        }

        private void OnChartDayIndexChanged()
        {
            int dIdx = _ctx.CbChartDay.SelectedIndex;
            int mIdx = _ctx.CbChartMonth.SelectedIndex;
            int yIdx = _ctx.CbChartYear.SelectedIndex;
            bool ok = dIdx >= 0 && mIdx >= 0 && yIdx >= 0
                    && dIdx < _chartDays.Count && mIdx < _chartMonths.Count && yIdx < _chartYears.Count;

            if (!ok) return;
            int year = _chartYears[yIdx];
            int month = _chartMonths[mIdx];
            int day = _chartDays[dIdx];
            FillPeriodChart(_ctx.ChartDataYieldDaily,
                InspectionStatisticsService.ComputeGroupedByHourOfDay(_statsDataRootPath,
                    new DateTime(year, month, day), new DateTime(year, month, day, 23, 59, 59),
                    BuildThresholdContext()));
        }

        // ══════════════════════════════════════════════════════════════
        // Available date/time helpers
        // ══════════════════════════════════════════════════════════════

        private List<string> GetAvailableDateStrings() =>
            _statAvailableTimes.Select(t => t.ToString("yyyy-MM-dd")).Distinct()
                .OrderByDescending(x => x).ToList();

        private List<string> GetAvailableTimeStrings(string dateStr) =>
            _statAvailableTimes
                .Where(t => t.ToString("yyyy-MM-dd") == dateStr)
                .Select(t => t.ToString("HH:mm:ss"))
                .Distinct().OrderByDescending(x => x).ToList();

        private List<int> GetAvailableYears() =>
            _statAvailableTimes.Select(t => t.Year).Distinct().ToList();

        private List<int> GetAvailableMonths(int y) =>
            _statAvailableTimes.Where(t => t.Year == y)
                               .Select(t => t.Month).Distinct().ToList();

        private List<int> GetAvailableDays(int y, int mo) =>
            _statAvailableTimes.Where(t => t.Year == y && t.Month == mo)
                               .Select(t => t.Day).Distinct().ToList();

        // ══════════════════════════════════════════════════════════════
        // Populate / GroupBox helpers
        // ══════════════════════════════════════════════════════════════

        public void PopulateAllGrabIdCombos(bool selectDataGrabId = false)
        {
            using (StatComboGuard.Enter())
            {
                _ctx.CbReviewGrabId.Items.Clear();
                _ctx.CbGrabIdStart.Items.Clear();
                _ctx.CbGrabIdEnd.Items.Clear();
                _ctx.CbDataGrabId.Items.Clear();
                foreach (var info in _grabIdInfos)
                {
                    _ctx.CbReviewGrabId.Items.Add(info.GrabId);
                    _ctx.CbGrabIdStart.Items.Add(info.GrabId);
                    _ctx.CbGrabIdEnd.Items.Add(info.GrabId);
                    _ctx.CbDataGrabId.Items.Add(info.GrabId);
                }
                // SyncGrabIdFromTime 需要外部 DateTimeNavigator，由 Form 呼叫
                UpdateGrabIdNavState();
                if (_ctx.CbGrabIdStart.Items.Count > 0)
                {
                    _ctx.CbGrabIdStart.SelectedIndex = _ctx.CbGrabIdStart.Items.Count - 1;
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;
                    if (selectDataGrabId)
                        _ctx.CbDataGrabId.SelectedIndex = _ctx.CbGrabIdStart.SelectedIndex;
                }
            }
        }

        public void SetReviewGroupBoxes(bool grabNavActive)
        {
            SetGroupBoxActive(_ctx.GrpReviewGrabNav, grabNavActive);
            SetGroupBoxActive(_ctx.GrpReviewTimePeriod, !grabNavActive);
        }

        /// <summary>讀取資料後預設切到單片模式：觸發 cbDataId SelectedIndexChanged →
        /// OnSingleSheetComboChanged → SetActiveStatGroupBox(GrpDataSingleSheet) + RefreshStats。</summary>
        public void SelectLatestInSingleSheetMode()
        {
            if (_ctx.CbDataGrabId.Items.Count > 0)
                _ctx.CbDataGrabId.SelectedIndex = 0;
        }

        private void SetActiveStatGroupBox(GroupBox active)
        {
            _activeStatMode = active;
            foreach (var box in new[] { _ctx.GroupBoxGrabIdRange, _ctx.GrpDataSingleSheet, _ctx.GroupBoxTimeRange })
                SetGroupBoxActive(box, box == active);
        }

        /// <summary>
        /// 由 GroupBox.Click 觸發：切換 active 模式並重算統計（camData / listViewGrabDetail
        /// / chartDataVertical / chartDataYieldYearly 等）。已是 active 則無動作。
        /// 切到範圍類模式時把對應 combo 攤開到資料夾的完整範圍（避免承襲單片模式的單筆設定）：
        ///   - GroupBoxGrabIdRange：cbDataIdStart = 最舊、cbDataIdEnd = 最新
        ///   - GroupBoxTimeRange：cbDataDateStart/Time = _statAvailableTimes.Min、cbDataDateEnd/Time = Max
        /// </summary>
        private void SwitchActiveStatGroupBox(GroupBox target)
        {
            if (target == null || _activeStatMode == target) return;
            SetActiveStatGroupBox(target);

            if (target == _ctx.GroupBoxGrabIdRange && _grabIdInfos.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = _grabIdInfos.Count - 1; // descending 最後一筆 = 最舊
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;                        // descending 第一筆 = 最新
                }
            }
            else if (target == _ctx.GroupBoxTimeRange && _statAvailableTimes.Count > 0)
            {
                // PopulateStatDateCombos 內已包 StatComboGuard，外面不必再包
                PopulateStatDateCombos(_statAvailableTimes.Min, _statAvailableTimes.Max);
            }
            RefreshStats();
        }

        // ── GroupBox 綠色高亮 ─────────────────────────────────────────

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
