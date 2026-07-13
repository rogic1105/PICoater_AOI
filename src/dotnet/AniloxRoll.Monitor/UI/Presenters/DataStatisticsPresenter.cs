using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Widgets;
using TanukiCv.Controls;
using TanukiCv.Controls.WinForms;

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
        private List<GrabDetail> _visibleDetails = new List<GrabDetail>();
        private bool _showFailOnly;
        private bool _preserveDetailListDuringSelection;

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
            _dateGrabIdNavigator = new DataDateGrabIdNavigator(_ctx,
                () => _grabIdInfos,
                RefreshStats,
                RefreshSelectedGrab,
                (grabId, earliest, latest, idx) => GrabIdSelectedFromData?.Invoke(grabId, earliest, latest, idx),
                (grabId, earliest, latest, idx) => GrabIdSelectedFromReview?.Invoke(grabId, earliest, latest, idx),
                SetGroupBoxActive, SetChipActive);
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
            InitGrabDetailListView();
            _muraChart = new MuraProfileChartPresenter(_ctx,
                () => _dateGrabIdNavigator.ActiveStatMode, () => _grabIdInfos, () => _statsDataRootPath);
            _muraChart.Init();
            _yieldPeriodCharts = new YieldPeriodChartPresenter(_ctx, () => _statAvailableTimes, () => _statsDataRootPath);
            _yieldPeriodCharts.Init();

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
            _statsDataRootPath = path;
            _statAvailableTimes = InspectionStatisticsService.LoadAvailableTimes(path);
            _grabIdInfos = InspectionStatisticsService.LoadGrabIdInfosDescending(path);

            PopulateAllGrabIdCombos();

            PopulateChartNavigators(_statAvailableTimes.Count > 0
                ? (DateTime?)_statAvailableTimes.Max : null);
            SelectLatestInSingleSheetMode();
            RefreshStats();
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
            int idx = _ctx.CbGrabIdStart.SelectedIndex;
            if (idx >= 0 && idx < _grabIdInfos.Count) return _grabIdInfos[idx].GrabId;
            idx = _ctx.CbDataGrabId.SelectedIndex;
            if (idx >= 0 && idx < _grabIdInfos.Count) return _grabIdInfos[idx].GrabId;
            return _grabIdInfos.Count > 0 ? _grabIdInfos[0].GrabId : string.Empty;
        }

        private string GetDetailListEndGrabId()
        {
            int idx = _ctx.CbGrabIdEnd.SelectedIndex;
            if (idx >= 0 && idx < _grabIdInfos.Count) return _grabIdInfos[idx].GrabId;
            idx = _ctx.CbDataGrabId.SelectedIndex;
            if (idx >= 0 && idx < _grabIdInfos.Count) return _grabIdInfos[idx].GrabId;
            return _grabIdInfos.Count > 0 ? _grabIdInfos[0].GrabId : string.Empty;
        }

        public void RefreshStats()
        {
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;

            // view-time threshold context：以當前 Settings 的閾值 + 欄正規值即時重算 Pass/Fail，
            // 不再用 CSV 內的 MaxExceed/MeanExceed（那是 capture-time baked-in）。
            var ctx = new ThresholdContext(
                _ctx.Settings.HessianMaxFactorV,
                _ctx.Settings.ErrorValueMeanV,
                _ctx.Settings.ErrorValueMaxV);

            // SingleSheet mode：用 cbDataId.SelectedIndex 算單 grab stats（start=end=該 grab）。
            // 不靠 cbDataIdStart/End 範圍；cbDataId 變更不連動範圍 cb（範圍獨立），
            // 故 listViewGrabDetail 點選後 stats 仍對齊到剛點的單 grab。
            if (_dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet
                && _ctx.CbDataGrabId.SelectedIndex >= 0
                && _ctx.CbDataGrabId.SelectedIndex < _grabIdInfos.Count)
            {
                if (!_preserveDetailListDuringSelection)
                {
                    var swList = Stopwatch.StartNew();
                    _currentDetails = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                        _statsDataRootPath, GetDetailListStartGrabId(), GetDetailListEndGrabId(), ctx);
                    ApplyFailFilter();
                    FlowTrace.Log($"DT list reload range={GetDetailListStartGrabId()}~{GetDetailListEndGrabId()} rows={_currentDetails.Count} ms={swList.ElapsedMilliseconds}");
                }
                RefreshSelectedGrab();
                return;
            }

            if (_ctx.CbGrabIdStart.SelectedIndex >= 0 && _ctx.CbGrabIdEnd.SelectedIndex >= 0
                && _grabIdInfos.Count > 0)
            {
                var startInfo = _grabIdInfos[_ctx.CbGrabIdStart.SelectedIndex];
                var endInfo = _grabIdInfos[_ctx.CbGrabIdEnd.SelectedIndex];

                var stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, startInfo.GrabId, endInfo.GrabId, ctx);
                _statsPresenter.Update(stats);
                if (!_preserveDetailListDuringSelection)
                {
                    var swList = Stopwatch.StartNew();
                    _currentDetails = InspectionStatisticsService.ComputeDetailedByGrabIdRange(
                        _statsDataRootPath, startInfo.GrabId, endInfo.GrabId, ctx);
                    ApplyFailFilter();
                    FlowTrace.Log($"DT list reload range={startInfo.GrabId}~{endInfo.GrabId} rows={_currentDetails.Count} ms={swList.ElapsedMilliseconds}");
                }

                int si = _ctx.CbGrabIdStart.SelectedIndex;
                int ei = _ctx.CbGrabIdEnd.SelectedIndex;
                int lo = Math.Min(si, ei); int hi = Math.Max(si, ei);
                var rangeInfos = _grabIdInfos.GetRange(lo, hi - lo + 1);
                _muraChart.Update(EvenSample(rangeInfos, 50));
                return;
            }

        }

        /// <summary>單片序號快路：List 範圍內容不變，只更新該筆統計、Mura curve 與反白。</summary>
        private void RefreshSelectedGrab()
        {
            if (string.IsNullOrWhiteSpace(_statsDataRootPath)) return;
            int selectedIndex = _ctx.CbDataGrabId.SelectedIndex;
            if (selectedIndex < 0 || selectedIndex >= _grabIdInfos.Count) return;

            var sw = Stopwatch.StartNew();
            var grab = _grabIdInfos[selectedIndex];
            var detail = _currentDetails.FirstOrDefault(item => item.GrabId == grab.GrabId);
            bool cacheHit = detail != null;
            Dictionary<int, CameraStats> stats;
            if (cacheHit)
            {
                stats = BuildSingleGrabStats(detail);
            }
            else
            {
                var threshold = new ThresholdContext(
                    _ctx.Settings.HessianMaxFactorV,
                    _ctx.Settings.ErrorValueMeanV,
                    _ctx.Settings.ErrorValueMaxV);
                stats = InspectionStatisticsService.ComputeByGrabIdRange(
                    _statsDataRootPath, grab.GrabId, grab.GrabId, threshold);
            }

            _statsPresenter.Update(stats);
            HighlightDetailRow(grab.GrabId);
            _muraChart.Update(null);
            FlowTrace.Log($"DT selected {grab.GrabId} stats={(cacheHit ? "cache" : "scan")} list=keep ms={sw.ElapsedMilliseconds}");
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

        // ══════════════════════════════════════════════════════════════
        // Detail ListView
        // ══════════════════════════════════════════════════════════════


        private void InitGrabDetailListView()
        {
            var lv = _ctx.ListViewGrabDetail;
            lv.View = View.Details;
            lv.FullRowSelect = true;
            lv.GridLines = true;
            lv.OwnerDraw = true;
            lv.VirtualMode = true;
            EnableDoubleBuffering(lv);
            lv.Columns.Clear();
            lv.VirtualListSize = 0;

            lv.Columns.Add("料件序號", -1, HorizontalAlignment.Center);
            for (int i = 1; i <= _ctx.CameraCount; i++)
                lv.Columns.Add($"{i}", -1, HorizontalAlignment.Center);
            FitGrabDetailColumnsToContent(lv);

            // 點選明細列表的列 → MouseDown 時 ListView 預設視覺先反白（顯示被選中），
            // MouseUp 才 commit 切到該序號（與 cbDataId 變更流程共用 OnSingleSheetComboChanged）。
            // cbDataId 變更不連動 cbDataIdStart/End（範圍獨立），故 commit 後範圍 cb 維持不變。
            lv.DrawColumnHeader -= ListViewGrabDetail_DrawColumnHeader;
            lv.DrawSubItem -= ListViewGrabDetail_DrawSubItem;
            lv.RetrieveVirtualItem -= ListViewGrabDetail_RetrieveVirtualItem;
            lv.DrawColumnHeader += ListViewGrabDetail_DrawColumnHeader;
            lv.DrawSubItem += ListViewGrabDetail_DrawSubItem;
            lv.RetrieveVirtualItem += ListViewGrabDetail_RetrieveVirtualItem;
            lv.MouseUp -= OnGrabDetailRowCommitted;
            lv.MouseUp += OnGrabDetailRowCommitted;
        }

        private string _lastListViewSelectedGrabId;

        private void OnGrabDetailRowCommitted(object sender, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left) return;
            if (StatComboGuard.IsSet) return;
            var lv = _ctx.ListViewGrabDetail;
            if (lv.SelectedIndices.Count == 0) return;
            int selectedIndex = lv.SelectedIndices[0];
            if (selectedIndex < 0 || selectedIndex >= _visibleDetails.Count) return;
            string grabId = _visibleDetails[selectedIndex].GrabId;
            if (string.IsNullOrEmpty(grabId)) return;

            // Toggle：第二次點同 row + 已是 SingleSheet → 切回 GroupBoxGrabIdRange（範圍模式，stats 用 cbDataIdStart/End）
            if (grabId == _lastListViewSelectedGrabId && _dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet)
            {
                FlowTrace.Log($"ui:【明細列表】同列再點 {grabId} → 回範圍模式");
                ExecuteWithDetailListRedrawSuspended(lv, () =>
                {
                    _lastListViewSelectedGrabId = null;
                    lv.SelectedIndices.Clear();  // 清掉反白，視覺回到「無選中」
                    _muraChart?.Clear();          // 先清圖，避免同列二次點選時殘留上一筆 CURVE
                    _dateGrabIdNavigator.SetActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
                    RefreshStats();
                });
                return;
            }
            _lastListViewSelectedGrabId = grabId;
            FlowTrace.Log($"ui:【明細列表】→ {grabId}");

            int idx = _ctx.CbDataGrabId.Items.IndexOf(grabId);
            if (idx < 0) return;
            if (_ctx.CbDataGrabId.SelectedIndex == idx)
            {
                // SelectedIndex 沒變 → 不會觸發 OnSingleSheetComboChanged，但仍需確保 active 模式為單片
                ExecuteWithDetailListRedrawSuspended(lv, () =>
                {
                if (_dateGrabIdNavigator.ActiveStatMode != _ctx.GrpDataSingleSheet)
                {
                    _dateGrabIdNavigator.SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
                }
                });
                return;
            }
            ExecuteWithDetailListRedrawSuspended(lv, () =>
            {
            _dateGrabIdNavigator.CommitDataGrabIdFromDetailList(grabId);
            });
        }

        private void ExecuteWithDetailListRedrawSuspended(ListView lv, Action action)
        {
            if (lv == null || action == null) return;
            if (!lv.IsHandleCreated || lv.IsDisposed)
            {
                bool previous = _preserveDetailListDuringSelection;
                _preserveDetailListDuringSelection = true;
                try { action(); }
                finally { _preserveDetailListDuringSelection = previous; }
                return;
            }

            bool savedPreserveDetailList = _preserveDetailListDuringSelection;
            _preserveDetailListDuringSelection = true;
            using (new RedrawScope(lv))
            {
                try
                {
                    action();
                }
                finally
                {
                    _preserveDetailListDuringSelection = savedPreserveDetailList;
                }
            }
        }

        /// <summary>cbDataId（單片序號）變更 → 在 listViewGrabDetail 對應列重用「點擊時的選取框」
        /// （框由 DrawSubItem 依 _lastListViewSelectedGrabId 自繪，非原生選取，故不新增框）+ EnsureVisible 捲到可見。
        /// 點擊路徑也走這（同值冪等）。僅單片分支呼叫；範圍模式 DrawSubItem 不畫框。</summary>
        private void HighlightDetailRow(string grabId)
        {
            int previousRow = _visibleDetails.FindIndex(d => d.GrabId == _lastListViewSelectedGrabId);
            _lastListViewSelectedGrabId = grabId;
            var lv = _ctx.ListViewGrabDetail;
            if (lv == null) return;
            int rowIdx = _visibleDetails.FindIndex(d => d.GrabId == grabId);   // 被 fail filter 濾掉則 -1（不可見不捲）
            if (rowIdx >= 0 && lv.IsHandleCreated && rowIdx < lv.VirtualListSize)
                EnsureDetailRowInBufferedViewport(lv, rowIdx);
            RedrawDetailRow(lv, previousRow);
            RedrawDetailRow(lv, rowIdx);
        }

        private static void EnableDoubleBuffering(ListView listView)
        {
            try
            {
                typeof(Control).GetProperty("DoubleBuffered", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?.SetValue(listView, true, null);
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[DataListDoubleBuffer] {ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>選中列接近上下邊界才捲動，並預留數列視窗緩衝，避免每跨一列就整窗重畫。</summary>
        private static void EnsureDetailRowInBufferedViewport(ListView listView, int rowIndex)
        {
            int itemHeight = Math.Max(18, listView.Font.Height + 6);
            int visibleRows = Math.Max(1, (listView.ClientSize.Height - 24) / itemHeight);
            int margin = Math.Min(5, Math.Max(1, visibleRows / 4));
            int topIndex = 0;
            try
            {
                if (listView.TopItem != null)
                    topIndex = listView.TopItem.Index;
            }
            catch (InvalidOperationException) { }

            int bottomIndex = Math.Min(listView.VirtualListSize - 1, topIndex + visibleRows - 1);
            if (rowIndex < topIndex + margin)
                listView.EnsureVisible(Math.Max(0, rowIndex - margin));
            else if (rowIndex > bottomIndex - margin)
                listView.EnsureVisible(Math.Min(listView.VirtualListSize - 1, rowIndex + margin));
        }

        private static void RedrawDetailRow(ListView listView, int rowIndex)
        {
            if (listView == null || !listView.IsHandleCreated) return;
            if (rowIndex < 0 || rowIndex >= listView.VirtualListSize) return;
            try { listView.RedrawItems(rowIndex, rowIndex, true); }
            catch (InvalidOperationException)
            {
                try { listView.Invalidate(listView.GetItemRect(rowIndex)); }
                catch (InvalidOperationException) { }
            }
            catch (ArgumentOutOfRangeException) { }
        }

        private void UpdateGrabDetailListView(List<GrabDetail> details)
        {
            var lv = _ctx.ListViewGrabDetail;
            if (lv.VirtualMode)
            {
                lv.BeginUpdate();
                try
                {
                    _visibleDetails = details ?? new List<GrabDetail>();
                    lv.SelectedIndices.Clear();
                    lv.VirtualListSize = _visibleDetails.Count;
                    FitGrabDetailColumnsToContent(lv);
                }
                finally
                {
                    lv.EndUpdate();
                }
                return;
            }

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

                item.Tag = rowHasFail;
                item.BackColor = rowHasFail ? _detailFail : _detailPass;
                lv.Items.Add(item);
            }

            lv.EndUpdate();
            lv.AutoResizeColumns(ColumnHeaderAutoResizeStyle.ColumnContent);
        }

        private void ListViewGrabDetail_RetrieveVirtualItem(object sender, RetrieveVirtualItemEventArgs e)
        {
            e.Item = BuildGrabDetailListViewItem(e.ItemIndex);
        }

        private ListViewItem BuildGrabDetailListViewItem(int index)
        {
            if (index < 0 || index >= _visibleDetails.Count)
                return new ListViewItem(string.Empty);

            var d = _visibleDetails[index];
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

            item.Tag = rowHasFail;
            item.BackColor = rowHasFail ? _detailFail : _detailPass;
            return item;
        }

        // VirtualMode 下 lv.Items 為空，AutoResizeColumns(ColumnContent) / 量 Items 的
        // FitListViewColumnsProportional 都失效。改用 _visibleDetails 取樣量測，
        // 還原「貼齊內容的緊湊欄寬」（與非 virtual 路徑 AutoResizeColumns(ColumnContent) 同觀感）。
        private void FitGrabDetailColumnsToContent(ListView lv)
        {
            if (lv.Columns.Count == 0) return;
            using (var g = lv.CreateGraphics())
            {
                const int pad = 16; // 比照 AutoResizeColumns 的內距餘裕
                string sample0 = _visibleDetails.Count > 0 && !string.IsNullOrEmpty(_visibleDetails[0].GrabId)
                    ? _visibleDetails[0].GrabId
                    : lv.Columns[0].Text;
                lv.Columns[0].Width = (int)Math.Ceiling(Math.Max(
                    g.MeasureString(lv.Columns[0].Text, lv.Font).Width,
                    g.MeasureString(sample0, lv.Font).Width)) + pad;

                float glyphW = g.MeasureString("×", lv.Font).Width;
                for (int i = 1; i < lv.Columns.Count; i++)
                    lv.Columns[i].Width = (int)Math.Ceiling(Math.Max(
                        g.MeasureString(lv.Columns[i].Text, lv.Font).Width, glyphW)) + pad;
            }
        }

        private static void ListViewGrabDetail_DrawColumnHeader(object sender, DrawListViewColumnHeaderEventArgs e)
        {
            e.DrawDefault = true;
        }

        private void ListViewGrabDetail_DrawSubItem(object sender, DrawListViewSubItemEventArgs e)
        {
            bool rowHasFail = e.Item.Tag is bool failed && failed;
            Color backColor = rowHasFail ? _detailFail : _detailPass;

            using (var backBrush = new SolidBrush(backColor))
                e.Graphics.FillRectangle(backBrush, e.Bounds);

            var textFlags = TextFormatFlags.VerticalCenter
                          | TextFormatFlags.HorizontalCenter
                          | TextFormatFlags.EndEllipsis
                          | TextFormatFlags.NoPrefix;
            TextRenderer.DrawText(e.Graphics, e.SubItem.Text, e.Item.ListView.Font,
                e.Bounds, e.Item.ForeColor, textFlags);

            bool isMarked = _dateGrabIdNavigator.ActiveStatMode == _ctx.GrpDataSingleSheet
                         && e.Item.Text == _lastListViewSelectedGrabId;
            if (!isMarked || e.ColumnIndex != e.Item.SubItems.Count - 1)
                return;

            Rectangle rowBounds = e.Item.Bounds;
            rowBounds.Width = e.Item.ListView.Columns.Cast<ColumnHeader>().Sum(c => c.Width);
            rowBounds.Width = Math.Min(rowBounds.Width, e.Item.ListView.ClientSize.Width - 1);
            rowBounds.Height -= 1;
            if (rowBounds.Width <= 0 || rowBounds.Height <= 0) return;

            Color borderColor = rowHasFail ? Color.FromArgb(211, 47, 47) : Color.FromArgb(46, 125, 50);
            using (var pen = new Pen(borderColor, 2))
                e.Graphics.DrawRectangle(pen, rowBounds);
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
        // Mura 空間分布曲線圖（拼接式，與 chartLiveColumn 相同格式）
        // ══════════════════════════════════════════════════════════════

        // Mura 分布圖繪圖職責已提取到 MuraProfileChartPresenter（2026-06-30）；以下為對外 public 門面轉發。

        /// <summary>由 PropertyGrid 變更觸發：刷新 chartDataColumn 閾值線 + view-time 正規值 rescale（不重算統計）。</summary>
        public void RefreshMuraProfileForSettingsChange() => _muraChart?.RefreshForSettingsChange();

        /// <summary>SingleSheet 模式：用 Review tab 已載入曲線資料更新 chartDataColumn，與 chartReviewColumn 一致。</summary>
        public void SyncMuraProfileFromReview(float[][] mean, float[][] max,
            double[] ops, double[] pos, float errMean, float errMax)
            => _muraChart?.SyncFromReview(mean, max, ops, pos, errMean, errMax);

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
            FlowTrace.Log($"ui:【篩選異常】→ {(_showFailOnly ? "只顯示異常" : "顯示全部")}");
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

        public void ApplyChartScaleFromSettings() => _yieldPeriodCharts?.ApplyChartScaleFromSettings();

        public void ApplyFixedScaleForChart(string chartName, int fixedMax) =>
            _yieldPeriodCharts?.ApplyFixedScaleForChart(chartName, fixedMax);

        public void PopulateChartNavigators() => _yieldPeriodCharts?.PopulateChartNavigators();

        public void PopulateChartNavigators(DateTime? hintDate) =>
            _yieldPeriodCharts?.PopulateChartNavigators(hintDate);

        /// <summary>由 PropertyGrid 設定變更觸發，重新整理 chartDataYieldYearly/Monthly/Daily，讓 Settings 立刻套用 Pass/Fail。</summary>
        public void RefreshPeriodCharts() => _yieldPeriodCharts?.RefreshPeriodCharts();
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
                    _ctx.CbGrabIdStart.SelectedIndex = _grabIdInfos.Count - 1;   // 最舊
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;                          // 最新
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
