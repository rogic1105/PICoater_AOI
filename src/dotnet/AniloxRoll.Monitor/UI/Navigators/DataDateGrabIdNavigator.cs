using System;
using System.Collections.Generic;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Navigators
{
    /// <summary>序號範圍的來源：給「哪個來源高亮成綠色」+ 單片 toggle 回範圍時記憶用。</summary>
    public enum GrabIdRangeSource { Global, Year, Month, Day, Custom }

    public sealed class DataDateGrabIdNavigator
    {
        private readonly DataStatisticsContext _ctx;
        private readonly Func<List<GrabIdInfo>> _getGrabIdInfos;
        private readonly Action _refreshStats;
        private readonly Action _refreshSelectedGrab;
        private readonly Action<string, DateTime, DateTime, int> _selectFromData;
        private readonly Action<string, DateTime, DateTime, int> _selectFromReview;
        private readonly Action<GroupBox, bool> _setGroupBoxActive;
        private readonly Action<Label, bool> _setChipActive;
        private GrabIdRangeSource _rangeSource = GrabIdRangeSource.Global;

        public EventGuard StatComboGuard { get; } = new EventGuard();
        public EventGuard GrabIdNavGuard { get; } = new EventGuard();
        public EventGuard GrabIdCrossGuard { get; } = new EventGuard();
        public GroupBox ActiveStatMode { get; private set; }

        public DataDateGrabIdNavigator(
            DataStatisticsContext ctx,
            Func<List<GrabIdInfo>> getGrabIdInfos,
            Action refreshStats,
            Action refreshSelectedGrab,
            Action<string, DateTime, DateTime, int> selectFromData,
            Action<string, DateTime, DateTime, int> selectFromReview,
            Action<GroupBox, bool> setGroupBoxActive,
            Action<Label, bool> setChipActive)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getGrabIdInfos = getGrabIdInfos ?? throw new ArgumentNullException(nameof(getGrabIdInfos));
            _refreshStats = refreshStats ?? throw new ArgumentNullException(nameof(refreshStats));
            _refreshSelectedGrab = refreshSelectedGrab ?? throw new ArgumentNullException(nameof(refreshSelectedGrab));
            _selectFromData = selectFromData ?? throw new ArgumentNullException(nameof(selectFromData));
            _selectFromReview = selectFromReview ?? throw new ArgumentNullException(nameof(selectFromReview));
            _setGroupBoxActive = setGroupBoxActive ?? throw new ArgumentNullException(nameof(setGroupBoxActive));
            _setChipActive = setChipActive ?? throw new ArgumentNullException(nameof(setChipActive));
            ActiveStatMode = ctx.GroupBoxGrabIdRange;
        }

        public void WireEvents()
        {
            _ctx.CbGrabIdStart.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: true);
            _ctx.CbGrabIdEnd.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: false);
            _ctx.CbDataGrabId.SelectedIndexChanged += (s, e) => OnSingleSheetComboChanged();
            _ctx.CbReviewGrabId.SelectedIndexChanged += (s, e) => OnReviewGrabIdChanged();
            _ctx.GrpReviewGrabNav.Click += (s, e) => OnReviewGrabIdChanged();

            _ctx.GrpDataSingleSheet.Click += (s, e) => SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            _ctx.GroupBoxGrabIdRange.Click += (s, e) => ApplyGlobalRange();   // 回全局（永遠重設，不受目前 mode 早退影響）

            // 年/月/日 label 點擊 → 套用該期間；範圍模式下再點同一個來源則解除綁定並保留目前範圍。
            if (_ctx.LblChartNavYear != null)  _ctx.LblChartNavYear.Click  += (s, e) => TogglePeriodRange(GrabIdRangeSource.Year);
            if (_ctx.LblChartNavMonth != null) _ctx.LblChartNavMonth.Click += (s, e) => TogglePeriodRange(GrabIdRangeSource.Month);
            if (_ctx.LblChartNavDay != null)   _ctx.LblChartNavDay.Click   += (s, e) => TogglePeriodRange(GrabIdRangeSource.Day);

            // lblChartNav 為 active 來源時，改對應的 cbDataYear/Month/Day → 範圍跟著更新
            if (_ctx.CbChartYear != null)  _ctx.CbChartYear.SelectedIndexChanged  += (s, e) => OnPeriodComboChangedForRange(GrabIdRangeSource.Year);
            if (_ctx.CbChartMonth != null) _ctx.CbChartMonth.SelectedIndexChanged += (s, e) => OnPeriodComboChangedForRange(GrabIdRangeSource.Month);
            if (_ctx.CbChartDay != null)   _ctx.CbChartDay.SelectedIndexChanged   += (s, e) => OnPeriodComboChangedForRange(GrabIdRangeSource.Day);

            UpdateSourceHighlights();   // 初始：反映預設 Global
        }

        /// <summary>lblChartNav 目前正是 active 來源時，改對應 cbDataYield → 重套該期間範圍。
        /// 非 active 來源的 combo 變更（含年變更連帶重填月/日的串聯）不觸發，避免亂改範圍。</summary>
        private void OnPeriodComboChangedForRange(GrabIdRangeSource source)
        {
            if (_rangeSource != source) return;
            ApplyPeriodRange(source);
        }

        public void PopulateAllGrabIdCombos(bool selectDataGrabId = false)
        {
            using (StatComboGuard.Enter())
            {
                var grabIdInfos = _getGrabIdInfos();
                // 批次填充：一萬筆時逐筆 Add 會 4 個 combo × N 次重繪 → UI 凍住幾秒。
                // 先組成陣列，各 combo 用 BeginUpdate + AddRange 一次配置（重繪 4 萬次 → 4 次）。
                var ids = new object[grabIdInfos.Count];
                for (int i = 0; i < grabIdInfos.Count; i++)
                    ids[i] = grabIdInfos[i].GrabId;
                foreach (var cb in new[] { _ctx.CbReviewGrabId, _ctx.CbGrabIdStart, _ctx.CbGrabIdEnd, _ctx.CbDataGrabId })
                {
                    cb.BeginUpdate();
                    cb.Items.Clear();
                    cb.Items.AddRange(ids);
                    cb.EndUpdate();
                }
                UpdateGrabIdNavState();
                if (_ctx.CbGrabIdStart.Items.Count > 0)
                {
                    _ctx.CbGrabIdStart.SelectedIndex = _ctx.CbGrabIdStart.Items.Count - 1;
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;
                    _ctx.CbDataGrabId.SelectedIndex = selectDataGrabId
                        ? _ctx.CbGrabIdStart.SelectedIndex
                        : 0;
                }
            }
        }

        public void SyncDataGrabIdFromReview(int idx, GrabIdInfo info)
        {
            using (GrabIdCrossGuard.Enter())
            {
                _ctx.CbDataGrabId.SelectedIndex = idx;
                SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
                _refreshSelectedGrab();
            }
        }

        public void UpdateGrabIdNavState()
        {
            int idx = _ctx.CbReviewGrabId.SelectedIndex;
            int count = _ctx.CbReviewGrabId.Items.Count;
            UpdateDataGrabIdNavState();
        }

        public void SyncGrabIdFromTime(DateTime current)
        {
            var grabIdInfos = _getGrabIdInfos();
            if (GrabIdNavGuard.IsSet || grabIdInfos.Count == 0) return;

            int bestIdx = -1;
            long bestDiff = long.MaxValue;
            for (int i = 0; i < grabIdInfos.Count; i++)
            {
                var info = grabIdInfos[i];
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

        public void CommitDataGrabIdFromDetailList(string grabId)
        {
            int idx = _ctx.CbDataGrabId.Items.IndexOf(grabId);
            if (idx < 0) return;
            if (_ctx.CbDataGrabId.SelectedIndex == idx)
            {
                if (ActiveStatMode != _ctx.GrpDataSingleSheet)
                {
                    SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
                }
                return;
            }

            _ctx.CbDataGrabId.SelectedIndex = idx;
        }

        public void SetActiveStatGroupBox(GroupBox active)
        {
            ActiveStatMode = active;
            UpdateSourceHighlights();
        }

        /// <summary>互斥高亮：同時間只有一個範圍來源是綠色。單片模式 → 只有 grpDataSingleSheet 綠、來源 chip 全滅
        /// （但 _rangeSource 保留，toggle 回範圍即還原）。範圍模式 → 依 _rangeSource 綠 groupBoxGrabIdRange(全局)
        /// 或對應 lblChartNav(年/月/日)；Custom 全滅。</summary>
        private void UpdateSourceHighlights()
        {
            bool single = ActiveStatMode == _ctx.GrpDataSingleSheet;
            _setGroupBoxActive(_ctx.GrpDataSingleSheet, single);
            _setGroupBoxActive(_ctx.GroupBoxGrabIdRange, !single && _rangeSource == GrabIdRangeSource.Global);
            _setChipActive(_ctx.LblChartNavYear,  !single && _rangeSource == GrabIdRangeSource.Year);
            _setChipActive(_ctx.LblChartNavMonth, !single && _rangeSource == GrabIdRangeSource.Month);
            _setChipActive(_ctx.LblChartNavDay,   !single && _rangeSource == GrabIdRangeSource.Day);
        }

        /// <summary>同一期間第二按解除綁定；只取消後續連動，不改目前序號範圍。</summary>
        private void TogglePeriodRange(GrabIdRangeSource source)
        {
            if (ActiveStatMode == _ctx.GroupBoxGrabIdRange && _rangeSource == source)
            {
                _rangeSource = GrabIdRangeSource.Custom;
                UpdateSourceHighlights();
                FlowTrace.Log($"ui:【期間-{GetPeriodLabel(source)}】→ 取消綁定 保留範圍 {_ctx.CbGrabIdStart.Text}~{_ctx.CbGrabIdEnd.Text}");
                return;
            }

            ApplyPeriodRange(source);
        }

        /// <summary>年/月/日 label 點擊：範圍序號只取該期間（值取自 cbDataYieldYear/Month/Day）。</summary>
        private void ApplyPeriodRange(GrabIdRangeSource source)
        {
            var infos = _getGrabIdInfos();
            if (infos.Count == 0) return;
            if (!TryGetSelectedPeriod(source, out int year, out int month, out int day)) return;

            // infos 為降冪（0=最新、Count-1=最舊）→ hi=符合期間的最大 idx(最舊)、lo=最小 idx(最新)
            int lo = -1, hi = -1;
            for (int i = 0; i < infos.Count; i++)
            {
                var d = infos[i].Earliest;
                bool match = d.Year == year
                    && (source == GrabIdRangeSource.Year || d.Month == month)
                    && (source != GrabIdRangeSource.Day || d.Day == day);
                if (!match) continue;
                if (lo < 0) lo = i;
                hi = i;
            }
            if (lo < 0) return;   // 該期間無資料（實務不會發生：年月日來自有資料）

            using (StatComboGuard.Enter())
            {
                _ctx.CbGrabIdStart.SelectedIndex = hi;   // 最舊
                _ctx.CbGrabIdEnd.SelectedIndex = lo;     // 最新
            }
            _rangeSource = source;
            FlowTrace.Log($"ui:【期間-{GetPeriodLabel(source)}】→ 範圍 {infos[hi].GrabId}~{infos[lo].GrabId}");
            SetActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);   // mode=範圍 + UpdateSourceHighlights
            _refreshStats();
        }

        /// <summary>點 groupBoxGrabIdRange：回全局（最舊→最新）。永遠重設，不受目前 mode 早退影響。</summary>
        private void ApplyGlobalRange()
        {
            var infos = _getGrabIdInfos();
            if (infos.Count == 0) return;
            FlowTrace.Log("ui:【期間-全局】→ 全範圍");
            _rangeSource = GrabIdRangeSource.Global;
            using (StatComboGuard.Enter())
            {
                _ctx.CbGrabIdStart.SelectedIndex = infos.Count - 1;
                _ctx.CbGrabIdEnd.SelectedIndex = 0;
            }
            SetActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
            _refreshStats();
        }

        private bool TryGetSelectedPeriod(GrabIdRangeSource source, out int year, out int month, out int day)
        {
            year = month = day = 0;
            if (!TryParseCombo(_ctx.CbChartYear, out year)) return false;
            if (source == GrabIdRangeSource.Year) return true;
            if (!TryParseCombo(_ctx.CbChartMonth, out month)) return false;
            if (source == GrabIdRangeSource.Month) return true;
            return TryParseCombo(_ctx.CbChartDay, out day);
        }

        private static string GetPeriodLabel(GrabIdRangeSource source)
        {
            return source == GrabIdRangeSource.Year ? "年"
                : source == GrabIdRangeSource.Month ? "月"
                : "日";
        }

        private static bool TryParseCombo(ComboBox cb, out int v)
        {
            v = 0;
            return cb?.SelectedItem != null && int.TryParse(cb.SelectedItem.ToString(), out v);
        }

        public void SwitchActiveStatGroupBox(GroupBox target)
        {
            if (target == null || ActiveStatMode == target) return;
            SetActiveStatGroupBox(target);

            var grabIdInfos = _getGrabIdInfos();
            if (target == _ctx.GroupBoxGrabIdRange && grabIdInfos.Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = grabIdInfos.Count - 1;
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;
                }
            }

            if (target == _ctx.GrpDataSingleSheet)
                _refreshSelectedGrab();
            else
                _refreshStats();
        }

        private void OnGrabIdComboChanged(bool isStart)
        {
            var grabIdInfos = _getGrabIdInfos();
            if (StatComboGuard.IsSet || grabIdInfos.Count == 0) return;
            FlowTrace.Log($"ui:【序號範圍-{(isStart ? "起始" : "結束")}】變更");
            _rangeSource = GrabIdRangeSource.Custom;   // 手動拖範圍 → 自訂，清來源高亮
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
            }

            _refreshStats();
        }

        private void OnSingleSheetComboChanged()
        {
            var grabIdInfos = _getGrabIdInfos();
            UpdateDataGrabIdNavState();
            if (StatComboGuard.IsSet || grabIdInfos.Count == 0) return;
            if (GrabIdCrossGuard.IsSet) return;

            SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            int idx = _ctx.CbDataGrabId.SelectedIndex;
            if (idx < 0) return;
            FlowTrace.Log($"ui:【報表序號】→ {(idx < grabIdInfos.Count ? grabIdInfos[idx].GrabId : idx.ToString())}");

            // cbDataId（單片序號）變更「不」連動 cbDataIdStart/End —— 範圍序號獨立，選單片不動範圍。
            _refreshSelectedGrab();

            if (!GrabIdCrossGuard.IsSet && _ctx.CbReviewGrabId.Items.Count > 0
                && idx < _ctx.CbReviewGrabId.Items.Count)
            {
                var info = grabIdInfos[idx];
                _selectFromData(info.GrabId, info.Earliest, info.Latest, idx);
            }
        }

        private void OnReviewGrabIdChanged()
        {
            var grabIdInfos = _getGrabIdInfos();
            UpdateGrabIdNavState();
            if (GrabIdNavGuard.IsSet) return;
            if (GrabIdCrossGuard.IsSet) return;
            if (grabIdInfos.Count == 0) return;
            int idx = _ctx.CbReviewGrabId.SelectedIndex;
            if (idx < 0 || idx >= grabIdInfos.Count) return;

            var info = grabIdInfos[idx];
            FlowTrace.Log($"ui:【單片序號】→ {info.GrabId}");   // intent 行；guard 之後＝只記手動選取
            _selectFromReview(info.GrabId, info.Earliest, info.Latest, idx);
        }

        private void UpdateDataGrabIdNavState()
        {
            int idx = _ctx.CbDataGrabId.SelectedIndex;
            int count = _ctx.CbDataGrabId.Items.Count;
        }
    }
}
