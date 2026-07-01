using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Navigators
{
    public sealed class DataDateGrabIdNavigator
    {
        private readonly DataStatisticsContext _ctx;
        private readonly Func<SortedSet<DateTime>> _getAvailableTimes;
        private readonly Func<List<GrabIdInfo>> _getGrabIdInfos;
        private readonly Action _refreshStats;
        private readonly Action<string, DateTime, DateTime, int> _selectFromData;
        private readonly Action<string, DateTime, DateTime, int> _selectFromReview;
        private readonly Action<GroupBox, bool> _setGroupBoxActive;

        private bool _suppressRangeOnSingleSheetSync;

        public EventGuard StatComboGuard { get; } = new EventGuard();
        public EventGuard GrabIdNavGuard { get; } = new EventGuard();
        public EventGuard GrabIdCrossGuard { get; } = new EventGuard();
        public GroupBox ActiveStatMode { get; private set; }

        public DataDateGrabIdNavigator(
            DataStatisticsContext ctx,
            Func<SortedSet<DateTime>> getAvailableTimes,
            Func<List<GrabIdInfo>> getGrabIdInfos,
            Action refreshStats,
            Action<string, DateTime, DateTime, int> selectFromData,
            Action<string, DateTime, DateTime, int> selectFromReview,
            Action<GroupBox, bool> setGroupBoxActive)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getAvailableTimes = getAvailableTimes ?? throw new ArgumentNullException(nameof(getAvailableTimes));
            _getGrabIdInfos = getGrabIdInfos ?? throw new ArgumentNullException(nameof(getGrabIdInfos));
            _refreshStats = refreshStats ?? throw new ArgumentNullException(nameof(refreshStats));
            _selectFromData = selectFromData ?? throw new ArgumentNullException(nameof(selectFromData));
            _selectFromReview = selectFromReview ?? throw new ArgumentNullException(nameof(selectFromReview));
            _setGroupBoxActive = setGroupBoxActive ?? throw new ArgumentNullException(nameof(setGroupBoxActive));
            ActiveStatMode = ctx.GroupBoxGrabIdRange;
        }

        public void WireEvents()
        {
            _ctx.CbStartDate.SelectedIndexChanged += (s, e) => OnStartComboChanged(1);
            _ctx.CbStartTime.SelectedIndexChanged += (s, e) => OnStartComboChanged(2);
            _ctx.CbEndDate.SelectedIndexChanged += (s, e) => OnEndComboChanged(1);
            _ctx.CbEndTime.SelectedIndexChanged += (s, e) => OnEndComboChanged(2);

            _ctx.CbGrabIdStart.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: true);
            _ctx.CbGrabIdEnd.SelectedIndexChanged += (s, e) => OnGrabIdComboChanged(isStart: false);
            _ctx.CbDataGrabId.SelectedIndexChanged += (s, e) => OnSingleSheetComboChanged();
            _ctx.CbReviewGrabId.SelectedIndexChanged += (s, e) => OnReviewGrabIdChanged();
            _ctx.GrpReviewGrabNav.Click += (s, e) => OnReviewGrabIdChanged();

            _ctx.GrpDataSingleSheet.Click += (s, e) => SwitchActiveStatGroupBox(_ctx.GrpDataSingleSheet);
            _ctx.GroupBoxGrabIdRange.Click += (s, e) => SwitchActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
            _ctx.GroupBoxTimeRange.Click += (s, e) => SwitchActiveStatGroupBox(_ctx.GroupBoxTimeRange);
        }

        public void PopulateStatDateCombos(DateTime start, DateTime end)
        {
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

        public void PopulateAllGrabIdCombos(bool selectDataGrabId = false)
        {
            using (StatComboGuard.Enter())
            {
                var grabIdInfos = _getGrabIdInfos();
                _ctx.CbReviewGrabId.Items.Clear();
                _ctx.CbGrabIdStart.Items.Clear();
                _ctx.CbGrabIdEnd.Items.Clear();
                _ctx.CbDataGrabId.Items.Clear();
                foreach (var info in grabIdInfos)
                {
                    _ctx.CbReviewGrabId.Items.Add(info.GrabId);
                    _ctx.CbGrabIdStart.Items.Add(info.GrabId);
                    _ctx.CbGrabIdEnd.Items.Add(info.GrabId);
                    _ctx.CbDataGrabId.Items.Add(info.GrabId);
                }
                UpdateGrabIdNavState();
                if (_ctx.CbGrabIdStart.Items.Count > 0)
                {
                    _ctx.CbGrabIdStart.SelectedIndex = _ctx.CbGrabIdStart.Items.Count - 1;
                    _ctx.CbGrabIdEnd.SelectedIndex = 0;
                    if (selectDataGrabId)
                        _ctx.CbDataGrabId.SelectedIndex = _ctx.CbGrabIdStart.SelectedIndex;
                }
            }
            if (_ctx.CbDataGrabId.Items.Count > 0)
                _ctx.CbDataGrabId.SelectedIndex = 0;
        }

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
                _refreshStats();
                SetActiveStatGroupBox(_ctx.GrpDataSingleSheet);
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
                    _refreshStats();
                }
                return;
            }

            _suppressRangeOnSingleSheetSync = true;
            try { _ctx.CbDataGrabId.SelectedIndex = idx; }
            finally { _suppressRangeOnSingleSheetSync = false; }
        }

        public void SetActiveStatGroupBox(GroupBox active)
        {
            ActiveStatMode = active;
            foreach (var box in new[] { _ctx.GroupBoxGrabIdRange, _ctx.GrpDataSingleSheet, _ctx.GroupBoxTimeRange })
            {
                _setGroupBoxActive(box, box == active);
            }
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
            else if (target == _ctx.GroupBoxTimeRange && _getAvailableTimes().Count > 0)
            {
                PopulateStatDateCombos(_getAvailableTimes().Min, _getAvailableTimes().Max);
            }

            _refreshStats();
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

        private void OnStartComboChanged(int fromLevel)
        {
            if (StatComboGuard.IsSet) return;
            SetActiveStatGroupBox(_ctx.GroupBoxTimeRange);
            if (_getAvailableTimes().Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(_ctx.CbStartDate, _ctx.CbStartTime, _ctx.CbStartTime.Text);
                    ClampEndToStart();
                }
            }
            _refreshStats();
        }

        private void OnEndComboChanged(int fromLevel)
        {
            if (StatComboGuard.IsSet) return;
            SetActiveStatGroupBox(_ctx.GroupBoxTimeRange);
            if (_getAvailableTimes().Count > 0)
            {
                using (StatComboGuard.Enter())
                {
                    if (fromLevel <= 1) RefreshStatTimeCombo(_ctx.CbEndDate, _ctx.CbEndTime, _ctx.CbEndTime.Text);
                    ClampStartToEnd();
                }
            }
            _refreshStats();
        }

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
            var availableTimes = _getAvailableTimes();
            if (!TryBuildDateTimeFromCombos(_ctx.CbStartDate, _ctx.CbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(_ctx.CbEndDate, _ctx.CbEndTime, out DateTime end)) return;
            if (start <= end) return;
            var view = availableTimes.GetViewBetween(start, DateTime.MaxValue);
            DateTime newEnd = view.Count > 0 ? view.Min : availableTimes.Max;
            SetCombosToDateTime(false, newEnd);
        }

        private void ClampStartToEnd()
        {
            var availableTimes = _getAvailableTimes();
            if (!TryBuildDateTimeFromCombos(_ctx.CbStartDate, _ctx.CbStartTime, out DateTime start)) return;
            if (!TryBuildDateTimeFromCombos(_ctx.CbEndDate, _ctx.CbEndTime, out DateTime end)) return;
            if (start <= end) return;
            var view = availableTimes.GetViewBetween(DateTime.MinValue, end);
            DateTime newStart = view.Count > 0 ? view.Max : availableTimes.Min;
            SetCombosToDateTime(true, newStart);
        }

        private void OnGrabIdComboChanged(bool isStart)
        {
            var grabIdInfos = _getGrabIdInfos();
            if (StatComboGuard.IsSet || grabIdInfos.Count == 0) return;
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

                var startInfo = grabIdInfos[_ctx.CbGrabIdStart.SelectedIndex];
                var endInfo = grabIdInfos[_ctx.CbGrabIdEnd.SelectedIndex];
                SetCombosToDateTime(true, startInfo.Earliest);
                SetCombosToDateTime(false, endInfo.Latest);
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

            if (!_suppressRangeOnSingleSheetSync)
            {
                using (StatComboGuard.Enter())
                {
                    _ctx.CbGrabIdStart.SelectedIndex = idx;
                    _ctx.CbGrabIdEnd.SelectedIndex = idx;
                    var info = grabIdInfos[idx];
                    SetCombosToDateTime(true, info.Earliest);
                    SetCombosToDateTime(false, info.Latest);
                }
            }

            _refreshStats();

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
            _selectFromReview(info.GrabId, info.Earliest, info.Latest, idx);
        }

        private void UpdateDataGrabIdNavState()
        {
            int idx = _ctx.CbDataGrabId.SelectedIndex;
            int count = _ctx.CbDataGrabId.Items.Count;
        }

        private List<string> GetAvailableDateStrings() =>
            _getAvailableTimes().Select(t => t.ToString("yyyy-MM-dd")).Distinct()
                .OrderByDescending(x => x).ToList();

        private List<string> GetAvailableTimeStrings(string dateStr) =>
            _getAvailableTimes()
                .Where(t => t.ToString("yyyy-MM-dd") == dateStr)
                .Select(t => t.ToString("HH:mm:ss"))
                .Distinct().OrderByDescending(x => x).ToList();

        private static bool TryBuildDateTimeFromCombos(ComboBox dateCb, ComboBox timeCb, out DateTime result)
        {
            string text = $"{dateCb.Text} {timeCb.Text}";
            if (DateTime.TryParseExact(text, "yyyy-MM-dd HH:mm:ss",
                    CultureInfo.InvariantCulture,
                    DateTimeStyles.None, out result))
                return true;
            return false;
        }
    }
}
