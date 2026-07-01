using System;
using System.Collections.Generic;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Presenters;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Navigators
{
    public sealed class DataDateGrabIdNavigator
    {
        private readonly DataStatisticsContext _ctx;
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
            Func<List<GrabIdInfo>> getGrabIdInfos,
            Action refreshStats,
            Action<string, DateTime, DateTime, int> selectFromData,
            Action<string, DateTime, DateTime, int> selectFromReview,
            Action<GroupBox, bool> setGroupBoxActive)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getGrabIdInfos = getGrabIdInfos ?? throw new ArgumentNullException(nameof(getGrabIdInfos));
            _refreshStats = refreshStats ?? throw new ArgumentNullException(nameof(refreshStats));
            _selectFromData = selectFromData ?? throw new ArgumentNullException(nameof(selectFromData));
            _selectFromReview = selectFromReview ?? throw new ArgumentNullException(nameof(selectFromReview));
            _setGroupBoxActive = setGroupBoxActive ?? throw new ArgumentNullException(nameof(setGroupBoxActive));
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
            _ctx.GroupBoxGrabIdRange.Click += (s, e) => SwitchActiveStatGroupBox(_ctx.GroupBoxGrabIdRange);
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
            foreach (var box in new[] { _ctx.GroupBoxGrabIdRange, _ctx.GrpDataSingleSheet })
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

            _refreshStats();
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
    }
}
