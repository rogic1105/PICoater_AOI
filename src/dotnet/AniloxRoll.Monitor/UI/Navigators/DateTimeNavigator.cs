using System;
using System.Collections.Generic;
using System.Globalization;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.UI.State;

namespace AniloxRoll.Monitor.UI.Navigators
{
    /// <summary>
    /// [View Helper] 時間篩選連動管理器（簡化版：cbReviewDate + cbReviewTime）。
    /// cbReviewDate 顯示 YYYY-MM-DD，cbReviewTime 顯示 HH:mm:ss.fff。
    /// </summary>
    public class DateTimeNavigator
    {
        private readonly ImageRepository _repository;
        private readonly ComboBox _cbReviewDate, _cbReviewTime;
        private bool _updating;
        public event Action PeriodSelectionChanged;

        public DateTimeNavigator(
            ImageRepository repository,
            ComboBox cbReviewDate, ComboBox cbReviewTime)
        {
            _repository = repository;
            _cbReviewDate = cbReviewDate;
            _cbReviewTime = cbReviewTime;

            _cbReviewDate.SelectedIndexChanged += (s, e) =>
            {
                if (!_updating) UpdateTimeCombo();
                if (!_updating) OnPeriodSelectionChanged();
            };
            _cbReviewTime.SelectedIndexChanged += (s, e) =>
            {
                if (!_updating) OnPeriodSelectionChanged();
            };
        }

        private void OnPeriodSelectionChanged() => PeriodSelectionChanged?.Invoke();

        public void Initialize(string lastYear)
        {
            _updating = true;
            try
            {
                var dates = _repository.GetDates();
                _cbReviewDate.Items.Clear();
                _cbReviewDate.Items.AddRange(dates.ToArray());
                if (dates.Count == 0) return;

                // 嘗試還原上次選擇的日期
                string lastDate = BuildLastDate();
                int idx = dates.IndexOf(lastDate);
                _cbReviewDate.SelectedIndex = idx >= 0 ? idx : 0;

                UpdateTimeCombo();
            }
            finally { _updating = false; }
        }

        private void UpdateTimeCombo(bool autoSelect = true)
        {
            var times = _repository.GetTimesForDate(_cbReviewDate.Text);
            _cbReviewTime.Items.Clear();
            _cbReviewTime.Items.AddRange(times.ToArray());
            if (times.Count == 0) return;
            if (!autoSelect) return;

            string lastTime = BuildLastTime();
            int idx = times.IndexOf(lastTime);
            _cbReviewTime.SelectedIndex = idx >= 0 ? idx : 0;
        }

        /// <summary>從 UserSessionState 組合上次存的日期字串。</summary>
        private static string BuildLastDate()
        {
            string y = UserSessionState.LastYear;
            string m = UserSessionState.LastMonth;
            string d = UserSessionState.LastDay;
            if (string.IsNullOrEmpty(y)) return string.Empty;
            return $"{y}-{m}-{d}";
        }

        /// <summary>從 UserSessionState 組合上次存的時間字串。</summary>
        private static string BuildLastTime()
        {
            string h = UserSessionState.LastHour;
            string min = UserSessionState.LastMin;
            string sec = UserSessionState.LastSec;
            if (string.IsNullOrEmpty(h)) return string.Empty;
            return $"{h}:{min}:{sec}";
        }

        public void SaveCurrentSelection()
        {
            ParseDateTimeTexts(out string y, out string m, out string d,
                               out string h, out string min, out string sec);
            UserSessionState.SaveDateTimeSelection(y, m, d, h, min, sec);
            UserSessionState.Save();
        }

        // ── 保持向後相容的 Get* 方法 ────────────────────────────────

        public string GetCurrentYear()  { ParseDate(out string y, out _, out _); return y; }
        public string GetCurrentMonth() { ParseDate(out _, out string m, out _); return m; }
        public string GetCurrentDay()   { ParseDate(out _, out _, out string d); return d; }
        public string GetCurrentHour()  { ParseTime(out string h, out _, out _); return h; }
        public string GetCurrentMin()   { ParseTime(out _, out string min, out _); return min; }
        public string GetCurrentSec()   { ParseTime(out _, out _, out string sec); return sec; }

        public DateTime GetCurrentPeriodOrDefault(DateTime fallback)
        {
            ParseDateTimeTexts(out string y, out string m, out string d,
                               out string h, out string min, out string secFff);
            if (!int.TryParse(y, out int yi) || !int.TryParse(m, out int mi) ||
                !int.TryParse(d, out int di) || !int.TryParse(h, out int hi) ||
                !int.TryParse(min, out int mni))
                return fallback;

            int si = 0, msi = 0;
            if (secFff != null)
            {
                int dot = secFff.IndexOf('.');
                if (dot >= 0)
                {
                    int.TryParse(secFff.Substring(0, dot), out si);
                    int.TryParse(secFff.Substring(dot + 1), out msi);
                }
                else
                    int.TryParse(secFff, out si);
            }
            try { return new DateTime(yi, mi, di, hi, mni, si, msi); }
            catch { return fallback; }
        }

        public void NavigateTo(DateTime dt)
        {
            UserSessionState.SaveDateTimeSelection(
                dt.ToString("yyyy"), dt.ToString("MM"), dt.ToString("dd"),
                dt.ToString("HH"), dt.ToString("mm"), dt.ToString("ss.fff"));
            UserSessionState.Save();
            Initialize(dt.ToString("yyyy"));
        }

        public void SetPeriodToCombo(DateTime dt)
        {
            _updating = true;
            try
            {
                string dateStr = dt.ToString("yyyy-MM-dd");
                string timeStr = dt.ToString("HH:mm:ss.fff");
                if (_cbReviewDate.Items.Contains(dateStr))
                    _cbReviewDate.SelectedItem = dateStr;
                else
                    _cbReviewDate.Text = dateStr;

                // 日期變更後必須刷新 time combo 的 Items，否則殘留前一天的時間列表
                UpdateTimeCombo(autoSelect: false);

                if (_cbReviewTime.Items.Contains(timeStr))
                    _cbReviewTime.SelectedItem = timeStr;
                else
                    _cbReviewTime.Text = timeStr;
            }
            finally { _updating = false; }
        }

        // ── 內部解析 ────────────────────────────────────────────────

        private void ParseDate(out string year, out string month, out string day)
        {
            year = ""; month = ""; day = "";
            string text = _cbReviewDate.Text ?? "";
            var parts = text.Split('-');
            if (parts.Length >= 3) { year = parts[0]; month = parts[1]; day = parts[2]; }
        }

        private void ParseTime(out string hour, out string min, out string secFff)
        {
            hour = ""; min = ""; secFff = "";
            string text = _cbReviewTime.Text ?? "";
            var parts = text.Split(':');
            if (parts.Length >= 3) { hour = parts[0]; min = parts[1]; secFff = parts[2]; }
        }

        private void ParseDateTimeTexts(out string y, out string m, out string d,
                                         out string h, out string min, out string sec)
        {
            ParseDate(out y, out m, out d);
            ParseTime(out h, out min, out sec);
        }
    }
}
