// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Forms\Helpers\TimeSelectionManager.cs

using System;
using System.Collections.Generic;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    /// <summary>
    /// 負責管理 年/月/日/時/分/秒 下拉選單的連動邏輯
    /// </summary>
    public class TimeSelectionManager
    {
        private readonly ImageRepository _repository;
        private readonly ComboBox _cbYear, _cbMonth, _cbDay, _cbHour, _cbMin, _cbSec;

        public TimeSelectionManager(
            ImageRepository repository,
            ComboBox year, ComboBox month, ComboBox day,
            ComboBox hour, ComboBox min, ComboBox sec)
        {
            _repository = repository;
            _cbYear = year;
            _cbMonth = month;
            _cbDay = day;
            _cbHour = hour;
            _cbMin = min;
            _cbSec = sec;

            BindEvents();
        }

        private void BindEvents()
        {
            _cbYear.SelectedIndexChanged += (s, e) => UpdateMonth();
            _cbMonth.SelectedIndexChanged += (s, e) => UpdateDay();
            _cbDay.SelectedIndexChanged += (s, e) => UpdateHour();
            _cbHour.SelectedIndexChanged += (s, e) => UpdateMinute();
            _cbMin.SelectedIndexChanged += (s, e) => UpdateSecond();
        }

        public void Initialize(string lastYear)
        {
            UpdateCombo(_cbYear, _repository.GetYears(), lastYear);
            if (_cbYear.Items.Count > 0 && _cbYear.SelectedIndex == -1)
                _cbYear.SelectedIndex = 0;
        }

        // 封裝各層級的更新邏輯，不再讓 Form 直接呼叫 Repository
        private void UpdateMonth() => UpdateCombo(_cbMonth, _repository.GetMonths(_cbYear.Text), Properties.Settings.Default.LastMonth);
        private void UpdateDay() => UpdateCombo(_cbDay, _repository.GetDays(_cbYear.Text, _cbMonth.Text), Properties.Settings.Default.LastDay);
        private void UpdateHour() => UpdateCombo(_cbHour, _repository.GetHours(_cbYear.Text, _cbMonth.Text, _cbDay.Text), Properties.Settings.Default.LastHour);
        private void UpdateMinute() => UpdateCombo(_cbMin, _repository.GetMinutes(_cbYear.Text, _cbMonth.Text, _cbDay.Text, _cbHour.Text), Properties.Settings.Default.LastMin);
        private void UpdateSecond() => UpdateCombo(_cbSec, _repository.GetSeconds(_cbYear.Text, _cbMonth.Text, _cbDay.Text, _cbHour.Text, _cbMin.Text), Properties.Settings.Default.LastSec);

        private void UpdateCombo(ComboBox cb, List<string> items, string lastVal)
        {
            cb.Items.Clear();
            cb.Items.AddRange(items.ToArray());
            if (items.Count == 0) return;
            cb.SelectedItem = items.Contains(lastVal) ? lastVal : items[0];
        }

        public void SaveCurrentSelection()
        {
            Properties.Settings.Default.LastYear = _cbYear.Text;
            Properties.Settings.Default.LastMonth = _cbMonth.Text;
            Properties.Settings.Default.LastDay = _cbDay.Text;
            Properties.Settings.Default.LastHour = _cbHour.Text;
            Properties.Settings.Default.LastMin = _cbMin.Text;
            Properties.Settings.Default.LastSec = _cbSec.Text;
            Properties.Settings.Default.Save();
        }

        public string GetCurrentYear() => _cbYear.Text;
        public string GetCurrentMonth() => _cbMonth.Text;
        public string GetCurrentDay() => _cbDay.Text;
        public string GetCurrentHour() => _cbHour.Text;
        public string GetCurrentMin() => _cbMin.Text;
        public string GetCurrentSec() => _cbSec.Text;
    }
}