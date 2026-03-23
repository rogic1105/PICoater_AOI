// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Forms\Helpers\TimeSelectionManager.cs

using System;
using System.Collections.Generic;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.UI.State;

namespace AniloxRoll.Monitor.UI.Navigators
{
    /// <summary>
    /// [View Helper] 時間篩選連動管理器。
    /// 負責處理 ComboBox (年/月/日/時/分/秒) 之間的級聯更新邏輯，
    /// 並將使用者的選擇持久化儲存 (Properties.Settings)。
    /// </summary>
    public class DateTimeNavigator
    {
        private readonly ImageRepository _repository;
        private readonly ComboBox _cbYear, _cbMonth, _cbDay, _cbHour, _cbMin, _cbSec;
        public event Action PeriodSelectionChanged;

        public DateTimeNavigator(
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
            _cbYear.SelectedIndexChanged += (s, e) => { UpdateMonth(); OnPeriodSelectionChanged(); };
            _cbMonth.SelectedIndexChanged += (s, e) => { UpdateDay(); OnPeriodSelectionChanged(); };
            _cbDay.SelectedIndexChanged += (s, e) => { UpdateHour(); OnPeriodSelectionChanged(); };
            _cbHour.SelectedIndexChanged += (s, e) => { UpdateMinute(); OnPeriodSelectionChanged(); };
            _cbMin.SelectedIndexChanged += (s, e) => { UpdateSecond(); OnPeriodSelectionChanged(); };
            _cbSec.SelectedIndexChanged += (s, e) => OnPeriodSelectionChanged();
        }

        private void OnPeriodSelectionChanged() => PeriodSelectionChanged?.Invoke();

        public void Initialize(string lastYear)
        {
            UpdateCombo(_cbYear, _repository.GetYears(), lastYear);
            if (_cbYear.Items.Count > 0 && _cbYear.SelectedIndex == -1)
                _cbYear.SelectedIndex = 0;
        }

        // 封裝各層級的更新邏輯，不再讓 Form 直接呼叫 Repository
        private void UpdateMonth() => UpdateCombo(_cbMonth, _repository.GetMonths(_cbYear.Text), UserSessionState.LastMonth);
        private void UpdateDay() => UpdateCombo(_cbDay, _repository.GetDays(_cbYear.Text, _cbMonth.Text), UserSessionState.LastDay);
        private void UpdateHour() => UpdateCombo(_cbHour, _repository.GetHours(_cbYear.Text, _cbMonth.Text, _cbDay.Text), UserSessionState.LastHour);
        private void UpdateMinute() => UpdateCombo(_cbMin, _repository.GetMinutes(_cbYear.Text, _cbMonth.Text, _cbDay.Text, _cbHour.Text), UserSessionState.LastMin);
        private void UpdateSecond() => UpdateCombo(_cbSec, _repository.GetSeconds(_cbYear.Text, _cbMonth.Text, _cbDay.Text, _cbHour.Text, _cbMin.Text), UserSessionState.LastSec);

        private void UpdateCombo(ComboBox cb, List<string> items, string lastVal)
        {
            cb.Items.Clear();
            cb.Items.AddRange(items.ToArray());
            if (items.Count == 0) return;
            cb.SelectedItem = items.Contains(lastVal) ? lastVal : items[0];
        }

        public void SaveCurrentSelection()
        {
            UserSessionState.SaveDateTimeSelection(
                _cbYear.Text,
                _cbMonth.Text,
                _cbDay.Text,
                _cbHour.Text,
                _cbMin.Text,
                _cbSec.Text
            );
            UserSessionState.Save();
        }

        public string GetCurrentYear() => _cbYear.Text;
        public string GetCurrentMonth() => _cbMonth.Text;
        public string GetCurrentDay() => _cbDay.Text;
        public string GetCurrentHour() => _cbHour.Text;
        public string GetCurrentMin() => _cbMin.Text;
        public string GetCurrentSec() => _cbSec.Text;

        public DateTime GetCurrentPeriodOrDefault(DateTime fallback)
        {
            if (int.TryParse(_cbYear.Text, out int y) && int.TryParse(_cbMonth.Text, out int m) && int.TryParse(_cbDay.Text, out int d) &&
                int.TryParse(_cbHour.Text, out int h) && int.TryParse(_cbMin.Text, out int min))
            {
                int s = 0, ms = 0;
                string secText = _cbSec.Text ?? "";
                int dot = secText.IndexOf('.');
                if (dot >= 0)
                {
                    int.TryParse(secText.Substring(0, dot), out s);
                    int.TryParse(secText.Substring(dot + 1), out ms);
                }
                else
                {
                    int.TryParse(secText, out s);
                }
                try { return new DateTime(y, m, d, h, min, s, ms); }
                catch { }
            }
            return fallback;
        }

        /// <summary>
        /// 以指定時間為目標，更新 session state 後重新初始化 cascade，
        /// 使各層 ComboBox 自動選到最接近該時間的項目。
        /// </summary>
        public void NavigateTo(DateTime dt)
        {
            UserSessionState.SaveDateTimeSelection(
                dt.ToString("yyyy"), dt.ToString("MM"), dt.ToString("dd"),
                dt.ToString("HH"), dt.ToString("mm"), dt.ToString("ss.fff"));
            Initialize(dt.ToString("yyyy"));
        }

        public void SetPeriodToCombo(DateTime dt)
        {
            _cbYear.Text = dt.ToString("yyyy");
            _cbMonth.Text = dt.ToString("MM");
            _cbDay.Text = dt.ToString("dd");
            _cbHour.Text = dt.ToString("HH");
            _cbMin.Text = dt.ToString("mm");
            _cbSec.Text = dt.ToString("ss.fff");
        }
    }
}
