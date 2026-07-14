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
    /// <summary>
    /// Owns the data-tab yield period charts: yearly, monthly, daily, scale mode, and chart navigation combos.
    /// </summary>
    public sealed class YieldPeriodChartPresenter
    {
        private const string PassSeriesName = "合格";
        private const string FailSeriesName = "異常";

        private readonly DataStatisticsContext _ctx;
        private readonly Func<SortedSet<DateTime>> _getAvailableTimes;
        private readonly Func<IList<GrabIdInfo>> _getGrabIds;
        private readonly Func<IDictionary<string, GrabDetail>> _getDetails;
        private readonly EventGuard _chartNavGuard = new EventGuard();

        private List<int> _chartYears = new List<int>();
        private List<int> _chartMonths = new List<int>();
        private List<int> _chartDays = new List<int>();
        private uint _lastChartToggleTick;
        private bool _hasChartToggleTick;
        private ChartScaleMode? _yearlyScaleOverride;
        private ChartScaleMode? _monthlyScaleOverride;
        private ChartScaleMode? _dailyScaleOverride;

        public YieldPeriodChartPresenter(
            DataStatisticsContext ctx,
            Func<SortedSet<DateTime>> getAvailableTimes,
            Func<IList<GrabIdInfo>> getGrabIds,
            Func<IDictionary<string, GrabDetail>> getDetails)
        {
            _ctx = ctx ?? throw new ArgumentNullException(nameof(ctx));
            _getAvailableTimes = getAvailableTimes ?? throw new ArgumentNullException(nameof(getAvailableTimes));
            _getGrabIds = getGrabIds ?? throw new ArgumentNullException(nameof(getGrabIds));
            _getDetails = getDetails ?? throw new ArgumentNullException(nameof(getDetails));
        }

        public void Init()
        {
            var cs = _ctx.Settings.Chart;
            InitOneChart(_ctx.ChartDataYieldYearly, yDefault: cs.YearlyYMax, xCount: 12, xStart: 1, xUnit: "月");
            InitOneChart(_ctx.ChartDataYieldMonthly, yDefault: cs.MonthlyYMax, xCount: 31, xStart: 1);
            InitOneChart(_ctx.ChartDataYieldDaily, yDefault: cs.DailyYMax, xCount: 24, xStart: 0);

            foreach (var chart in new[]
            {
                _ctx.ChartDataYieldYearly,
                _ctx.ChartDataYieldMonthly,
                _ctx.ChartDataYieldDaily
            })
            {
                ApplyScale(chart, GetEffectiveScaleMode(chart), ReadChartData(chart));
            }

            _ctx.ChartDataYieldYearly.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldMonthly.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldDaily.MouseClick -= PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldYearly.MouseClick += PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldMonthly.MouseClick += PeriodChart_ToggleAutoScale;
            _ctx.ChartDataYieldDaily.MouseClick += PeriodChart_ToggleAutoScale;

            _ctx.CbChartYear.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) { FlowTrace.Log($"ui:【良率導航-年】→ {_ctx.CbChartYear.SelectedItem}"); OnChartYearIndexChanged(); } };
            _ctx.CbChartMonth.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) { FlowTrace.Log($"ui:【良率導航-月】→ {_ctx.CbChartMonth.SelectedItem}"); OnChartMonthIndexChanged(); } };
            _ctx.CbChartDay.SelectedIndexChanged += (s, e) => { if (!_chartNavGuard.IsSet) { FlowTrace.Log($"ui:【良率導航-日】→ {_ctx.CbChartDay.SelectedItem}"); OnChartDayIndexChanged(); } };
        }

        private void PeriodChart_ToggleAutoScale(object sender, MouseEventArgs e)
        {
            uint now = unchecked((uint)Environment.TickCount);
            if (_hasChartToggleTick && unchecked(now - _lastChartToggleTick) < 500u) return;
            _lastChartToggleTick = now;
            _hasChartToggleTick = true;

            var chart = (Chart)sender;
            if (chart.ChartAreas.Count == 0) return;

            ChartScaleMode settingMode = _ctx.Settings.Chart.ScaleMode;
            ChartScaleMode effectiveMode = GetEffectiveScaleMode(chart);
            ChartScaleMode nextMode = effectiveMode == ChartScaleMode.Auto
                ? ChartScaleMode.Fixed
                : ChartScaleMode.Auto;

            // 圖表點擊是暫時顯示鍵，不回寫 PropertyGrid setting。
            // 切回 setting 本身時收掉 override，之後設定變更便會自然生效。
            SetScaleOverride(chart, nextMode == settingMode ? (ChartScaleMode?)null : nextMode);
            ApplyScale(chart, nextMode, ReadChartData(chart));

            ChartScaleMode? scaleOverride = GetScaleOverride(chart);
            FlowTrace.Log($"ui:【良率圖-{GetChartPeriodName(chart)}】→ Y軸={GetScaleModeName(nextMode)} "
                + $"setting={GetScaleModeName(settingMode)} override={(scaleOverride.HasValue ? GetScaleModeName(scaleOverride.Value) : "off")}");
        }

        private static void InitOneChart(Chart chart, int xLabelAngle = 0, int yDefault = 10,
            int xCount = 0, int xStart = 1, string xUnit = "")
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
            if (!string.IsNullOrEmpty(xUnit))
            {
                area.AxisX.Title = xUnit;                            // X 軸單位（年圖＝月）
                area.AxisX.TitleAlignment = StringAlignment.Near;    // 靠左 → 顯示在左下角
                area.AxisX.TitleFont = new Font("Arial", 8f);
                area.AxisX.TitleForeColor = Color.FromArgb(90, 90, 90);
            }
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
            area.InnerPlotPosition.Y = 18f;       // 上緣多留空白（原 12）
            area.InnerPlotPosition.Width = 93f;
            area.InnerPlotPosition.Height = 60f;  // 縮 6 補回，底部 X 標籤/單位不被擠（下緣仍 78）
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

            var sPass = new Series(PassSeriesName);
            sPass.ChartType = SeriesChartType.StackedColumn;
            sPass.Color = Color.FromArgb(102, 187, 106);
            sPass.ChartArea = "Main";
            sPass.Legend = "L";
            sPass.YAxisType = AxisType.Secondary;
            chart.Series.Add(sPass);

            var sFail = new Series(FailSeriesName);
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
            var sPass = chart.Series[PassSeriesName];
            var sFail = chart.Series[FailSeriesName];
            sPass.Points.Clear();
            sFail.Points.Clear();
            foreach (var p in data)
            {
                sPass.Points.AddXY(p.Label, p.Pass);
                sFail.Points.AddXY(p.Label, p.Fail);
            }

            ApplyScale(chart, GetEffectiveScaleMode(chart), data);
        }

        private static void ApplyAutoScale(Chart chart, List<PeriodStats> data)
        {
            int maxTotal = 0;
            foreach (var p in data)
                maxTotal = Math.Max(maxTotal, p.Pass + p.Fail);
            int niceMax = Math.Max(5, (int)(Math.Ceiling(maxTotal / 5.0) * 5));
            SetChartYRange(chart, niceMax * 1.05, niceMax / 5.0, niceMax);
        }

        private static void ApplyFixedScale(Chart chart, int fixedMax)
        {
            SetChartYRange(chart, fixedMax, fixedMax / 5.0, fixedMax);
        }

        private void ApplyScale(Chart chart, ChartScaleMode mode, List<PeriodStats> data)
        {
            if (mode == ChartScaleMode.Auto)
                ApplyAutoScale(chart, data);
            else
                ApplyFixedScale(chart, GetFixedScaleMax(chart));
        }

        private ChartScaleMode GetEffectiveScaleMode(Chart chart) =>
            GetScaleOverride(chart) ?? _ctx.Settings.Chart.ScaleMode;

        private ChartScaleMode? GetScaleOverride(Chart chart)
        {
            if (chart == _ctx.ChartDataYieldYearly) return _yearlyScaleOverride;
            if (chart == _ctx.ChartDataYieldMonthly) return _monthlyScaleOverride;
            return _dailyScaleOverride;
        }

        private void SetScaleOverride(Chart chart, ChartScaleMode? mode)
        {
            if (chart == _ctx.ChartDataYieldYearly)
                _yearlyScaleOverride = mode;
            else if (chart == _ctx.ChartDataYieldMonthly)
                _monthlyScaleOverride = mode;
            else
                _dailyScaleOverride = mode;
        }

        private int GetFixedScaleMax(Chart chart) =>
            chart == _ctx.ChartDataYieldYearly ? _ctx.Settings.Chart.YearlyYMax
          : chart == _ctx.ChartDataYieldMonthly ? _ctx.Settings.Chart.MonthlyYMax
          : _ctx.Settings.Chart.DailyYMax;

        private string GetChartPeriodName(Chart chart) =>
            chart == _ctx.ChartDataYieldYearly ? "年"
          : chart == _ctx.ChartDataYieldMonthly ? "月"
          : "日";

        private static string GetScaleModeName(ChartScaleMode mode) =>
            mode == ChartScaleMode.Auto ? "Auto" : "Fixed";

        private static List<PeriodStats> ReadChartData(Chart chart)
        {
            var data = new List<PeriodStats>();
            var sPass = chart.Series[PassSeriesName];
            var sFail = chart.Series[FailSeriesName];
            for (int i = 0; i < sPass.Points.Count; i++)
                data.Add(new PeriodStats
                {
                    Label = "",
                    Pass = (int)sPass.Points[i].YValues[0],
                    Fail = (int)sFail.Points[i].YValues[0]
                });
            return data;
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
            foreach (var chart in new[] { _ctx.ChartDataYieldYearly, _ctx.ChartDataYieldMonthly, _ctx.ChartDataYieldDaily })
            {
                if (chart.ChartAreas.Count == 0) continue;
                ApplyScale(chart, GetEffectiveScaleMode(chart), ReadChartData(chart));
            }
        }

        public void ApplyChartScaleForChart(string chartName)
        {
            var chart = chartName == "Yearly" ? _ctx.ChartDataYieldYearly
                      : chartName == "Monthly" ? _ctx.ChartDataYieldMonthly
                      : _ctx.ChartDataYieldDaily;
            ApplyScale(chart, GetEffectiveScaleMode(chart), ReadChartData(chart));
        }

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
            FillPeriodChart(_ctx.ChartDataYieldYearly,
                InspectionStatisticsService.ComputeGroupedByMonthOfYear(
                    _getGrabIds(), _getDetails(),
                    new DateTime(year, 1, 1),
                    new DateTime(year, 12, 31, 23, 59, 59)));

            OnChartMonthIndexChanged(hint);
        }

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
                InspectionStatisticsService.ComputeGroupedByDayOfMonth(
                    _getGrabIds(), _getDetails(),
                    new DateTime(year, month, 1),
                    new DateTime(year, month, lastDay, 23, 59, 59)));

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
                InspectionStatisticsService.ComputeGroupedByHourOfDay(
                    _getGrabIds(), _getDetails(),
                    new DateTime(year, month, day),
                    new DateTime(year, month, day, 23, 59, 59)));
        }

        private List<int> GetAvailableYears() =>
            _getAvailableTimes().Select(t => t.Year).Distinct().ToList();

        private List<int> GetAvailableMonths(int y) =>
            _getAvailableTimes().Where(t => t.Year == y)
                               .Select(t => t.Month).Distinct().ToList();

        private List<int> GetAvailableDays(int y, int mo) =>
            _getAvailableTimes().Where(t => t.Year == y && t.Month == mo)
                               .Select(t => t.Day).Distinct().ToList();
    }
}
