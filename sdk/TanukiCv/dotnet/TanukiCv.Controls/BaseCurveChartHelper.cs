using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace TanukiCv.Controls
{
    /// <summary>
    /// Column/Row 曲線圖共用基底：chart 初始化骨架、閾值線、Mean/Max series 建立。
    /// 子類實作方向特定的 ChartArea、series anchor、InnerPlotPosition 補償。
    /// </summary>
    public abstract class BaseCurveChartHelper
    {
        protected readonly Chart _chart;
        // 與 InspectionDefaults.ErrorValueMeanV/MaxV 對齊；runtime 會被 owner 的 RefreshThresholds() 覆寫，
        // 此預設值只在 Build() 初始 Y 軸計算前短暫使用
        protected float _errorValueMean = 0.2f;
        protected float _errorValueMax  = 0.4f;
        protected bool  _innerPlotPositionFrozen = false;

        protected static readonly Font  UnitFont  = new Font("Segoe UI", 7f);
        protected static readonly Brush UnitBrush = new SolidBrush(Color.Gray);

        protected BaseCurveChartHelper(Chart chart)
        {
            _chart = chart;
        }

        /// <summary>更新閾值（PropertyGrid 修改配方後呼叫）。</summary>
        public void SetThresholds(float errorValueMean, float errorValueMax)
        {
            _errorValueMean = errorValueMean;
            _errorValueMax  = errorValueMax;
            RefreshThresholds();
        }

        // ── 建立骨架 ──────────────────────────────────────────────────────

        /// <summary>
        /// 初始化 chart（清空 → 建立 ChartArea → 加入 series → 設閾值 → 掛 PostPaint）。
        /// 子類建構子呼叫。
        /// </summary>
        protected void Build()
        {
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();
            _chart.Legends.Clear();
            _chart.Margin  = new Padding(0);
            _chart.Padding = new Padding(0);

            _chart.ChartAreas.Add(BuildChartArea());
            AddAnchorSeries();
            AddMeanMaxSeries();
            RefreshThresholds();
            _chart.PostPaint += OnPostPaint;
            _chart.PostPaint += OnPostPaintUnit;
        }

        /// <summary>建立方向特定的 ChartArea。</summary>
        protected abstract ChartArea BuildChartArea();

        /// <summary>建立 anchor series（確保軸 scale 正確建立）。</summary>
        protected abstract void AddAnchorSeries();

        /// <summary>更新閾值 StripLines + 調整軸範圍。</summary>
        protected abstract void RefreshThresholds();

        /// <summary>InnerPlotPosition 凍結 + zoom 補正。</summary>
        protected abstract void OnPostPaint(object sender, ChartPaintEventArgs e);

        /// <summary>繪製 "mm" 單位標籤。</summary>
        protected abstract void OnPostPaintUnit(object sender, ChartPaintEventArgs e);

        // ── 共用 series 建立 ─────────────────────────────────────────────

        /// <summary>建立 Mean/Max FastLine series（綁 Primary Y 軸）。</summary>
        private void AddMeanMaxSeries()
        {
            _chart.Series.Add(new Series("Mean")
            {
                ChartType       = SeriesChartType.FastLine,
                Color           = Color.DeepSkyBlue,
                BorderDashStyle = ChartDashStyle.Dash,
                YAxisType       = AxisType.Primary
            });

            _chart.Series.Add(new Series("Max")
            {
                ChartType       = SeriesChartType.FastLine,
                Color           = Color.Blue,
                BorderDashStyle = ChartDashStyle.Solid,
                YAxisType       = AxisType.Primary
            });
        }

        // ── 共用閾值線工廠 ───────────────────────────────────────────────

        protected static StripLine MakeStripLine(double offset, ChartDashStyle dash) => new StripLine
        {
            IntervalOffset  = offset,
            StripWidth      = 0,
            Interval        = 0,
            BorderColor     = Color.Red,
            BorderWidth     = 1,
            BorderDashStyle = dash,
            BackColor       = Color.Transparent,
        };
    }
}
