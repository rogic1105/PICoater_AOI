using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace TanukiCv.Controls
{
    /// <summary>⚠ 不變量（踩過 3 次的坑）：chart 任何重畫**必須原子帶視野**（UpdateDataAndView 一次給資料+範圍），
    /// **嚴禁**「先 Clear/重設 → 之後再補視野」——中間狀態會閃給使用者看（重載/強化切換回預設一閃）。
    /// 視野來源用呼叫端快取的「當前視野」（Live=_liveViewLeftMm、Review=SameSourceViewRange 注入），不要事後補發。</summary>
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

        /// <summary>是否顯示紅色閾值線（mura 用）。純剖面（L0）等場景可關；設值即重整。子類 RefreshThresholds 依此 gate。</summary>
        protected bool _showThresholds = true;
        public bool ShowThresholds { get => _showThresholds; set { _showThresholds = value; if (_chart.ChartAreas.Count > 0) RefreshThresholds(); } }

        protected static readonly Font  UnitFont  = new Font("Segoe UI", 7f);
        protected static readonly Brush UnitBrush = new SolidBrush(Color.Gray);

        protected BaseCurveChartHelper(Chart chart)
        {
            _chart = chart;
        }

        /// <summary>清空曲線（Mean/Max 點全清）→ chart 歸零（如游標移出影像時）。保留 anchor series 維持軸範圍。</summary>
        public void Clear()
        {
            if (_chart.Series.IndexOf("Mean") >= 0) _chart.Series["Mean"].Points.Clear();
            if (_chart.Series.IndexOf("Max")  >= 0) _chart.Series["Max"].Points.Clear();
            _chart.Invalidate();
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
