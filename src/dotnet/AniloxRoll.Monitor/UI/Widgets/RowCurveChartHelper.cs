using System;
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 法向（axial）Mura 曲線圖：row-wise ridge data，旋轉 90° 顯示。
    /// X 軸（底部）= curve value（0–1 normalized），Y 軸（左側）= 法向位置 mm。
    /// Y 軸標籤反轉：視覺上 0 在上、max 在下（透過 Customize 事件修改標籤文字）。
    /// InnerPlotPosition 補償機制對齊 canvas 垂直 viewport。
    /// </summary>
    public class RowCurveChartHelper : BaseCurveChartHelper
    {
        private double _rowPitchMm = 0.01;
        private double _totalMm   = 100;

        public double RowPitchMm => _rowPitchMm;
        public double TotalMm   => _totalMm;

        // InnerPlotPosition 補償（垂直方向）
        private double _cachedFTop              = 0.0;
        private double _cachedFBottom           = 1.0;

        // 上次設定的「邏輯視野」
        private double _logicalTopMm  = double.NaN;
        private double _logicalBotMm  = double.NaN;

        public RowCurveChartHelper(Chart chart) : base(chart)
        {
            Build();
            _chart.Customize += OnCustomizeLabels;
        }

        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
        {
            if (speedMPerMin > 0 && lineRateHz > 0)
                _rowPitchMm = (speedMPerMin / 60.0 * 1000.0) / lineRateHz;
        }

        /// <summary>
        /// 更新 row-wise 曲線資料。meanData[i] / maxData[i] 為 row i 的值（0–255 raw）。
        /// </summary>
        public void UpdateData(float[] meanData, float[] maxData)
        {
            if (meanData == null || meanData.Length == 0) return;

            int n = meanData.Length;
            _totalMm = n * _rowPitchMm;

            _chart.Series.SuspendUpdates();

            var meanSeries = _chart.Series["Mean"];
            var maxSeries  = _chart.Series["Max"];
            meanSeries.Points.Clear();
            maxSeries.Points.Clear();

            for (int i = 0; i < n; i++)
            {
                double yMm = (n - 1 - i) * _rowPitchMm;
                double xMean = meanData[i] / 255.0;
                meanSeries.Points.AddXY(xMean, yMm);
                if (maxData != null && i < maxData.Length)
                    maxSeries.Points.AddXY(maxData[i] / 255.0, yMm);
            }

            var area = _chart.ChartAreas[0];
            area.AxisY.Minimum = 0;
            area.AxisY.Maximum = _totalMm;

            _chart.Series.ResumeUpdates();
        }

        /// <summary>
        /// 更新 Y 軸視野範圍（對應 canvas 垂直 viewport），單位為 mm。
        /// </summary>
        public void UpdateViewRange(double canvasTopMm, double canvasBotMm)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(canvasTopMm) || double.IsNaN(canvasBotMm) || canvasTopMm >= canvasBotMm) return;

            _logicalTopMm = canvasTopMm;
            _logicalBotMm = canvasBotMm;

            GetAdjustedZoom(canvasTopMm, canvasBotMm, out double zMin, out double zMax);
            var axisY = _chart.ChartAreas[0].AxisY;
            axisY.Minimum = Math.Min(0, zMin);
            axisY.Maximum = Math.Max(_totalMm, zMax);
            try { axisY.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[RowCurveChartHelper.UpdateViewRange] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ── InnerPlotPosition 補償 ────────────────────────────────────────────

        protected override void OnPostPaint(object sender, ChartPaintEventArgs e)
        {
            if (_innerPlotPositionFrozen) return;
            if (_chart.ChartAreas.Count == 0) return;

            var inner = _chart.ChartAreas[0].InnerPlotPosition;
            if (inner.Height < 1.0) return;

            const float topPadding = 0.5f;

            double newFTop    = (inner.Y + topPadding) / 100.0;
            double newFBottom = (inner.Y + inner.Height) / 100.0;
            bool   changed   = Math.Abs(newFTop    - _cachedFTop)    > 0.001 ||
                               Math.Abs(newFBottom - _cachedFBottom) > 0.001;

            _cachedFTop    = newFTop;
            _cachedFBottom = newFBottom;
            _innerPlotPositionFrozen = true;

            var area = _chart.ChartAreas[0];
            area.InnerPlotPosition.Auto   = false;
            area.InnerPlotPosition.X      = inner.X;
            area.InnerPlotPosition.Y      = inner.Y + topPadding;
            area.InnerPlotPosition.Width  = inner.Width;
            area.InnerPlotPosition.Height = Math.Max(1f, inner.Height - topPadding);

            if (changed && !double.IsNaN(_logicalTopMm) && _logicalTopMm < _logicalBotMm)
            {
                double top = _logicalTopMm;
                double bot = _logicalBotMm;
                // 守 IsHandleCreated：StitchMode 切換期 chart 可能正在 re-layout，handle 短暫無效
                if (_chart.IsHandleCreated && !_chart.IsDisposed)
                {
                    try
                    {
                        _chart.BeginInvoke(new Action(() =>
                        {
                            GetAdjustedZoom(top, bot, out double zMin, out double zMax);
                            var axisY = _chart.ChartAreas[0].AxisY;
                            axisY.Minimum = Math.Min(0, zMin);
                            axisY.Maximum = Math.Max(_totalMm, zMax);
                            try { axisY.ScaleView.Zoom(zMin, zMax); }
                            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[RowCurveChartHelper.OnPostPaint] {ex.GetType().Name}: {ex.Message}"); }
                        }));
                    }
                    catch (InvalidOperationException) { /* guard 通過後 Handle 已銷毀的競態窗口（ObjectDisposedException 亦繼承自此）*/ }
                }
            }
        }

        protected override void OnPostPaintUnit(object sender, ChartPaintEventArgs e)
        {
            if (e.ChartElement != _chart.ChartAreas[0]) return;
            var g = e.ChartGraphics.Graphics;
            var sz = g.MeasureString("mm", UnitFont);
            g.DrawString("mm", UnitFont, UnitBrush, 2, _chart.Height - sz.Height - 1);
        }

        // ── 方向特定：zoom 計算、標籤反轉 ────────────────────────────────────

        private void GetAdjustedZoom(double canvasTopMm, double canvasBotMm,
                                     out double zoomMin, out double zoomMax)
        {
            double chartHighY = _totalMm - canvasTopMm;
            double chartLowY  = _totalMm - canvasBotMm;
            double span = chartLowY - chartHighY;
            zoomMax = chartHighY + _cachedFTop    * span;
            zoomMin = chartHighY + _cachedFBottom * span;
        }

        private void OnCustomizeLabels(object sender, EventArgs e)
        {
            if (_chart.ChartAreas.Count == 0) return;
            var axisY = _chart.ChartAreas[0].AxisY;
            foreach (CustomLabel label in axisY.CustomLabels)
            {
                double mid = (label.FromPosition + label.ToPosition) / 2.0;
                double inverted = _totalMm - mid;
                label.Text = Math.Round(inverted).ToString("F0");
            }
        }

        // ── 方向特定實作 ─────────────────────────────────────────────────────

        protected override ChartArea BuildChartArea()
        {
            var area = new ChartArea("Main");
            area.Position.Auto   = false;
            area.Position.X      = 0f;
            area.Position.Y      = 0f;
            area.Position.Width  = 100f;
            area.Position.Height = 100f;

            area.InnerPlotPosition.Auto   = false;
            area.InnerPlotPosition.X      = 15f;
            area.InnerPlotPosition.Y      = 2f;
            area.InnerPlotPosition.Width  = 78f;
            area.InnerPlotPosition.Height = 88f;

            area.AxisX.Minimum                  = 0;
            area.AxisX.IsMarginVisible          = false;
            area.AxisX.LabelStyle.Enabled       = true;
            area.AxisX.LabelStyle.Format        = "F1";
            area.AxisX.LabelStyle.Font          = new Font("Segoe UI", 6.5f);
            area.AxisX.IsLabelAutoFit           = false;
            area.AxisX.MajorTickMark.Enabled    = true;
            area.AxisX.MinorTickMark.Enabled    = false;
            area.AxisX.MajorGrid.Enabled        = true;
            area.AxisX.MajorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisX.MajorGrid.Interval       = 0.2;
            area.AxisX.Interval                 = 0.2;
            area.AxisX.ScrollBar.Enabled        = false;
            area.AxisX.ScaleView.Zoomable       = false;

            area.AxisY.Minimum                  = 0;
            area.AxisY.Maximum                  = 100;
            area.AxisY.IsMarginVisible          = false;
            area.AxisY.LabelStyle.Enabled       = true;
            area.AxisY.LabelStyle.Format        = "F0";
            area.AxisY.LabelStyle.Font          = new Font("Segoe UI", 6.5f);
            area.AxisY.IsLabelAutoFit           = false;
            area.AxisY.MajorTickMark.Enabled    = true;
            area.AxisY.MinorTickMark.Enabled    = false;
            area.AxisY.MajorGrid.Enabled        = true;
            area.AxisY.MajorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisY.ScrollBar.Enabled        = false;
            area.AxisY.ScaleView.Zoomable       = false;

            return area;
        }

        protected override void AddAnchorSeries()
        {
            var anchorY = new Series("_anchorY")
            {
                ChartType = SeriesChartType.Point, YAxisType = AxisType.Primary,
                Color = Color.Transparent, MarkerSize = 0, IsVisibleInLegend = false
            };
            anchorY.Points.AddXY(0, 0);
            anchorY.Points.AddXY(0, 100);
            _chart.Series.Add(anchorY);
        }

        protected override void RefreshThresholds()
        {
            if (_chart.ChartAreas.Count == 0) return;

            var area = _chart.ChartAreas[0];

            area.AxisX.StripLines.Clear();
            area.AxisX.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));
            area.AxisX.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));

            double xMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisX.Maximum = xMax;
        }
    }
}
