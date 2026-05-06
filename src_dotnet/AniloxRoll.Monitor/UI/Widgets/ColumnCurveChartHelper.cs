using System;
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 切向（tangential）Mura 曲線圖：X 軸 = 位置 mm，Y 軸 = normalized value。
    /// 右側 Y2 軸顯示刻度、紅色閾值線、InnerPlotPosition 補償對齊 canvas 水平 viewport。
    /// </summary>
    public class ColumnCurveChartHelper : BaseCurveChartHelper
    {
        private double _opsInMm        = 0.01;
        private double _dataMinX       = 0;
        private double _dataMaxX       = 100;

        // InnerPlotPosition 補償（水平方向）
        private double _cachedFLeft              = 0.0;
        private double _cachedFRight             = 1.0;

        // 上次設定的「邏輯視野」（canvas 的 leftMm/rightMm）
        private double _logicalLeftMm  = double.NaN;
        private double _logicalRightMm = double.NaN;

        public ColumnCurveChartHelper(Chart chart) : base(chart)
        {
            Build();
        }

        // ── 公開設定 ─────────────────────────────────────────────────────────

        public void SetOps(double opsInUm) => _opsInMm = opsInUm / 1000.0;

        // ── 資料更新 ──────────────────────────────────────────────────────────

        public void UpdateData(float[] meanData, float[] maxData, double startPos)
            => UpdateDataAndView(meanData, maxData, startPos, double.NaN, double.NaN);

        /// <summary>
        /// 資料與視野範圍合為單次重繪，避免先顯示全圖再 zoom 造成的閃爍。
        /// viewLeftMm/viewRightMm 為 NaN 時退回全圖。
        /// </summary>
        public void UpdateDataAndView(float[] meanData, float[] maxData, double startPos,
                                      double viewLeftMm, double viewRightMm)
        {
            if (meanData == null || meanData.Length == 0) return;

            int n = meanData.Length;
            _dataMinX = startPos;
            _dataMaxX = startPos + n * _opsInMm;

            var xs    = new double[n];
            var yMean = new double[n];
            var yMax  = new double[n];

            for (int i = 0; i < n; i++)
            {
                xs[i]    = startPos + i * _opsInMm;
                yMean[i] = meanData[i] / 255.0;
                if (maxData != null && i < maxData.Length)
                    yMax[i] = maxData[i] / 255.0;
            }

            _chart.Series.SuspendUpdates();

            _chart.Series["Mean"].Points.Clear();
            _chart.Series["Max"].Points.Clear();
            _chart.Series["Mean"].Points.DataBindXY(xs, yMean);
            if (maxData != null && maxData.Length > 0)
                _chart.Series["Max"].Points.DataBindXY(xs, yMax);

            var area = _chart.ChartAreas[0];

            bool hasView = !double.IsNaN(viewLeftMm) && !double.IsNaN(viewRightMm) && viewLeftMm < viewRightMm;
            if (hasView)
            {
                _logicalLeftMm  = viewLeftMm;
                _logicalRightMm = viewRightMm;
                GetAdjustedZoom(viewLeftMm, viewRightMm, out double zMin, out double zMax);
                area.AxisX.Minimum = Math.Min(_dataMinX, zMin);
                area.AxisX.Maximum = Math.Max(_dataMaxX, zMax);
                try { area.AxisX.ScaleView.Zoom(zMin, zMax); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[MuraChart] UpdateDataAndView Zoom({zMin:F2}, {zMax:F2}) failed: {ex.GetType().Name}: {ex.Message}");
                    area.AxisX.Minimum = _dataMinX;
                    area.AxisX.Maximum = _dataMaxX;
                    area.AxisX.ScaleView.ZoomReset();
                }
            }
            else
            {
                _logicalLeftMm  = double.NaN;
                _logicalRightMm = double.NaN;
                area.AxisX.Minimum = _dataMinX;
                area.AxisX.Maximum = _dataMaxX;
                area.AxisX.ScaleView.ZoomReset();
            }

            _chart.Series.ResumeUpdates();
        }

        // ── Canvas 聯動（X 軸 zoom）────────────────────────────────────────────

        public void UpdateViewRange(double minMm, double maxMm)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            _logicalLeftMm  = minMm;
            _logicalRightMm = maxMm;
            GetAdjustedZoom(minMm, maxMm, out double zMin, out double zMax);
            var axisX = _chart.ChartAreas[0].AxisX;
            axisX.Minimum = Math.Min(_dataMinX, zMin);
            axisX.Maximum = Math.Max(_dataMaxX, zMax);
            try { axisX.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[MuraChart] UpdateViewRange Zoom({zMin:F2}, {zMax:F2}) failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ── InnerPlotPosition 補償 ────────────────────────────────────────────

        protected override void OnPostPaint(object sender, ChartPaintEventArgs e)
        {
            if (_innerPlotPositionFrozen) return;
            if (_chart.ChartAreas.Count == 0) return;

            var inner = _chart.ChartAreas[0].InnerPlotPosition;
            if (inner.Width < 1.0) return;

            const float leftPadding = 0.5f;

            double newFLeft  = (inner.X + leftPadding) / 100.0;
            double newFRight = (inner.X + inner.Width) / 100.0;
            bool   changed   = Math.Abs(newFLeft  - _cachedFLeft)  > 0.001 ||
                               Math.Abs(newFRight - _cachedFRight) > 0.001;

            _cachedFLeft  = newFLeft;
            _cachedFRight = newFRight;
            _innerPlotPositionFrozen = true;

            var area = _chart.ChartAreas[0];
            area.InnerPlotPosition.Auto   = false;
            area.InnerPlotPosition.X      = inner.X + leftPadding;
            area.InnerPlotPosition.Y      = inner.Y;
            area.InnerPlotPosition.Width  = Math.Max(1f, inner.Width - leftPadding);
            area.InnerPlotPosition.Height = inner.Height;

            if (changed && !double.IsNaN(_logicalLeftMm) && _logicalLeftMm < _logicalRightMm)
            {
                double left  = _logicalLeftMm;
                double right = _logicalRightMm;
                _chart.BeginInvoke(new Action(() => ReapplyZoom(left, right)));
            }
        }

        protected override void OnPostPaintUnit(object sender, ChartPaintEventArgs e)
        {
            if (e.ChartElement != _chart.ChartAreas[0]) return;
            var g = e.ChartGraphics.Graphics;
            float chartW = _chart.Width;
            float chartH = _chart.Height;
            var sz = g.MeasureString("mm", UnitFont);
            g.DrawString("mm", UnitFont, UnitBrush, chartW - sz.Width - 2, chartH - sz.Height - 1);
        }

        private void ReapplyZoom(double logicalLeft, double logicalRight)
        {
            GetAdjustedZoom(logicalLeft, logicalRight, out double zMin, out double zMax);
            var axisX = _chart.ChartAreas[0].AxisX;
            axisX.Minimum = Math.Min(_dataMinX, zMin);
            axisX.Maximum = Math.Max(_dataMaxX, zMax);
            try { axisX.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[MuraChart] ReapplyZoom({logicalLeft:F2}, {logicalRight:F2}) failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void GetAdjustedZoom(double leftMm, double rightMm,
                                     out double zoomMin, out double zoomMax)
        {
            double s = rightMm - leftMm;
            zoomMin = leftMm + _cachedFLeft  * s;
            zoomMax = leftMm + _cachedFRight * s;
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

            area.AxisX.Minimum                  = 0;
            area.AxisX.Maximum                  = 100;
            area.AxisX.IsMarginVisible          = false;
            area.AxisX.LabelStyle.Format        = "F0";
            area.AxisX.IsLabelAutoFit           = true;
            area.AxisX.LabelAutoFitMinFontSize  = 6;
            area.AxisX.MajorGrid.Enabled        = true;
            area.AxisX.MajorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisX.MinorGrid.Enabled        = true;
            area.AxisX.MinorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisX.ScrollBar.Enabled        = false;
            area.AxisX.ScaleView.Zoomable       = true;

            area.AxisY.Minimum                    = 0;
            area.AxisY.Maximum                    = 1.0;
            area.AxisY.Interval                   = 0.2;
            area.AxisY.LabelStyle.Enabled         = false;
            area.AxisY.LineColor                  = Color.Transparent;
            area.AxisY.MajorTickMark.Enabled      = false;
            area.AxisY.MinorTickMark.Enabled      = false;
            area.AxisY.MajorGrid.Enabled          = true;
            area.AxisY.MajorGrid.LineColor        = Color.FromArgb(220, 220, 220);

            area.AxisY2.Enabled            = AxisEnabled.True;
            area.AxisY2.Minimum            = 0;
            area.AxisY2.Maximum            = 1.0;
            area.AxisY2.Interval           = 0.2;
            area.AxisY2.LabelStyle.Format  = "F1";
            area.AxisY2.LabelStyle.Angle   = 0;
            area.AxisY2.IsLabelAutoFit     = false;
            area.AxisY2.MajorGrid.Enabled  = false;

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
            anchorY.Points.AddXY(0, 1.0);
            _chart.Series.Add(anchorY);

            var anchorY2 = new Series("_anchorY2")
            {
                ChartType = SeriesChartType.Point, YAxisType = AxisType.Secondary,
                Color = Color.Transparent, MarkerSize = 0, IsVisibleInLegend = false
            };
            anchorY2.Points.AddXY(0, 0);
            anchorY2.Points.AddXY(0, 1.0);
            _chart.Series.Add(anchorY2);
        }

        protected override void RefreshThresholds()
        {
            if (_chart.ChartAreas.Count == 0) return;

            var area  = _chart.ChartAreas[0];
            var axisY = area.AxisY;

            axisY.StripLines.Clear();
            axisY.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));
            axisY.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));

            double yMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisY.Maximum  = yMax;
            area.AxisY2.Maximum = yMax;
        }
    }
}
