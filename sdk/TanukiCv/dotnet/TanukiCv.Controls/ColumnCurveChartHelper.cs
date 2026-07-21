using System;
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 欄（tangential）Mura 曲線圖：X 軸 = 位置 mm，Y 軸 = normalized value。
    /// 右側 Y2 軸顯示刻度、紅色閾值線、InnerPlotPosition 補償對齊 canvas 水平 viewport。
    /// </summary>
    public class ColumnCurveChartHelper : BaseCurveChartHelper
    {
        private const int AbsoluteMaxDisplayPoints = 2000;
        private const int MinDisplayPoints = 128;

        private double _opsInMm        = 0.01;
        private double _dataMinX       = 0;
        private double _dataMaxX       = 100;

        // InnerPlotPosition 補償（水平方向）
        private double _cachedFLeft              = 0.0;
        private double _cachedFRight             = 1.0;

        // 上次設定的「邏輯視野」（canvas 的 leftMm/rightMm）
        private double _logicalLeftMm  = double.NaN;
        private double _logicalRightMm = double.NaN;
        private double[] _displayXs = new double[0];
        private double[] _displayMean = new double[0];
        private double[] _displayMax = new double[0];

        /// <summary>最近一次更新實際送進圖表的顯示點數。</summary>
        public int DisplayPointCount => _displayXs.Length;

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

            // 依圖表寬度降採樣：每兩個水平像素一個桶；Mean 取桶平均、Max 取桶最大，
            // 因此不漏窄凸波。這只是顯示瘦身，完整資料與判定不經此路。
            int maxDisplayPoints = GetDisplayPointLimit();
            int stride = Math.Max(1, (n + maxDisplayPoints - 1) / maxDisplayPoints);
            int displayCount = (n + stride - 1) / stride;
            EnsureDisplayBuffers(displayCount);
            if (stride > 1)
            {
                for (int b = 0; b < displayCount; b++)
                {
                    int i0 = b * stride, i1 = Math.Min(i0 + stride, n);
                    double sum = 0, bmax = 0; int cnt = 0;
                    for (int j = i0; j < i1; j++)
                    {
                        sum += meanData[j]; cnt++;
                        if (maxData != null && j < maxData.Length && maxData[j] > bmax) bmax = maxData[j];
                    }
                    int mid = (i0 + i1 - 1) / 2;
                    _displayXs[b]   = startPos + mid * _opsInMm;
                    _displayMean[b] = sum / cnt / 255.0;
                    _displayMax[b]  = bmax / 255.0;
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    _displayXs[i]   = startPos + i * _opsInMm;
                    _displayMean[i] = meanData[i] / 255.0;
                    _displayMax[i]  = maxData != null && i < maxData.Length
                        ? maxData[i] / 255.0
                        : 0.0;
                }
            }

            _chart.Series.SuspendUpdates();

            BindOrUpdatePoints(_chart.Series["Mean"], _displayXs, _displayMean);
            if (maxData != null && maxData.Length > 0)
                BindOrUpdatePoints(_chart.Series["Max"], _displayXs, _displayMax);
            else
                _chart.Series["Max"].Points.Clear();

            var area = _chart.ChartAreas[0];

            bool hasView = !double.IsNaN(viewLeftMm) && !double.IsNaN(viewRightMm) && viewLeftMm < viewRightMm;
            if (hasView)
            {
                _logicalLeftMm  = viewLeftMm;
                _logicalRightMm = viewRightMm;
                GetAdjustedZoom(viewLeftMm, viewRightMm, out double zMin, out double zMax);
                ApplyViewportBounds(area.AxisX, zMin, zMax);
                ApplyXAxisTickInterval(area.AxisX, zMin, zMax);
                try { area.AxisX.ScaleView.Zoom(zMin, zMax); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[MuraChart] UpdateDataAndView Zoom({zMin:F2}, {zMax:F2}) failed: {ex.GetType().Name}: {ex.Message}");
                    ApplyAxisBounds(area.AxisX, _dataMinX, _dataMaxX);
                    ApplyXAxisTickInterval(area.AxisX, _dataMinX, _dataMaxX);
                    area.AxisX.ScaleView.ZoomReset();
                }
            }
            else
            {
                _logicalLeftMm  = double.NaN;
                _logicalRightMm = double.NaN;
                ApplyAxisBounds(area.AxisX, _dataMinX, _dataMaxX);
                ApplyXAxisTickInterval(area.AxisX, _dataMinX, _dataMaxX);
                area.AxisX.ScaleView.ZoomReset();
            }

            _chart.Series.ResumeUpdates();
            _chart.Invalidate();
        }

        private void EnsureDisplayBuffers(int count)
        {
            if (_displayXs.Length == count) return;
            _displayXs = new double[count];
            _displayMean = new double[count];
            _displayMax = new double[count];
        }

        private int GetDisplayPointLimit()
        {
            int width = _chart.ClientSize.Width;
            if (width <= 0) return AbsoluteMaxDisplayPoints;
            return Math.Min(AbsoluteMaxDisplayPoints, Math.Max(MinDisplayPoints, width / 2));
        }

        private static void BindOrUpdatePoints(Series series, double[] xs, double[] ys)
        {
            DataPointCollection points = series.Points;
            points.SuspendUpdates();
            try
            {
                if (points.Count != xs.Length)
                {
                    points.Clear();
                    points.DataBindXY(xs, ys);
                    return;
                }

                for (int i = 0; i < xs.Length; i++)
                {
                    DataPoint point = points[i];
                    point.XValue = xs[i];
                    point.YValues[0] = ys[i];
                }
            }
            finally
            {
                points.ResumeUpdates();
            }
        }

        // ── Canvas 聯動（X 軸 zoom）────────────────────────────────────────────

        public void UpdateViewRange(double minMm, double maxMm)
            => UpdateViewRangeCore(minMm, maxMm, invalidateBeforeUpdate: false);

        /// <summary>
        /// Updates the visible physical range and forces it to paint immediately.
        /// Use when a prepared range must appear before later image or data work invalidates the chart.
        /// </summary>
        public void UpdateViewRangeImmediate(double minMm, double maxMm)
        {
            bool plotWasFrozen = _innerPlotPositionFrozen;
            UpdateViewRangeCore(minMm, maxMm, invalidateBeforeUpdate: true);
            // The first real paint discovers MSChart's InnerPlotPosition. Reapply in the same
            // UI action so the prepared range already includes plot-area compensation instead
            // of changing again when image/curve data is presented.
            if (!plotWasFrozen && _innerPlotPositionFrozen)
                UpdateViewRangeCore(minMm, maxMm, invalidateBeforeUpdate: true);
        }

        private void UpdateViewRangeCore(double minMm, double maxMm, bool invalidateBeforeUpdate)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            _logicalLeftMm  = minMm;
            _logicalRightMm = maxMm;
            GetAdjustedZoom(minMm, maxMm, out double zMin, out double zMax);
            var axisX = _chart.ChartAreas[0].AxisX;
            // ⚠ 拖曳即時跟隨效能：設 Minimum/Maximum 會觸發 MSChart 整張重排版（比 ScaleView.Zoom 貴一級）。
            //   30fps 連續跟隨時 Min/Max 其實不變（資料沒換）→ 只在真的變了才設，跟隨只走便宜的 Zoom。
            // While synchronized to the image, the image viewport is the coordinate SSoT.
            // Curve extent must not resize the axis before the debounced image is presented.
            ApplyViewportBounds(axisX, zMin, zMax);
            ApplyXAxisTickInterval(axisX, zMin, zMax);
            try { axisX.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[MuraChart] UpdateViewRange Zoom({zMin:F2}, {zMax:F2}) failed: {ex.GetType().Name}: {ex.Message}");
            }
            // 拖曳即時跟隨：滑鼠訊息佔滿佇列時 WM_PAINT（最低優先級）會飢餓 → chart 放開滑鼠才動。
            // Update() 同步畫掉 pending paint → 真即時（鐵則：互動中不可抑制曲線連動）。
            if (invalidateBeforeUpdate)
                _chart.Invalidate();
            _chart.Update();
        }

        // ── InnerPlotPosition 補償 ────────────────────────────────────────────

        protected override void OnPostPaint(object sender, ChartPaintEventArgs e)
        {
            if (_innerPlotPositionFrozen) return;
            if (_chart.ChartAreas.Count == 0) return;

            var inner = _chart.ChartAreas[0].InnerPlotPosition;
            if (_chart.ClientSize.Width < 100 || _chart.ClientSize.Height < 40) return;
            if (inner.Width < 20.0 || inner.Height < 20.0) return;

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
                // 守 IsHandleCreated：StitchMode 切換期 chart 可能正在 re-layout，handle 短暫無效
                if (_chart.IsHandleCreated && !_chart.IsDisposed)
                {
                    try { _chart.BeginInvoke(new Action(() => ReapplyZoom(left, right))); }
                    catch (InvalidOperationException) { /* guard 通過後 Handle 已銷毀的競態窗口（ObjectDisposedException 亦繼承自此）*/ }
                }
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
            // ⚠ 拖曳即時跟隨效能：設 Minimum/Maximum 會觸發 MSChart 整張重排版（比 ScaleView.Zoom 貴一級）。
            //   30fps 連續跟隨時 Min/Max 其實不變（資料沒換）→ 只在真的變了才設，跟隨只走便宜的 Zoom。
            ApplyViewportBounds(axisX, zMin, zMax);
            ApplyXAxisTickInterval(axisX, zMin, zMax);
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

        private void ApplyAxisBounds(Axis axis, double min, double max)
        {
            if (axis == null) return;
            if (axis.Minimum != min) axis.Minimum = min;
            if (axis.Maximum != max) axis.Maximum = max;
        }

        private void ApplyViewportBounds(Axis axis, double first, double second)
        {
            ApplyAxisBounds(axis, Math.Min(first, second), Math.Max(first, second));
        }

        private void ApplyXAxisTickInterval(Axis axis, double min, double max)
        {
            if (axis == null) return;
            double span = max - min;
            double dataSpan = _dataMaxX - _dataMinX;
            if (!double.IsNaN(dataSpan) && !double.IsInfinity(dataSpan))
                span = Math.Max(span, dataSpan);
            if (double.IsNaN(span) || double.IsInfinity(span) || span <= 0)
                span = 1.0;

            double interval = NiceInterval(span / 5.0);
            double minorInterval = interval / 2.0;
            string labelFormat = interval < 0.1 ? "F2" : interval < 1.0 ? "F1" : "F0";

            if (axis.Interval != interval) axis.Interval = interval;
            if (axis.LabelStyle.Interval != interval) axis.LabelStyle.Interval = interval;
            if (axis.MajorGrid.Interval != interval) axis.MajorGrid.Interval = interval;
            if (axis.MajorTickMark.Interval != interval) axis.MajorTickMark.Interval = interval;
            if (axis.MinorGrid.Interval != minorInterval) axis.MinorGrid.Interval = minorInterval;
            if (!axis.LabelStyle.Enabled) axis.LabelStyle.Enabled = true;
            if (axis.LabelStyle.Format != labelFormat) axis.LabelStyle.Format = labelFormat;
            if (!axis.MajorTickMark.Enabled) axis.MajorTickMark.Enabled = true;
            if (axis.IsLabelAutoFit) axis.IsLabelAutoFit = false;
        }

        private static double NiceInterval(double raw)
        {
            if (double.IsNaN(raw) || double.IsInfinity(raw) || raw <= 0) return 1.0;

            double exponent = Math.Floor(Math.Log10(raw));
            double baseValue = Math.Pow(10.0, exponent);
            double fraction = raw / baseValue;

            double niceFraction;
            if (fraction <= 1.0) niceFraction = 1.0;
            else if (fraction <= 2.0) niceFraction = 2.0;
            else if (fraction <= 5.0) niceFraction = 5.0;
            else niceFraction = 10.0;

            return niceFraction * baseValue;
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
            area.InnerPlotPosition.X      = 1.5f;
            area.InnerPlotPosition.Y      = 5f;
            area.InnerPlotPosition.Width  = 92f;
            area.InnerPlotPosition.Height = 72f;

            area.AxisX.Minimum                  = 0;
            area.AxisX.Maximum                  = 100;
            area.AxisX.IsMarginVisible          = false;
            area.AxisX.LabelStyle.Format        = "F0";
            // 固定 label 樣式（比照 RowCurveChartHelper）：不用 IsLabelAutoFit —— auto-fit 會依控制項字體
            // 決定角度/排列，當 ProportionalScaler 把 Font 縮放到不一致時，窄的那個會退化成「逐字元豎排」
            // （[1][2][3] 一字一行）。固定字體 + Angle 0 → 兩個欄圖一致的乾淨橫排，不受縮放影響。
            area.AxisX.LabelStyle.Font          = new Font("Segoe UI", 9f);   // 固定大小（不隨視窗縮放，但兩圖一致、不逐字豎排）；要調大小改這裡
            area.AxisX.LabelStyle.Angle         = 0;
            area.AxisX.IsLabelAutoFit           = false;
            area.AxisX.Interval                 = 20;
            area.AxisX.LabelStyle.Interval      = 20;
            area.AxisX.MajorTickMark.Enabled    = true;
            area.AxisX.MajorTickMark.Interval   = 20;
            area.AxisX.MajorGrid.Enabled        = true;
            area.AxisX.MajorGrid.Interval       = 20;
            area.AxisX.MajorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisX.MinorGrid.Enabled        = true;
            area.AxisX.MinorGrid.Interval       = 10;
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
            if (_showThresholds)
            {
                axisY.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));
                axisY.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));
            }

            double yMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisY.Maximum  = yMax;
            area.AxisY2.Maximum = yMax;
        }
    }
}
