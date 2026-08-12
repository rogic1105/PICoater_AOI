using System;
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 列（axial）Mura 曲線圖：row-wise ridge data，旋轉 90° 顯示。
    /// X 軸（底部）= curve value（0–1 normalized），Y 軸（左側）= 列位置 mm。
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

        /// <summary>直接設 row pitch（mm/影像列）；純剖面用（CursorProfile.OpsYmm）。</summary>
        public void SetRowPitch(double mmPerRow) { if (mmPerRow > 0) _rowPitchMm = mmPerRow; }

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
            LastDataOccLo = LastDataOccHi = double.NaN;   // 實際值域重算

            // 顯示降採樣（[ReviewSync] 實測：全點上 chart 重繪 22~67ms/次 → 拖曳跟隨吃滿 UI）：
            // 上限 ~2000 點（同 overview 慣例）。桶內 mean=平均、max=取大（保峰值）、位置=桶中心；
            // 純顯示瘦身，資料/判定不經此路。
            const int MaxDisplayPoints = 2000;
            int stride = Math.Max(1, (n + MaxDisplayPoints - 1) / MaxDisplayPoints);
            for (int i = 0; i < n; i += stride)
            {
                int end = Math.Min(i + stride, n);
                double sum = 0; double bucketMax = 0; int cnt = 0;
                for (int j = i; j < end; j++)
                {
                    sum += meanData[j]; cnt++;
                    if (maxData != null && j < maxData.Length && maxData[j] > bucketMax) bucketMax = maxData[j];
                }
                int mid = (i + end - 1) / 2;
                double yMm = ZeroAtTop ? (n - 1 - mid) * _rowPitchMm : mid * _rowPitchMm;   // 方向同源映射
                meanSeries.Points.AddXY(sum / cnt / 255.0, yMm);
                if (sum > 0) { if (double.IsNaN(LastDataOccLo) || yMm < LastDataOccLo) LastDataOccLo = yMm;
                               if (double.IsNaN(LastDataOccHi) || yMm > LastDataOccHi) LastDataOccHi = yMm; }
                if (maxData != null)
                    maxSeries.Points.AddXY(bucketMax / 255.0, yMm);
            }

            var area = _chart.ChartAreas[0];
            area.AxisY.Minimum = 0;
            area.AxisY.Maximum = _totalMm;

            _chart.Series.ResumeUpdates();
        }

        /// <summary>
        /// Replaces row-curve values while retaining the current physical Y viewport. This is
        /// used for normalization or threshold changes where image geometry did not change.
        /// </summary>
        public void UpdateDataPreservingView(float[] meanData, float[] maxData)
        {
            if (_chart.ChartAreas.Count == 0) return;
            var axis = _chart.ChartAreas[0].AxisY;
            double minimum = axis.Minimum;
            double maximum = axis.Maximum;
            double viewMinimum = axis.ScaleView.ViewMinimum;
            double viewMaximum = axis.ScaleView.ViewMaximum;

            UpdateData(meanData, maxData);

            axis.Minimum = minimum;
            axis.Maximum = maximum;
            if (!double.IsNaN(viewMinimum) && !double.IsNaN(viewMaximum) &&
                viewMinimum < viewMaximum)
            {
                try { axis.ScaleView.Zoom(viewMinimum, viewMaximum); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning(
                        $"[RowCurveChartHelper.UpdateDataPreservingView] {ex.GetType().Name}: {ex.Message}");
                }
            }
            _chart.Invalidate();
        }

        /// <summary>
        /// 更新 Y 軸視野範圍（對應 canvas 垂直 viewport），單位為 mm。
        /// </summary>
        public void UpdateDataAndViewRange(float[] meanData, float[] maxData,
            double canvasTopMm, double canvasBotMm)
        {
            if (meanData == null || meanData.Length == 0) return;
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(canvasTopMm) || double.IsNaN(canvasBotMm) || canvasTopMm == canvasBotMm)
            {
                UpdateData(meanData, maxData);
                return;
            }

            int n = meanData.Length;
            _totalMm = n * _rowPitchMm;

            _logicalTopMm = canvasTopMm;
            _logicalBotMm = canvasBotMm;
            GetAdjustedZoom(canvasTopMm, canvasBotMm, out double zMin, out double zMax);

            var axisY = _chart.ChartAreas[0].AxisY;
            ApplyViewportBounds(axisY, zMin, zMax);
            try { axisY.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[RowCurveChartHelper.UpdateDataAndViewRange] {ex.GetType().Name}: {ex.Message}"); }

            _chart.Series.SuspendUpdates();

            var meanSeries = _chart.Series["Mean"];
            var maxSeries  = _chart.Series["Max"];
            meanSeries.Points.Clear();
            maxSeries.Points.Clear();
            LastDataOccLo = LastDataOccHi = double.NaN;   // 實際值域重算

            const int MaxDisplayPoints = 2000;
            int stride = Math.Max(1, (n + MaxDisplayPoints - 1) / MaxDisplayPoints);
            for (int i = 0; i < n; i += stride)
            {
                int end = Math.Min(i + stride, n);
                double sum = 0; double bucketMax = 0; int cnt = 0;
                for (int j = i; j < end; j++)
                {
                    sum += meanData[j]; cnt++;
                    if (maxData != null && j < maxData.Length && maxData[j] > bucketMax) bucketMax = maxData[j];
                }
                int mid = (i + end - 1) / 2;
                double yMm = ZeroAtTop ? (n - 1 - mid) * _rowPitchMm : mid * _rowPitchMm;   // 方向同源映射
                meanSeries.Points.AddXY(sum / cnt / 255.0, yMm);
                if (sum > 0) { if (double.IsNaN(LastDataOccLo) || yMm < LastDataOccLo) LastDataOccLo = yMm;
                               if (double.IsNaN(LastDataOccHi) || yMm > LastDataOccHi) LastDataOccHi = yMm; }
                if (maxData != null)
                    maxSeries.Points.AddXY(bucketMax / 255.0, yMm);
            }

            _chart.Series.ResumeUpdates();
            _chart.Update();
        }

        public void UpdateViewRange(double canvasTopMm, double canvasBotMm)
            => UpdateViewRangeCore(canvasTopMm, canvasBotMm, invalidateBeforeUpdate: false);

        /// <summary>
        /// Updates the visible physical range and forces it to paint immediately.
        /// Use when a prepared range must appear before later image or data work invalidates the chart.
        /// </summary>
        public void UpdateViewRangeImmediate(double canvasTopMm, double canvasBotMm)
        {
            bool plotWasFrozen = _innerPlotPositionFrozen;
            UpdateViewRangeCore(canvasTopMm, canvasBotMm, invalidateBeforeUpdate: true);
            // The first real paint discovers MSChart's InnerPlotPosition. Reapply in the same
            // UI action so replacement row data cannot reveal a second, compensated range.
            if (!plotWasFrozen && _innerPlotPositionFrozen)
                UpdateViewRangeCore(canvasTopMm, canvasBotMm, invalidateBeforeUpdate: true);
        }

        private void UpdateViewRangeCore(double canvasTopMm, double canvasBotMm, bool invalidateBeforeUpdate)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(canvasTopMm) || double.IsNaN(canvasBotMm) || canvasTopMm == canvasBotMm) return;

            _logicalTopMm = canvasTopMm;
            _logicalBotMm = canvasBotMm;

            GetAdjustedZoom(canvasTopMm, canvasBotMm, out double zMin, out double zMax);
            var axisY = _chart.ChartAreas[0].AxisY;
            // 同 ColumnCurveChartHelper：Min/Max 變了才設（避免拖曳跟隨時每次整張重排版）。
            // While synchronized to the image, the image viewport is the coordinate SSoT.
            // A fast curve replacement may change total data length, but it must not move the
            // coordinate scale before the debounced image selection is committed.
            ApplyViewportBounds(axisY, zMin, zMax);
            try { axisY.ScaleView.Zoom(zMin, zMax); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[RowCurveChartHelper.UpdateViewRange] {ex.GetType().Name}: {ex.Message}"); }
            if (invalidateBeforeUpdate)
                _chart.Invalidate();
            _chart.Update();   // 同 ColumnCurveChartHelper：防拖曳時 WM_PAINT 飢餓（chart 放開滑鼠才動）
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
                            ApplyViewportBounds(axisY, zMin, zMax);
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
            // canvasTop/Bot＝畫面上/下緣的「輸入空間」值（邊界身份保留，非排序值）。
            // 由上而下（ZeroAtTop）：內部值=total−輸入（原調校）；由下而上：直通（天然軸向，零轉換）。
            double chartHighY = ZeroAtTop ? _totalMm - canvasTopMm : canvasTopMm;
            double chartLowY  = ZeroAtTop ? _totalMm - canvasBotMm : canvasBotMm;
            double span = chartLowY - chartHighY;
            zoomMax = chartHighY + _cachedFTop    * span;
            zoomMin = chartHighY + _cachedFBottom * span;
        }

        private static void ApplyViewportBounds(Axis axis, double first, double second)
        {
            double min = Math.Min(first, second);
            double max = Math.Max(first, second);
            if (axis.Minimum != min) axis.Minimum = min;
            if (axis.Maximum != max) axis.Maximum = max;
        }

        /// <summary>方向旗標（唯一決策點，資料映射/視窗換算/標籤三者同源——排版/渲染零接觸）：
        /// true（原調校行為）＝0 在圖表「頂端」（由上而下）：資料 (n-1-i)、視窗 total−、標籤反轉；
        /// false＝0 在「底端」（由下而上）＝軸的天然方向：資料/視窗/標籤**全直通、零轉換**。
        /// 2026-07-08 抵銷層歸零重構：外層（adapter）不得再有任何鏡射。</summary>
        public bool ZeroAtTop { get; set; } = true;

        /// <summary>最近一次資料更新「實際畫上 chart」的非零值域（量實際非意圖——供 M-state dataChart 對數；
        /// 若與方向規則推算的預期關係不符＝映射層壞）。NaN=無資料。</summary>
        public double LastDataOccLo { get; private set; } = double.NaN;
        public double LastDataOccHi { get; private set; } = double.NaN;

        private void OnCustomizeLabels(object sender, EventArgs e)
        {
            if (_chart.ChartAreas.Count == 0) return;
            var axisY = _chart.ChartAreas[0].AxisY;
            foreach (CustomLabel label in axisY.CustomLabels)
            {
                double mid = (label.FromPosition + label.ToPosition) / 2.0;
                double shown = ZeroAtTop ? _totalMm - mid : mid;
                label.Text = Math.Round(shown).ToString("F0");
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
            // 左邊界留給位置軸(AxisY, mm)的 label：回顧整卷拼接時 _totalMm 可達 4~5 位數，
            // 控制項僅 117px 寬，原 15% (~18px) 塞不下 → 數字溢出左緣被切。放寬到 18% (~21px) 容得下。
            area.InnerPlotPosition.X      = 18f;
            area.InnerPlotPosition.Y      = 2f;
            area.InnerPlotPosition.Width  = 75f;   // X+Width=93，右緣留白同舊
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
            if (_showThresholds)
            {
                if (_showMaxMetric)
                    area.AxisX.StripLines.Add(MakeStripLine(_errorValueMax, ChartDashStyle.Solid));
                if (_showMeanMetric)
                    area.AxisX.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));
            }

            double activeThreshold = _showMeanMetric && _showMaxMetric
                ? Math.Max(_errorValueMean, _errorValueMax)
                : _showMeanMetric ? _errorValueMean : _errorValueMax;
            double xMax = Math.Max(1.0, activeThreshold * 1.1);
            area.AxisX.Maximum = xMax;
        }
    }
}
