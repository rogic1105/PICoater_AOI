using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 法向（axial）Mura 曲線圖：row-wise ridge data，旋轉 90° 顯示。
    /// X 軸（底部）= curve value（0–1 normalized），Y 軸（左側）= 法向位置 mm。
    /// Y 軸標籤反轉：視覺上 0 在上、max 在下（透過 Customize 事件修改標籤文字）。
    /// InnerPlotPosition 補償機制對齊 canvas 垂直 viewport。
    /// </summary>
    public class RowMuraChartHelper
    {
        private readonly Chart _chart;
        private float  _errorValueMean = 1.0f;
        private float  _errorValueMax  = 2.0f;
        private double _rowPitchMm     = 0.01;  // mm per row
        private double _totalMm        = 100;   // 目前資料總高度 mm

        public double RowPitchMm => _rowPitchMm;
        public double TotalMm   => _totalMm;

        // InnerPlotPosition 補償：plot 區域佔控制項高度的比例（PostPaint 凍結）。
        // _cachedFTop = plot 上邊 / 控制項高度，_cachedFBottom = plot 下邊 / 控制項高度。
        private double _cachedFTop              = 0.0;
        private double _cachedFBottom           = 1.0;
        private bool   _innerPlotPositionFrozen = false;

        // 上次設定的「邏輯視野」，供 PostPaint 凍結後透過 BeginInvoke 補正 zoom。
        private double _logicalTopMm  = double.NaN;
        private double _logicalBotMm  = double.NaN;

        public RowMuraChartHelper(Chart chart)
        {
            _chart = chart;
            Build();
            _chart.Customize += OnCustomizeLabels;
        }

        public void SetRowPitch(double rowPitchMm)
        {
            if (rowPitchMm > 0)
                _rowPitchMm = rowPitchMm;
        }

        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
        {
            if (speedMPerMin > 0 && lineRateHz > 0)
                _rowPitchMm = (speedMPerMin / 60.0 * 1000.0) / lineRateHz;
        }

        public void SetThresholds(float errorValueMean, float errorValueMax)
        {
            _errorValueMean = errorValueMean;
            _errorValueMax  = errorValueMax;
            RefreshThresholds();
        }

        /// <summary>
        /// 更新 row-wise 曲線資料。meanData[i] / maxData[i] 為 row i 的值（0–255 raw）。
        /// row i → Y = i * rowPitchMm：不反轉。chart 上方=大 Y=後段 row。
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

            // row i → Y = i * rowPitchMm（不反轉）
            for (int i = 0; i < n; i++)
            {
                double yMm = i * _rowPitchMm;
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
        /// canvasTopMm / canvasBotMm = canvas 上下邊緣對應的 displayed pixel mm。
        /// 內部自動反轉至 chart 座標 + InnerPlotPosition 補償。
        /// 允許負值（canvas 顯示超出影像範圍時）。
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
            catch { /* ignore */ }
        }

        /// <summary>
        /// canvas Y mm → chart Y zoom 範圍（含反轉 + InnerPlotPosition 補償）。
        /// data 不反轉（row i → Y=i*rowPitch），IsReversed=true 使 Y=0 在上方。
        /// 反轉：chartHighY = totalMm - canvasTopMm → chart 視覺上方（小 Y with IsReversed），
        ///        chartLowY  = totalMm - canvasBotMm → chart 視覺下方（大 Y with IsReversed）。
        /// 再套用 InnerPlotPosition 比例補償（與 MuraChartHelper.GetAdjustedZoom 同理）。
        /// </summary>
        private void GetAdjustedZoom(double canvasTopMm, double canvasBotMm,
                                     out double zoomMin, out double zoomMax)
        {
            double chartHighY = _totalMm - canvasTopMm;  // chart 視覺上方
            double chartLowY  = _totalMm - canvasBotMm;  // chart 視覺下方

            // InnerPlotPosition 補償：控制項上邊→chartHighY，下邊→chartLowY
            double span = chartLowY - chartHighY;  // 負值（highY > lowY）
            zoomMax = chartHighY + _cachedFTop    * span;
            zoomMin = chartHighY + _cachedFBottom * span;
        }

        /// <summary>
        /// Customize 事件：反轉 Y 軸標籤，使視覺上 0 在上（chart 頂部）、max 在下。
        /// Chart 控件在 Customize 事件中將自動標籤轉為 CustomLabels，可直接修改 Text。
        /// </summary>
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

        // ── 建立 ──────────────────────────────────────────────────────────────

        private void Build()
        {
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();
            _chart.Legends.Clear();
            _chart.Margin  = new Padding(0);
            _chart.Padding = new Padding(0);

            _chart.ChartAreas.Add(BuildChartArea());
            AddSeries();
            RefreshThresholds();
            _chart.PostPaint += OnPostPaint;
            _chart.PostPaint += OnPostPaintUnit;
        }

        // ── InnerPlotPosition 補償 ────────────────────────────────────────────

        /// <summary>
        /// 首次有效渲染後：量測 InnerPlotPosition、凍結版面，
        /// 若快取值改變則透過 BeginInvoke 補正當下 zoom。
        /// </summary>
        private void OnPostPaint(object sender, ChartPaintEventArgs e)
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

            // 凍結版面
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
                _chart.BeginInvoke(new Action(() =>
                {
                    GetAdjustedZoom(top, bot, out double zMin, out double zMax);
                    var axisY = _chart.ChartAreas[0].AxisY;
                    axisY.Minimum = Math.Min(0, zMin);
                    axisY.Maximum = Math.Max(_totalMm, zMax);
                    try { axisY.ScaleView.Zoom(zMin, zMax); }
                    catch { /* ignore */ }
                }));
            }
        }

        private static readonly Font  _unitFont  = new Font("Segoe UI", 7f);
        private static readonly Brush _unitBrush = new SolidBrush(Color.Gray);

        private void OnPostPaintUnit(object sender, ChartPaintEventArgs e)
        {
            if (e.ChartElement != _chart.ChartAreas[0]) return;
            var g = e.ChartGraphics.Graphics;
            // Y 軸頂端（左上角）繪製 "mm"（Y=0 顯示在上方）
            g.DrawString("mm", _unitFont, _unitBrush, 2, 1);
        }

        // ── ChartArea ─────────────────────────────────────────────────────────

        private static ChartArea BuildChartArea()
        {
            var area = new ChartArea("Main");
            area.Position.Auto   = false;
            area.Position.X      = 0f;
            area.Position.Y      = 0f;
            area.Position.Width  = 100f;
            area.Position.Height = 100f;

            // InnerPlotPosition：左邊留給 AxisY mm 標籤，下邊留給 AxisX curve value 標籤
            area.InnerPlotPosition.Auto   = false;
            area.InnerPlotPosition.X      = 15f;   // 左邊界
            area.InnerPlotPosition.Y      = 2f;
            area.InnerPlotPosition.Width  = 78f;   // 右邊界留 7%
            area.InnerPlotPosition.Height = 80f;   // 下邊界留 18%

            // X 軸（底部）= curve value（0–1+）
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

            // Y 軸（左側）= 法向位置 mm（0 在下，max 在上）
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

        private void AddSeries()
        {
            // Anchor series：確保 Y 軸在無資料時有 scale
            var anchorY = new Series("_anchorY")
            {
                ChartType         = SeriesChartType.Point,
                YAxisType         = AxisType.Primary,
                Color             = Color.Transparent,
                MarkerSize        = 0,
                IsVisibleInLegend = false
            };
            anchorY.Points.AddXY(0, 0);
            anchorY.Points.AddXY(0, 100);
            _chart.Series.Add(anchorY);

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

        private void RefreshThresholds()
        {
            if (_chart.ChartAreas.Count == 0) return;

            var area = _chart.ChartAreas[0];

            area.AxisX.StripLines.Clear();
            area.AxisX.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));
            area.AxisX.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));

            double xMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisX.Maximum = xMax;
        }

        private static StripLine MakeStripLine(double offset, ChartDashStyle dash) => new StripLine
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
