using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 法向（axial）Mura 曲線圖：row-wise ridge data，旋轉 90° 顯示。
    /// Y 軸 = 法向位置 mm（上→下），X 軸 = curve value（0–1 normalized）。
    /// 對齊 panelMainDisplay 的垂直方向。
    /// 法向 mm = row_index × rowPitchMm，其中 rowPitchMm = (speed_m_per_min / 60 × 1000) / lineRateHz。
    /// </summary>
    public class RowMuraChartHelper
    {
        private readonly Chart _chart;
        private float  _errorValueMean = 1.0f;
        private float  _errorValueMax  = 2.0f;
        private double _rowPitchMm     = 0.01;  // mm per row

        public RowMuraChartHelper(Chart chart)
        {
            _chart = chart;
            Build();
        }

        /// <summary>
        /// 設定法向每行間距（mm）。rowPitchMm = (speedMPerMin / 60 * 1000) / lineRateHz。
        /// </summary>
        public void SetRowPitch(double rowPitchMm)
        {
            if (rowPitchMm > 0)
                _rowPitchMm = rowPitchMm;
        }

        /// <summary>從 A輪速度 (m/min) 及取樣頻率 (Hz) 計算 row pitch (mm)。</summary>
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
        /// Y 座標以 mm 為單位（row_index × _rowPitchMm）。
        /// </summary>
        public void UpdateData(float[] meanData, float[] maxData)
        {
            if (meanData == null || meanData.Length == 0) return;

            int n = meanData.Length;

            _chart.Series.SuspendUpdates();

            var meanSeries = _chart.Series["Mean"];
            var maxSeries  = _chart.Series["Max"];
            meanSeries.Points.Clear();
            maxSeries.Points.Clear();

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
            area.AxisY.Maximum = n * _rowPitchMm;

            _chart.Series.ResumeUpdates();
        }

        /// <summary>
        /// 更新 Y 軸視野範圍（對應 canvas 垂直 viewport），單位為 mm。
        /// </summary>
        public void UpdateViewRange(double minMm, double maxMm)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            var axisY = _chart.ChartAreas[0].AxisY;
            try { axisY.ScaleView.Zoom(minMm, maxMm); }
            catch { /* ignore */ }
        }

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
            _chart.PostPaint += OnPostPaintUnit;
        }

        private static readonly Font  _unitFont  = new Font("Segoe UI", 7f);
        private static readonly Brush _unitBrush = new SolidBrush(Color.Gray);

        private void OnPostPaintUnit(object sender, ChartPaintEventArgs e)
        {
            if (e.ChartElement != _chart.ChartAreas[0]) return;
            var g = e.ChartGraphics.Graphics;
            // Y 軸底端繪製 "mm"
            float chartH = _chart.Height;
            var sz = g.MeasureString("mm", _unitFont);
            g.DrawString("mm", _unitFont, _unitBrush, 2, chartH - sz.Height - 1);
        }

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
            area.InnerPlotPosition.X      = 15f;   // 左邊界（AxisY mm labels 空間）
            area.InnerPlotPosition.Y      = 2f;
            area.InnerPlotPosition.Width  = 81f;   // 右邊界留 4%
            area.InnerPlotPosition.Height = 84f;   // 下邊界留 14%（AxisX labels + mm 文字）

            // X 軸（水平，底部）= curve value（0–1+）
            area.AxisX.Minimum                  = 0;
            area.AxisX.IsMarginVisible          = false;
            area.AxisX.LabelStyle.Enabled       = true;
            area.AxisX.LabelStyle.Format        = "F1";
            area.AxisX.MajorTickMark.Enabled    = true;
            area.AxisX.MinorTickMark.Enabled    = false;
            area.AxisX.MajorGrid.Enabled        = true;
            area.AxisX.MajorGrid.LineColor      = Color.FromArgb(220, 220, 220);
            area.AxisX.MajorGrid.Interval       = 0.2;
            area.AxisX.Interval                 = 0.2;
            area.AxisX.ScrollBar.Enabled        = false;
            area.AxisX.ScaleView.Zoomable       = false;

            // Y 軸（垂直，左側）= 法向位置 mm，反轉使 row 0 在上方
            area.AxisY.Minimum                  = 0;
            area.AxisY.IsReversed               = true;
            area.AxisY.IsMarginVisible          = false;
            area.AxisY.LabelStyle.Enabled       = true;
            area.AxisY.LabelStyle.Format        = "F0";
            area.AxisY.MajorTickMark.Enabled    = true;
            area.AxisY.MinorTickMark.Enabled    = false;
            area.AxisY.MajorGrid.Enabled        = false;
            area.AxisY.ScrollBar.Enabled        = false;
            area.AxisY.ScaleView.Zoomable       = false;

            return area;
        }

        private void AddSeries()
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

        private void RefreshThresholds()
        {
            if (_chart.ChartAreas.Count == 0) return;

            var area  = _chart.ChartAreas[0];
            var axisX = area.AxisX;

            axisX.StripLines.Clear();
            axisX.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));
            axisX.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));

            double xMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisX.Maximum  = xMax;
            area.AxisX2.Maximum = xMax;
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
