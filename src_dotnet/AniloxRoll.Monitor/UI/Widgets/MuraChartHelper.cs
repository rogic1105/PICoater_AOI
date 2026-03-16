using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    public class MuraChartHelper
    {
        private readonly Chart _chart;
        private double _opsInMm = 0.01;

        private double _dataMinX = 0;
        private double _dataMaxX = 100;

        private float _errorValueMean = 1.0f;
        private float _errorValueMax  = 2.0f;

        public MuraChartHelper(Chart chart)
        {
            _chart = chart;
            ConfigureChart();
        }

        public void SetOps(double opsInUm)
        {
            _opsInMm = opsInUm / 1000.0;
        }

        /// <summary>
        /// 更新 ErrorValueMax（紅色實線）與 ErrorValueMean（紅色虛線）參考線。
        /// 在 PropertyGrid 修改配方參數後呼叫。
        /// </summary>
        public void SetThresholds(float errorValueMean, float errorValueMax)
        {
            _errorValueMean = errorValueMean;
            _errorValueMax  = errorValueMax;
            UpdateThresholdLines();
        }

        private void ConfigureChart()
        {
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();
            _chart.Legends.Clear();

            _chart.Margin  = new Padding(0);
            _chart.Padding = new Padding(0);

            ChartArea area = new ChartArea("MainArea");
            area.Position.Auto   = false;
            area.Position.X      = 0;
            area.Position.Y      = 0;
            area.Position.Width  = 100;
            area.Position.Height = 100;

            area.AxisX.ScrollBar.Enabled  = false;
            area.AxisX.ScaleView.Zoomable = true;
            area.AxisX.IsMarginVisible    = false;

            area.AxisX.Minimum = double.NaN;
            area.AxisX.Maximum = double.NaN;

            area.AxisY.Minimum = 0;
            area.AxisY.Maximum = double.NaN;   // auto，由 UpdateThresholdLines 動態調整

            area.AxisX.LabelStyle.Format   = "F1";
            area.AxisY.LabelStyle.Format   = "F2";
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);

            _chart.ChartAreas.Add(area);

            // ── 資料曲線 ──────────────────────────────────────────────────
            var sMean = new Series("Mean")
            {
                ChartType = SeriesChartType.FastLine,
                Color     = Color.Blue
            };
            _chart.Series.Add(sMean);

            var sMax = new Series("Max")
            {
                ChartType = SeriesChartType.FastLine,
                Color     = Color.Orange
            };
            _chart.Series.Add(sMax);

            // ── 閾值參考線 ────────────────────────────────────────────────
            var sErrMax = new Series("ErrorMax")
            {
                ChartType       = SeriesChartType.FastLine,
                Color           = Color.Red,
                BorderWidth     = 1,
                BorderDashStyle = ChartDashStyle.Solid
            };
            _chart.Series.Add(sErrMax);

            var sErrMean = new Series("ErrorMean")
            {
                ChartType       = SeriesChartType.FastLine,
                Color           = Color.Red,
                BorderWidth     = 1,
                BorderDashStyle = ChartDashStyle.Dash
            };
            _chart.Series.Add(sErrMean);

            UpdateThresholdLines();
        }

        public void UpdateData(float[] meanData, float[] maxData, double startPos)
        {
            if (meanData == null || meanData.Length == 0) return;

            _chart.Series.SuspendUpdates();

            Series sMean = _chart.Series["Mean"];
            Series sMax  = _chart.Series["Max"];

            sMean.Points.Clear();
            sMax.Points.Clear();

            int      count       = meanData.Length;
            double[] xValues     = new double[count];
            double[] yMeanValues = new double[count];
            double[] yMaxValues  = new double[count];

            _dataMinX = startPos;
            _dataMaxX = startPos + (count * _opsInMm);

            for (int i = 0; i < count; i++)
            {
                xValues[i]     = startPos + (i * _opsInMm);
                yMeanValues[i] = meanData[i] / 255.0;

                if (maxData != null && i < maxData.Length)
                    yMaxValues[i] = maxData[i] / 255.0;
            }

            sMean.Points.DataBindXY(xValues, yMeanValues);

            if (maxData != null && maxData.Length > 0)
                sMax.Points.DataBindXY(xValues, yMaxValues);

            var area = _chart.ChartAreas[0];
            area.AxisX.Minimum = double.NaN;
            area.AxisX.Maximum = double.NaN;
            area.AxisX.ScaleView.ZoomReset();

            _chart.Series.ResumeUpdates();

            // 資料更新後同步延伸參考線至新的 X 範圍
            UpdateThresholdLines();
        }

        /// <summary>依目前 _dataMinX/_dataMaxX 更新閾值線端點，並調整 Y 軸上限。</summary>
        private void UpdateThresholdLines()
        {
            var sErrMax  = _chart.Series["ErrorMax"];
            var sErrMean = _chart.Series["ErrorMean"];

            sErrMax.Points.Clear();
            sErrMean.Points.Clear();

            sErrMax.Points.AddXY(_dataMinX, _errorValueMax);
            sErrMax.Points.AddXY(_dataMaxX, _errorValueMax);

            sErrMean.Points.AddXY(_dataMinX, _errorValueMean);
            sErrMean.Points.AddXY(_dataMaxX, _errorValueMean);

            // Y 軸上限：資料在 0–1，但閾值可能超過 1，自動擴展
            float threshTop  = Math.Max(_errorValueMean, _errorValueMax);
            double yAxisMax  = Math.Max(1.0, threshTop * 1.1);
            _chart.ChartAreas[0].AxisY.Maximum = yAxisMax;
        }

        public void UpdateViewRange(double minMm, double maxMm)
        {
            if (_chart.ChartAreas.Count == 0) return;

            var axisX = _chart.ChartAreas[0].AxisX;

            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            double newWorldMin = Math.Min(_dataMinX, minMm);
            double newWorldMax = Math.Max(_dataMaxX, maxMm);

            axisX.Minimum = newWorldMin;
            axisX.Maximum = newWorldMax;

            try
            {
                axisX.ScaleView.Zoom(minMm, maxMm);
            }
            catch (Exception)
            {
                // 萬一計算溢位，捕捉例外避免程式崩潰
            }
        }
    }
}
