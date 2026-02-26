using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    public class MuraChartHelper
    {
        private readonly Chart _chart;
        private double _opsInMm = 0.01;

        // [新增] 記錄數據的邊界，用來判斷是否需要擴展軸
        private double _dataMinX = 0;
        private double _dataMaxX = 100;

        public MuraChartHelper(Chart chart)
        {
            _chart = chart;
            ConfigureChart();
        }

        public void SetOps(double opsInUm)
        {
            _opsInMm = opsInUm / 1000.0;
        }

        private void ConfigureChart()
        {
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();
            _chart.Legends.Clear();

            _chart.Margin = new Padding(0);
            _chart.Padding = new Padding(0);

            ChartArea area = new ChartArea("MainArea");
            area.Position.Auto = false;
            area.Position.X = 0;
            area.Position.Y = 0;
            area.Position.Width = 100;
            area.Position.Height = 100;

            // 隱藏拉霸
            area.AxisX.ScrollBar.Enabled = false;
            area.AxisX.ScaleView.Zoomable = true;
            area.AxisX.IsMarginVisible = false;

            // [關鍵] 初始時不設限，讓它自動依照數據調整 (UpdateData 會重設)
            area.AxisX.Minimum = Double.NaN;
            area.AxisX.Maximum = Double.NaN;

            area.AxisY.Minimum = 0;
            area.AxisY.Maximum = 1;

            area.AxisX.LabelStyle.Format = "F1";
            area.AxisY.LabelStyle.Format = "F2";
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);
            area.AxisY.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);

            _chart.ChartAreas.Add(area);

            Series sMean = new Series("Mean");
            sMean.ChartType = SeriesChartType.FastLine;
            sMean.Color = Color.Blue;
            _chart.Series.Add(sMean);

            Series sMax = new Series("Max");
            sMax.ChartType = SeriesChartType.FastLine;
            sMax.Color = Color.Orange;
            _chart.Series.Add(sMax);
        }

        public void UpdateData(float[] meanData, float[] maxData, double startPos)
        {
            if (meanData == null || meanData.Length == 0) return;

            _chart.Series.SuspendUpdates();

            Series sMean = _chart.Series["Mean"];
            Series sMax = _chart.Series["Max"];

            sMean.Points.Clear();
            sMax.Points.Clear();

            int count = meanData.Length;
            double[] xValues = new double[count];
            double[] yMeanValues = new double[count];
            double[] yMaxValues = new double[count];

            // [新增] 更新數據邊界紀錄
            _dataMinX = startPos;
            _dataMaxX = startPos + (count * _opsInMm);

            for (int i = 0; i < count; i++)
            {
                xValues[i] = startPos + (i * _opsInMm);
                yMeanValues[i] = meanData[i] / 255.0;

                if (maxData != null && i < maxData.Length)
                {
                    yMaxValues[i] = maxData[i] / 255.0;
                }
            }

            sMean.Points.DataBindXY(xValues, yMeanValues);

            if (maxData != null && maxData.Length > 0)
            {
                sMax.Points.DataBindXY(xValues, yMaxValues);
            }

            // [關鍵] 數據更新時，先重置軸限制為 Auto，讓它貼齊數據
            // 後續 UpdateViewRange 會再根據視野撐大它
            var area = _chart.ChartAreas[0];
            area.AxisX.Minimum = double.NaN;
            area.AxisX.Maximum = double.NaN;
            area.AxisX.ScaleView.ZoomReset();

            _chart.Series.ResumeUpdates();
        }

        public void UpdateViewRange(double minMm, double maxMm)
        {
            if (_chart.ChartAreas.Count == 0) return;

            var axisX = _chart.ChartAreas[0].AxisX;

            // 防呆
            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            // [關鍵修正] 計算包含「數據」與「視野」的聯集範圍
            // 這樣當視野 (minMm) 小於數據起點 (_dataMinX) 時，AxisX.Minimum 會被撐大，允許顯示空白
            double newWorldMin = Math.Min(_dataMinX, minMm);
            double newWorldMax = Math.Max(_dataMaxX, maxMm);

            // 設定軸的物理極限 (這樣 Zoom 才有空間可以運作)
            // 注意：我們只在需要擴展時才設定，避免不必要的抖動，但為了安全起見，直接設定最保險
            axisX.Minimum = newWorldMin;
            axisX.Maximum = newWorldMax;

            // 最後再執行縮放
            try
            {
                axisX.ScaleView.Zoom(minMm, maxMm);
            }
            catch (Exception)
            {
                // 萬一計算還是溢位，至少捕捉例外不讓程式崩潰
                // 通常是因為 minMm/maxMm 數值過大 (例如 1.7E+308)
            }
        }
    }
}