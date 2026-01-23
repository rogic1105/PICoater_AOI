using System;
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    public class MuraChartHelper
    {
        private readonly Chart _chart;

        public MuraChartHelper(Chart chart)
        {
            _chart = chart;
            ConfigureChart();
        }

        public void SetOps(double opsInUm)
        {
            // 在原始測試中，我們暫時忽略這個參數
        }

        private void ConfigureChart()
        {
            // 1. 清空既有的設定 (這是必須的，否則會跟 Designer 的重複)
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();
            _chart.Legends.Clear();

            // 2. 加入一個最陽春的 ChartArea
            ChartArea area = new ChartArea("MainArea");
            _chart.ChartAreas.Add(area);

            // 3. 建立 Series 1: Mean (使用 FastLine 效能較好)
            Series sMean = new Series("Mean");
            sMean.ChartType = SeriesChartType.FastLine;
            sMean.Color = Color.Blue;
            _chart.Series.Add(sMean);

            // 4. 建立 Series 2: Max
            Series sMax = new Series("Max");
            sMax.ChartType = SeriesChartType.FastLine;
            sMax.Color = Color.Orange;
            _chart.Series.Add(sMax);
        }

        public void UpdateData(float[] meanData, float[] maxData)
        {
            if (meanData == null || meanData.Length == 0) return;

            // 暫停更新以提升效能
            _chart.Series.SuspendUpdates();

            Series sMean = _chart.Series["Mean"];
            Series sMax = _chart.Series["Max"];

            sMean.Points.Clear();
            sMax.Points.Clear();

            // 5. [最原始的綁定] 直接使用 DataBindY
            // 這會讓 X 軸顯示為索引 (0, 1, 2...)，Y 軸顯示原始數值
            // 不做任何運算，看看數據到底是多大
            sMean.Points.DataBindY(meanData);

            if (maxData != null && maxData.Length > 0)
            {
                sMax.Points.DataBindY(maxData);
            }

            // 恢復更新
            _chart.Series.ResumeUpdates();

            // 6. [除錯訊息] 將第一筆數據印到 Console，確認數據不是全 0 或全 NaN
            Console.WriteLine($"[Chart Debug] Mean[0]={meanData[0]}, Max[0]={(maxData != null && maxData.Length > 0 ? maxData[0].ToString() : "null")}");
        }
    }
}