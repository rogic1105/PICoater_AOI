// AniloxRoll.Monitor\Forms\Helpers\MuraChartHelper.cs

using System; // Math
using System.Drawing;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    public class MuraChartHelper
    {
        private readonly Chart _chart;
        private int _dataWidth = 0; // 記錄數據長度

        public MuraChartHelper(Chart chart)
        {
            _chart = chart;
        }

        public void Initialize(int imageWidth)
        {
            _dataWidth = imageWidth;
            _chart.Series.Clear();
            _chart.ChartAreas.Clear();

            ChartArea area = new ChartArea("MainArea");

            // X 軸設定
            area.AxisX.Title = "Pixel Position (X)";
            area.AxisX.Minimum = 0;
            area.AxisX.Maximum = imageWidth;

            // [關鍵修正] 徹底解決 "捲軸大小應在 5 到 20" 的崩潰問題
            // 1. 先啟用 ScrollBar (為了設定屬性)
            area.AxisX.ScrollBar.Enabled = true;
            // 2. 設定固定大小，避開自動計算的 Bug
            area.AxisX.ScrollBar.Size = 10;
            // 3. 移除所有按鈕，只保留捲軸本身 (減少計算複雜度)
            area.AxisX.ScrollBar.ButtonStyle = ScrollBarButtonStyles.None;
            // 4. 設定位置在圖表外部 (避免遮擋)
            area.AxisX.ScrollBar.IsPositionedInside = false;
            // 5. [最重要] 最後再把它設為不顯示 (Enabled=false 有時會觸發 Bug，用 Enabled=true 但畫在看不見的地方比較穩)
            // 但為了安全，我們這裡將 Enabled 設為 false，並配合上面的 Size=10
            area.AxisX.ScrollBar.Enabled = false;

            // 其他樣式
            area.AxisX.MajorGrid.LineColor = Color.LightGray;
            area.AxisX.LabelStyle.Format = "0";

            // Y 軸設定
            area.AxisY.Title = "Response Value";
            area.AxisY.MajorGrid.LineColor = Color.LightGray;
            area.AxisY.IsStartedFromZero = false;

            _chart.ChartAreas.Add(area);

            Series series = new Series("MuraCurve");
            series.ChartType = SeriesChartType.FastLine;
            series.Color = Color.Blue;
            series.BorderWidth = 1;
            series.IsVisibleInLegend = false;
            _chart.Series.Add(series);
        }
        public void UpdateData(float[] data)
        {
            if (data == null || data.Length == 0) return;

            _chart.Series.SuspendUpdates();
            Series s = _chart.Series["MuraCurve"];
            s.Points.Clear();
            s.Points.DataBindY(data);
            _chart.Series.ResumeUpdates();

            // [注意] 這裡不要呼叫 RecalculateAxesScale，因為我們會由 SyncView 來控制 X 軸
        }

        // [新增] 同步視野方法
        public void SyncView(float startPixel, float viewLength)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (_dataWidth == 0) return;

            var area = _chart.ChartAreas[0];

            try
            {
                // [修正] 防呆：防止 viewLength 過小導致 MSChart 除以零或溢位
                if (viewLength < 1.0f) viewLength = 1.0f;
                if (double.IsNaN(startPixel) || double.IsNaN(viewLength)) return;

                // [修正] 限制 startPixel 和 viewLength 不要超出合理範圍太誇張
                // 雖然我們要同步黑色區域，但如果數值大到 int.MaxValue，GDI+ 會畫不出來

                // 設定視圖位置
                // 注意：MSChart 的 Zoom(start, length) 第二個參數是"長度"
                area.AxisX.ScaleView.Zoom(startPixel, viewLength);

                // [關鍵] 這裡不要呼叫 RecalculateAxesScale，會導致閃爍或重置
            }
            catch
            {
                // 吞掉 MSChart 偶發的計算錯誤
            }
        }


    }
}