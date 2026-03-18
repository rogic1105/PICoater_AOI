using System;
using System.Drawing;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// chartMura 封裝：右側 Y 軸、X/Y grid、Mean/Max 曲線、紅色閾值線。
    /// </summary>
    public class MuraChartHelper
    {
        private readonly Chart _chart;

        private double _opsInMm        = 0.01;
        private double _dataMinX       = 0;
        private double _dataMaxX       = 100;
        private float  _errorValueMean = 1.0f;
        private float  _errorValueMax  = 2.0f;

        public MuraChartHelper(Chart chart)
        {
            _chart = chart;
            Build();
        }

        // ── 公開設定 ─────────────────────────────────────────────────────────

        public void SetOps(double opsInUm) => _opsInMm = opsInUm / 1000.0;

        /// <summary>更新閾值（PropertyGrid 修改配方後呼叫）。</summary>
        public void SetThresholds(float errorValueMean, float errorValueMax)
        {
            _errorValueMean = errorValueMean;
            _errorValueMax  = errorValueMax;
            RefreshThresholds();
        }

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
            area.AxisX.Minimum = _dataMinX;
            area.AxisX.Maximum = _dataMaxX;

            bool hasView = !double.IsNaN(viewLeftMm) && !double.IsNaN(viewRightMm) && viewLeftMm < viewRightMm;
            if (hasView)
            {
                try { area.AxisX.ScaleView.Zoom(viewLeftMm, viewRightMm); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[MuraChart] UpdateDataAndView Zoom({viewLeftMm:F2}, {viewRightMm:F2}) failed: {ex.GetType().Name}: {ex.Message}");
                    area.AxisX.ScaleView.ZoomReset();
                }
            }
            else
            {
                area.AxisX.ScaleView.ZoomReset();
            }

            _chart.Series.ResumeUpdates();   // 單次重繪
        }

        // ── Canvas 聯動（X 軸 zoom）────────────────────────────────────────────

        public void UpdateViewRange(double minMm, double maxMm)
        {
            if (_chart.ChartAreas.Count == 0) return;
            if (double.IsNaN(minMm) || double.IsNaN(maxMm) || minMm >= maxMm) return;

            var axisX = _chart.ChartAreas[0].AxisX;
            axisX.Minimum = Math.Min(_dataMinX, minMm);
            axisX.Maximum = Math.Max(_dataMaxX, maxMm);
            try { axisX.ScaleView.Zoom(minMm, maxMm); }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[MuraChart] UpdateViewRange Zoom({minMm:F2}, {maxMm:F2}) failed: {ex.GetType().Name}: {ex.Message}");
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
        }

        private static ChartArea BuildChartArea()
        {
            var area = new ChartArea("Main");
            area.Position.Auto   = false;
            area.Position.X      = 0f;
            area.Position.Y      = 0f;
            area.Position.Width  = 100f;
            area.Position.Height = 100f;

            // X 軸：整數標籤
            area.AxisX.Minimum             = 0;
            area.AxisX.Maximum             = 100;
            area.AxisX.IsMarginVisible     = false;
            area.AxisX.LabelStyle.Format   = "F0";
            area.AxisX.MajorGrid.Enabled   = true;
            area.AxisX.MajorGrid.LineColor = Color.FromArgb(220, 220, 220);
            area.AxisX.ScrollBar.Enabled   = false;
            area.AxisX.ScaleView.Zoomable  = true;

            // AxisY（左）：完全不顯示（軸線/刻度/標籤全隱藏）
            // 仍需設定 scale，否則 grid 和 StripLines 無法渲染
            area.AxisY.Minimum                    = 0;
            area.AxisY.Interval                   = 0.2;
            area.AxisY.LabelStyle.Enabled         = false;
            area.AxisY.LineColor                  = Color.Transparent;
            area.AxisY.MajorTickMark.Enabled      = false;
            area.AxisY.MinorTickMark.Enabled      = false;
            area.AxisY.MajorGrid.Enabled          = true;
            area.AxisY.MajorGrid.LineColor        = Color.FromArgb(220, 220, 220);

            // AxisY2（右）：顯示右側刻度標籤（小數第一位），不畫 grid
            area.AxisY2.Enabled            = AxisEnabled.True;
            area.AxisY2.Minimum            = 0;
            area.AxisY2.Interval           = 0.2;
            area.AxisY2.LabelStyle.Format  = "F1";
            area.AxisY2.MajorGrid.Enabled  = false;

            return area;
        }

        private void AddSeries()
        {
            // AxisY（Primary）需要有 series 才會啟動 scale → grid / StripLines 才會渲染。
            // 加透明 anchor 並跨越完整 Y 範圍，確保啟動時 scale 正確建立。
            var anchorY = new Series("_anchorY")
            {
                ChartType         = SeriesChartType.Point,
                YAxisType         = AxisType.Primary,
                Color             = Color.Transparent,
                MarkerSize        = 0,
                IsVisibleInLegend = false
            };
            anchorY.Points.AddXY(0, 0);
            anchorY.Points.AddXY(0, 2.2);   // 建立 Y 範圍上界
            _chart.Series.Add(anchorY);

            // AxisY2（Secondary）同樣需要 anchor 才會顯示右側標籤。
            var anchorY2 = new Series("_anchorY2")
            {
                ChartType         = SeriesChartType.Point,
                YAxisType         = AxisType.Secondary,
                Color             = Color.Transparent,
                MarkerSize        = 0,
                IsVisibleInLegend = false
            };
            anchorY2.Points.AddXY(0, 0);
            anchorY2.Points.AddXY(0, 2.2);
            _chart.Series.Add(anchorY2);

            // Mean / Max 曲線綁 Primary，與 AxisY grid / StripLines 對齊
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

        // ── 閾值線 ────────────────────────────────────────────────────────────

        private void RefreshThresholds()
        {
            if (_chart.ChartAreas.Count == 0) return;

            var area  = _chart.ChartAreas[0];
            var axisY = area.AxisY;

            axisY.StripLines.Clear();
            axisY.StripLines.Add(MakeStripLine(_errorValueMax,  ChartDashStyle.Solid));   // 紅色實線
            axisY.StripLines.Add(MakeStripLine(_errorValueMean, ChartDashStyle.Dash));    // 紅色虛線

            double yMax = Math.Max(1.0, Math.Max(_errorValueMean, _errorValueMax) * 1.1);
            area.AxisY.Maximum  = yMax;
            area.AxisY2.Maximum = yMax;
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
