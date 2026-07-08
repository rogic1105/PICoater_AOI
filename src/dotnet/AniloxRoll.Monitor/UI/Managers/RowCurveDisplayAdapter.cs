using System;
using AniloxRoll.Monitor.Core.Data;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// App-layer row chart policy shared by live/review displays.
    /// 2026-07-08 抵銷層歸零定案（詳 /row-chart-coordinates）：方向的資料/視窗/標籤三轉換
    /// **全部同源在 helper.ZeroAtTop**（由上而下=原調校那套；由下而上=軸天然方向、零轉換）。
    /// adapter 零鏡射、零排序（邊界值保留「邊的身份」直通 helper）——外層再出現任何
    /// total−/Reverse/排序＝重蹈「層層包反轉」事故。
    /// 唯一真轉換：瀑布×由下而上的資料反向（顯示順序 buffer → 邏輯空間，非抵銷層）。
    /// </summary>
    public sealed class RowCurveDisplayAdapter
    {
        private readonly RowCurveChartHelper _chart;
        private readonly Func<VerticalDisplayDirection> _getDirection;

        /// <summary>flow 留痕識別名（"LC row"/"RV row"）；null=不留痕。</summary>
        public string FlowName { get; set; }
        /// <summary>資料是否已是「顯示順序」（瀑布 band 緩衝）；原始擷取順序（即時/回顧）=false。</summary>
        public bool DataIsDisplayOrdered { get; set; }

        private VerticalDisplayDirection? _lastLoggedDir;
        private int _lastLoggedN = -1;
        private int _lastFlowMs;

        public RowCurveDisplayAdapter(RowCurveChartHelper chart, Func<VerticalDisplayDirection> getDirection)
        {
            _chart = chart ?? throw new ArgumentNullException(nameof(chart));
            _getDirection = getDirection ?? throw new ArgumentNullException(nameof(getDirection));
        }

        public double RowPitchMm => _chart.RowPitchMm;
        public void SetThresholds(float mean, float max) => _chart.SetThresholds(mean, max);
        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
            => _chart.SetRowPitchFromSpeed(speedMPerMin, lineRateHz);
        public void SetRowPitch(double mmPerRow) => _chart.SetRowPitch(mmPerRow);

        private void ApplyDirection()
            => _chart.ZeroAtTop = _getDirection() == VerticalDisplayDirection.TopToBottom;

        /// <summary>唯一真轉換：瀑布（顯示順序）×由下而上 → 反向成邏輯順序。其餘直通。</summary>
        private bool NeedDataReverse()
            => DataIsDisplayOrdered && _getDirection() == VerticalDisplayDirection.BottomToTop;

        private float[] CopyForDisplay(float[] data)
        {
            if (data == null || !NeedDataReverse()) return data;
            var copy = new float[data.Length];
            Array.Copy(data, copy, data.Length);
            Array.Reverse(copy);
            return copy;
        }

        private void FlowApply(string kind, int n, double topMm, double botMm)
        {
            if (FlowName == null) return;
            var dir = _getDirection();
            int now = Environment.TickCount;
            if (dir == _lastLoggedDir && n == _lastLoggedN && now - _lastFlowMs < 1000) return;
            _lastLoggedDir = dir; _lastLoggedN = n; _lastFlowMs = now;
            Core.Services.FlowTrace.Log(
                $"{FlowName} {kind} dir={dir} n={n} total={_chart.TotalMm:F0}mm view {topMm:F0}~{botMm:F0}");
        }

        public void UpdateViewRange(double topMm, double botMm)
        {
            ApplyDirection();
            FlowApply("rowView", -1, topMm, botMm);
            _chart.UpdateViewRange(topMm, botMm);
        }

        public void UpdateData(float[] mean, float[] max)
        {
            ApplyDirection();
            _chart.UpdateData(CopyForDisplay(mean), CopyForDisplay(max));
        }

        public void UpdateDataAndViewRange(float[] mean, float[] max, double topMm, double botMm)
        {
            ApplyDirection();
            FlowApply("rowChart", mean?.Length ?? 0, topMm, botMm);
            _chart.UpdateDataAndViewRange(CopyForDisplay(mean), CopyForDisplay(max), topMm, botMm);
        }
    }
}
