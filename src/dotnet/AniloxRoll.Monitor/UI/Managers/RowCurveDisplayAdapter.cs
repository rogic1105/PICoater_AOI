using System;
using AniloxRoll.Monitor.Core.Data;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// App-layer row chart policy shared by live/review displays.
    /// SDK RowCurveChartHelper only draws; this adapter applies product display direction.
    /// </summary>
    public sealed class RowCurveDisplayAdapter
    {
        private readonly RowCurveChartHelper _chart;
        private readonly Func<VerticalDisplayDirection> _getDirection;

        /// <summary>flow 留痕識別名（"LC row"/"RV row"）；null=不留痕。方向/長度變化才記一行，不洗版。</summary>
        public string FlowName { get; set; }
        private VerticalDisplayDirection? _lastLoggedDir;
        private int _lastLoggedN = -1;

        public RowCurveDisplayAdapter(RowCurveChartHelper chart, Func<VerticalDisplayDirection> getDirection)
        {
            _chart = chart ?? throw new ArgumentNullException(nameof(chart));
            _getDirection = getDirection ?? throw new ArgumentNullException(nameof(getDirection));
        }

        private int _lastFlowMs;

        private void FlowApply(string kind, int n, double topIn, double botIn, double topOut, double botOut)
        {
            if (FlowName == null) return;
            var dir = _getDirection();
            int now = Environment.TickCount;
            // 方向/長度變化必記；其餘每秒至多一行（拖曳中的視野跟隨也要有樣本，才能遠端驗方向鏈）
            if (dir == _lastLoggedDir && n == _lastLoggedN && now - _lastFlowMs < 1000) return;
            _lastLoggedDir = dir; _lastLoggedN = n; _lastFlowMs = now;
            Core.Services.FlowTrace.Log(
                $"{FlowName} {kind} dir={dir} n={n} total={_chart.TotalMm:F0}mm view {topIn:F0}~{botIn:F0} → chart {topOut:F0}~{botOut:F0}");
        }

        public double RowPitchMm => _chart.RowPitchMm;

        public void SetThresholds(float mean, float max) => _chart.SetThresholds(mean, max);

        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
            => _chart.SetRowPitchFromSpeed(speedMPerMin, lineRateHz);

        public void SetRowPitch(double mmPerRow) => _chart.SetRowPitch(mmPerRow);

        /// <summary>資料是否已是「顯示順序」（index 0＝畫面最上列）。瀑布 band 緩衝＝true；
        /// 即時/回顧的原始擷取順序曲線＝false。決定反向規則（見 CopyForDisplay）。</summary>
        public bool DataIsDisplayOrdered { get; set; }

        // ── 垂直物理座標定版（2026-07-08）：0 錨定「擷取第一列」。資料映射恆 data[i]→i×pitch（helper），
        //    方向只由一個旗標（AxisY.IsReversed）決定值渲染方向——取代舊「Reverse×鏡射×(n-1-i)」多層轉換。
        //    由上而下＝0 在圖頂；由下而上＝0 在圖底；瀑布（顯示順序資料，0=畫面頂）＝恆 0 在圖頂。
        //    視野值來自畫面同一套物理座標（ImageDisplayView 已轉），此處只歸一排序（chart 要 lo<hi）。 ──

        private void ApplyDirection()
        {
            bool zeroAtTop = DataIsDisplayOrdered
                || _getDirection() == VerticalDisplayDirection.TopToBottom;
            _chart.SetPositionZeroAtTop(zeroAtTop);
        }

        public void UpdateViewRange(double topMm, double botMm)
        {
            double lo = Math.Min(topMm, botMm), hi = Math.Max(topMm, botMm);
            ApplyDirection();
            FlowApply("rowView", -1, topMm, botMm, lo, hi);   // n=-1＝視野-only（拖曳跟隨路）
            _chart.UpdateViewRange(lo, hi);
        }

        public void UpdateData(float[] mean, float[] max)
        {
            ApplyDirection();
            _chart.UpdateData(mean, max);
        }

        public void UpdateDataAndViewRange(float[] mean, float[] max, double topMm, double botMm)
        {
            double lo = Math.Min(topMm, botMm), hi = Math.Max(topMm, botMm);
            ApplyDirection();
            FlowApply("rowChart", mean?.Length ?? 0, topMm, botMm, lo, hi);
            _chart.UpdateDataAndViewRange(mean, max, lo, hi);
        }
    }
}
