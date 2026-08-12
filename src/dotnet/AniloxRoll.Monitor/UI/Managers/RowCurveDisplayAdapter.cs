using System;
using AniloxRoll.Monitor.Core.Data;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// App-layer row chart policy shared by live/review displays.
    /// 2026-07-08 抵銷層歸零定案（詳 $row-chart-coordinates）：方向的資料/視窗/標籤三轉換
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
        public void SetVisibleMetrics(bool showMean, bool showMax)
        {
            _chart.SetVisibleMetrics(showMean, showMax);
            if (FlowName != null)
                Core.Services.FlowTrace.Dvt(
                    $"{FlowName} metrics mean={showMean} max={showMax}");
        }
        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
            => _chart.SetRowPitchFromSpeed(speedMPerMin, lineRateHz);
        public void SetRowPitch(double mmPerRow) => _chart.SetRowPitch(mmPerRow);
        public void Clear() => _chart.Clear();

        private void ApplyDirection()
            => _chart.ZeroAtTop = _getDirection() == VerticalDisplayDirection.TopToBottom;

        // 資料反向已全數退場（2026-07-09 定案）：它是舊 helper (n-1-i) 映射時代的「抵銷層殘餘」——
        // 歸零後 helper 映射已直通，再 Reverse＝多翻一次（瀑布/由下而上內容反轉的回歸根因，
        // 剝層前重建版 A/B+四格符號推導定罪）。adapter 從此真正零轉換。

        private void FlowApply(string kind, int n, double topMm, double botMm, float[] mean = null)
        {
            if (FlowName == null) return;
            var dir = _getDirection();
            int now = Environment.TickCount;
            if (dir == _lastLoggedDir && n == _lastLoggedN && now - _lastFlowMs < 1000) return;
            _lastLoggedDir = dir; _lastLoggedN = n; _lastFlowMs = now;
            // 資料非零「物理範圍」（方向可判量：內容 pad 在哪端一眼見；改A壞B即時可抓）
            string occ = "";
            if (mean != null && mean.Length > 0)
            {
                int i0 = -1, i1 = -1;
                for (int i = 0; i < mean.Length; i++) if (mean[i] != 0) { i0 = i; break; }
                for (int i = mean.Length - 1; i >= 0; i--) if (mean[i] != 0) { i1 = i; break; }
                double p = _chart.RowPitchMm;
                if (i0 < 0) occ = " dataPhys=空";
                else
                {
                    // dataChart＝helper「實際畫上」的值域（量實際非意圖——第二次注入測驗抓到的量測學錯誤：
                    // 原版是 adapter 按規則自算的預期值，映射層壞了照樣印對＝假綠）
                    occ = $" dataPhys {i0 * p:F0}~{i1 * p:F0}mm dataChart {_chart.LastDataOccLo:F0}~{_chart.LastDataOccHi:F0}";
                }
            }
            Core.Services.FlowTrace.Dvt(
                $"{FlowName} {kind} dir={dir} n={n} total={_chart.TotalMm:F0}mm view {topMm:F0}~{botMm:F0}{occ}");
        }

        public void UpdateViewRange(double topMm, double botMm)
        {
            ApplyDirection();
            _chart.UpdateViewRange(topMm, botMm);
            // 必須在同步 Chart.Update 返回後才留痕；舊順序只證明事件進入 adapter，會把
            // 「chart 沒真正畫完」誤判為成功。
            FlowApply("rowView", -1, topMm, botMm);
        }

        public void UpdateViewRangeImmediate(double topMm, double botMm)
        {
            ApplyDirection();
            _chart.UpdateViewRangeImmediate(topMm, botMm);
            FlowApply("rowView", -1, topMm, botMm);
        }

        public void UpdateData(float[] mean, float[] max)
        {
            ApplyDirection();
            _chart.UpdateData(mean, max);
        }

        public void UpdateDataPreservingView(float[] mean, float[] max)
        {
            ApplyDirection();
            _chart.UpdateDataPreservingView(mean, max);
        }

        public void UpdateDataAndViewRange(float[] mean, float[] max, double topMm, double botMm)
        {
            ApplyDirection();
            _chart.UpdateDataAndViewRange(mean, max, topMm, botMm);
            FlowApply("rowChart", mean?.Length ?? 0, topMm, botMm, mean);   // 在更新後記＝dataChart 讀「實際」
        }
    }
}
