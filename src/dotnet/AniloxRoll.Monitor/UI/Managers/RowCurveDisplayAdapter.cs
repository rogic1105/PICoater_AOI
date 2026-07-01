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

        public void UpdateViewRange(double topMm, double botMm)
        {
            TransformViewRange(ref topMm, ref botMm, _chart.TotalMm);
            _chart.UpdateViewRange(topMm, botMm);
        }

        public void UpdateData(float[] mean, float[] max)
            => _chart.UpdateData(CopyForDisplay(mean), CopyForDisplay(max));

        public void UpdateDataAndViewRange(float[] mean, float[] max, double topMm, double botMm)
        {
            var displayMean = CopyForDisplay(mean);
            var displayMax = CopyForDisplay(max);
            double totalMm = displayMean == null ? _chart.TotalMm : displayMean.Length * _chart.RowPitchMm;
            TransformViewRange(ref topMm, ref botMm, totalMm);
            _chart.UpdateDataAndViewRange(displayMean, displayMax, topMm, botMm);
        }

        private float[] CopyForDisplay(float[] data)
        {
            if (data == null) return null;
            var copy = new float[data.Length];
            Array.Copy(data, copy, data.Length);
            if (_getDirection() == VerticalDisplayDirection.BottomToTop)
                Array.Reverse(copy);
            return copy;
        }

        private void TransformViewRange(ref double topMm, ref double botMm, double totalMm)
        {
            if (_getDirection() != VerticalDisplayDirection.BottomToTop) return;
            if (double.IsNaN(topMm) || double.IsNaN(botMm) || topMm >= botMm) return;
            if (totalMm <= 0) return;

            double newTop = totalMm - botMm;
            double newBot = totalMm - topMm;
            topMm = newTop;
            botMm = newBot;
        }
    }
}
