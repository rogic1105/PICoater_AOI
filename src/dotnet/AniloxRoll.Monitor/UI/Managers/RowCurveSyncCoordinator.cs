namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// Keeps row-curve data updates atomic with the current image view range.
    /// </summary>
    public sealed class RowCurveSyncCoordinator
    {
        private readonly RowCurveDisplayAdapter _display;
        private bool _rangeSuspended;
        private bool _hasViewRange;
        private double _topMm;
        private double _botMm;
        private float[] _pendingMean;
        private float[] _pendingMax;
        private float[] _currentMean;
        private float[] _currentMax;
        private bool _currentRequiresViewRange;

        public RowCurveSyncCoordinator(RowCurveDisplayAdapter display)
        {
            _display = display;
        }

        public double RowPitchMm => _display?.RowPitchMm ?? 0;

        public void SetThresholds(float mean, float max) => _display?.SetThresholds(mean, max);

        public void SetRowPitchFromSpeed(double speedMPerMin, double lineRateHz)
            => _display?.SetRowPitchFromSpeed(speedMPerMin, lineRateHz);

        public void SetRowPitch(double mmPerRow) => _display?.SetRowPitch(mmPerRow);

        public void SuspendUntilNextData()
        {
            _rangeSuspended = true;
        }

        public void Resume()
        {
            _rangeSuspended = false;
            FlushPending();
        }

        public void ClearPending()
        {
            _pendingMean = null;
            _pendingMax = null;
            _currentMean = null;
            _currentMax = null;
            _rangeSuspended = false;
            _display?.Clear();
        }

        public void SetViewRange(double topMm, double botMm)
        {
            if (_display == null) return;
            // 物理座標（2026-07-08 定版）：由下而上時上緣值 > 下緣值＝合法；只擋 NaN/零跨距，排序由 adapter 歸一
            if (double.IsNaN(topMm) || double.IsNaN(botMm) || topMm == botMm) return;

            _topMm = topMm;
            _botMm = botMm;
            _hasViewRange = true;

            if (!_rangeSuspended)
                _display.UpdateViewRange(topMm, botMm);

            FlushPending();
        }

        public bool TryApplyCurrentViewRange()
        {
            if (_display == null) return true;
            if (!_hasViewRange) return true;
            _display.UpdateViewRange(_topMm, _botMm);
            return true;
        }

        public bool UpdateData(float[] mean, float[] max, bool requireViewRange)
        {
            if (_display == null) return true;

            if (requireViewRange && !_hasViewRange)
            {
                _pendingMean = mean;
                _pendingMax = max;
                _rangeSuspended = true;
                return true;
            }

            _pendingMean = null;
            _pendingMax = null;
            _rangeSuspended = false;
            _currentMean = mean;
            _currentMax = max;
            _currentRequiresViewRange = requireViewRange;

            if (requireViewRange)
            {
                _display.UpdateDataAndViewRange(mean, max, _topMm, _botMm);
                return true;
            }

            _display.UpdateData(mean, max);
            return false;
        }

        public void RefreshDirection()
        {
            if (_display == null) return;
            if (_currentMean != null)
            {
                if (_currentRequiresViewRange && _hasViewRange)
                    _display.UpdateDataAndViewRange(_currentMean, _currentMax, _topMm, _botMm);
                else
                    _display.UpdateData(_currentMean, _currentMax);
                return;
            }

            if (_hasViewRange)
                _display.UpdateViewRange(_topMm, _botMm);
        }

        private void FlushPending()
        {
            if (_display == null) return;
            if (_pendingMean == null || !_hasViewRange) return;

            var mean = _pendingMean;
            var max = _pendingMax;
            _pendingMean = null;
            _pendingMax = null;
            _rangeSuspended = false;
            _currentMean = mean;
            _currentMax = max;
            _currentRequiresViewRange = true;
            _display.UpdateDataAndViewRange(mean, max, _topMm, _botMm);
        }
    }
}
