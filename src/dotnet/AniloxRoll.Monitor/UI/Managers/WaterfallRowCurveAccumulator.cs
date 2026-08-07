using System;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// Owns accumulated waterfall row curves. Neutral data is retained so a display-only
    /// normalization change can rescale the current history without appending another band.
    /// </summary>
    internal sealed class WaterfallRowCurveAccumulator
    {
        private float[] _neutralMean;
        private float[] _neutralMax;
        private float[] _displayMean;
        private float[] _displayMax;
        private int _writePosition;
        private bool _hasData;

        public float[] Mean => _displayMean;

        public float[] Max => _displayMax;

        public int WritePosition => _writePosition;

        public bool HasData => _hasData;

        public void Append(
            float[] neutralMeanBand,
            float[] neutralMaxBand,
            int capacity,
            bool ring,
            float displayFactor)
        {
            if (neutralMeanBand == null || neutralMeanBand.Length == 0) return;

            capacity = Math.Max(1, capacity);
            EnsureCapacity(capacity);

            int bandLength = Math.Min(neutralMeanBand.Length, capacity);
            if (!ring && _writePosition + bandLength > capacity)
            {
                ClearBuffers();
                _writePosition = 0;
                _hasData = false;
            }

            float factor = NormalizeDisplayFactor(displayFactor);
            for (int i = 0; i < bandLength; i++)
            {
                int destination = ring
                    ? (_writePosition + i) % capacity
                    : _writePosition + i;
                if (destination < 0 || destination >= capacity) break;

                float mean = neutralMeanBand[i];
                float maximum = neutralMaxBand != null && i < neutralMaxBand.Length
                    ? neutralMaxBand[i]
                    : 0f;
                _neutralMean[destination] = mean;
                _neutralMax[destination] = maximum;
                _displayMean[destination] = mean * factor;
                _displayMax[destination] = maximum * factor;
            }

            _writePosition = ring
                ? (_writePosition + bandLength) % capacity
                : Math.Min(capacity, _writePosition + bandLength);
            _hasData = true;
        }

        public void Rescale(float displayFactor)
        {
            if (!_hasData || _neutralMean == null) return;

            float factor = NormalizeDisplayFactor(displayFactor);
            for (int i = 0; i < _neutralMean.Length; i++)
            {
                _displayMean[i] = _neutralMean[i] * factor;
                _displayMax[i] = _neutralMax[i] * factor;
            }
        }

        public void Reset()
        {
            _neutralMean = null;
            _neutralMax = null;
            _displayMean = null;
            _displayMax = null;
            _writePosition = 0;
            _hasData = false;
        }

        private void EnsureCapacity(int capacity)
        {
            if (_neutralMean != null && _neutralMean.Length == capacity) return;

            _neutralMean = new float[capacity];
            _neutralMax = new float[capacity];
            _displayMean = new float[capacity];
            _displayMax = new float[capacity];
            _writePosition = 0;
            _hasData = false;
        }

        private void ClearBuffers()
        {
            Array.Clear(_neutralMean, 0, _neutralMean.Length);
            Array.Clear(_neutralMax, 0, _neutralMax.Length);
            Array.Clear(_displayMean, 0, _displayMean.Length);
            Array.Clear(_displayMax, 0, _displayMax.Length);
        }

        private static float NormalizeDisplayFactor(float displayFactor)
            => displayFactor > 0f ? displayFactor : 1f;
    }
}
