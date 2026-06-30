using System;
using System.Drawing;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    public enum MultiClickDistanceMode
    {
        Rectangle,
        Radius
    }

    public sealed class MultiClickDetector
    {
        private readonly int _clickIntervalMs;
        private readonly Size _maxDistance;
        private readonly MultiClickDistanceMode _distanceMode;

        private int _clickCount;
        private int _lastClickTick;
        private Point _lastClickPos;
        private bool _consumed;

        public MultiClickDetector()
            : this(SystemInformation.DoubleClickTime, SystemInformation.DoubleClickSize, MultiClickDistanceMode.Rectangle)
        {
        }

        public MultiClickDetector(int clickIntervalMs, Size maxDistance, MultiClickDistanceMode distanceMode)
        {
            _clickIntervalMs = clickIntervalMs > 0 ? clickIntervalMs : SystemInformation.DoubleClickTime;
            _maxDistance = maxDistance.Width > 0 && maxDistance.Height > 0
                ? maxDistance
                : SystemInformation.DoubleClickSize;
            _distanceMode = distanceMode;
        }

        public int RegisterClick(Point pos)
        {
            int now = Environment.TickCount;
            if (_consumed
                || now - _lastClickTick > _clickIntervalMs
                || !IsWithinDistance(pos))
            {
                _clickCount = 0;
                _consumed = false;
            }

            _lastClickTick = now;
            _lastClickPos = pos;
            return ++_clickCount;
        }

        public void Consume() => _consumed = true;

        public void Reset()
        {
            _clickCount = 0;
            _consumed = false;
        }

        private bool IsWithinDistance(Point pos)
        {
            int dx = pos.X - _lastClickPos.X;
            int dy = pos.Y - _lastClickPos.Y;
            if (_distanceMode == MultiClickDistanceMode.Radius)
            {
                int radius = _maxDistance.Width;
                return dx * dx + dy * dy <= radius * radius;
            }

            return Math.Abs(dx) <= _maxDistance.Width
                && Math.Abs(dy) <= _maxDistance.Height;
        }
    }
}
