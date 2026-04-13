using System;
using System.Drawing;
using System.Windows.Forms;

namespace AniloxRoll.Monitor.UI.Widgets
{
    internal sealed class MultiClickDetector
    {
        private const int ClickIntervalMs = 300;
        private int _clickCount;
        private int _lastClickTick;
        private Point _lastClickPos;
        private bool _consumed;

        public int RegisterClick(Point pos)
        {
            int now = Environment.TickCount;
            int dx = pos.X - _lastClickPos.X;
            int dy = pos.Y - _lastClickPos.Y;
            int distSq = dx * dx + dy * dy;
            int threshold = SystemInformation.DoubleClickSize.Width;

            if (_consumed
                || now - _lastClickTick > ClickIntervalMs
                || distSq > threshold * threshold)
            {
                _clickCount = 0;
                _consumed = false;
            }

            _lastClickTick = now;
            _lastClickPos = pos;
            return ++_clickCount;
        }

        /// <summary>標記本輪點擊已消費，下次 RegisterClick 從 1 重新開始。</summary>
        public void Consume() => _consumed = true;

        public void Reset() { _clickCount = 0; _consumed = false; }
    }
}
