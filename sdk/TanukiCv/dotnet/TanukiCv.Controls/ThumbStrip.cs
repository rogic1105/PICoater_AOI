using System;
using System.Drawing;
using System.Windows.Forms;

namespace TanukiCv.Controls
{
    /// <summary>
    /// 多相機縮圖條（純 CPU、吃 8bpp 灰階 bytes，0 依賴 MIL / GPU / app）：在傳入的每個 panel 疊一個雙緩衝
    /// <see cref="ThumbView"/>，相機每幀（可能背景執行緒）只 <see cref="PushFrame"/> 存快照 + 標 dirty，
    /// 內部 33ms timer 在 UI 執行緒一次批量建圖（CPU stride 降採樣到 <see cref="ThumbMaxW"/>）+ 換圖。
    ///
    /// **不閃的關鍵**：縮圖不每幀每台 BeginInvoke（8 台灌爆 UI → 閃），改 timer 批量；建圖在 UI 執行緒、
    /// 葉子 ThumbView 自繪雙緩衝（不對容器開雙緩衝——那會反讓子控制項閃）。
    ///
    /// app（LiveCameraManager 經 MultiCamLiveView）與範例（MilGrabberPbForm）共用此唯一來源縮圖路徑。
    /// </summary>
    public sealed class ThumbStrip : IDisposable
    {
        private sealed class Frame { public readonly byte[] Bytes; public readonly int W, H; public Frame(byte[] b, int w, int h) { Bytes = b; W = w; H = h; } }

        private readonly ThumbView[] _thumbs;
        private readonly Frame[] _latest;
        private readonly bool[] _dirty;
        private readonly int _count;
        private System.Windows.Forms.Timer _timer;
        private volatile bool _disposed;

        /// <summary>縮圖被點 → 要求切換選中相機（0-based 索引，與 panels 傳入順序一致）。</summary>
        public event Action<int> SelectRequested;

        /// <summary>建縮圖時 stride 降採樣到此寬以下（小圖、便宜）。預設 320。</summary>
        public int ThumbMaxW { get; set; } = 320;

        /// <summary>上下翻轉（線掃由下往上拍 / GPU 輸出 bottom-up 時）。</summary>
        public bool FlipVertical { get; set; }

        /// <summary>在每個傳入 panel 疊一個 ThumbView（Dock=Fill, BringToFront）。panel 既有的狀態列 / 邊框由 caller 自理。</summary>
        public ThumbStrip(Panel[] panels)
        {
            if (panels == null) throw new ArgumentNullException(nameof(panels));
            _count = panels.Length;
            _thumbs = new ThumbView[_count];
            _latest = new Frame[_count];
            _dirty = new bool[_count];

            for (int i = 0; i < _count; i++)
            {
                if (panels[i] == null) continue;
                var tv = new ThumbView { Dock = DockStyle.Fill };
                int idx = i;
                tv.MouseClick += (s, e) => SelectRequested?.Invoke(idx);
                panels[i].Controls.Add(tv);
                tv.BringToFront();
                _thumbs[i] = tv;
            }

            _timer = new System.Windows.Forms.Timer { Interval = 33 };
            _timer.Tick += (s, e) => Flush();
            _timer.Start();
        }

        /// <summary>相機每幀（可能背景執行緒）：存灰階快照 + 標 dirty。**不在此建圖/換圖**（避免每幀每台 BeginInvoke → 閃）。</summary>
        public void PushFrame(int idx, byte[] gray, int w, int h)
        {
            if (_disposed || idx < 0 || idx >= _count || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            _latest[idx] = new Frame(copy, w, h);
            _dirty[idx] = true;
        }

        /// <summary>清掉某台縮圖（釋放時 caller 呼叫）。</summary>
        public void Clear(int idx)
        {
            if (idx < 0 || idx >= _count) return;
            _latest[idx] = null; _dirty[idx] = false;
            ThumbView tv = _thumbs[idx];
            if (tv != null && !tv.IsDisposed) tv.SetImage(null);
        }

        // ── UI timer（33ms）：批量建圖 + 換圖（僅 dirty 才動）──
        private void Flush()
        {
            if (_disposed) return;
            for (int i = 0; i < _count; i++)
            {
                if (!_dirty[i]) continue;
                _dirty[i] = false;
                Frame f = _latest[i];
                ThumbView tv = _thumbs[i];
                if (f == null || tv == null || tv.IsDisposed) continue;
                try { tv.SetImage(BuildThumb(f)); } catch { /* 換圖失敗忽略 */ }
            }
        }

        /// <summary>把灰階幀 stride 降採樣到 ThumbMaxW 以下（CPU、零 GPU）再組灰階 bitmap。</summary>
        private Bitmap BuildThumb(Frame f)
        {
            int cap = ThumbMaxW > 0 ? ThumbMaxW : 320;
            int ds = Math.Max(1, (f.W + cap - 1) / cap);
            if (ds == 1) return GrayBitmap.From(f.Bytes, f.W, f.H, FlipVertical);
            int dw = Math.Max(1, f.W / ds), dh = Math.Max(1, f.H / ds);
            var small = new byte[dw * dh];
            byte[] src = f.Bytes; int w = f.W;
            for (int y = 0; y < dh; y++)
            {
                int srow = (y * ds) * w, drow = y * dw;
                for (int x = 0; x < dw; x++) small[drow + x] = src[srow + x * ds];
            }
            return GrayBitmap.From(small, dw, dh, FlipVertical);
        }

        public void Dispose()
        {
            _disposed = true;
            if (_timer != null) { _timer.Stop(); _timer.Dispose(); _timer = null; }
            for (int i = 0; i < _thumbs.Length; i++)
            {
                var tv = _thumbs[i];
                if (tv != null) { tv.Parent?.Controls.Remove(tv); tv.Dispose(); _thumbs[i] = null; }
                _latest[i] = null;
            }
        }
    }
}
