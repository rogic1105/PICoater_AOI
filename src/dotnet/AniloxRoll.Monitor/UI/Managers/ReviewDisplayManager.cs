using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 回顧主畫面接 sdk <see cref="LiveDisplayView"/>（與監控同源，絞殺榕收官 #13 Stage1）。
    /// 執行時在 camReviewMain 與 camReview1~7 位置疊 Panel 宿主 LiveDisplayView（同 Parent/Bounds/Anchor）。
    /// 4c 已轉正＝唯一顯示路徑（過渡旗標已刪）；舊控制項實體待 4d 連 Designer 一併清除。
    /// 回顧直接繼承：動態 LOD（70 張合圖顯示成本 ~180ms→~1ms 路線）、縮圖↔主畫面雙向連動、mm overlay、雙三擊、游標剖面。
    /// </summary>
    internal sealed class ReviewDisplayManager : IDisposable
    {
        private readonly Control _mainUnder;
        private readonly Control[] _thumbsUnder;
        private Panel _mainHost;
        private Panel[] _thumbHosts;
        private LiveDisplayView _view;
        private bool _disposed;

        /// <summary>視野可見範圍（mm）pass-through（View 為 lazy，外部訂這裡）：left,right,top,bot →
        /// 回顧曲線圖 zoom 連動（拖曳中也即時，鐵則：不可為效能抑制）。</summary>
        public event Action<double, double, double, double> ViewRangeMmChanged;

        /// <summary>游標狀態（mm 位置/範圍/亮度/倍率）pass-through → 上層更新狀態列 lblPixelInfo。</summary>
        public event Action<LiveDisplayView.CursorStatus> CursorStatusChanged;

        /// <summary>sdk 顯示元件（接事件 / 進階用；未建立前 null）。</summary>
        public LiveDisplayView View => _view;

        public ReviewDisplayManager(Control mainUnder, Control[] thumbsUnder)
        {
            _mainUnder = mainUnder ?? throw new ArgumentNullException(nameof(mainUnder));
            _thumbsUnder = thumbsUnder ?? new Control[0];
        }

        /// <summary>lazy 建宿主 + LiveDisplayView（首次餵圖時呼叫；screenMmPerPx 屆時已由 SystemInfo 設定）。</summary>
        public void EnsureCreated(double screenMmPerPx)
        {
            if (_view != null || _disposed) return;

            _mainHost = OverlayPanel(_mainUnder);
            _thumbHosts = new Panel[_thumbsUnder.Length];
            for (int i = 0; i < _thumbsUnder.Length; i++)
                _thumbHosts[i] = _thumbsUnder[i] != null ? OverlayPanel(_thumbsUnder[i]) : null;

            _view = new LiveDisplayView(_mainHost, _thumbHosts, screenMmPerPx);
            _view.ThumbSelectedColor = Color.Orange;   // 與監控同款；選取視覺唯一來源 = sdk ThumbView
            _view.MergeAll = true;                     // 缺台黑占位（與影像/曲線分界一致）
            _view.FlipVertical = false;                // 回顧影像載入時已翻轉（StitchCamera baked-in），勿再翻
            _view.EnableLod(GrayResizeCpu.Resize);     // 回顧白賺 LOD；CPU provider＝無 GPU 機也跑
            _view.ViewRangeMmChanged += (l, r, tp, bt) => ViewRangeMmChanged?.Invoke(l, r, tp, bt);
            _view.CursorStatusChanged += s => CursorStatusChanged?.Invoke(s);
        }

        private static Panel OverlayPanel(Control under)
        {
            var p = new Panel
            {
                Bounds = under.Bounds,
                Anchor = under.Anchor,
                Dock = under.Dock,
                BackColor = Color.Black,
            };
            under.Parent.Controls.Add(p);
            p.BringToFront();
            // 跟隨底下控制項（ProportionalScaler 縮放 / 佈局變更 → 同步 Bounds，不跑版）
            under.SizeChanged += (s, e) => p.Bounds = under.Bounds;
            under.LocationChanged += (s, e) => p.Bounds = under.Bounds;
            return p;
        }

        /// <summary>
        /// 餵一組回顧灰階幀（RSC 解碼段已轉好的不可變 bytes；null 槽=間空黑占位）+ CFG 座標。
        /// 純推幀零轉換 → UI 執行緒無負擔、與 Bitmap 生命週期零 race。
        /// </summary>
        public void PushFrames(byte[][] gray, int[] w, int[] h, double[] opsUm, double[] posMm,
            bool mergeMode, double screenMmPerPx, int feedScale, double rowPitchMm)
        {
            if (_disposed || gray == null) return;
            EnsureCreated(screenMmPerPx);
            _view.SetLayout(posMm, opsUm, Math.Max(1, feedScale), rowPitchMm); // feedScale=降採樣倍率；rowPitchMm=真實 mm/列
            _view.SetMergeMode(mergeMode);
            for (int i = 0; i < gray.Length; i++)
                if (gray[i] != null) _view.PushFrame(i + 1, gray[i], w[i], h[i]);
        }

        /// <summary>chart 重建後補發當前視野（強化切換/重載後曲線恢復跟隨，免等滑鼠互動）。</summary>
        public void RefireViewRange() => _view?.RefireViewRange();

        public void SetMergeMode(bool on) => _view?.SetMergeMode(on);
        public void SetSelected(int camId) => _view?.SetSelected(camId);

        /// <summary>Bitmap（8bpp 索引灰階或 24/32bpp）→ 8bpp 灰階 bytes。</summary>
        internal static byte[] ToGray8(Bitmap bmp, out int w, out int h)
        {
            w = bmp.Width; h = bmp.Height;
            if (w <= 0 || h <= 0) return null;
            var rect = new Rectangle(0, 0, w, h);
            var dst = new byte[w * h];

            if (bmp.PixelFormat == PixelFormat.Format8bppIndexed)
            {
                var bd = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                try
                {
                    for (int y = 0; y < h; y++)
                        System.Runtime.InteropServices.Marshal.Copy(bd.Scan0 + y * bd.Stride, dst, y * w, w);
                }
                finally { bmp.UnlockBits(bd); }
                return dst;
            }

            // 24/32bpp：取 G 通道（灰階 JPEG 解成 RGB 時 R=G=B，足夠；非灰圖也可接受近似）
            var bd2 = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                int stride = bd2.Stride;
                var row = new byte[stride];
                for (int y = 0; y < h; y++)
                {
                    System.Runtime.InteropServices.Marshal.Copy(bd2.Scan0 + y * stride, row, 0, stride);
                    int o = y * w;
                    for (int x = 0; x < w; x++) dst[o + x] = row[x * 3 + 1];
                }
            }
            finally { bmp.UnlockBits(bd2); }
            return dst;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _view?.Dispose(); _view = null;
            if (_mainHost != null) { _mainHost.Parent?.Controls.Remove(_mainHost); _mainHost.Dispose(); _mainHost = null; }
            if (_thumbHosts != null)
                foreach (var p in _thumbHosts)
                    if (p != null) { p.Parent?.Controls.Remove(p); p.Dispose(); }
        }
    }

}
