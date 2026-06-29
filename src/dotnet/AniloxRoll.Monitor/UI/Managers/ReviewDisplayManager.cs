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
        private readonly Panel _mainHost;          // 2b-ii-B：直接是 Designer 上的 Panel（落地生根，不再 runtime overlay）
        private readonly Panel[] _thumbHosts;
        private LiveDisplayView _view;
        private bool _disposed;

        /// <summary>視野可見範圍（mm）pass-through（View 為 lazy，外部訂這裡）：left,right,top,bot →
        /// 回顧曲線圖 zoom 連動（拖曳中也即時，鐵則：不可為效能抑制）。</summary>
        public event Action<double, double, double, double> ViewRangeMmChanged;

        /// <summary>游標狀態（mm 位置/範圍/亮度/倍率）pass-through → 上層更新狀態列 lblPixelInfo。</summary>
        public event Action<LiveDisplayView.CursorStatus> CursorStatusChanged;

        /// <summary>sdk 顯示元件（接事件 / 進階用；未建立前 null）。</summary>
        public LiveDisplayView View => _view;

        /// <summary>當前選中相機 index（0-based；未建立前回 0）。供 RSC 重畫 per-cam 曲線用。</summary>
        public int SelectedCamIndex => (_view?.SelectedCamId ?? 1) - 1;

        public ReviewDisplayManager(Panel mainHost, Panel[] thumbHosts)
        {
            _mainHost = mainHost ?? throw new ArgumentNullException(nameof(mainHost));
            _thumbHosts = thumbHosts ?? new Panel[0];
        }

        /// <summary>lazy 建 LiveDisplayView（首次餵圖時呼叫；screenMmPerPx 屆時已由 SystemInfo 設定）。
        /// 宿主＝Designer 上的 camReviewMain/camReview1~7 Panel（2b-ii-B 後直接用，不再 overlay）。</summary>
        public void EnsureCreated(double screenMmPerPx)
        {
            if (_view != null || _disposed) return;

            _view = new LiveDisplayView(_mainHost, _thumbHosts, screenMmPerPx);
            _view.ThumbSelectedColor = Color.Orange;   // 與監控同款；選取視覺唯一來源 = sdk ThumbView
            _view.MergeAll = true;                     // 缺台黑占位（與影像/曲線分界一致）
            _view.EnableLod(GrayResizeCpu.Resize);     // 回顧白賺 LOD；CPU provider＝無 GPU 機也跑
            _view.ViewRangeMmChanged += (l, r, tp, bt) => ViewRangeMmChanged?.Invoke(l, r, tp, bt);
            _view.CursorStatusChanged += s => CursorStatusChanged?.Invoke(s);
        }

        /// <summary>
        /// 餵一組回顧灰階幀（RSC 解碼段已轉好的不可變 bytes；null 槽=間空黑占位）+ CFG 座標。
        /// 純推幀零轉換 → UI 執行緒無負擔、與 Bitmap 生命週期零 race。
        /// </summary>
        public void PushFrames(byte[][] gray, int[] w, int[] h, double[] opsUm, double[] posMm,
            bool mergeMode, double screenMmPerPx, int feedScale, double rowPitchMm, bool flipVertical)
        {
            if (_disposed || gray == null) return;
            EnsureCreated(screenMmPerPx);
            _view.FlipVertical = flipVertical;
            _view.SetLayout(posMm, opsUm, Math.Max(1, feedScale), rowPitchMm); // feedScale=降採樣倍率；rowPitchMm=真實 mm/列
            _view.SetMergeMode(mergeMode);
            for (int i = 0; i < gray.Length; i++)
                if (gray[i] != null) _view.PushFrame(i + 1, gray[i], w[i], h[i]);
        }

        /// <summary>chart 重建後補發當前視野（強化切換/重載後曲線恢復跟隨，免等滑鼠互動）。</summary>
        public void RefireViewRange() => _view?.RefireViewRange();

        public void SetFlipVertical(bool flipVertical)
        {
            if (_view == null) return;
            _view.FlipVertical = flipVertical;
            _view.RefireViewRange();
        }

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
            // 宿主 Panel 是 Designer 擁有（Form 自行 dispose）；這裡只釋放 LiveDisplayView。
            _view?.Dispose(); _view = null;
        }
    }

}
