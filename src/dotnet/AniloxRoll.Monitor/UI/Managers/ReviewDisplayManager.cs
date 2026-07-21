using System;
using System.Drawing;
using System.Windows.Forms;
using TanukiCv.Controls;
using TanukiCv.Core;

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 回顧主畫面接 sdk <see cref="ImageDisplayView"/>（與監控同源，絞殺榕收官 #13 Stage1）。
    /// 執行時在 camReviewMain 與 camReview1~7 位置疊 Panel 宿主 ImageDisplayView（同 Parent/Bounds/Anchor）。
    /// 4c 已轉正＝唯一顯示路徑（過渡旗標已刪）；舊控制項實體待 4d 連 Designer 一併清除。
    /// 回顧直接繼承：動態 LOD（70 張合圖顯示成本 ~180ms→~1ms 路線）、縮圖↔主畫面雙向連動、mm overlay、雙三擊、游標剖面。
    /// </summary>
    internal sealed class ReviewDisplayManager : IDisposable
    {
        private readonly Panel _mainHost;          // 2b-ii-B：直接是 Designer 上的 Panel（落地生根，不再 runtime overlay）
        private readonly Panel[] _thumbHosts;
        private ImageDisplayView _view;
        private bool _disposed;

        /// <summary>視野可見範圍（mm）pass-through（View 為 lazy，外部訂這裡）：left,right,top,bot →
        /// 回顧曲線圖 zoom 連動（拖曳中也即時，鐵則：不可為效能抑制）。</summary>
        public event Action<double, double, double, double> ViewRangeMmChanged;

        /// <summary>sdk 顯示元件（接事件 / 進階用；未建立前 null）。</summary>
        public ImageDisplayView View => _view;

        /// <summary>當前選中相機 index（0-based；未建立前回 0）。供 RSC 重畫 per-cam 曲線用。</summary>
        public int SelectedCamIndex => (_view?.SelectedCamId ?? 1) - 1;

        public ReviewDisplayManager(Panel mainHost, Panel[] thumbHosts)
        {
            _mainHost = mainHost ?? throw new ArgumentNullException(nameof(mainHost));
            _thumbHosts = thumbHosts ?? new Panel[0];
        }

        /// <summary>lazy 建 ImageDisplayView（首次餵圖時呼叫；screenMmPerPx 屆時已由 SystemInfo 設定）。
        /// 宿主＝Designer 上的 camReviewMain/camReview1~7 Panel（2b-ii-B 後直接用，不再 overlay）。</summary>
        public void EnsureCreated(double screenMmPerPx)
        {
            if (_view != null || _disposed) return;

            _view = new ImageDisplayView(_mainHost, _thumbHosts, screenMmPerPx);
            _view.ThumbSelectedColor = Color.Orange;   // 與監控同款；選取視覺唯一來源 = sdk ThumbView
            _view.MergeAll = true;                     // 缺台黑占位（與影像/曲線分界一致）
            _view.EnableLod(GrayResizeCpu.Resize);     // 回顧白賺 LOD；CPU provider＝無 GPU 機也跑
            _view.ViewRangeMmChanged += (l, r, tp, bt) => ViewRangeMmChanged?.Invoke(l, r, tp, bt);
            // 互動流跡（RV 前綴）：autoFit 原因/lodRebind/clearFrame/wheelZoom 與監控同一套 sdk 掛勾
            _view.FlowLog = s => Core.Services.FlowTrace.Log("RV " + s);
            Core.Services.FlowTrace.Log($"RV EnsureImageDisplay create（thumbs={_thumbHosts.Length}）");
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
            _view.VerticalZeroAtBottom = flipVertical;   // 垂直座標約定同方向（由下而上＝0 錨定畫面底）
            _view.SetLayout(posMm, opsUm, Math.Max(1, feedScale), rowPitchMm); // feedScale=降採樣倍率；rowPitchMm=真實 mm/列
            _view.SetMergeMode(mergeMode);
            var present = new bool[_thumbHosts.Length + 1];
            int count = Math.Min(gray.Length, _thumbHosts.Length);
            int pushed = 0;
            for (int i = 0; i < count; i++)
            {
                if (gray[i] == null) continue;
                present[i + 1] = true;
                _view.PushFrame(i + 1, gray[i], w[i], h[i]);
                pushed++;
            }
            _view.ClearFramesExcept(present);
            _view.RefreshNow();
            // Same pixel dimensions do not rebind LOD, but OPS/start/row pitch may still differ
            // between grabs. Re-publish the range from ImageDisplayView's single conversion path
            // before the new column/row curves are applied, so no chart paints with stale units.
            _view.RefireViewRange();
            Core.Services.FlowTrace.Log($"RV pushFrames {pushed}/{count}（merge={mergeMode}, feedScale={feedScale}）");
        }

        /// <summary>
        /// 只計算回顧合圖在 fit 狀態下的四邊，不建立 ImageDisplayView，也不觸發重繪。
        /// </summary>
        public bool TryComputeFitViewRange(
            int[] widths, int[] heights, double[] opsUm, double[] posMm,
            bool mergeMode, int feedScale, double rowPitchMm, bool flipVertical,
            out ImageViewRange range)
        {
            range = default(ImageViewRange);
            if (_disposed || !mergeMode || _mainHost.IsDisposed) return false;
            return ImageDisplayView.TryComputeMergeFitViewRange(
                widths, heights, posMm, opsUm,
                Math.Max(1, feedScale), rowPitchMm,
                mergeAll: true, mergeStrategy: MergeOverlap.Midline,
                verticalZeroAtBottom: flipVertical,
                viewport: _mainHost.ClientSize,
                range: out range);
        }

        /// <summary>chart 重建後補發當前視野（強化切換/重載後曲線恢復跟隨，免等滑鼠互動）。</summary>
        public void RefireViewRange() => _view?.RefireViewRange();

        /// <summary>回顧頁由隱藏轉為可見時補畫既有內容，不重讀檔或重設視野。</summary>
        public void RefreshVisible()
        {
            Core.Services.FlowTrace.Log($"RV tabVisible repaint view={_view != null}");
            _view?.RefreshVisible();
        }

        public void SetFlipVertical(bool flipVertical)
        {
            if (_view == null) return;
            _view.FlipVertical = flipVertical;
            _view.VerticalZeroAtBottom = flipVertical;   // 垂直座標約定同方向（由下而上＝0 錨定畫面底）
            _view.RefireViewRange();
        }

        public void SetMergeMode(bool on) => _view?.SetMergeMode(on);
        public void SetSelected(int camId) => _view?.SetSelected(camId);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // 宿主 Panel 是 Designer 擁有（Form 自行 dispose）；這裡只釋放 ImageDisplayView。
            _view?.Dispose(); _view = null;
        }
    }

}
