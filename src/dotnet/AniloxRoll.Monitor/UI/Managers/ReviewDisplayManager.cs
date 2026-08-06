using System;
using System.Drawing;
using System.Globalization;
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
        private bool _suppressViewRangeEvents;
        private bool _recordTransitionPending;
        private bool _recordTransitionDropLogged;
        private string _recordTransitionGrabId;
        private IntensityColorMap _mainColorMap = IntensityColorMap.Grayscale;
        private CanvasOverlayMode _overlayMode = CanvasOverlayMode.Coordinates;
        private bool _applyingOverlayMode;
        private Func<string> _informationTextProvider;

        /// <summary>視野可見範圍（mm）pass-through（View 為 lazy，外部訂這裡）：left,right,top,bot →
        /// 回顧曲線圖 zoom 連動（拖曳中也即時，鐵則：不可為效能抑制）。</summary>
        public event Action<double, double, double, double> ViewRangeMmChanged;
        public event Action<CanvasOverlayMode> OverlayModeChanged;

        /// <summary>sdk 顯示元件（接事件 / 進階用；未建立前 null）。</summary>
        public ImageDisplayView View => _view;

        public CanvasOverlayMode OverlayMode
        {
            get => _overlayMode;
            set
            {
                _overlayMode = value;
                if (_view == null) return;
                _applyingOverlayMode = true;
                try { _view.Canvas.OverlayMode = value; }
                finally { _applyingOverlayMode = false; }
            }
        }

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
            _view.Canvas.InformationTextProvider = _informationTextProvider;
            _view.Canvas.OverlayMode = _overlayMode;
            _view.Canvas.OverlayModeChanged += OnCanvasOverlayModeChanged;
            _view.MainColorMap = _mainColorMap;
            _view.ThumbSelectedColor = Color.Orange;   // 與監控同款；選取視覺唯一來源 = sdk ThumbView
            _view.MergeAll = true;                     // 缺台黑占位（與影像/曲線分界一致）
            _view.EnableLod(GrayResizeCpu.Resize);     // 回顧白賺 LOD；CPU provider＝無 GPU 機也跑
            _view.ViewRangeMmChanged += (l, r, tp, bt) =>
            {
                if (_suppressViewRangeEvents) return;
                if (_recordTransitionPending)
                {
                    if (!_recordTransitionDropLogged)
                    {
                        _recordTransitionDropLogged = true;
                        Core.Services.FlowTrace.Dvt(
                            $"RV staleView drop {_recordTransitionGrabId}");
                    }
                    return;
                }
                ViewRangeMmChanged?.Invoke(l, r, tp, bt);
            };
            // 互動流跡（RV 前綴）：autoFit 原因/lodRebind/clearFrame/wheelZoom 與監控同一套 sdk 掛勾
            _view.FlowLog = s => Core.Services.FlowTrace.Display("RV", s);
            Core.Services.FlowTrace.Log($"RV EnsureImageDisplay create（thumbs={_thumbHosts.Length}）");
        }

        /// <summary>
        /// 餵一組回顧灰階幀（RSC 解碼段已轉好的不可變 bytes；null 槽=間空黑占位）+ CFG 座標。
        /// 純推幀零轉換 → UI 執行緒無負擔、與 Bitmap 生命週期零 race。
        /// </summary>
        /// <summary>
        /// A new record already owns the charts, but its pixels are still loading. View events from
        /// the previously displayed record must not overwrite the prepared fit range.
        /// </summary>
        public void BeginRecordTransition(string grabId)
        {
            _recordTransitionPending = true;
            _recordTransitionDropLogged = false;
            _recordTransitionGrabId = grabId ?? "";
        }

        public void PushFrames(byte[][] gray, int[] w, int[] h, double[] opsUm, double[] posMm,
            bool mergeMode, double screenMmPerPx, int feedScale, double rowPitchMm, bool flipVertical,
            double trimHeadMm, double trimTailMm, bool preserveChartView = false)
        {
            if (_disposed || gray == null) return;
            EnsureCreated(screenMmPerPx);
            bool publishRecordView = !preserveChartView;
            bool fittedRecord = false;
            ImageViewRange preservedView = default(ImageViewRange);
            bool hasPreservedView = preserveChartView && _view.TryGetViewRangeMm(out preservedView);
            _suppressViewRangeEvents = true;
            try
            {
                _view.FlipVertical = flipVertical;
                _view.VerticalZeroAtBottom = flipVertical;   // 垂直座標約定同方向（由下而上＝0 錨定畫面底）
                _view.SetHorizontalDisplayCrop(trimHeadMm, trimTailMm);
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
                Core.Services.FlowTrace.Log(
                    $"RV pushFrames {pushed}/{count}（merge={mergeMode}, feedScale={feedScale}, " +
                    $"chartView={(preserveChartView ? "keep" : "publish")}）");
                if (preserveChartView)
                {
                    bool restored = hasPreservedView && _view.TryRestoreViewRangeMm(preservedView);
                    ImageViewRange restoredView = default(ImageViewRange);
                    bool measured = restored && _view.TryGetViewRangeMm(out restoredView);
                    double maxDelta = measured ? MaxViewDelta(preservedView, restoredView) : double.PositiveInfinity;
                    Core.Services.FlowTrace.Dvt(string.Format(
                        CultureInfo.InvariantCulture,
                        "RV variantView keep beforeX={0:F2}~{1:F2} beforeY={2:F2}~{3:F2} " +
                        "afterX={4:F2}~{5:F2} afterY={6:F2}~{7:F2} maxDelta={8}",
                        preservedView.LeftMm, preservedView.RightMm,
                        preservedView.TopMm, preservedView.BottomMm,
                        measured ? restoredView.LeftMm : double.NaN,
                        measured ? restoredView.RightMm : double.NaN,
                        measured ? restoredView.TopMm : double.NaN,
                        measured ? restoredView.BottomMm : double.NaN,
                        double.IsInfinity(maxDelta)
                            ? "missing"
                            : maxDelta.ToString("F3", CultureInfo.InvariantCulture)));
                }
                if (publishRecordView)
                {
                    // A new record always starts fitted. Equal-sized records do not rebind LOD,
                    // so relying on lodRebind would retain the previous record's pan/zoom.
                    _view.ResetViewToFit();
                    Core.Services.FlowTrace.Display("RV", "fit(record-change)");
                    fittedRecord = true;
                }
            }
            finally
            {
                _suppressViewRangeEvents = false;
            }

            if (fittedRecord)
            {
                _recordTransitionPending = false;
                _recordTransitionDropLogged = false;
                _recordTransitionGrabId = null;
                _view.RefireViewRange();
            }
        }

        private static double MaxViewDelta(ImageViewRange before, ImageViewRange after)
        {
            return Math.Max(
                Math.Max(Math.Abs(before.LeftMm - after.LeftMm), Math.Abs(before.RightMm - after.RightMm)),
                Math.Max(Math.Abs(before.TopMm - after.TopMm), Math.Abs(before.BottomMm - after.BottomMm)));
        }

        /// <summary>
        /// 只計算回顧合圖在 fit 狀態下的四邊，不建立 ImageDisplayView，也不觸發重繪。
        /// </summary>
        public bool TryComputeFitViewRange(
            int[] widths, int[] heights, double[] opsUm, double[] posMm,
            bool mergeMode, int feedScale, double rowPitchMm, bool flipVertical,
            double trimHeadMm, double trimTailMm,
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
                trimHeadMm: trimHeadMm,
                trimTailMm: trimTailMm,
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

        public void SetInformationTextProvider(Func<string> provider)
        {
            _informationTextProvider = provider;
            if (_view != null) _view.Canvas.InformationTextProvider = provider;
        }

        /// <summary>只替現有回顧主畫面換調色盤；不重讀圖片、Curve 或改變視野。</summary>
        public void SetMainColorMap(IntensityColorMap colorMap)
        {
            _mainColorMap = colorMap;
            if (_view == null) return;
            _view.MainColorMap = colorMap;
            _view.RefreshNow();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // 宿主 Panel 是 Designer 擁有（Form 自行 dispose）；這裡只釋放 ImageDisplayView。
            if (_view != null)
            {
                _view.Canvas.OverlayModeChanged -= OnCanvasOverlayModeChanged;
                _view.Dispose();
                _view = null;
            }
        }

        private void OnCanvasOverlayModeChanged(CanvasOverlayMode mode)
        {
            if (_applyingOverlayMode) return;
            _overlayMode = mode;
            OverlayModeChanged?.Invoke(mode);
        }
    }

}
