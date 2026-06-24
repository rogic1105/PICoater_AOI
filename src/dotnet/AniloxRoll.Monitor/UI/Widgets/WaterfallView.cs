using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using TanukiCv.Controls;   // SmartCanvas / GrayResizeCpu / GrayBitmap
using TanukiCv.Core;       // MergeLayout / MergeOverlap

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 監控主畫面「瀑布圖」：全幅合圖每幀往下接、即時往下捲動（線掃像印表機吐紙）。
    ///
    /// 設計：
    /// - **記憶體不預配**：每幀合成一個 band（降採樣後的全幅合圖），chunk append 進 list；秒數不限 → 記憶體隨時間
    ///   線性成長，看資源監控找極限（全解析每幀 ~300MB 2-3 幀就 OOM，故降採樣到 TargetBandW，可長時間測壓力）。
    /// - **顯示**：SmartCanvas 動態 LOD（虛擬長圖＝全幅寬 × 累積總高；provider 只畫可見區）。
    /// - **掉偵補黑**：某台沒最新幀那欄留黑（= band 初值 0）；framecount 級的即時掉偵偵測 = segment 3。
    ///
    /// 假設：一個 waterfall session 內各 band 同高（grabHeight 不變）→ provider 用 gy/_bandH 反查 band。
    /// </summary>
    public sealed class WaterfallView : IDisposable
    {
        private const int TargetBandW = 2048;   // band 降採樣目標寬（第一版求可用 + 可長時間測壓力；全解析另議）

        private readonly SmartCanvas _canvas;
        private readonly int _camCount;
        private readonly System.Windows.Forms.Timer _commitTimer;
        private readonly object _lock = new object();

        private readonly List<byte[]> _bands = new List<byte[]>(); // 每個 = _bandW × _bandH 灰階
        private int _bandW, _bandH, _totalH;

        private readonly byte[][] _latest;       // 各相機最新全解析度幀
        private readonly int[] _lw, _lh;
        private double[] _startMm, _opsUm;
        private double _refOpsMm = 0.024;
        private bool _disposed;

        public WaterfallView(Panel host, int camCount, int commitMs = 150)
        {
            _camCount = Math.Max(1, camCount);
            _latest = new byte[_camCount][];
            _lw = new int[_camCount];
            _lh = new int[_camCount];

            _canvas = new SmartCanvas { Dock = DockStyle.Fill, BackColor = Color.Black };
            host.Controls.Add(_canvas);
            _canvas.BringToFront();
            _canvas.EnableLod(1, 1, ProvideRegion); // 初始虛擬尺寸 1×1，第一個 band 進來才 UpdateLodVirtualSize

            _commitTimer = new System.Windows.Forms.Timer { Interval = Math.Max(30, commitMs) };
            _commitTimer.Tick += (s, e) => CommitBand();
            _commitTimer.Start();
        }

        /// <summary>合圖佈局（各台 start mm + 基準像素尺寸 mm/px）。對齊 live 全域合圖。</summary>
        public void SetLayout(double[] startMm, double[] opsUm, double refOpsMm)
        {
            lock (_lock)
            {
                _startMm = startMm; _opsUm = opsUm;
                if (refOpsMm > 0) _refOpsMm = refOpsMm;
            }
        }

        /// <summary>各相機每幀（可能背景執行緒）：存最新全解析度灰階快照（重用緩衝故複製）。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h)
        {
            if (_disposed || camId < 1 || camId > _camCount || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            lock (_lock) { _latest[camId - 1] = copy; _lw[camId - 1] = w; _lh[camId - 1] = h; }
        }

        // ── 合成一個 band（降採樣全幅合圖）並 append ──
        private void CommitBand()
        {
            if (_disposed) return;
            int bw, bh;
            lock (_lock)
            {
                if (!TryComposeBand(out var band, out bw, out bh)) return;
                if (_bandW != 0 && bw != _bandW) return; // 寬變了（佈局改）→ 本版跳過，避免 provider 反查錯
                _bands.Add(band);
                _bandW = bw; _bandH = bh; _totalH += bh;
            }
            _canvas.UpdateLodVirtualSize(_bandW, _totalH); // 虛擬長圖長高（內含 FitToScreen；auto-scroll 到底＝segment 3）
        }

        private bool TryComposeBand(out byte[] band, out int bw, out int bh)
        {
            band = null; bw = 0; bh = 0;
            if (_startMm == null) return false;

            // 收集有最新幀的相機 + 全域原點
            var cams = new List<MergeLayout.CamGeom>();
            double minStart = double.MaxValue;
            for (int i = 0; i < _camCount; i++)
            {
                if (_latest[i] == null) continue;
                double sm = i < _startMm.Length ? _startMm[i] : 0;
                if (sm < minStart) minStart = sm;
            }
            if (minStart == double.MaxValue) return false;
            for (int i = 0; i < _camCount; i++)
            {
                if (_latest[i] == null) continue;
                cams.Add(new MergeLayout.CamGeom { CameraId = i + 1, StartMm = i < _startMm.Length ? _startMm[i] : 0, WidthPx = _lw[i] });
            }
            if (cams.Count == 0 || _refOpsMm <= 0) return false;

            // 先算全解析度總寬決定降採樣 scale，使降採樣後 ~TargetBandW
            MergeLayout.Compute(cams, minStart, _refOpsMm, 1, MergeOverlap.Midline, out int fullW);
            if (fullW <= 0) return false;
            int scale = Math.Max(1, (int)Math.Ceiling(fullW / (double)TargetBandW));

            // 降採樣後佈局（scale 帶入 → 座標已是降採樣空間）
            var places = MergeLayout.Compute(cams, minStart, _refOpsMm, scale, MergeOverlap.Midline, out int dsW);
            if (dsW <= 0) return false;

            // 各相機降採樣成 strip
            var ds = new byte[_camCount][];
            var dsw = new int[_camCount];
            var dsh = new int[_camCount];
            int bandH = 0;
            for (int i = 0; i < _camCount; i++)
            {
                if (_latest[i] == null) continue;
                int nw = Math.Max(1, _lw[i] / scale);
                int nh = Math.Max(1, _lh[i] / scale);
                ds[i] = GrayResizeCpu.Resize(_latest[i], _lw[i], _lh[i], nw, nh);
                dsw[i] = nw; dsh[i] = nh;
                if (bandH == 0) bandH = nh;
            }
            if (bandH == 0) return false;

            var outBand = new byte[dsW * bandH]; // 黑底（掉偵那欄自然留黑）
            foreach (var p in places)
            {
                int i = p.CameraId - 1;
                if (i < 0 || i >= _camCount || ds[i] == null || p.SrcWidth <= 0) continue;
                int rows = Math.Min(bandH, dsh[i]);
                for (int y = 0; y < rows; y++)
                {
                    int sx = p.SrcLeft, dx = p.DestX, cw = p.SrcWidth;
                    if (sx < 0) { dx -= sx; cw += sx; sx = 0; }
                    if (sx + cw > dsw[i]) cw = dsw[i] - sx;
                    if (dx < 0) { sx -= dx; cw += dx; dx = 0; }
                    if (dx + cw > dsW) cw = dsW - dx;
                    if (cw <= 0) continue;
                    Array.Copy(ds[i], y * dsw[i] + sx, outBand, y * dsW + dx, cw);
                }
            }
            band = outBand; bw = dsW; bh = bandH;
            return true;
        }

        // ── SmartCanvas LOD provider：給虛擬區域 → 回傳該區域縮到 dest 的 bitmap ──
        private Bitmap ProvideRegion(Rectangle r, Size dest)
        {
            if (dest.Width <= 0 || dest.Height <= 0) return null;
            byte[] region;
            int rw = Math.Max(1, r.Width), rh = Math.Max(1, r.Height);
            lock (_lock)
            {
                if (_bandH <= 0 || _bands.Count == 0) return null;
                region = new byte[rw * rh]; // 黑底
                for (int yy = 0; yy < rh; yy++)
                {
                    int gy = r.Y + yy;
                    if (gy < 0 || gy >= _totalH) continue;
                    int bi = gy / _bandH;
                    if (bi < 0 || bi >= _bands.Count) continue;
                    int rib = gy % _bandH;
                    byte[] bandData = _bands[bi];
                    int sx = r.X, cw = rw;
                    if (sx < 0) { cw += sx; sx = 0; }
                    if (sx + cw > _bandW) cw = _bandW - sx;
                    if (cw <= 0) continue;
                    int dstX = sx - r.X;
                    Array.Copy(bandData, rib * _bandW + sx, region, yy * rw + dstX, cw);
                }
            }
            byte[] scaled = GrayResizeCpu.Resize(region, rw, rh, dest.Width, dest.Height);
            if (scaled == null) return null;
            return GrayBitmap.From(scaled, dest.Width, dest.Height);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _commitTimer.Stop(); _commitTimer.Dispose(); } catch { }
            try { _canvas.DisableLod(); } catch { }
            try { if (_canvas.Parent != null) _canvas.Parent.Controls.Remove(_canvas); _canvas.Dispose(); } catch { }
            lock (_lock) { _bands.Clear(); _totalH = 0; _bandW = 0; _bandH = 0; }
        }
    }
}
