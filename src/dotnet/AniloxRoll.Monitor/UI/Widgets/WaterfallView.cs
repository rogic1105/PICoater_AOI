using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data; // WaterfallFullMode
using TanukiCv.Controls;   // SmartCanvas / GrayResizeCpu / GrayBitmap
using TanukiCv.Core;       // MergeLayout / MergeOverlap

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 監控主畫面「瀑布圖」：全幅合圖每幀往下接、即時捲動（線掃像印表機吐紙）。
    ///
    /// 模型（2026-06-24 更新）：**固定總高度**（WaterfallTotalHeight，預設 30000）的虛擬長圖
    /// → 記憶體有上界、可預配一塊固定 buffer（總高 × band寬）；點兩下 fit 有確定目標高度。
    /// 填滿後行為（WaterfallFullMode）：
    ///   - **Restart 重來**：清空黑幕、從頭重畫。
    ///   - **Ring 循環**：從頭覆蓋最舊的、連續捲（環狀 buffer + 顯示時 unroll：最舊在上、最新在下）。
    ///
    /// 顯示：SmartCanvas 動態 LOD（虛擬圖固定 band寬×總高；provider 只畫可見區，從固定 buffer 反查）。
    /// 預設 fit 整張 + 點兩下 fit / 三擊實體 1:1（三擊需 mm 校正 → 後續 mm 尺規一起接）。
    /// 掉偵補黑：某台這個 band cycle 沒新幀那欄留黑（fresh 旗標）。
    /// </summary>
    public sealed class WaterfallView : IDisposable
    {
        private const int TargetBandW = 2048;   // band 降採樣目標寬（求可用 + band 寬有界）

        private readonly SmartCanvas _canvas;
        private readonly int _camCount;
        private readonly int _totalHeight;       // 虛擬長圖固定總高
        private readonly WaterfallFullMode _fullMode;
        private readonly System.Windows.Forms.Timer _commitTimer;
        private readonly object _lock = new object();

        private byte[] _buffer;                  // 固定 buffer = _bandW × _totalHeight（band 寬確定後配）
        private int _bandW;
        private int _writeRow;                   // 下一個 band 寫入的起始 row
        private bool _wrapped;                    // Ring：是否已繞過一圈（決定顯示要不要 unroll）

        private readonly byte[][] _latest;       // 各相機最新全解析度幀
        private readonly int[] _lw, _lh;
        private readonly bool[] _fresh;          // 自上個 band 以來這台有沒有新幀（掉偵補黑）
        private double[] _startMm;
        private double _refOpsMm = 0.024;
        private bool _disposed;
        private bool _virtualSet;
        private int _composeLog;

        public WaterfallView(Panel host, int camCount, int totalHeight, WaterfallFullMode fullMode, int commitMs = 150)
        {
            _camCount = Math.Max(1, camCount);
            _totalHeight = Math.Max(1000, totalHeight);
            _fullMode = fullMode;
            _latest = new byte[_camCount][];
            _lw = new int[_camCount];
            _lh = new int[_camCount];
            _fresh = new bool[_camCount];

            _canvas = new SmartCanvas { Dock = DockStyle.Fill, BackColor = Color.Black };
            _canvas.ShowOverlay = true;               // 游標座標 + 亮度
            _canvas.DoubleClickFitToScreen = true;    // 點兩下 fit 整張（固定總高 → 目標確定）
            _canvas.TripleClickPhysical1x = true;     // 三擊實體 1:1（需 mm 校正，後續接）
            host.Controls.Add(_canvas);
            _canvas.BringToFront();
            _canvas.EnableLod(1, 1, ProvideRegion);

            _commitTimer = new System.Windows.Forms.Timer { Interval = Math.Max(30, commitMs) };
            _commitTimer.Tick += (s, e) => CommitBand();
            _commitTimer.Start();
        }

        /// <summary>合圖佈局（各台 start mm + 基準像素尺寸 mm/px）。對齊 live 全域合圖。</summary>
        public void SetLayout(double[] startMm, double[] opsUm, double refOpsMm)
        {
            lock (_lock) { _startMm = startMm; if (refOpsMm > 0) _refOpsMm = refOpsMm; }
        }

        /// <summary>各相機每幀（可能背景執行緒）：存最新全解析度灰階快照（重用緩衝故複製）。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h)
        {
            if (_disposed || camId < 1 || camId > _camCount || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            lock (_lock) { _latest[camId - 1] = copy; _lw[camId - 1] = w; _lh[camId - 1] = h; _fresh[camId - 1] = true; }
        }

        private void CommitBand()
        {
            if (_disposed) return;
            lock (_lock)
            {
                if (!TryComposeBand(out var band, out var bw, out var bh)) return;
                // band 寬變了（佈局改）或首次 → (重)配固定 buffer
                if (_buffer == null || bw != _bandW)
                {
                    _bandW = bw;
                    _buffer = new byte[_bandW * _totalHeight];
                    _writeRow = 0; _wrapped = false; _virtualSet = false;
                }
                WriteBand(band, bh);
            }
            if (!_virtualSet) { _canvas.UpdateLodVirtualSize(_bandW, _totalHeight); _virtualSet = true; } // 固定虛擬尺寸 + fit 一次
            else _canvas.RefreshLod();                                                                     // 內容變、視角不變
        }

        // 把 band（_bandW × bh）寫進固定 buffer，依模式處理填滿。
        private void WriteBand(byte[] band, int bh)
        {
            if (bh <= 0) return;
            if (_fullMode == WaterfallFullMode.Restart)
            {
                if (_writeRow + bh > _totalHeight) { Array.Clear(_buffer, 0, _buffer.Length); _writeRow = 0; } // 滿 → 黑幕重來
                int rows = Math.Min(bh, _totalHeight - _writeRow);
                Array.Copy(band, 0, _buffer, _writeRow * _bandW, rows * _bandW);
                _writeRow += rows;
            }
            else // Ring：環狀寫，逐行 wrap
            {
                for (int y = 0; y < bh; y++)
                {
                    int dst = ((_writeRow + y) % _totalHeight) * _bandW;
                    Array.Copy(band, y * _bandW, _buffer, dst, _bandW);
                }
                int nw = _writeRow + bh;
                if (nw >= _totalHeight) _wrapped = true;
                _writeRow = nw % _totalHeight;
            }
        }

        private bool TryComposeBand(out byte[] band, out int bw, out int bh)
        {
            band = null; bw = 0; bh = 0;
            if (_startMm == null) return false;

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

            MergeLayout.Compute(cams, minStart, _refOpsMm, 1, MergeOverlap.Midline, out int fullW);
            if (fullW <= 0) return false;
            int scale = Math.Max(1, (int)Math.Ceiling(fullW / (double)TargetBandW));

            if (_composeLog < 6)
            {
                _composeLog++;
                int haveLatest = 0, freshCnt = 0;
                for (int i = 0; i < _camCount; i++) { if (_latest[i] != null) haveLatest++; if (_fresh[i]) freshCnt++; }
                System.Diagnostics.Trace.WriteLine($"[Waterfall] compose haveLatest={haveLatest} fresh={freshCnt} refOps={_refOpsMm:F4} fullW={fullW} scale={scale} totalH={_totalHeight} mode={_fullMode}");
            }

            // 降採樣空間佈局：寬度也要 ÷scale（MergeLayout 只對 xOff ÷scale、WidthPx 不動）
            var camsDs = new List<MergeLayout.CamGeom>(cams.Count);
            foreach (var cg in cams)
                camsDs.Add(new MergeLayout.CamGeom { CameraId = cg.CameraId, StartMm = cg.StartMm, WidthPx = Math.Max(1, cg.WidthPx / scale) });
            var places = MergeLayout.Compute(camsDs, minStart, _refOpsMm, scale, MergeOverlap.Midline, out int dsW);
            if (dsW <= 0) return false;

            var ds = new byte[_camCount][];
            var dsw = new int[_camCount];
            var dsh = new int[_camCount];
            int bandH = 0;
            for (int i = 0; i < _camCount; i++)
            {
                // 只放這個 cycle 有新幀的相機；其餘留黑（掉偵補黑）。佈局仍以所有曾有幀的相機算 → 寬/位置穩定。
                if (_latest[i] == null || !_fresh[i]) continue;
                int nw = Math.Max(1, _lw[i] / scale);
                int nh = Math.Max(1, _lh[i] / scale);
                ds[i] = GrayResizeCpu.Resize(_latest[i], _lw[i], _lh[i], nw, nh);
                dsw[i] = nw; dsh[i] = nh;
                if (bandH == 0) bandH = nh;
            }
            if (bandH == 0) return false; // 沒有任何 fresh 相機 → 不出 band（避免重複）

            var outBand = new byte[dsW * bandH]; // 黑底
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
            // 這個 band cycle 消費完 → 清 fresh（下個 cycle 沒新幀的相機＝掉偵補黑）
            for (int i = 0; i < _camCount; i++) _fresh[i] = false;
            band = outBand; bw = dsW; bh = bandH;
            return true;
        }

        // SmartCanvas LOD provider：給虛擬區域 → 回傳該區域縮到 dest 的 bitmap（從固定 buffer 反查；Ring 已繞圈則 unroll）
        private Bitmap ProvideRegion(Rectangle r, Size dest)
        {
            if (dest.Width <= 0 || dest.Height <= 0) return null;
            int rw = Math.Max(1, r.Width), rh = Math.Max(1, r.Height);
            byte[] region;
            lock (_lock)
            {
                if (_buffer == null || _bandW <= 0) return null;
                bool ring = _fullMode == WaterfallFullMode.Ring && _wrapped;
                region = new byte[rw * rh]; // 黑底
                for (int yy = 0; yy < rh; yy++)
                {
                    int gy = r.Y + yy;
                    if (gy < 0 || gy >= _totalHeight) continue;
                    int bufRow = ring ? (_writeRow + gy) % _totalHeight : gy; // Ring 繞圈：最舊(=_writeRow)在上、最新在下
                    int sx = r.X, cw = rw;
                    if (sx < 0) { cw += sx; sx = 0; }
                    if (sx + cw > _bandW) cw = _bandW - sx;
                    if (cw <= 0) continue;
                    Array.Copy(_buffer, bufRow * _bandW + sx, region, yy * rw + (sx - r.X), cw);
                }
            }
            byte[] scaled = GrayResizeCpu.Resize(region, rw, rh, dest.Width, dest.Height);
            return scaled == null ? null : GrayBitmap.From(scaled, dest.Width, dest.Height);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _commitTimer.Stop(); _commitTimer.Dispose(); } catch { }
            try { _canvas.DisableLod(); } catch { }
            try { if (_canvas.Parent != null) _canvas.Parent.Controls.Remove(_canvas); _canvas.Dispose(); } catch { }
            lock (_lock) { _buffer = null; _bandW = 0; _writeRow = 0; _wrapped = false; }
        }
    }
}
