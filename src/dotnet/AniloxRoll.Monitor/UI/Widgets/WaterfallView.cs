using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Data;       // WaterfallFullMode
using AniloxRoll.Monitor.Core.Services;   // FrameTickIndex（共用聚類規則 ComputeThreshold）
using TanukiCv.Controls;   // SmartCanvas / GrayBitmap
using TanukiCv.Core;       // MergeLayout / MergeOverlap

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 監控主畫面「瀑布圖」：全幅 7 相機合圖每幀往下接、即時捲動（線掃像印表機吐紙）。
    ///
    /// 儲存（2026-06-24 全解析重寫）：全解析**分塊**存（chunk，避開 2GB byte[] 上限），LOD 只在顯示時降採樣。
    /// 虛擬長圖 = 全解析合圖寬 `_fullW`（7 槽全排、沒畫面補黑）× 固定總高 `_totalHeight`（預設 30000，LOD 之前的全解析 Y）。
    /// 填滿：Restart 重來＝清黑幕從頂重畫；Ring 循環＝繞回頂端覆蓋最舊、顯示時寫頭畫亮掃描線接縫。
    ///
    /// 跨相機幀對齊（2026-06-25 與回顧同源）：**用硬體 frame-start tick 半週期聚類成「時間槽」**，
    /// 一槽＝一個物理瞬間（每槽輸出一條 band）；某台在某槽沒幀＝它在那掉幀 → 該欄補黑。
    /// 聚類容差 thr = 幀週期/2，與回顧 <see cref="FrameTickIndex"/> 共用 <see cref="FrameTickIndex.ComputeThreshold"/>（單一來源）。
    /// 週期線上估計＝各相機相鄰 tick 差的「運行最小值」（掉幀只會放大 delta，故 min 天然抗掉幀）。
    /// 串流（幀多執行緒即時到）用 pending 槽緩衝 + hold-back grace flush（非「下一幀立即 flush」，避免同瞬間晚到的幀被誤判掉幀補黑）。
    ///
    /// ⚠ 跨板限制：cam1-4(板0)/cam5-7(板1) tick epoch 不同、不可跨板相減。目前同板(≤4 台)正確；
    /// 7 台跨板需先做 board offset 正規化（回顧 FrameTickIndex 亦有此潛在問題，獨立議題）。
    /// </summary>
    public sealed class WaterfallView : IDisposable
    {
        private const int ChunkRows = 512;          // 每塊高度（row）；全幅~101000 → 每塊 ~51MB，30000 高約 60 塊
        private const long HoldGraceMs = 12;        // 槽最後一幀後再等這麼久才 flush（接同瞬間晚到的幀，防偽掉幀補黑）
        private const long BootstrapGraceMs = 40;   // 週期未知前用 wall-clock 視窗聚類（tick 單位/週期還沒學到時）
        private const long StaleFlushMs = 250;      // 久未更新的滯留槽強制 flush（grab 暫停也讓畫面前進）

        private static readonly Stopwatch _clock = Stopwatch.StartNew();

        private readonly SmartCanvas _canvas;
        private readonly int _camCount;             // 配置槽數（7，含黑布槽）
        private readonly int _totalHeight;
        private readonly WaterfallFullMode _fullMode;
        private readonly double _screenMmPerPx;
        private readonly System.Windows.Forms.Timer _flushTimer; // 安全網：flush 滯留槽 + 推 LOD 刷新
        private readonly object _lock = new object();

        // 全解析分塊儲存：_chunks[ci] = byte[_fullW * ChunkRows]（lazy 配；null=黑）。
        private byte[][] _chunks;
        private int _fullW;
        private int _writeRow;                       // 下一個 band 寫入起始 row（也是 Ring 顯示接縫位置）

        // ── 跨相機 tick 聚類（串流）──
        private sealed class Slot
        {
            public long Anchor;                      // 槽最小 tick
            public long MaxTick;                     // 槽最大 tick
            public long LastWallMs;                  // 最後一幀到達 wall-clock
            public readonly Dictionary<int, Frame> Frames = new Dictionary<int, Frame>();
        }
        private struct Frame { public byte[] Gray; public int W, H; }
        private readonly List<Slot> _pending = new List<Slot>();   // 開啟中的時間槽（依 Anchor 排序，小）
        private readonly HashSet<int> _seenCams = new HashSet<int>(); // 曾出過幀的相機（判「滿槽」用）
        private readonly long[] _perCamLastTick;     // 各相機上一幀 tick（估週期）
        private long _periodTicks;                    // 線上估計幀週期（運行最小 delta）
        private long _maxTickSeen;                    // watermark

        // ── 背景寫入佇列（compose 在 lock 內輕量；memcpy 在背景）──
        private readonly Queue<BandJob> _writeQueue = new Queue<BandJob>();
        private bool _writerRunning;
        private sealed class BandJob
        {
            public int FullW, BandH, BandStartRow;
            public bool Ring;
            public List<Span> Spans;
        }
        private struct Span { public byte[] Src; public int Sw, Sh, DestX, SrcLeft, SrcWidth; }

        private int _defaultFrameW = 16384;           // 尚無幀的槽位用此寬度排佈局 → 7 槽寬度穩定
        private double[] _startMm;
        private double _refOpsMm = 0.024;
        private bool _disposed;
        private volatile bool _virtualSet;
        private int _diagLog;

        public WaterfallView(Panel host, int camCount, int totalHeight, WaterfallFullMode fullMode,
            double screenMmPerPx = 0)
        {
            _camCount = Math.Max(1, camCount);
            _totalHeight = Math.Max(1000, totalHeight);
            _fullMode = fullMode;
            _screenMmPerPx = screenMmPerPx;
            _perCamLastTick = new long[_camCount];

            _canvas = new SmartCanvas { Dock = DockStyle.Fill, BackColor = Color.Black };
            _canvas.ShowOverlay = true;               // 游標座標 + 亮度
            _canvas.FitRelativeZoom = true;           // 滾輪相對 fit 縮放（fit=1×，與 live/review 一致）
            _canvas.DoubleClickFitToScreen = true;    // 點兩下 fit 整張
            _canvas.TripleClickPhysical1x = true;     // 三擊實體 1:1（需 mm 校正；SetLayout 後設）
            host.Controls.Add(_canvas);
            _canvas.BringToFront();
            _canvas.EnableLod(1, 1, ProvideRegion);

            _flushTimer = new System.Windows.Forms.Timer { Interval = 30 };
            _flushTimer.Tick += (s, e) => { TryFlush(_clock.ElapsedMilliseconds); PushLodRefresh(); };
            _flushTimer.Start();
        }

        /// <summary>合圖佈局（各台 start mm + 基準像素尺寸 mm/px）。對齊 live 全域合圖；opsUm 目前未用（保留簽名相容）。</summary>
        public void SetLayout(double[] startMm, double[] opsUm, double refOpsMm)
        {
            lock (_lock)
            {
                _startMm = startMm;
                if (refOpsMm > 0) _refOpsMm = refOpsMm;
            }
            if (_screenMmPerPx > 0 && refOpsMm > 0)
                try { _canvas.SetPhysicalCalibration(refOpsMm, _screenMmPerPx); } catch { }
        }

        /// <summary>各相機每幀（MIL hook 多執行緒）：複製幀 + 用硬體 tick 歸入時間槽（同瞬間聚一起）。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h, long tick)
        {
            if (_disposed || camId < 1 || camId > _camCount || gray == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(gray, copy, Math.Min(gray.Length, n));
            long nowMs = _clock.ElapsedMilliseconds;
            lock (_lock)
            {
                _seenCams.Add(camId);
                if (w > _defaultFrameW) _defaultFrameW = w;

                // 線上週期估計：運行最小正 delta（掉幀放大 delta、不縮小 → min 抗掉幀；下限 0.5× 防 glitch）
                long last = _perCamLastTick[camId - 1];
                if (tick > 0 && last > 0 && tick > last)
                {
                    long d = tick - last;
                    if (_periodTicks == 0) _periodTicks = d;
                    else if (d < _periodTicks && d >= _periodTicks / 2) _periodTicks = d;
                }
                if (tick > 0) _perCamLastTick[camId - 1] = tick;
                if (tick > _maxTickSeen) _maxTickSeen = tick;

                long thr = FrameTickIndex.ComputeThreshold(_periodTicks);

                // 找同槽：tick 在某槽 anchor 半週期內（週期未知→wall-clock 視窗），且該槽尚無此相機
                Slot target = null;
                foreach (var s in _pending)
                {
                    if (s.Frames.ContainsKey(camId)) continue;
                    bool same = thr > 0 ? Math.Abs(tick - s.Anchor) <= thr
                                        : (nowMs - s.LastWallMs) <= BootstrapGraceMs;
                    if (same) { target = s; break; }
                }
                if (target == null)
                {
                    target = new Slot { Anchor = tick > 0 ? tick : _maxTickSeen, MaxTick = tick, LastWallMs = nowMs };
                    _pending.Add(target);
                    _pending.Sort((a, b) => a.Anchor.CompareTo(b.Anchor));
                }
                target.Frames[camId] = new Frame { Gray = copy, W = w, H = h };
                if (tick > target.MaxTick) target.MaxTick = tick;
                if (tick > 0 && tick < target.Anchor) target.Anchor = tick;
                target.LastWallMs = nowMs;
            }
            TryFlush(nowMs);
        }

        // 把「已完整 / 已過 grace / 已被下一瞬間超越 / 滯留太久」的槽（依 Anchor 由舊到新）送去寫成 band。
        private void TryFlush(long nowMs)
        {
            List<Slot> ready = null;
            lock (_lock)
            {
                long thr = FrameTickIndex.ComputeThreshold(_periodTicks);
                while (_pending.Count > 0)
                {
                    var s = _pending[0];
                    bool full = _seenCams.Count > 0 && s.Frames.Count >= _seenCams.Count;
                    bool nextInstant = thr > 0 && (_maxTickSeen - s.MaxTick > thr);
                    bool graced = (nowMs - s.LastWallMs) >= HoldGraceMs;
                    bool stale = (nowMs - s.LastWallMs) >= StaleFlushMs;
                    // 滿槽或「已證明進入下一瞬間」都要等 grace（接同瞬間晚到的幀）；滯留太久則無條件 flush。
                    bool flush = stale || ((full || nextInstant) && graced);
                    if (!flush) break;
                    _pending.RemoveAt(0);
                    (ready ?? (ready = new List<Slot>())).Add(s);
                }
                if (ready != null)
                    foreach (var s in ready)
                    {
                        var job = ComposeJob(s);   // 在 lock 內：算佈局 + 推進寫頭（原子、輕量）
                        if (job != null) _writeQueue.Enqueue(job);
                    }
            }
            KickWriter();
        }

        // 在 lock 內：用槽內各相機幀算 7 槽佈局 + 推進寫頭 + 配分塊。回 BandJob 給背景寫 memcpy。
        private BandJob ComposeJob(Slot slot)
        {
            if (_startMm == null || _refOpsMm <= 0) { DiagNull("startMm/refOps 未備（未餵佈局）"); return null; }

            // band 高 = 槽內幀最大高度
            int bandH = 0;
            foreach (var kv in slot.Frames) if (kv.Value.H > bandH) bandH = kv.Value.H;
            if (bandH == 0) return null;

            // 7 槽全排：所有配置相機進佈局（沒幀的槽用學到的寬或 _defaultFrameW）→ fullW = 完整 7 台寬
            double minStart = double.MaxValue;
            for (int i = 0; i < _camCount; i++)
            {
                double sm = i < _startMm.Length ? _startMm[i] : 0;
                if (sm < minStart) minStart = sm;
            }
            if (minStart == double.MaxValue) return null;

            var cams = new List<MergeLayout.CamGeom>(_camCount);
            for (int i = 0; i < _camCount; i++)
            {
                int wpx = slot.Frames.TryGetValue(i + 1, out var f) ? f.W : _defaultFrameW;
                cams.Add(new MergeLayout.CamGeom { CameraId = i + 1, StartMm = i < _startMm.Length ? _startMm[i] : 0, WidthPx = wpx });
            }
            var places = MergeLayout.Compute(cams, minStart, _refOpsMm, 1, MergeOverlap.Midline, out int fullW);
            if (fullW <= 0) return null;

            // fullW 變了（佈局改）或首次 → (重)配分塊 + 重置寫頭
            if (_chunks == null || _fullW != fullW)
            {
                _fullW = fullW;
                int nChunks = (_totalHeight + ChunkRows - 1) / ChunkRows;
                _chunks = new byte[nChunks][];
                _writeRow = 0; _virtualSet = false;
            }

            // 填滿處理 + band 起始 row（原子）
            bool ring = _fullMode == WaterfallFullMode.Ring;
            if (!ring && _writeRow + bandH > _totalHeight)
            {
                for (int i = 0; i < _chunks.Length; i++) _chunks[i] = null; // Restart：滿 → 黑幕重來
                _writeRow = 0;
            }
            int bandStart = _writeRow;

            // 槽內有幀的相機 → 擺放 span；沒幀的槽不放 → 那欄留黑（掉偵補黑）
            var spans = new List<Span>();
            foreach (var p in places)
            {
                int i = p.CameraId - 1;
                if (i < 0 || i >= _camCount || p.SrcWidth <= 0) continue;
                if (!slot.Frames.TryGetValue(p.CameraId, out var f)) continue;
                spans.Add(new Span { Src = f.Gray, Sw = f.W, Sh = f.H, DestX = p.DestX, SrcLeft = p.SrcLeft, SrcWidth = p.SrcWidth });
            }

            if (ring) _writeRow = (_writeRow + bandH) % _totalHeight;
            else _writeRow += bandH;

            if (_diagLog < 8)
            {
                _diagLog++;
                System.Diagnostics.Trace.WriteLine($"[Waterfall] band cams={slot.Frames.Count}/{_seenCams.Count} period={_periodTicks} fullW={fullW} bandH={bandH} start={bandStart} mode={_fullMode}");
            }
            return new BandJob { FullW = fullW, BandH = bandH, BandStartRow = bandStart, Ring = ring, Spans = spans };
        }

        private void KickWriter()
        {
            lock (_lock)
            {
                if (_writerRunning || _writeQueue.Count == 0) return;
                _writerRunning = true;
            }
            Task.Run(() =>
            {
                try
                {
                    while (true)
                    {
                        BandJob job;
                        lock (_lock)
                        {
                            if (_disposed || _writeQueue.Count == 0) { _writerRunning = false; return; }
                            job = _writeQueue.Dequeue();
                        }
                        WriteBand(job);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.TraceWarning($"[Waterfall.Writer] {ex.GetType().Name}: {ex.Message}");
                    lock (_lock) _writerRunning = false;
                }
            });
        }

        // 背景：把 band 逐行寫進分塊儲存（每行短持鎖，讓 PushFrame/provider 能交錯）。
        private void WriteBand(BandJob job)
        {
            int fullW = job.FullW;
            for (int y = 0; y < job.BandH; y++)
            {
                if (_disposed) return;
                int gy = job.Ring ? (job.BandStartRow + y) % _totalHeight : (job.BandStartRow + y);
                if (gy < 0 || gy >= _totalHeight) break; // Restart clamp
                int ci = gy / ChunkRows, off = gy % ChunkRows;
                lock (_lock)
                {
                    if (_chunks == null || _fullW != fullW) return; // 佈局已換 → 放棄這份 job
                    var chunk = _chunks[ci];
                    if (chunk == null) { chunk = new byte[fullW * ChunkRows]; _chunks[ci] = chunk; }
                    int rowBase = off * fullW;
                    Array.Clear(chunk, rowBase, fullW); // 黑底（補黑 + 槽間空隙 + Ring 覆蓋舊內容）
                    foreach (var s in job.Spans)
                    {
                        if (y >= s.Sh) continue;
                        int sx = s.SrcLeft, dx = s.DestX, cw = s.SrcWidth;
                        if (sx < 0) { dx -= sx; cw += sx; sx = 0; }
                        if (dx < 0) { sx -= dx; cw += dx; dx = 0; }
                        if (sx + cw > s.Sw) cw = s.Sw - sx;
                        if (dx + cw > fullW) cw = fullW - dx;
                        if (cw <= 0) continue;
                        Array.Copy(s.Src, y * s.Sw + sx, chunk, rowBase + dx, cw);
                    }
                }
            }
            // LOD 刷新（含首次設虛擬尺寸）由 flushTimer 的 PushLodRefresh 在 UI 執行緒做，避免每 band BeginInvoke。
        }

        // UI 執行緒（flushTimer）：固定虛擬尺寸一次 + 刷 LOD（內容變、視角不變）。
        private void PushLodRefresh()
        {
            if (_disposed || _fullW <= 0) return;
            if (!_virtualSet) { _canvas.UpdateLodVirtualSize(_fullW, _totalHeight); _virtualSet = true; }
            else _canvas.RefreshLod();
        }

        // SmartCanvas LOD provider：給虛擬區域 r（全解析座標）+ dest 大小 → 邊讀邊降採樣（nearest）出 dest 大小 bitmap。
        private Bitmap ProvideRegion(Rectangle r, Size dest)
        {
            int dw = dest.Width, dh = dest.Height;
            if (dw <= 0 || dh <= 0) return null;
            int rw = Math.Max(1, r.Width), rh = Math.Max(1, r.Height);
            byte[] outp;
            int seamDestY = -1;
            lock (_lock)
            {
                if (_chunks == null || _fullW <= 0) return null;
                outp = new byte[dw * dh]; // 黑底
                for (int dy = 0; dy < dh; dy++)
                {
                    long sy = r.Y + (long)dy * rh / dh;
                    if (sy < 0 || sy >= _totalHeight) continue;
                    int ci = (int)(sy / ChunkRows), off = (int)(sy % ChunkRows);
                    if (ci < 0 || ci >= _chunks.Length) continue;
                    var chunk = _chunks[ci];
                    if (chunk == null) continue;
                    int rowBase = off * _fullW;
                    int orow = dy * dw;
                    for (int dx = 0; dx < dw; dx++)
                    {
                        long sx = r.X + (long)dx * rw / dw;
                        if (sx < 0 || sx >= _fullW) continue;
                        outp[orow + dx] = chunk[rowBase + (int)sx];
                    }
                }
                if (_fullMode == WaterfallFullMode.Ring)   // Ring 接縫：寫頭畫亮掃描線 → 一看就知道在循環
                {
                    long hy = _writeRow;
                    seamDestY = (int)((hy - r.Y) * dh / rh);
                }
            }
            if (seamDestY >= 0 && seamDestY < dh)
            {
                int orow = seamDestY * dw;
                for (int dx = 0; dx < dw; dx++) outp[orow + dx] = 255;
            }
            return GrayBitmap.From(outp, dw, dh);
        }

        // 節流診斷：沒出 band 的原因 → 沒畫面時看 trace 判斷。
        private void DiagNull(string reason)
        {
            if (_diagLog >= 8) return;
            _diagLog++;
            System.Diagnostics.Trace.WriteLine($"[Waterfall] no band: {reason}");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _flushTimer.Stop(); _flushTimer.Dispose(); } catch { }
            try { _canvas.DisableLod(); } catch { }
            try { if (_canvas.Parent != null) _canvas.Parent.Controls.Remove(_canvas); _canvas.Dispose(); } catch { }
            lock (_lock) { _chunks = null; _fullW = 0; _writeRow = 0; _pending.Clear(); _writeQueue.Clear(); }
        }
    }
}
