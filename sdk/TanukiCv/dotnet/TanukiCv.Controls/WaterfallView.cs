using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;
using System.Windows.Forms;
using TanukiCv.Core;       // MergeLayout / MergeOverlap

namespace TanukiCv.Controls
{
    public enum WaterfallFrameLayer
    {
        Raw = 0,
        Column = 1,
        Row = 2
    }

    /// <summary>
    /// 監控主畫面「瀑布圖」：全幅 7 相機合圖每幀往下接、即時捲動（線掃像印表機吐紙）。
    ///
    /// 儲存（2026-06-24 全解析重寫）：全解析**分塊**存（chunk，避開 2GB byte[] 上限），LOD 只在顯示時降採樣。
    /// 虛擬長圖 = 全解析合圖寬 `_fullW`（7 槽全排、沒畫面補黑）× 固定總高 `_totalHeight`（預設 30000，LOD 之前的全解析 Y）。
    /// 填滿：Restart 重來＝清黑幕從頂重畫；Ring 循環＝繞回頂端覆蓋最舊、顯示時寫頭畫亮掃描線接縫。
    ///
    /// 跨相機幀對齊（2026-06-25 定版＝**tick 網格錨定**）：
    ///   每幀獨立算 `seq = round((tick - origin) / period)` → 同一物理掃描的各台幀（tick 差＝相位 φ，同板實測 ~6 萬≪半週期）
    ///   落同一格 seq → 放同一條 band；某台在某格沒幀＝該欄補黑。對同步相機穩定不抖（consecutive 同台幀差＝1 週期、恰好落格）。
    ///   tick 用途：①估週期＝運行**最小** delta（同台連續幀差＝1 週期；掉幀/gap 只放大 delta → min 取真週期；
    ///   **不可加下限守門**，否則先看到 gap 當種子會誤拒真週期）；②錨定 seq。
    ///   **延後 bootstrap**：週期學到前先緩衝幀，學到當下設原點(緩衝最小 tick)+ 依 tick 一次入槽，並**丟棄重建殘留的脫隊舊幀**
    ///   （比最新還舊 >2 週期）→ 修「grab 中切模式重建 WaterfallView 時，某台第一幀是幾個週期前的舊幀 → bootstrap 硬塞同 seq → 整條歪 N 格」。
    ///   φ 只造成固定垂直小偏移（非掉幀），不再有假黑欄/錯位。
    ///
    /// ⚠ 跨板限制：cam1-4(板0)/cam5-7(板1) tick epoch 不同、不可跨板相減；7 台跨板需各板自錨（同板自估 period+origin）。
    /// </summary>
    public sealed class WaterfallView : IDisposable
    {
        private const int ChunkRows = 512;          // 每塊高度（row）；全幅~101000 → 每塊 ~51MB，30000 高約 60 塊
        private const long StaleBaseMs = 1500;      // 滯留槽 flush 保險（週期 wall 未知時）；正常由序號 watermark flush
        private const long JoinGraceMs = 150;       // slot 建立後至少等這麼久才憑 complete flush（讓晚到/剛上線的相機加入同 seq，防啟動錯位）
        private const long StabilizeMs = 1500;      // 相機集合穩定這麼久後恢復即時 flush（不再每 band 等 grace，降延遲）

        private static readonly Stopwatch _clock = Stopwatch.StartNew();

        private readonly ImageCanvas _canvas;
        private readonly int _camCount;             // 配置槽數（7，含黑布槽）
        private readonly int _totalHeight;
        private readonly WaterfallFullMode _fullMode;
        private readonly double _screenMmPerPx;
        private readonly System.Windows.Forms.Timer _flushTimer; // 安全網：flush 滯留槽 + 推 LOD 刷新
        private readonly object _lock = new object();
        private bool _flipVertical;
        private double _rowPitchMm;

        // 三種顯示共用同一套時間軸與寫頭；各層分塊 lazy 配置，切換只換讀取層，不清歷史。
        // _layerChunks[layer][ci] = byte[_fullW * ChunkRows]（null=黑）。
        private byte[][][] _layerChunks;
        private WaterfallFrameLayer _displayLayer = WaterfallFrameLayer.Raw;
        private int _fullW;
        private int _writeRow;                       // 下一個 band 寫入起始 row（也是 Ring 顯示接縫位置）
        private int _lastStateLogMs;                 // 狀態快照節流（每秒一行）

        /// <summary>狀態快照（改A壞B即時可抓）：占用列/總高＋最新內容渲染在畫面哪端（方向可判量）。</summary>
        private void FlowState()
        {
            if (FlowLog == null) return;
            int now = Environment.TickCount;
            if (now - _lastStateLogMs < 1000) return;
            _lastStateLogMs = now;
            FlowLog($"state 占用=0~{_writeRow}/{_totalHeight} 最新內容畫面端={(_flipVertical ? "底" : "頂")}");
        }


        // ── 跨相機序號配對（串流）──
        private sealed class Slot
        {
            public long Seq;                         // 全域 band 序號（各台 seq 對齊到此格）
            public long FirstWallMs, LastWallMs;     // 診斷：槽存活時間
            public string FlushReason = "?";
            public readonly Dictionary<int, Frame> Frames = new Dictionary<int, Frame>();
        }
        private struct Frame
        {
            public byte[] Raw, Column, Row;
            public int W, H;
            public long Tick;
        }
        private struct BufFrame { public int Cam; public Frame F; }
        private readonly List<BufFrame> _preBuffer = new List<BufFrame>(); // 週期學到前緩衝（學到才依 tick 入槽）
        private readonly SortedDictionary<long, Slot> _pending = new SortedDictionary<long, Slot>(); // 開啟槽（依 seq）
        private readonly HashSet<int> _seenCams = new HashSet<int>(); // 曾出過幀的相機
        private readonly long[] _perCamLastTick;     // 各相機上一幀 tick（估週期 + 偵掉幀）
        private readonly long[] _perCamLastWall;     // 各相機上一幀 wall-clock（估週期 wall）
        private readonly long[] _perCamSeq;          // 各相機目前 seq（watermark；初值 -1=尚無幀）
        private long _periodTicks;                    // 線上估計幀週期（運行最小 tick delta）
        private long _periodWallMs;                   // 線上估計幀週期（運行最小 wall delta，算 stale）
        private long _originTick;                     // 序號原點＝第一幀 tick
        private bool _originSet;
        private long _lastNewCamWallMs;               // 最後一台新相機被發現的 wall-clock（判相機集合是否穩定）

        // ── 背景寫入佇列（compose 在 lock 內輕量；memcpy 在背景）──
        private readonly Queue<BandJob> _writeQueue = new Queue<BandJob>();
        private bool _writerRunning;
        private int _contentGeneration;
        private sealed class BandJob
        {
            public int Generation, FullW, BandH, BandStartRow;
            public bool Ring;
            public List<Span> Spans;
        }
        private struct Span
        {
            public byte[] Raw, Column, Row;
            public int Sw, Sh, DestX, SrcLeft, SrcWidth;
        }

        private int _defaultFrameW = 16384;           // 尚無幀的槽位用此寬度排佈局 → 7 槽寬度穩定
        private double[] _startMm;
        private double _refOpsMm = 0.024;
        private readonly List<CameraPlacement> _cameraPlacements = new List<CameraPlacement>();
        private CanvasInfo _lastCanvasInfo;
        private bool _hasCanvasInfo;
        private bool _disposed;
        private volatile bool _virtualSet;
        private volatile bool _lodContentDirty;
        private long _awaitedLodGeneration;

        public event Action<int> SelectRequested;
        public event Action<double, double, double, double> ViewRangeMmChanged;
        public event Action<ImageDisplayView.CursorStatus> CursorStatusChanged;
        public event Action ContentPresented;

        public ImageCanvas Canvas => _canvas;

        public Func<string> InformationTextProvider
        {
            get => _canvas.InformationTextProvider;
            set => _canvas.InformationTextProvider = value;
        }

        public bool FlipVertical
        {
            get => _flipVertical;
            set
            {
                if (_flipVertical == value) return;
                _flipVertical = value;
                _lodContentDirty = true;
                PushLodRefresh();
                RefireViewRange();
            }
        }

        public WaterfallView(Panel host, int camCount, int totalHeight, WaterfallFullMode fullMode,
            double screenMmPerPx = 0)
        {
            _camCount = Math.Max(1, camCount);
            _totalHeight = Math.Max(1000, totalHeight);
            _fullMode = fullMode;
            _screenMmPerPx = screenMmPerPx;
            _perCamLastTick = new long[_camCount];
            _perCamLastWall = new long[_camCount];
            _perCamSeq = new long[_camCount];
            for (int i = 0; i < _camCount; i++) _perCamSeq[i] = -1;

            _canvas = new ImageCanvas { Dock = DockStyle.Fill, BackColor = Color.Black };
            _canvas.CameraFrameRegionsProvider = GetCameraFrameRegions;
            _canvas.ShowOverlay = true;               // 游標座標 + 亮度
            _canvas.FitRelativeZoom = false;          // 監控允許縮到 fit 以下，與即時／回顧畫布一致
            _canvas.DoubleClickFitToScreen = true;    // 點兩下 fit 整張
            _canvas.TripleClickPhysical1x = true;     // 三擊實體 1:1（需 mm 校正；SetLayout 後設）
            host.Controls.Add(_canvas);
            _canvas.BringToFront();
            _canvas.StatusChanged += OnCanvasStatus;
            _canvas.LodTileApplied += OnLodTileApplied;
            _canvas.MouseClick += OnCanvasMouseClick;

            _flushTimer = new System.Windows.Forms.Timer { Interval = 30 };
            _flushTimer.Tick += (s, e) =>
            {
                TryFlush(_clock.ElapsedMilliseconds);
                PushLodRefresh();
                // 縮圖高亮反向連動的快拖補刷（同 ImageDisplayView.UpdateReverseThumbSync 的 33ms 保險）：
                // StatusChanged 互動事件限流/合併時，中心相機變更不被跳過。計算極便宜（找 placement）。
                UpdateCenterCam(_canvas.Zoom, _canvas.PanOffset);
            };
            _flushTimer.Start();
        }

        /// <summary>合圖佈局（各台 start mm + 基準像素尺寸 mm/px）。對齊 live 全域合圖；
        /// 在首幀前即建立由設定決定的黑底座標畫布，使 grab 前後使用同一組視野。</summary>
        public void SetLayout(double[] startMm, double[] opsUm, double refOpsMm)
        {
            int layoutW;
            bool sizeChanged;
            double calibrationOps;
            lock (_lock)
            {
                _startMm = startMm;
                if (refOpsMm > 0) _refOpsMm = refOpsMm;
                calibrationOps = _refOpsMm;
                layoutW = RebuildCameraPlacementsLocked();
                sizeChanged = layoutW > 0 && (_layerChunks == null || _fullW != layoutW);
                if (sizeChanged)
                {
                    _fullW = layoutW;
                    _layerChunks = CreateLayerChunks();
                    _writeRow = 0;
                }
            }

            if (_screenMmPerPx > 0 && calibrationOps > 0)
                try { _canvas.SetPhysicalCalibration(calibrationOps, _screenMmPerPx); } catch { }
            if (layoutW <= 0) return;

            if (!_virtualSet || !_canvas.LodActive)
            {
                _virtualSet = true;
                _canvas.EnableLod(layoutW, _totalHeight, ProvideRegion);
            }
            else if (sizeChanged)
            {
                _canvas.UpdateLodVirtualSize(layoutW, _totalHeight);
            }
            else
            {
                // START/OPS may change without changing the pixel width; republish the same view with new physical coordinates.
                _canvas.SetView(_canvas.Zoom, _canvas.PanOffset);
            }
        }

        private IReadOnlyList<RectangleF> GetCameraFrameRegions()
        {
            lock (_lock)
            {
                var regions = new List<RectangleF>(_cameraPlacements.Count);
                foreach (CameraPlacement placement in _cameraPlacements)
                {
                    regions.Add(new RectangleF(
                        placement.DestX,
                        0,
                        Math.Max(1, placement.SrcWidth),
                        _totalHeight));
                }
                return regions;
            }
        }

        /// <summary>只改 LOD tile 調色盤；累積的三層 8-bit 資料不轉換、不重建。</summary>
        public IntensityColorMap ColorMap
        {
            get => _colorMap;
            set
            {
                if (_colorMap == value) return;
                _colorMap = value;
                _canvas.BrightnessSelector = GrayBitmap.GetBrightnessSelector(value);
                _lodContentDirty = true;
                PushLodRefresh();
            }
        }
        private IntensityColorMap _colorMap = IntensityColorMap.Grayscale;

        public WaterfallFrameLayer DisplayLayer => _displayLayer;

        /// <summary>切換原圖／欄強化／列強化；保留累積內容、寫頭、tick 對齊與目前視野。</summary>
        public void SetDisplayLayer(WaterfallFrameLayer layer)
        {
            if (_disposed) return;
            WaterfallFrameLayer previous;
            int writeRow;
            lock (_lock)
            {
                if (_displayLayer == layer) return;
                previous = _displayLayer;
                _displayLayer = layer;
                writeRow = _writeRow;
            }
            FlowLog?.Invoke(
                $"layer {previous.ToString().ToLowerInvariant()}->{layer.ToString().ToLowerInvariant()} " +
                $"writeRow={writeRow} history=preserved");
            _lodContentDirty = true;
            PushLodRefresh();
        }

        /// <summary>Set material-direction row pitch so waterfall Y range matches the live row chart.</summary>
        public void SetRowPitch(double mmPerRow)
        {
            if (mmPerRow <= 0) return;
            _rowPitchMm = mmPerRow;
            RefireViewRange();
        }

        /// <summary>Re-publish the current visible range without waiting for mouse movement.</summary>
        public void RefireViewRange()
        {
            if (TryComputeViewRange(_canvas.Zoom, _canvas.PanOffset, out double leftMm, out double rightMm, out double topMm, out double botMm))
                ViewRangeMmChanged?.Invoke(leftMm, rightMm, topMm, botMm);
            UpdateCenterCam(_canvas.Zoom, _canvas.PanOffset);
        }

        /// <summary>視野中心所在相機（1-based）變更時觸發（pan/zoom/置中）。縮圖高亮反向連動用；
        /// 程式化來源，上層只需更新高亮、勿再呼 CenterOnCamera（防遞迴）。</summary>
        public event Action<int> CenterCamChanged;

        /// <summary>互動流跡（診斷用，可為 null）：wheel 手勢（轉發自 ImageCanvas）供上層流程契約驗證。</summary>
        public Action<string> FlowLog
        {
            get => _canvas?.FlowLog;
            set { if (_canvas != null) _canvas.FlowLog = value; }
        }
        private int _lastCenterCamId = -1;

        /// <summary>視野水平置中到指定相機（1-based）欄位中心；保持縮放與垂直位置（縮圖點選→主畫面連動）。</summary>
        public void CenterOnCamera(int camId)
        {
            if (_disposed || _canvas == null) return;
            bool found = false;
            CameraPlacement hit = default;
            lock (_lock)
            {
                foreach (var p in _cameraPlacements)
                    if (p.CameraId == camId) { hit = p; found = true; break; }
            }
            if (!found) return;
            float zoom = _canvas.Zoom;
            if (zoom <= 0) return;
            float centerX = hit.DestX + Math.Max(1, hit.SrcWidth) / 2f;
            _canvas.SetView(zoom, new PointF(_canvas.Width / 2f - centerX * zoom, _canvas.PanOffset.Y));
            RefireViewRange();
        }

        private void UpdateCenterCam(float zoom, PointF pan)
        {
            if (_disposed || !_virtualSet || _canvas == null || zoom <= 0) return;
            int imageX = (int)((_canvas.Width / 2f - pan.X) / zoom);
            int camId = ResolveCameraAtX(imageX);
            if (camId > 0 && camId != _lastCenterCamId)
            {
                _lastCenterCamId = camId;
                CenterCamChanged?.Invoke(camId);
            }
        }

        /// <summary>重 grab：清掉舊瀑布內容 + 重置對齊狀態（origin/period/seq/pending/緩衝），下次幀重新 bootstrap。
        /// 重 grab 時呼叫 → 舊圖清空（符合預期）+ 避免新幀接在舊網格上、兩台重啟相位不一而錯位。</summary>
        public void Reset()
        {
            lock (_lock)
            {
                _contentGeneration++;
                ClearLayerChunksLocked();
                _writeRow = 0;
                _pending.Clear(); _preBuffer.Clear(); _writeQueue.Clear();
                _seenCams.Clear();
                for (int i = 0; i < _camCount; i++) { _perCamLastTick[i] = 0; _perCamLastWall[i] = 0; _perCamSeq[i] = -1; }
                _periodTicks = 0; _periodWallMs = 0; _originTick = 0; _originSet = false; _lastNewCamWallMs = 0;
            }
            _lodContentDirty = true;
            PushLodRefresh();
        }

        /// <summary>各相機每幀（MIL hook 多執行緒）：複製幀 + tick 網格錨定歸入 band（同掃描同 seq；缺幀補黑）。</summary>
        public void PushFrame(int camId, byte[] gray, int w, int h, long tick)
            => PushFrameVariants(camId, gray, gray, gray, w, h, tick);

        /// <summary>
        /// 同一物理幀的原圖、欄強化與列強化。三層只做一次 tick 對齊並落在同一個 band；
        /// 傳入陣列會在回呼內同步複製，呼叫端可安全重用緩衝。
        /// </summary>
        public void PushFrameVariants(
            int camId, byte[] raw, byte[] column, byte[] row, int w, int h, long tick)
        {
            if (_disposed || camId < 1 || camId > _camCount || raw == null || w <= 0 || h <= 0) return;
            int n = w * h;
            byte[] rawCopy = CopyFrame(raw, n);
            byte[] columnCopy = ReferenceEquals(column, raw) ? rawCopy : CopyFrame(column ?? raw, n);
            byte[] rowCopy = ReferenceEquals(row, raw) ? rawCopy
                : ReferenceEquals(row, column) ? columnCopy
                : CopyFrame(row ?? raw, n);
            long nowMs = _clock.ElapsedMilliseconds;
            lock (_lock)
            {
                if (_seenCams.Add(camId)) _lastNewCamWallMs = nowMs; // 新相機被發現 → 重置穩定計時（啟動期給 join grace）
                if (w > _defaultFrameW)
                {
                    _defaultFrameW = w;
                    RebuildCameraPlacementsLocked();
                }
                int ci = camId - 1;

                long lastTick = _perCamLastTick[ci];
                long lastWall = _perCamLastWall[ci];

                // 週期估計：運行最小正 delta（同台連續幀差＝1 個週期；掉幀/gap 只放大 delta → min 取到真週期）。
                // ★ 不可加「d>=period/2」下限：若先看到多週期 gap 當種子，真週期(較小)會被誤拒 → period 永遠錯。
                if (tick > 0 && lastTick > 0 && tick > lastTick)
                {
                    long d = tick - lastTick;
                    if (_periodTicks == 0 || d < _periodTicks) _periodTicks = d;
                }
                if (lastWall > 0 && nowMs > lastWall)
                {
                    long dw = nowMs - lastWall;
                    if (dw >= 1) _periodWallMs = _periodWallMs == 0 ? dw : Math.Min(_periodWallMs, dw);
                }

                if (tick > 0) _perCamLastTick[ci] = tick;
                _perCamLastWall[ci] = nowMs;

                var f = new Frame
                {
                    Raw = rawCopy,
                    Column = columnCopy,
                    Row = rowCopy,
                    W = w,
                    H = h,
                    Tick = tick
                };
                // tick 網格錨定：每幀獨立 seq=round((tick-origin)/period)（同步相機 φ≪半週期→同掃描同 seq）。
                // 週期未學到前先緩衝；學到當下設原點 + 把緩衝依 tick 一次入槽（並丟棄重建殘留的脫隊舊幀）。
                if (_periodTicks <= 0) _preBuffer.Add(new BufFrame { Cam = camId, F = f });
                else if (!_originSet) { _preBuffer.Add(new BufFrame { Cam = camId, F = f }); DrainPreBuffer(); }
                else PlaceFrame(camId, f);
            }
            TryFlush(nowMs);
        }

        private static byte[] CopyFrame(byte[] source, int length)
        {
            var copy = new byte[length];
            Array.Copy(source, copy, Math.Min(source.Length, length));
            return copy;
        }

        // tick 網格錨定 → seq → 入槽。watermark _perCamSeq 取最新（最大）。
        private void PlaceFrame(int camId, Frame f)
        {
            int ci = camId - 1;
            long seq = f.Tick > 0 && _periodTicks > 0
                ? (long)Math.Round((double)(f.Tick - _originTick) / _periodTicks)
                : (_perCamSeq[ci] < 0 ? 0 : _perCamSeq[ci] + 1); // tick 無效時退回序號 +1
            if (seq > _perCamSeq[ci]) _perCamSeq[ci] = seq;
            long now = _clock.ElapsedMilliseconds;
            if (!_pending.TryGetValue(seq, out var slot))
            {
                slot = new Slot { Seq = seq, FirstWallMs = now };
                _pending[seq] = slot;
            }
            slot.Frames[camId] = f; // 同 seq 同 cam 只留一幀（防呆覆蓋）
            slot.LastWallMs = now;
        }

        // 週期到手 → 設原點（緩衝最小 tick，但丟棄比最新還舊 >2 週期的脫隊幀＝重建殘留）+ 把緩衝依 tick 入槽。
        private void DrainPreBuffer()
        {
            if (_preBuffer.Count == 0) { _originSet = _periodTicks > 0; return; }
            long maxT = long.MinValue;
            foreach (var b in _preBuffer) if (b.F.Tick > maxT) maxT = b.F.Tick;
            long cutoff = maxT - 2 * _periodTicks; // 丟棄重建時殘留的舊幀（比最新還舊 >2 週期）
            long minT = long.MaxValue;
            foreach (var b in _preBuffer) if (b.F.Tick >= cutoff && b.F.Tick < minT) minT = b.F.Tick;
            _originTick = minT == long.MaxValue ? maxT : minT;
            _originSet = true;
            foreach (var b in _preBuffer) if (b.F.Tick >= cutoff) PlaceFrame(b.Cam, b.F);
            _preBuffer.Clear();
        }

        // 依 seq 由小到大，把「每台 seen 相機都已在此 seq 或已推進過此 seq（=該台在這格沒幀=真掉幀）」的槽送去寫；
        // 滯留太久（相機停了）則 stale 強制 flush。某台缺幀那欄補黑。
        private void TryFlush(long nowMs)
        {
            List<Slot> ready = null;
            lock (_lock)
            {
                long stale = _periodWallMs > 0 ? Math.Max(750, _periodWallMs * 3) : StaleBaseMs;
                bool stable = (nowMs - _lastNewCamWallMs) >= StabilizeMs; // 相機集合穩定 → 即時 flush；否則啟動期給 join grace
                while (_pending.Count > 0)
                {
                    long firstSeq = FirstPendingSeq();
                    var s = _pending[firstSeq];
                    // complete：每台 seen 相機都在此 seq 或已越過；但啟動期（集合未穩）要等 join grace，
                    // 讓晚到/剛上線的相機加入同 seq（防 cam1 先到就把 slot flush 成單台、cam2 另開一條＝錯位一格）。
                    bool complete = _seenCams.Count > 0 && AllSeenInOrPast(s.Seq)
                                    && (stable || (nowMs - s.FirstWallMs) >= JoinGraceMs);
                    bool isStale = (nowMs - s.FirstWallMs) >= stale;
                    if (!(complete || isStale)) break;
                    s.FlushReason = complete ? "complete" : "stale";
                    _pending.Remove(firstSeq);
                    (ready ?? (ready = new List<Slot>())).Add(s);
                }
                if (ready != null)
                    foreach (var s in ready)
                    {
                        var job = ComposeJob(s);
                        if (job != null) _writeQueue.Enqueue(job);
                    }
            }
            KickWriter();
        }

        // 每台 seen 相機：要嘛此 seq 有幀，要嘛已推進過此 seq（_perCamSeq>seq＝它已產出更後面的幀→此格它沒幀=真掉幀）。
        private bool AllSeenInOrPast(long seq)
        {
            foreach (var c in _seenCams)
            {
                if (_pending.TryGetValue(seq, out var s) && s.Frames.ContainsKey(c)) continue;
                if (_perCamSeq[c - 1] > seq) continue; // 已越過 → 此格該台真掉幀（補黑）
                return false;                          // 還沒到（相位/處理慢）→ 等
            }
            return true;
        }

        private long FirstPendingSeq()
        {
            foreach (var k in _pending.Keys) return k; // SortedDictionary 依 key 升序
            return 0;
        }

        // 在 lock 內：用槽內各相機幀算 7 槽佈局 + 推進寫頭 + 配分塊。回 BandJob 給背景寫 memcpy。
        private BandJob ComposeJob(Slot slot)
        {
            if (_startMm == null || _refOpsMm <= 0) return null; // 未餵佈局

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

            if (_layerChunks == null || _fullW != fullW)
            {
                _fullW = fullW;
                _layerChunks = CreateLayerChunks();
                _writeRow = 0; _virtualSet = false;
            }

            bool ring = _fullMode == WaterfallFullMode.Ring;
            if (!ring && _writeRow + bandH > _totalHeight)
            {
                ClearLayerChunksLocked(); // Restart：滿 → 三層黑幕一起重來
                _writeRow = 0;
            }
            int bandStart = _writeRow;

            var spans = new List<Span>();
            foreach (var p in places)
            {
                int i = p.CameraId - 1;
                if (i < 0 || i >= _camCount || p.SrcWidth <= 0) continue;
                if (!slot.Frames.TryGetValue(p.CameraId, out var f)) continue; // 沒幀的槽不放 → 補黑
                spans.Add(new Span
                {
                    Raw = f.Raw,
                    Column = f.Column,
                    Row = f.Row,
                    Sw = f.W,
                    Sh = f.H,
                    DestX = p.DestX,
                    SrcLeft = p.SrcLeft,
                    SrcWidth = p.SrcWidth
                });
            }

            if (ring) _writeRow = (_writeRow + bandH) % _totalHeight;
            else _writeRow += bandH;
            FlowState();   // 狀態快照（每秒一行）：占用/總高+畫面端＝方向可判量

            return new BandJob
            {
                Generation = _contentGeneration,
                FullW = fullW,
                BandH = bandH,
                BandStartRow = bandStart,
                Ring = ring,
                Spans = spans
            };
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
                    if (job.Generation != _contentGeneration) return;
                    if (_layerChunks == null || _fullW != fullW) return; // 佈局已換 → 放棄這份 job
                    int rowBase = off * fullW;
                    for (int layerIndex = 0; layerIndex < _layerChunks.Length; layerIndex++)
                    {
                        byte[][] layerChunks = _layerChunks[layerIndex];
                        var chunk = layerChunks[ci];
                        if (chunk == null) { chunk = new byte[fullW * ChunkRows]; layerChunks[ci] = chunk; }
                        Array.Clear(chunk, rowBase, fullW); // 黑底（補黑 + 槽間空隙 + Ring 覆蓋舊內容）
                        foreach (var s in job.Spans)
                        {
                            if (y >= s.Sh) continue;
                            byte[] source = GetSpanSource(s, layerIndex);
                            if (source == null) continue;
                            int sx = s.SrcLeft, dx = s.DestX, cw = s.SrcWidth;
                            if (sx < 0) { dx -= sx; cw += sx; sx = 0; }
                            if (dx < 0) { sx -= dx; cw += dx; dx = 0; }
                            if (sx + cw > s.Sw) cw = s.Sw - sx;
                            if (dx + cw > fullW) cw = fullW - dx;
                            if (cw <= 0) continue;
                            Array.Copy(source, y * s.Sw + sx, chunk, rowBase + dx, cw);
                        }
                    }
                }
            }
            _lodContentDirty = true;
            // LOD 刷新由 flushTimer 的 PushLodRefresh 在 UI 執行緒做，避免每 band BeginInvoke。
        }

        // UI 執行緒（flushTimer）：固定虛擬尺寸一次 + 刷 LOD（內容變、視角不變）。
        private void PushLodRefresh()
        {
            if (_disposed || _fullW <= 0) return;
            if (!_virtualSet)
            {
                // EnableLod synchronously publishes CanvasInfo through FitToScreen.
                // This fallback is only used when the actual frame width differs from the configured 16384-pixel layout.
                _virtualSet = true;
                _canvas.EnableLod(_fullW, _totalHeight, ProvideRegion);
                _awaitedLodGeneration = _canvas.LodContentGeneration;
                _lodContentDirty = false;
            }
            else if (_lodContentDirty)
            {
                _lodContentDirty = false;
                _canvas.RefreshLod();
                _awaitedLodGeneration = _canvas.LodContentGeneration;
            }
        }

        private void OnLodTileApplied(long contentGeneration)
        {
            if (_awaitedLodGeneration <= 0 || contentGeneration < _awaitedLodGeneration)
                return;
            _awaitedLodGeneration = 0;
            try { ContentPresented?.Invoke(); }
            catch (System.Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning(
                    $"[WaterfallView.ContentPresented] {ex.GetType().Name}: {ex.Message}");
            }
        }

        // ImageCanvas LOD provider：給虛擬區域 r（全解析座標）+ dest 大小 → 邊讀邊降採樣（nearest）出 dest 大小 bitmap。
        private Bitmap ProvideRegion(Rectangle r, Size dest)
        {
            int dw = dest.Width, dh = dest.Height;
            if (dw <= 0 || dh <= 0) return null;
            int rw = Math.Max(1, r.Width), rh = Math.Max(1, r.Height);
            byte[] outp;
            int seamDestY = -1;
            lock (_lock)
            {
                if (_layerChunks == null || _fullW <= 0) return null;
                byte[][] chunks = _layerChunks[(int)_displayLayer];
                outp = new byte[dw * dh]; // 黑底
                for (int dy = 0; dy < dh; dy++)
                {
                    long sy = r.Y + (long)dy * rh / dh;
                    if (_flipVertical) sy = _totalHeight - 1 - sy;
                    if (sy < 0 || sy >= _totalHeight) continue;
                    int ci = (int)(sy / ChunkRows), off = (int)(sy % ChunkRows);
                    if (ci < 0 || ci >= chunks.Length) continue;
                    var chunk = chunks[ci];
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
                    long hy = _flipVertical ? (_totalHeight - 1 - _writeRow) : _writeRow;
                    seamDestY = (int)((hy - r.Y) * dh / rh);
                }
            }
            if (seamDestY >= 0 && seamDestY < dh)
            {
                int orow = seamDestY * dw;
                for (int dx = 0; dx < dw; dx++) outp[orow + dx] = 255;
            }
            return GrayBitmap.From(outp, dw, dh, false, _colorMap);
        }

        private byte[][][] CreateLayerChunks()
        {
            int nChunks = (_totalHeight + ChunkRows - 1) / ChunkRows;
            return new[] { new byte[nChunks][], new byte[nChunks][], new byte[nChunks][] };
        }

        private void ClearLayerChunksLocked()
        {
            if (_layerChunks == null) return;
            for (int layer = 0; layer < _layerChunks.Length; layer++)
                for (int i = 0; i < _layerChunks[layer].Length; i++)
                    _layerChunks[layer][i] = null;
        }

        private static byte[] GetSpanSource(Span span, int layerIndex)
        {
            if (layerIndex == (int)WaterfallFrameLayer.Column) return span.Column ?? span.Raw;
            if (layerIndex == (int)WaterfallFrameLayer.Row) return span.Row ?? span.Raw;
            return span.Raw;
        }

        private int RebuildCameraPlacementsLocked()
        {
            _cameraPlacements.Clear();
            if (_startMm == null || _startMm.Length == 0 || _refOpsMm <= 0) return 0;

            double minStart = double.MaxValue;
            for (int i = 0; i < _camCount; i++)
            {
                double sm = i < _startMm.Length ? _startMm[i] : 0;
                if (sm < minStart) minStart = sm;
            }
            if (minStart == double.MaxValue) return 0;

            var cams = new List<MergeLayout.CamGeom>(_camCount);
            for (int i = 0; i < _camCount; i++)
            {
                cams.Add(new MergeLayout.CamGeom
                {
                    CameraId = i + 1,
                    StartMm = i < _startMm.Length ? _startMm[i] : 0,
                    WidthPx = _defaultFrameW
                });
            }

            _cameraPlacements.AddRange(MergeLayout.Compute(
                cams, minStart, _refOpsMm, 1, MergeOverlap.Midline, out int fullW));
            return fullW;
        }

        private void OnCanvasStatus(CanvasInfo info)
        {
            _lastCanvasInfo = info;
            _hasCanvasInfo = true;

            if (!TryBuildCursorStatus(info.ImageX, info.ImageY, info, out var status))
            {
                _canvas.SetRangeOverlay("", "", "", "", "");
                return;
            }

            _canvas.SetRangeOverlay(status.PhysMag > 0 ? $"{status.PhysMag:F2}x" : "",
                $"{status.ViewLeftMm:F1}", $"{status.ViewRightMm:F1}", $"{status.ViewTopMm:F1}", $"{status.ViewBotMm:F1}");
            _canvas.SetCursorMm(status.CurMmX, status.CurMmY);
            ViewRangeMmChanged?.Invoke(status.ViewLeftMm, status.ViewRightMm, status.ViewTopMm, status.ViewBotMm);
            UpdateCenterCam(info.Zoom, info.PanOffset);
            CursorStatusChanged?.Invoke(status);
        }

        private void OnCanvasMouseClick(object sender, MouseEventArgs e)
        {
            if (!_hasCanvasInfo) return;
            int imageX = (int)((e.X - _lastCanvasInfo.PanOffset.X) / _lastCanvasInfo.Zoom);
            int imageY = (int)((e.Y - _lastCanvasInfo.PanOffset.Y) / _lastCanvasInfo.Zoom);

            if (e.Button == MouseButtons.Left)
            {
                int camId = ResolveCameraAtX(imageX);
                if (camId > 0) SelectRequested?.Invoke(camId);
                return;
            }

            if (e.Button == MouseButtons.Right &&
                TryBuildCursorStatus(imageX, imageY, _lastCanvasInfo, out var status))
                CursorStatusChanged?.Invoke(status);
        }

        private int ResolveCameraAtX(int imageX)
        {
            if (_cameraPlacements.Count == 0 || imageX < 0) return -1;
            foreach (var p in _cameraPlacements)
            {
                int left = p.DestX;
                int right = p.DestX + Math.Max(1, p.SrcWidth);
                if (imageX >= left && imageX < right)
                    return p.CameraId;
            }

            int nearestId = -1;
            int nearestDist = int.MaxValue;
            foreach (var p in _cameraPlacements)
            {
                int center = p.DestX + Math.Max(1, p.SrcWidth) / 2;
                int dist = Math.Abs(imageX - center);
                if (dist < nearestDist)
                {
                    nearestDist = dist;
                    nearestId = p.CameraId;
                }
            }
            return nearestId;
        }

        private bool TryBuildCursorStatus(int imageX, int imageY, CanvasInfo info, out ImageDisplayView.CursorStatus status)
        {
            status = null;
            if (_disposed || _canvas == null || info.Zoom <= 0) return false;
            if (_startMm == null || _startMm.Length == 0 || _refOpsMm <= 0) return false;

            int camId = ResolveCameraAtX(imageX);
            if (!TryComputeViewRange(info.Zoom, info.PanOffset, out double leftMm, out double rightMm, out double topMm, out double botMm))
                return false;
            double yPitch = _rowPitchMm > 0 ? _rowPitchMm : _refOpsMm;
            double minStartMm = MinStartMm();
            double curMmX = PixelMmMapper.PixelToMm(imageX, minStartMm, _refOpsMm);
            double curMmY = ToLogicalY(imageY) * yPitch;   // 游標與視野同一座標約定（flip 時 0 在畫面底；原漏換算）

            _canvas.SetPhysicalCalibration(_refOpsMm, _screenMmPerPx);
            status = new ImageDisplayView.CursorStatus
            {
                CurMmX = curMmX,
                CurMmY = curMmY,
                ViewLeftMm = leftMm,
                ViewRightMm = rightMm,
                ViewTopMm = topMm,
                ViewBotMm = botMm,
                PhysMag = _canvas.PhysicalMagnification,
                CursorX = imageX,
                CursorY = imageY,
                Brightness = info.Brightness,
                SelectedCamId = camId > 0 ? camId : 0,
            };
            return true;
        }

        private double ToLogicalY(double visualY)
            => _flipVertical ? (_totalHeight - 1 - visualY) : visualY;


        private bool TryComputeViewRange(float zoom, PointF panOffset,
            out double leftMm, out double rightMm, out double topMm, out double botMm)
        {
            leftMm = rightMm = topMm = botMm = 0;
            if (_disposed || !_virtualSet || _canvas == null || zoom <= 0) return false;
            if (_startMm == null || _startMm.Length == 0 || _refOpsMm <= 0) return false;

            double minStartMm = MinStartMm();
            leftMm = PixelMmMapper.PixelToMm((0 - panOffset.X) / zoom, minStartMm, _refOpsMm);
            rightMm = PixelMmMapper.PixelToMm((_canvas.Width - panOffset.X) / zoom, minStartMm, _refOpsMm);

            double visualTop = (0 - panOffset.Y) / zoom;
            double visualBot = (_canvas.Height - panOffset.Y) / zoom;
            double logicalTop = ToLogicalY(visualTop);
            double logicalBot = ToLogicalY(visualBot);
            double yPitch = _rowPitchMm > 0 ? _rowPitchMm : _refOpsMm;
            // 邊界值保留「邊的身份」（top＝畫面上緣的 logical 值；flip 時上緣＝大值）——
            // 原 min/max 排序毀掉邊界語意 → overlay 上下邊顯錯（與游標兩套座標）。排序需求由消費端自理。
            topMm = logicalTop * yPitch;
            botMm = logicalBot * yPitch;
            return topMm != botMm;
        }

        private double MinStartMm()
        {
            double min = double.MaxValue;
            lock (_lock)
            {
                if (_startMm != null)
                {
                    for (int i = 0; i < _startMm.Length; i++)
                        if (_startMm[i] < min) min = _startMm[i];
                }
            }
            return min == double.MaxValue ? 0 : min;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _flushTimer.Stop(); _flushTimer.Dispose(); } catch { }
            try
            {
                _canvas.StatusChanged -= OnCanvasStatus;
                _canvas.LodTileApplied -= OnLodTileApplied;
                _canvas.MouseClick -= OnCanvasMouseClick;
            }
            catch { }
            try { _canvas.DisableLod(); } catch { }
            try { if (_canvas.Parent != null) _canvas.Parent.Controls.Remove(_canvas); _canvas.Dispose(); } catch { }
            lock (_lock) { _layerChunks = null; _fullW = 0; _writeRow = 0; _pending.Clear(); _preBuffer.Clear(); _writeQueue.Clear(); }
        }
    }
}
