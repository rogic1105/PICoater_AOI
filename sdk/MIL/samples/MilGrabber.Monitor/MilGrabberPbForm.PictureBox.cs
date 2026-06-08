using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Controls; // SmartCanvas（zoom/pan/overlay）
using TanukiCv.Core;     // SystemInfo（螢幕 mm/px）、PixelMmMapper

namespace MilGrabber.Monitor
{
    // PictureBox 顯示路徑：不讓 MIL 直接畫 panel，改訂閱 MilCamera.FrameReady →
    //   GetFrameBytes（8-bit 灰階）→ core_cv_api 的 GPU resize 縮圖 → 組灰階 Bitmap →
    //   thumbnail 用 PictureBox、主畫面用 SmartCanvas（zoom/pan/overlay）。
    //   合圖：各台縮圖橫向簡單拼接成一張，顯示在主畫面 SmartCanvas。
    // 用來測「縮圖後改由 GDI/PictureBox/SmartCanvas 繪製」即時取像會不會卡（對照 MIL 原生直繪的 MilGrabber.Monitor）。
    public partial class MilGrabberPbForm
    {
        // 每相機縮圖緩衝（同一相機的 FrameReady 為序列觸發 → 同 idx 緩衝無併發）
        private readonly IntPtr[] _srcPinned = new IntPtr[SubPanelCount];
        private readonly IntPtr[] _dstPinned = new IntPtr[SubPanelCount];
        private readonly int[]    _srcCap    = new int[SubPanelCount];
        private readonly int[]    _dstCap    = new int[SubPanelCount];
        private readonly byte[][] _srcBytes  = new byte[SubPanelCount][];

        private volatile int  _resizeScale = 4;   // 縮圖倍率（由 numResize 更新；控制項在 Designer）
        private volatile bool _mergeMode;          // 主畫面顯示合圖 vs 選中相機（由 chkMerge 更新）
        private volatile bool _flipDisplay;        // PictureBox 模式上下翻轉（CoreCV_Resize_GPU 輸出 bottom-up，由 chkFlipVertical 控制）

        // ── 動態 LOD（單張模式；由 chkLod 控制）：主畫面改用 SmartCanvas LOD provider，
        //    停住才裁可見區+GPU 縮到 panel 解析度 → 縮小看全圖便宜、放大看細節清晰且便宜。 ──
        private volatile bool _lodEnabled;
        private volatile FrameData _lodSnap;        // 選中相機最新「全解析度」灰階快照（整顆 ref 原子換→尺寸+bytes 一致，provider 在 UI 執行緒讀）
        private IntPtr _lodSrcPinned, _lodDstPinned; // LOD 裁+縮專用 pinned（provider 現跑背景執行緒）
        private readonly object _lodBufLock = new object(); // 保護 pinned：背景 provider vs 釋放（防 use-after-free）
        private volatile bool _lodReleased;          // 已釋放 pinned → 等待中的 provider 取得鎖後不再 re-alloc
        private int _lodSrcCap, _lodDstCap;
        private int _lodActiveCamIdx = -1;         // 目前 LOD 綁的相機 index（換相機要重設虛擬尺寸）

        private SmartCanvas _mainCanvas;           // 主畫面（panelMain 內，PictureBox 模式）
        private int _mainW = -1, _mainH = -1;      // 主畫面上次影像尺寸（變了才 FitToScreen）
        private double _screenMmPerPx;             // 螢幕實體 mm/px（SystemInfo 查一次）→ 三擊實體 1:1 用

        // ── 顯示模式（初始化前選）：MIL 直繪 vs PictureBox 縮圖繪 ──
        private bool _milMode;                                              // true=MIL 直繪、false=PictureBox
        private readonly Panel[] _displayPanels = new Panel[SubPanelCount]; // MIL 模式子畫面（MIL 直繪目標）
        private Panel _mainPanelMil;                                        // MIL 模式主畫面（MIL secondary 目標）
        // _rbModePb / _rbModeMil / _lblTiming 已移到 Designer（可在 [設計] 看到/調位置）

        // ── 計時（每 500ms 視窗的最大值=worst-case，比較 MIL vs PictureBox）──
        private double _resizeMaxMs; // 縮圖 worst-case：CoreCV_Resize_GPU + 組 bitmap（每窗重置）
        private readonly System.Diagnostics.Stopwatch _resizeSw = new System.Diagnostics.Stopwatch();

        // 合圖用：各台最新縮圖（整顆 ref 原子換 → 合圖端讀到完整一幀）
        private sealed class FrameData
        {
            public readonly byte[] Bytes; public readonly int W, H;
            public FrameData(byte[] b, int w, int h) { Bytes = b; W = w; H = h; }
        }
        private readonly FrameData[] _latest = new FrameData[SubPanelCount];
        private readonly bool[] _readySinceMerge = new bool[SubPanelCount]; // 自上次合圖後各台是否已產生新幀
        private int _lastMergeMs;                                            // 上次合圖時間（逾時 fallback 用）
        private System.Windows.Forms.Timer _displayTimer; // 單台刷 + 合圖逾時 fallback（仿主程式 _mergedDisplayTimer）

        /// <summary>建立主畫面 SmartCanvas + 定頻 timer + 顯示模式 radio + 計時 label（建構式呼叫）。</summary>
        private void SetupPbMain()
        {
            _mainCanvas = new SmartCanvas { Dock = DockStyle.Fill }; // PictureBox 模式主畫面
            _mainCanvas.FitRelativeZoom = true;        // 滾輪相對 fit（fit=1×；滾 N 格放大一致，上限隨 resize 而異）
            _mainCanvas.DoubleClickFitToScreen = true; // 雙擊回到 fit（提取自主程式回顧畫布）
            _mainCanvas.TripleClickPhysical1x = true;  // 三擊跳實體 1:1（需 SetPhysicalCalibration）
            _screenMmPerPx = SystemInfo.GetScreenMetrics().MmPerPx; // 螢幕 mm/px 查一次
            _mainCanvas.StatusChanged += OnMainCanvasStatus; // 四邊 mm 邊界 overlay + 游標 mm（驗證座標映射）
            panelMain.Controls.Add(_mainCanvas);
            _mainCanvas.BringToFront();

            _displayTimer = new System.Windows.Forms.Timer { Interval = 33 }; // ~30fps
            _displayTimer.Tick += DisplayTimer_Tick;
            // 顯示模式 radio（_rbModePb/_rbModeMil）+ 計時 label（_lblTiming）在 Designer 宣告/佈局。
        }

        /// <summary>依「初始化前選的模式」建立顯示控制項（btnInit 開頭呼叫）。
        /// MIL 模式：每容器疊 Panel（MIL 直繪目標）+ 主畫面疊 Panel；PictureBox 模式：用現成 PictureBox + SmartCanvas。</summary>
        private void CreateDisplaysForMode()
        {
            _lodReleased = false; // 重新 init → 允許 LOD provider 再配置 pinned
            _milMode = _rbModeMil.Checked;
            _rbModeMil.Enabled = _rbModePb.Enabled = false; // 初始化後不可改模式
            if (!_milMode) return;

            for (int i = 0; i < SubPanelCount; i++)
            {
                var p = new Panel { Dock = DockStyle.Fill, BackColor = Color.Black };
                int idx = i;
                p.MouseClick += (s, e) => SelectCamera(idx);
                _camContainers[i].Controls.Add(p);
                p.BringToFront(); // 蓋住 PictureBox
                _displayPanels[i] = p;
            }
            _mainPanelMil = new Panel { Dock = DockStyle.Fill, BackColor = Color.Black };
            panelMain.Controls.Add(_mainPanelMil);
            _mainPanelMil.BringToFront(); // 蓋住 SmartCanvas
        }

        /// <summary>只做合圖「逾時 fallback」。單台 / 合圖正常更新都在 OnCameraFrame（隨相機 fps）；
        /// 不再 30fps 空轉重畫 —— 相機 1fps 時冗餘重建 _viewCache 是主要顯示成本（見 PbTiming）。</summary>
        private void DisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_isReleasing || _milMode || _mainCanvas == null || _mainCanvas.IsDisposed) return; // MIL 模式主畫面由 MIL 直繪

            // 合圖模式某台停了 → 集合永遠湊不齊 → 超過 200ms 強制刷一次（避免合圖凍住）。
            // 但只有「真的有新幀進來、只是沒湊齊一輪」才刷；完全沒新幀就跳過 → 靜止=真靜止，
            // 不對 1fps 相機做 5fps 冗餘重建（低配機省 CPU）。
            if (_mergeMode && Environment.TickCount - _lastMergeMs > 200 && AnyReadySinceMerge())
            {
                ClearReadyFlags();
                _lastMergeMs = Environment.TickCount;
                Bitmap m = BuildMergeBitmap();
                if (m != null) ApplyMainImage(m);
            }
        }

        private bool AllActiveFramed()
        {
            bool any = false;
            for (int i = 0; i < SubPanelCount; i++)
            {
                if (_latest[i] == null) continue; // 非 active（沒產生過幀）
                any = true;
                if (!_readySinceMerge[i]) return false;
            }
            return any;
        }

        private void ClearReadyFlags()
        {
            for (int i = 0; i < SubPanelCount; i++) _readySinceMerge[i] = false;
        }

        private bool AnyReadySinceMerge()
        {
            for (int i = 0; i < SubPanelCount; i++) if (_readySinceMerge[i]) return true;
            return false;
        }

        /// <summary>切 UI 換主畫面 SmartCanvas 影像；尺寸變了才 FitToScreen（不重置使用者 zoom/pan）。
        /// 可從相機回呼或 UI timer 呼叫（一律 BeginInvoke）。</summary>
        private void ApplyMainImage(Bitmap bmp)
        {
            SmartCanvas c = _mainCanvas;
            if (c == null || !c.IsHandleCreated || c.IsDisposed) { bmp.Dispose(); return; }
            try
            {
                c.BeginInvoke((Action)(() =>
                {
                    var old = c.Image; c.Image = bmp; old?.Dispose();
                    if (bmp.Width != _mainW || bmp.Height != _mainH)
                    {
                        _mainW = bmp.Width; _mainH = bmp.Height;
                        c.FitToScreen();
                    }
                }));
            }
            catch { bmp.Dispose(); }
        }

        // ── Designer 控制項事件 ──
        private void numResize_ValueChanged(object sender, EventArgs e) { _resizeScale = (int)numResize.Value; UpdatePhysicalCalibration(); }
        private void numFovMm_ValueChanged(object sender, EventArgs e) => UpdatePhysicalCalibration();

        /// <summary>依 FOV + 選中相機影像寬 + 顯示模式算 mm/影像像素，餵 SmartCanvas（三擊實體 1:1）。
        /// LOD 模式畫布操作的是虛擬全解析度→mmPerImagePx=opsInMm；非 LOD 顯示縮圖→×scale。</summary>
        private void UpdatePhysicalCalibration()
        {
            if (_mainCanvas == null) return;
            double fovMm = (numFovMm != null) ? (double)numFovMm.Value : 0;
            int fw = (_selectedCam >= 0 && _cams != null && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                ? _cams[_selectedCam].FrameWidth : 0;
            if (fovMm <= 0 || fw <= 0 || _screenMmPerPx <= 0) { _mainCanvas.SetPhysicalCalibration(0, 0); return; }
            double opsInMm = fovMm / fw;                                  // mm / 全解析度像素
            double mmPerImagePx = opsInMm * (_lodEnabled ? 1.0 : _resizeScale);
            _mainCanvas.SetPhysicalCalibration(mmPerImagePx, _screenMmPerPx);
        }

        /// <summary>StatusChanged：算四邊 mm 邊界 + 游標 mm，推給 SmartCanvas overlay。
        /// 驗證座標映射：縮放/平移時 mm 即時跟動且不跑調 → pixel↔mm 對 → .bin 靠同座標就會對齊。
        /// 公式單一來源 PixelMmMapper；canvas像素→全解析度像素 sf=(LOD?1:scale)，與 SetPhysicalCalibration 同邏輯。</summary>
        private void OnMainCanvasStatus(CanvasInfo info)
        {
            if (_mainCanvas == null) return;
            double fovMm = (numFovMm != null) ? (double)numFovMm.Value : 0;
            int fw = (_selectedCam >= 0 && _cams != null && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                ? _cams[_selectedCam].FrameWidth : 0;
            if (fovMm <= 0 || fw <= 0 || info.Zoom <= 0)
            {
                _mainCanvas.SetRangeOverlay("", "", "", "", "");
                return;
            }
            double opsInMm = fovMm / fw;
            double sf = _lodEnabled ? 1.0 : _resizeScale;          // canvas 像素 → 全解析度像素
            // 四邊：canvas 邊 → 全解析度像素 → mm（start=0，單相機視角；Y 暫用同 ops 方形像素假設，驗證跟動用）
            double leftMm  = PixelMmMapper.PixelToMm((0 - info.PanOffset.X) / info.Zoom * sf, 0, opsInMm);
            double rightMm = PixelMmMapper.PixelToMm((_mainCanvas.Width  - info.PanOffset.X) / info.Zoom * sf, 0, opsInMm);
            double topMm   = PixelMmMapper.PixelToMm((0 - info.PanOffset.Y) / info.Zoom * sf, 0, opsInMm);
            double botMm   = PixelMmMapper.PixelToMm((_mainCanvas.Height - info.PanOffset.Y) / info.Zoom * sf, 0, opsInMm);

            double physMag = _mainCanvas.PhysicalMagnification;
            _mainCanvas.SetRangeOverlay(physMag > 0 ? $"{physMag:F2}x" : "",
                $"{leftMm:F1}", $"{rightMm:F1}", $"{topMm:F1}", $"{botMm:F1}");

            // 游標 mm（跟滑鼠）
            double curX = info.ImageX * sf * opsInMm;
            double curY = info.ImageY * sf * opsInMm;
            _mainCanvas.SetCursorMm($"({curX:F2}, {curY:F2})");
        }
        private void chkMerge_CheckedChanged(object sender, EventArgs e)
        {
            _mergeMode = chkMerge.Checked;
            if (_mergeMode && _mainCanvas != null && _mainCanvas.LodActive)
            {
                _mainCanvas.DisableLod(); // 合圖暫不支援 LOD → 退回一般顯示
                _lodActiveCamIdx = -1;
            }
        }

        private void chkLod_CheckedChanged(object sender, EventArgs e)
        {
            _lodEnabled = chkLod.Checked;
            if (!_lodEnabled && _mainCanvas != null && _mainCanvas.LodActive)
            {
                _mainCanvas.DisableLod();
                _lodActiveCamIdx = -1;
                _mainW = _mainH = -1; // 下次 ApplyMainImage 會重新 FitToScreen
            }
            UpdatePhysicalCalibration(); // 模式變→mm/px 基準變
            // 開啟時不立刻動作：等選中相機下一幀在 OnCameraFrame 啟用（要有全解析度快照）
        }

        // ── 動態 LOD ─────────────────────────────────────────────────────────
        /// <summary>UI 執行緒：依當前快照啟用 / 刷新主畫面 LOD（合圖模式跳過）。</summary>
        private void UpdateLodOnUi(int idx, int fullW, int fullH)
        {
            SmartCanvas c = _mainCanvas;
            if (c == null || !c.IsHandleCreated || c.IsDisposed) return;
            try
            {
                c.BeginInvoke((Action)(() =>
                {
                    if (_isReleasing || _mergeMode || !_lodEnabled) return;
                    if (!c.LodActive || _lodActiveCamIdx != idx)
                    {
                        _lodActiveCamIdx = idx;
                        c.EnableLod(fullW, fullH, LodProvide); // 內含 fit + 首次 RecomputeLodTile
                    }
                    else
                    {
                        c.RefreshLod(); // 新幀、視角沒變 → 用當前視角重產 tile
                    }
                }));
            }
            catch { /* handle 已毀 */ }
        }

        /// <summary>LOD provider（UI 執行緒）：從全解析度快照裁出可見虛擬區 srcRect → GPU 縮到 target → 灰階 Bitmap。
        /// 放大時 srcRect 小 → 縮得少 → 清晰；縮小時 srcRect≈全圖 → 縮到 panel → 便宜。</summary>
        private Bitmap LodProvide(Rectangle srcRect, Size target)
        {
            FrameData snap = _lodSnap;
            if (snap == null) return null;
            byte[] full = snap.Bytes; int fw = snap.W, fh = snap.H;

            // clamp srcRect 進 [0,fw]×[0,fh]
            int sx = Math.Max(0, Math.Min(srcRect.X, fw - 1));
            int sy = Math.Max(0, Math.Min(srcRect.Y, fh - 1));
            int sw = Math.Max(1, Math.Min(srcRect.Width,  fw - sx));
            int sh = Math.Max(1, Math.Min(srcRect.Height, fh - sy));
            int tw = Math.Max(1, target.Width), th = Math.Max(1, target.Height);
            int cropPix = sw * sh, dstPix = tw * th;

            byte[] dst;
            lock (_lodBufLock) // pinned 存取與「釋放」互斥（provider 現跑背景執行緒）
            {
                if (_lodReleased) return null; // 已釋放 → 不再碰/配置 pinned
                if (_lodSrcCap < cropPix)
                {
                    if (_lodSrcPinned != IntPtr.Zero) NativeResize.CoreCV_FreePinned(_lodSrcPinned);
                    _lodSrcPinned = NativeResize.CoreCV_AllocPinned((ulong)cropPix); _lodSrcCap = cropPix;
                }
                if (_lodDstCap < dstPix)
                {
                    if (_lodDstPinned != IntPtr.Zero) NativeResize.CoreCV_FreePinned(_lodDstPinned);
                    _lodDstPinned = NativeResize.CoreCV_AllocPinned((ulong)dstPix); _lodDstCap = dstPix;
                }
                if (_lodSrcPinned == IntPtr.Zero || _lodDstPinned == IntPtr.Zero) return null;

                // 裁出可見區到連續緩衝（逐列複製）→ GPU 縮 → 拷回 dst
                var crop = new byte[cropPix];
                for (int y = 0; y < sh; y++) Array.Copy(full, (sy + y) * fw + sx, crop, y * sw, sw);
                Marshal.Copy(crop, 0, _lodSrcPinned, cropPix);
                NativeResize.CoreCV_Resize_GPU(_lodSrcPinned, sw, sh, _lodDstPinned, tw, th);
                dst = new byte[dstPix];
                Marshal.Copy(_lodDstPinned, dst, 0, dstPix);
            }
            return BuildGrayBitmap(dst, tw, th); // 組 bitmap 不碰 pinned，移出鎖；flip 由 _flipDisplay 處理
        }

        /// <summary>更新計時 label（StatusTimer 每 500ms 呼叫）：比較 MIL 直繪 vs PictureBox 的縮圖/顯示成本。</summary>
        private void UpdateTimingLabel()
        {
            if (_lblTiming == null) return;
            double fps = (_selectedCam >= 0 && _cams != null && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                ? _cams[_selectedCam].CurrentFps : 0;
            if (_milMode)
            {
                _lblTiming.Text = $"模式: MIL 直繪\nFPS: {fps:F1}\n(縮圖/顯示由 MIL\n 硬體直繪，無 CPU 成本)";
                Trace.WriteLine($"[PbTiming] mode=MIL FPS={fps:F1}");
            }
            else
            {
                // 每 500ms 視窗的「最大值」(worst-case，判斷卡頓);讀完即重置 → 反映近期互動
                double resizeMax = _resizeMaxMs; _resizeMaxMs = 0;
                double paintMax  = _mainCanvas?.ResetMaxPaintMs() ?? 0;
                float  zoomRel   = _mainCanvas?.ZoomRelativeToFit ?? 1f;   // 螢幕縮放：相對 fit（fit=1×）
                double physMag   = _mainCanvas?.PhysicalMagnification ?? 0; // 實體倍率（mm 校正；0=未校正/沒FOV）
                string merge     = _mergeMode ? "合圖" : "單張";     // 主畫面合圖狀態
                // LOD 狀態：勾選(_lodEnabled) + 是否真的在畫布生效(LodActive)。ON=生效中、待命=勾了但還沒綁、OFF=沒勾
                bool lodOn       = _mainCanvas != null && _mainCanvas.LodActive;
                string lod       = lodOn ? "ON" : (_lodEnabled ? "待命" : "OFF");
                string physStr   = physMag > 0 ? $"{physMag:F2}x" : "-（需FOV）";
                _lblTiming.Text =
                    $"模式: PictureBox\n畫面: {merge}\nLOD: {lod}\n縮圖倍率: {_resizeScale}\nzoom×fit: {zoomRel:F2}\n實體倍率: {physStr}\n縮圖(max): {resizeMax:F1} ms\n顯示(max): {paintMax:F1} ms\nFPS: {fps:F1}";
                Trace.WriteLine($"[PbTiming] mode=PB {merge} LOD={lod} scale={_resizeScale} zoom×fit={zoomRel:F2} 實體={physStr} 縮圖max={resizeMax:F1}ms 顯示max={paintMax:F1}ms FPS={fps:F1}");
            }
        }

        /// <summary>每幀（MIL 回呼執行緒）：縮圖 → 灰階 Bitmap → thumbnail + 主畫面（合圖 / 選中）。</summary>
        private void OnCameraFrame(int idx, MilCamera cam, MIL_ID buffer)
        {
            if (_isReleasing) return;
            int fw = cam.FrameWidth, fh = cam.FrameHeight;
            if (fw <= 0 || fh <= 0) return;

            int scale = _resizeScale; if (scale < 1) scale = 1;
            int dw = Math.Max(1, fw / scale), dh = Math.Max(1, fh / scale);
            int srcPix = fw * fh, dstPix = dw * dh;

            try
            {
                if (_srcBytes[idx] == null || _srcBytes[idx].Length < srcPix) _srcBytes[idx] = new byte[srcPix];
                if (_srcCap[idx] < srcPix)
                {
                    if (_srcPinned[idx] != IntPtr.Zero) NativeResize.CoreCV_FreePinned(_srcPinned[idx]);
                    _srcPinned[idx] = NativeResize.CoreCV_AllocPinned((ulong)srcPix);
                    _srcCap[idx] = srcPix;
                }
                if (_dstCap[idx] < dstPix)
                {
                    if (_dstPinned[idx] != IntPtr.Zero) NativeResize.CoreCV_FreePinned(_dstPinned[idx]);
                    _dstPinned[idx] = NativeResize.CoreCV_AllocPinned((ulong)dstPix);
                    _dstCap[idx] = dstPix;
                }
                if (_srcPinned[idx] == IntPtr.Zero || _dstPinned[idx] == IntPtr.Zero) return; // alloc 失敗（無 GPU / DLL）

                _resizeSw.Restart();                                             // ── 縮圖計時起 ──
                cam.GetFrameBytes(buffer, _srcBytes[idx]);                       // 8-bit 灰階原圖
                Marshal.Copy(_srcBytes[idx], 0, _srcPinned[idx], srcPix);
                NativeResize.CoreCV_Resize_GPU(_srcPinned[idx], fw, fh, _dstPinned[idx], dw, dh); // GPU 縮圖

                var dstBytes = new byte[dstPix];
                Marshal.Copy(_dstPinned[idx], dstBytes, 0, dstPix);
                Bitmap thumb = BuildGrayBitmap(dstBytes, dw, dh);                // 組 thumbnail bitmap
                _resizeSw.Stop();                                                // ── 縮圖計時止（resize+組bitmap）──
                double rms = _resizeSw.Elapsed.TotalMilliseconds;
                if (rms > _resizeMaxMs) _resizeMaxMs = rms;                       // 視窗內最大（worst-case）

                _latest[idx] = new FrameData(dstBytes, dw, dh);                  // 供合圖讀（原子 ref 換）
                SwapImage((idx < _displayBoxes.Length) ? _displayBoxes[idx] : null, thumb); // 子畫面 thumbnail

                // 合圖同步：等所有在線相機都產生新幀（湊齊一輪）才合 → _latest 全是同一輪 → 對齊，
                // 避免「半輪取樣」（有些相機已更新、有些還在 resize）造成的時間差。
                _readySinceMerge[idx] = true;
                if (_mergeMode)
                {
                    if (AllActiveFramed())
                    {
                        ClearReadyFlags();
                        _lastMergeMs = Environment.TickCount;
                        Bitmap merged = BuildMergeBitmap();
                        if (merged != null) ApplyMainImage(merged);
                    }
                }
                else if (idx == _selectedCam)
                {
                    if (_lodEnabled)
                    {
                        // LOD：存「全解析度」快照（整顆 ref 原子換）→ UI 執行緒啟用/刷新 LOD。
                        // 主畫面不走 ApplyMainImage（SmartCanvas 改用 provider 產 tile）。
                        var full = new byte[srcPix];
                        Array.Copy(_srcBytes[idx], full, srcPix);
                        _lodSnap = new FrameData(full, fw, fh);
                        UpdateLodOnUi(idx, fw, fh);
                    }
                    else
                    {
                        // 單台模式：只在「被選中的相機產生新幀」時更新主畫面（隨相機 fps，非 30fps 空轉）。
                        // 互動（zoom/pan）由 SmartCanvas 自己 Invalidate 重繪，不靠這條路徑。
                        ApplyMainImage(BuildGrayBitmap(dstBytes, dw, dh));
                    }
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[PbFrame] CAM idx{idx}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>各台最新縮圖橫向簡單拼接成一張（A：不做位置重疊）。</summary>
        private Bitmap BuildMergeBitmap()
        {
            var frames = new FrameData[SubPanelCount];
            int totalW = 0, maxH = 0, count = 0;
            for (int i = 0; i < SubPanelCount; i++)
            {
                FrameData f = _latest[i];
                frames[i] = f;
                if (f != null) { totalW += f.W; if (f.H > maxH) maxH = f.H; count++; }
            }
            if (count == 0 || totalW <= 0 || maxH <= 0) return null;

            var merged = new Bitmap(totalW, maxH, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(merged))
            {
                g.Clear(Color.Black);
                int x = 0;
                for (int i = 0; i < SubPanelCount; i++)
                {
                    FrameData f = frames[i];
                    if (f == null) continue;
                    using (var fb = BuildGrayBitmap(f.Bytes, f.W, f.H))
                        g.DrawImageUnscaled(fb, x, 0);
                    x += f.W;
                }
            }
            return merged;
        }

        /// <summary>切到 UI 執行緒換 PictureBox.Image（dispose 舊圖）；box 無效時自行 dispose bmp。</summary>
        private static void SwapImage(PictureBox box, Bitmap bmp)
        {
            if (box != null && box.IsHandleCreated && !box.IsDisposed)
            {
                try { box.BeginInvoke((Action)(() => { var old = box.Image; box.Image = bmp; old?.Dispose(); })); }
                catch { bmp.Dispose(); }
            }
            else bmp.Dispose();
        }

        /// <summary>8-bit 灰階 byte[] → Format8bppIndexed Bitmap（灰階調色盤）。
        /// _flipDisplay 時上下翻轉（CoreCV_Resize_GPU 輸出 bottom-up，由「上下翻轉」勾選控制；翻轉只是來源列反向、零額外成本）。</summary>
        private static readonly Color[] _grayEntries = BuildGrayEntries(); // 256 灰階一次算好（每幀重算是純浪費）
        private static Color[] BuildGrayEntries()
        {
            var e = new Color[256];
            for (int i = 0; i < 256; i++) e[i] = Color.FromArgb(i, i, i);
            return e;
        }

        private Bitmap BuildGrayBitmap(byte[] data, int w, int h)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
            ColorPalette pal = bmp.Palette;                       // ColorPalette 不可跨 Bitmap 共用，但 entry 內容用快取免重算
            for (int i = 0; i < 256; i++) pal.Entries[i] = _grayEntries[i];
            bmp.Palette = pal;

            bool flip = _flipDisplay;
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try
            {
                for (int y = 0; y < h; y++)
                {
                    int srcRow = flip ? (h - 1 - y) : y;
                    Marshal.Copy(data, srcRow * w, IntPtr.Add(bd.Scan0, y * bd.Stride), w);
                }
            }
            finally { bmp.UnlockBits(bd); }
            return bmp;
        }

        /// <summary>釋放 pinned 緩衝 + 顯示圖（ReleaseAll 呼叫）。</summary>
        private void ReleasePictureBoxDisplays()
        {
            for (int i = 0; i < SubPanelCount; i++)
            {
                if (_srcPinned[i] != IntPtr.Zero) { NativeResize.CoreCV_FreePinned(_srcPinned[i]); _srcPinned[i] = IntPtr.Zero; _srcCap[i] = 0; }
                if (_dstPinned[i] != IntPtr.Zero) { NativeResize.CoreCV_FreePinned(_dstPinned[i]); _dstPinned[i] = IntPtr.Zero; _dstCap[i] = 0; }
                _srcBytes[i] = null;
                _latest[i] = null;
                _readySinceMerge[i] = false;

                PictureBox box = (_displayBoxes != null && i < _displayBoxes.Length) ? _displayBoxes[i] : null;
                if (box != null) { var old = box.Image; box.Image = null; old?.Dispose(); }

                // MIL 模式疊上去的 Panel：移除 + dispose（下次 init 可重選模式）
                if (_displayPanels[i] != null)
                {
                    _camContainers[i]?.Controls.Remove(_displayPanels[i]);
                    _displayPanels[i].Dispose();
                    _displayPanels[i] = null;
                }
            }
            // LOD：停用（新 recompute 不再啟動）+ 鎖內釋放 pinned（等背景 provider 用完）+ 清快照
            if (_mainCanvas != null && _mainCanvas.LodActive) _mainCanvas.DisableLod();
            lock (_lodBufLock)
            {
                _lodReleased = true;
                if (_lodSrcPinned != IntPtr.Zero) { NativeResize.CoreCV_FreePinned(_lodSrcPinned); _lodSrcPinned = IntPtr.Zero; _lodSrcCap = 0; }
                if (_lodDstPinned != IntPtr.Zero) { NativeResize.CoreCV_FreePinned(_lodDstPinned); _lodDstPinned = IntPtr.Zero; _lodDstCap = 0; }
            }
            _lodSnap = null; _lodActiveCamIdx = -1;

            if (_mainCanvas != null) { var old = _mainCanvas.Image; _mainCanvas.Image = null; old?.Dispose(); }
            if (_mainPanelMil != null)
            {
                panelMain?.Controls.Remove(_mainPanelMil);
                _mainPanelMil.Dispose();
                _mainPanelMil = null;
            }
            if (_rbModeMil != null) _rbModeMil.Enabled = true;  // 釋放後可重選模式
            if (_rbModePb  != null) _rbModePb.Enabled  = true;
        }
    }
}
