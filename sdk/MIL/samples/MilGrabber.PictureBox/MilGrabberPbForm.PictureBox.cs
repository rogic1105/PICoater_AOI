using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Controls; // SmartCanvas（zoom/pan/overlay）

namespace MilGrabber.PictureBoxTest
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

        private SmartCanvas _mainCanvas;           // 主畫面（panelMain 內）
        private int _mainW = -1, _mainH = -1;      // 主畫面上次影像尺寸（變了才 FitToScreen）

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

        /// <summary>建立主畫面 SmartCanvas + 定頻刷新 timer（建構式呼叫）。numResize/chkMerge 控制項在 Designer。</summary>
        private void SetupPbMain()
        {
            _mainCanvas = new SmartCanvas { Dock = DockStyle.Fill };
            panelMain.Controls.Add(_mainCanvas);
            _mainCanvas.BringToFront();

            _displayTimer = new System.Windows.Forms.Timer { Interval = 33 }; // ~30fps
            _displayTimer.Tick += DisplayTimer_Tick;
        }

        /// <summary>單台模式定頻刷；合圖模式只做「逾時 fallback」（合圖正常路徑在 OnCameraFrame 湊齊一輪即合）。</summary>
        private void DisplayTimer_Tick(object sender, EventArgs e)
        {
            if (_isReleasing || _mainCanvas == null || _mainCanvas.IsDisposed) return;

            if (_mergeMode)
            {
                // 某台停了 → 集合永遠湊不齊 → 超過 200ms 強制刷一次（避免合圖凍住）
                if (Environment.TickCount - _lastMergeMs > 200)
                {
                    ClearReadyFlags();
                    _lastMergeMs = Environment.TickCount;
                    Bitmap m = BuildMergeBitmap();
                    if (m != null) ApplyMainImage(m);
                }
            }
            else
            {
                int sel = _selectedCam;
                FrameData f = (sel >= 0 && sel < SubPanelCount) ? _latest[sel] : null;
                if (f != null) ApplyMainImage(BuildGrayBitmap(f.Bytes, f.W, f.H));
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
        private void numResize_ValueChanged(object sender, EventArgs e) => _resizeScale = (int)numResize.Value;
        private void chkMerge_CheckedChanged(object sender, EventArgs e) => _mergeMode = chkMerge.Checked;

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

                cam.GetFrameBytes(buffer, _srcBytes[idx]);                       // 8-bit 灰階原圖
                Marshal.Copy(_srcBytes[idx], 0, _srcPinned[idx], srcPix);
                NativeResize.CoreCV_Resize_GPU(_srcPinned[idx], fw, fh, _dstPinned[idx], dw, dh); // GPU 縮圖

                var dstBytes = new byte[dstPix];
                Marshal.Copy(_dstPinned[idx], dstBytes, 0, dstPix);
                _latest[idx] = new FrameData(dstBytes, dw, dh);                  // 供合圖讀（原子 ref 換）

                // 子畫面 thumbnail
                SwapImage((idx < _displayBoxes.Length) ? _displayBoxes[idx] : null, BuildGrayBitmap(dstBytes, dw, dh));

                // 合圖同步：等所有在線相機都產生新幀（湊齊一輪）才合 → _latest 全是同一輪 → 對齊，
                // 避免「半輪取樣」（有些相機已更新、有些還在 resize）造成的時間差。
                _readySinceMerge[idx] = true;
                if (_mergeMode && AllActiveFramed())
                {
                    ClearReadyFlags();
                    _lastMergeMs = Environment.TickCount;
                    Bitmap merged = BuildMergeBitmap();
                    if (merged != null) ApplyMainImage(merged);
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

        /// <summary>8-bit 灰階 byte[] → Format8bppIndexed Bitmap（灰階調色盤）。</summary>
        private static Bitmap BuildGrayBitmap(byte[] data, int w, int h)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
            ColorPalette pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(i, i, i);
            bmp.Palette = pal;

            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try
            {
                for (int y = 0; y < h; y++)
                    Marshal.Copy(data, y * w, IntPtr.Add(bd.Scan0, y * bd.Stride), w);
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
            }
            if (_mainCanvas != null) { var old = _mainCanvas.Image; _mainCanvas.Image = null; old?.Dispose(); }
        }
    }
}
