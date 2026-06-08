using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using TanukiCv.Controls; // SmartCanvas
using TanukiCv.Core;     // PixelMmMapper

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 監控主畫面的 SmartCanvas 顯示路徑（he_MainDisplay==SmartCanvas 時取代 MIL 直繪）。
    /// 在 camLiveMain 疊一個 SmartCanvas、camLive1~7 疊 PictureBox thumbnail；吃各相機每幀
    /// 顯示 bytes（<see cref="AniloxCamera.OnDisplayFrame"/>）→ 8bpp 灰階 bitmap → 單相機 / CPU 合圖顯示。
    /// 純 CPU、跟回顧畫布同源 SmartCanvas（zoom/pan/雙三擊/mm overlay 內建）。
    /// 合圖座標借 MultiCameraMerger 佈局（MinStartMm/RefOpsMm/各台 xOffset），與 MIL 合圖 / .bin 同源。
    /// </summary>
    internal sealed class LiveSmartDisplay : IDisposable
    {
        private readonly Panel _mainPanel;
        private readonly Panel[] _camPanels;
        private SmartCanvas _canvas;
        private readonly PictureBox[] _thumbs = new PictureBox[7];

        private sealed class Frame { public readonly byte[] Bytes; public readonly int W, H; public Frame(byte[] b, int w, int h) { Bytes = b; W = w; H = h; } }
        private readonly Frame[] _latest = new Frame[7]; // index = camId-1

        private volatile int _selectedCamId = 1;
        private volatile bool _mergeMode;
        private volatile bool _disposed;

        // 合圖佈局（來源 = MultiCameraMerger，經 LiveCameraManager 推入）
        private volatile bool _mergeReady;
        private double _minStartMm, _refOpsMm; private int _totalW, _totalH;
        private double[] _startPosMm, _opsUm;   // 各相機（mm 起始 / µm ops）
        private double _rowPitchMm;             // Y 方向 mm/px（線掃 row pitch）
        private readonly double _screenMmPerPx;

        private int _mainW = -1, _mainH = -1;   // 主畫面上次影像尺寸（變了才 FitToScreen）
        private System.Windows.Forms.Timer _timer;

        /// <summary>thumbnail 被點 → 要求切換選中相機（camId）。LiveCameraManager 訂閱。</summary>
        public event Action<int> SelectRequested;
        /// <summary>StatusChanged 視野範圍（mm）→ overview chart 聯動。LiveCameraManager 訂閱。</summary>
        public event Action<double, double> ViewRangeMmChanged;

        public LiveSmartDisplay(Panel mainPanel, Panel[] camPanels, double screenMmPerPx)
        {
            _mainPanel = mainPanel; _camPanels = camPanels; _screenMmPerPx = screenMmPerPx;

            _canvas = new SmartCanvas { Dock = DockStyle.Fill };
            _canvas.FitRelativeZoom = true;
            _canvas.DoubleClickFitToScreen = true;
            _canvas.TripleClickPhysical1x = true;
            _canvas.ClampPan = false; // 自由拖曳（可上下/左右拖，含 fit 時）—— 監控要能隨意拖看
            _canvas.StatusChanged += OnCanvasStatus;
            _mainPanel.Controls.Add(_canvas);
            _canvas.BringToFront();

            for (int i = 0; i < 7 && i < _camPanels.Length; i++)
            {
                var pb = new PictureBox { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.Black };
                int camId = i + 1;
                pb.MouseClick += (s, e) => SelectRequested?.Invoke(camId);
                _camPanels[i].Controls.Add(pb);
                pb.BringToFront();
                _thumbs[i] = pb;
            }

            _timer = new System.Windows.Forms.Timer { Interval = 33 }; // ~30fps 主畫面刷新（防多相機非同步閃爍）
            _timer.Tick += (s, e) => RefreshMain();
            _timer.Start();
        }

        public void SetSelected(int camId) => _selectedCamId = camId;
        public void SetMergeMode(bool on) => _mergeMode = on;

        public void SetMergeLayout(double minStartMm, double refOpsMm, int totalW, int totalH,
                                   double[] startPosMm, double[] opsUm, double rowPitchMm)
        {
            _minStartMm = minStartMm; _refOpsMm = refOpsMm; _totalW = totalW; _totalH = totalH;
            _startPosMm = startPosMm; _opsUm = opsUm; _rowPitchMm = rowPitchMm;
            _mergeReady = totalW > 0 && refOpsMm > 0;
        }

        /// <summary>相機每幀回呼（MIL 執行緒）：複製成 immutable 快照 + 更新 thumbnail。</summary>
        public void OnCameraFrame(int camId, byte[] bytes, int w, int h)
        {
            if (_disposed || camId < 1 || camId > 7 || bytes == null || w <= 0 || h <= 0) return;
            int n = w * h;
            var copy = new byte[n];
            Array.Copy(bytes, copy, Math.Min(bytes.Length, n));
            _latest[camId - 1] = new Frame(copy, w, h);

            // thumbnail（UI 執行緒）
            PictureBox pb = (camId - 1 < _thumbs.Length) ? _thumbs[camId - 1] : null;
            if (pb != null && pb.IsHandleCreated && !pb.IsDisposed)
            {
                Bitmap thumb = BuildGray(copy, w, h);
                try { pb.BeginInvoke((Action)(() => { var old = pb.Image; pb.Image = thumb; old?.Dispose(); })); }
                catch { thumb.Dispose(); }
            }
        }

        // ── 主畫面刷新（UI timer）──
        private void RefreshMain()
        {
            if (_disposed || _canvas == null || _canvas.IsDisposed) return;
            Bitmap bmp = _mergeMode ? BuildMerge() : BuildSingle();
            if (bmp == null) return;
            var old = _canvas.Image;
            _canvas.Image = bmp;
            old?.Dispose();
            if (bmp.Width != _mainW || bmp.Height != _mainH)
            {
                _mainW = bmp.Width; _mainH = bmp.Height;
                _canvas.FitToScreen();
            }
        }

        private Bitmap BuildSingle()
        {
            int idx = _selectedCamId - 1;
            Frame f = (idx >= 0 && idx < _latest.Length) ? _latest[idx] : null;
            return f != null ? BuildGray(f.Bytes, f.W, f.H) : null;
        }

        /// <summary>CPU 合圖：用 MultiCameraMerger 佈局把各台貼到 xOffset=(startPosMm-MinStartMm)/RefOpsMm。
        /// 重疊區後台相機覆蓋（簡化；MIL 的中點分界為後續精修）。</summary>
        private Bitmap BuildMerge()
        {
            if (!_mergeReady || _startPosMm == null || _totalW <= 0 || _totalH <= 0) return BuildSingle();
            var merged = new Bitmap(_totalW, _totalH, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(merged))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                for (int i = 0; i < _latest.Length && i < _startPosMm.Length; i++)
                {
                    Frame f = _latest[i];
                    if (f == null) continue;
                    int xOff = (int)Math.Round((_startPosMm[i] - _minStartMm) / _refOpsMm);
                    using (var cam = BuildGray(f.Bytes, f.W, f.H))
                        g.DrawImageUnscaled(cam, xOff, 0);
                }
            }
            return merged;
        }

        // ── mm overlay + 實體校正（StatusChanged）──
        private void OnCanvasStatus(CanvasInfo info)
        {
            if (_disposed || _canvas == null || info.Zoom <= 0) return;

            double startMm, opsInMm;
            if (_mergeMode && _mergeReady)
            {
                startMm = _minStartMm; opsInMm = _refOpsMm; // 合圖座標系
            }
            else
            {
                int idx = _selectedCamId - 1;
                if (_opsUm == null || _startPosMm == null || idx < 0 || idx >= _opsUm.Length) { _canvas.SetRangeOverlay("", "", "", "", ""); return; }
                opsInMm = _opsUm[idx] / 1000.0; startMm = _startPosMm[idx];
            }
            if (opsInMm <= 0) { _canvas.SetRangeOverlay("", "", "", "", ""); return; }

            // X：canvas 邊 → 影像像素(全解析度，sf=1) → mm（PixelMmMapper 單一公式）
            double leftMm  = PixelMmMapper.PixelToMm((0 - info.PanOffset.X) / info.Zoom, startMm, opsInMm);
            double rightMm = PixelMmMapper.PixelToMm((_canvas.Width - info.PanOffset.X) / info.Zoom, startMm, opsInMm);
            double yPitch = _rowPitchMm > 0 ? _rowPitchMm : opsInMm; // Y 用 row pitch；缺則退回 ops
            double topMm = (0 - info.PanOffset.Y) / info.Zoom * yPitch;
            double botMm = (_canvas.Height - info.PanOffset.Y) / info.Zoom * yPitch;

            _canvas.SetPhysicalCalibration(opsInMm, _screenMmPerPx); // 三擊實體 1:1（sf=1）
            double physMag = _canvas.PhysicalMagnification;
            _canvas.SetRangeOverlay(physMag > 0 ? $"{physMag:F2}x" : "",
                $"{leftMm:F1}", $"{rightMm:F1}", $"{topMm:F1}", $"{botMm:F1}");

            double curMmX = PixelMmMapper.PixelToMm(info.ImageX, startMm, opsInMm);
            double curMmY = info.ImageY * yPitch;
            _canvas.SetCursorMm($"({curMmX:F2}, {curMmY:F2})");

            ViewRangeMmChanged?.Invoke(leftMm, rightMm); // overview 聯動
        }

        // ── 8-bit 灰階 byte[] → 8bppIndexed Bitmap（灰階調色盤一次算好）──
        private static readonly Color[] _grayEntries = BuildGrayEntries();
        private static Color[] BuildGrayEntries() { var e = new Color[256]; for (int i = 0; i < 256; i++) e[i] = Color.FromArgb(i, i, i); return e; }

        private static Bitmap BuildGray(byte[] data, int w, int h)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
            ColorPalette pal = bmp.Palette;
            for (int i = 0; i < 256; i++) pal.Entries[i] = _grayEntries[i];
            bmp.Palette = pal;
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            try { for (int y = 0; y < h; y++) Marshal.Copy(data, y * w, IntPtr.Add(bd.Scan0, y * bd.Stride), w); }
            finally { bmp.UnlockBits(bd); }
            return bmp;
        }

        public void Dispose()
        {
            _disposed = true;
            if (_timer != null) { _timer.Stop(); _timer.Dispose(); _timer = null; }
            for (int i = 0; i < _thumbs.Length; i++)
            {
                var pb = _thumbs[i];
                if (pb != null) { var old = pb.Image; pb.Image = null; old?.Dispose(); pb.Parent?.Controls.Remove(pb); pb.Dispose(); _thumbs[i] = null; }
            }
            if (_canvas != null)
            {
                _canvas.StatusChanged -= OnCanvasStatus;
                var old = _canvas.Image; _canvas.Image = null; old?.Dispose();
                _mainPanel?.Controls.Remove(_canvas);
                _canvas.Dispose(); _canvas = null;
            }
        }
    }
}
