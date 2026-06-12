using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Controls; // LiveDisplayView / GrayResize / GrayResizeCpu
using TanukiCv.Core;     // SystemInfo / PerfTimer

namespace MilGrabber.Monitor
{
    // PictureBox 顯示路徑（絞殺榕重寫版）：不讓 MIL 直接畫 panel，改訂閱 MilCamera.FrameReady →
    //   GetFrameBytes（8-bit 灰階全解析度）→ 餵共用 LiveDisplayView（主畫面 SmartCanvas + 縮圖條 + CPU 合圖 + LOD）。
    // **不再每幀做全解析度 GPU resize / pinned**（那是舊路的資源殺手：16384×3000=49MB/幀，撐爆 CUDA pinned）。
    //   顯示/縮圖/合圖一律 CPU；LOD 只裁「可見區」小圖，resize 用可插拔委派（GPU TanukiCv / CPU GrayResizeCpu，二選一跑測）。
    public partial class MilGrabberPbForm
    {
        // 每相機原始灰階緩衝（同相機 FrameReady 序列觸發 → 同 idx 無併發；PushFrame 內會複製，故可重用）
        private readonly byte[][] _srcBytes = new byte[SubPanelCount][];

        private volatile int  _resizeScale = 1;     // 餵入降採樣倍率（PbSettings.ResizeScale；1=全解析度。feedScale 給 LiveDisplayView）
        private volatile bool _mergeMode;            // 合圖 vs 單張（PbSettings.Display）
        private volatile bool _flipDisplay;          // 上下翻轉（PbSettings.Flip）

        // ── 動態 LOD（PbSettings.Lod = GPU/CPU）：LiveDisplayView 提供 LOD 機制（純 CPU），resize 由本檔插拔委派 ──
        private volatile bool _lodEnabled;
        private volatile bool _lodUseGpu = true;     // true=GPU(TanukiCv_Resize_GPU)、false=CPU(GrayResizeCpu)
        // GPU LOD provider 專用 pinned（只裁「可見區」→ 一次一塊，非每幀全幀；on-settle 觸發）
        private IntPtr _lodSrcPinned, _lodDstPinned;
        private int _lodSrcCap, _lodDstCap;
        private readonly object _lodBufLock = new object();
        private volatile bool _lodReleased;
        // LOD resize 計時：GPU/CPU 各一 → label 分開顯示（勾 GPU 時 CPU=0，反之）→ 視覺化驗證哪條在跑
        private readonly PerfTimer _lodGpuTimer = new PerfTimer();
        private readonly PerfTimer _lodCpuTimer = new PerfTimer();

        private LiveDisplayView _live;               // 共用顯示元件（主畫面+縮圖+合圖+LOD）；存活到 form 關閉
        private double _screenMmPerPx;               // 螢幕實體 mm/px（SystemInfo 查一次）

        // ── 顯示模式（初始化前選）：MIL 直繪 vs PictureBox（LiveDisplayView） ──
        private bool _milMode;                                              // true=MIL 直繪、false=PictureBox
        private readonly Panel[] _displayPanels = new Panel[SubPanelCount]; // MIL 模式子畫面（MIL 直繪目標）
        private Panel _mainPanelMil;                                        // MIL 模式主畫面（MIL secondary 目標）

        /// <summary>建立共用 LiveDisplayView（PictureBox 模式主畫面+縮圖；建構式呼叫）。</summary>
        private void SetupPbMain()
        {
            _screenMmPerPx = SystemInfo.GetScreenMetrics().MmPerPx;
            _live = new LiveDisplayView(panelMain, _camContainers, _screenMmPerPx);
            _live.SelectRequested += camId => SelectCamera(camId - 1); // 1-based camId → 0-based idx

            // 游標剖面（L0）→ 重用 app 的曲線圖設計（Column/RowCurveChartHelper），閾值線關（純剖面不顯 mura 門檻）
            _profileChartX = new ColumnCurveChartHelper(chartProfileX) { ShowThresholds = false }; // 下方：沿 X
            _profileChartY = new RowCurveChartHelper(chartProfileY)    { ShowThresholds = false }; // 右側：沿 Y
            _live.CursorProfileChanged += OnCursorProfile;
        }

        private ColumnCurveChartHelper _profileChartX;
        private RowCurveChartHelper _profileChartY;

        /// <summary>游標剖面（UI 執行緒，StatusChanged 來）→ 共用曲線圖（降採樣 ~600 點防卡）+ zoom 同步。</summary>
        private void OnCursorProfile(LiveDisplayView.CursorProfile p)
        {
            if (_isReleasing) return;
            if (p == null) { _profileChartX?.Clear(); _profileChartY?.Clear(); return; } // 游標出界/離開畫布 → 剖面歸零
            if (_profileChartX != null && p.RowProfile != null && p.RowProfile.Length > 0)
            {
                int step = Math.Max(1, p.RowProfile.Length / 600);
                _profileChartX.SetOps(p.OpsXmm * step * 1000.0);  // µm；降採樣 → 間距 ×step
                _profileChartX.UpdateDataAndView(Downsample(p.RowProfile, step), null, p.StartXmm, p.ViewLeftMm, p.ViewRightMm);
            }
            if (_profileChartY != null && p.ColProfile != null && p.ColProfile.Length > 0)
            {
                int step = Math.Max(1, p.ColProfile.Length / 600);
                _profileChartY.SetRowPitch(p.OpsYmm * step);
                _profileChartY.UpdateData(Downsample(p.ColProfile, step), null);
                _profileChartY.UpdateViewRange(p.ViewTopMm, p.ViewBotMm);
            }
        }

        private static float[] Downsample(byte[] b, int step)
        {
            int n = (b.Length + step - 1) / step;
            var f = new float[n]; int k = 0;
            for (int i = 0; i < b.Length && k < n; i += step) f[k++] = b[i];
            return f;
        }

        // ── 設定面板（tabParams「設定」tab；全部選項融入 PropertyGrid，仿主程式 propertyGridSettings 的 SSoT 設計）──
        private readonly PbSettings _settings = new PbSettings();
        private volatile int _mergeStrategy;   // 0=Midline、1=RightOverLeft、2=LeftOverRight（對應 MergeOverlap）
        private volatile bool _mergeAllMode;   // 合圖全部：含無畫面相機（黑色占空間）
        private bool _modeLocked;              // 繪圖模式（PbSettings.Init）初始化後鎖定（CreateDisplaysForMode 設、btnRelease 解）
        private PbSettings.InitMode _lockedInitMode; // 鎖定當下的模式（軟鎖：改了就還原）

        /// <summary>「設定」tab 接線：填 ops/start 預設 + 綁 PropertyGrid（合圖方式 / 動態LOD / 重疊策略 / 上下翻轉 / FOV / 縮圖倍率 + ops/start）。</summary>
        private void SetupMergeTab()
        {
            _settings.Ops.Set(new[] { 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625, 24.4140625 });
            _settings.Start.Set(new[] { 0.0, 345, 690, 1035, 1380, 1725, 2070, 2415 });

            propertyGridMerge.SelectedObject = _settings;
            propertyGridMerge.PropertyValueChanged += (s, e) => ApplyPbSettings(); // 改任一設定 → 一次套用全部
            ApplyPbSettings();
        }

        /// <summary>PropertyGrid 改任一設定 → 一次套用全部到 LiveDisplayView（取代原本散落的 chk/num handler；互斥選項用 enum）。</summary>
        private void ApplyPbSettings()
        {
            // 繪圖模式（PictureBox / MIL 直繪）初始化後鎖定（軟鎖：改了就還原 + 刷新顯示）
            if (_modeLocked && _settings.Init != _lockedInitMode)
            {
                _settings.Init = _lockedInitMode;
                propertyGridMerge.Refresh();
            }

            // 顯示模式（單張 / 合圖 / 合圖全部，互斥 → enum）
            _mergeAllMode  = _settings.Display == PbSettings.DisplayMode.MergeAll;
            _mergeMode     = _settings.Display != PbSettings.DisplayMode.Single;
            _mergeStrategy = (int)_settings.Overlap;
            _resizeScale   = Math.Max(1, _settings.ResizeScale);
            _flipDisplay   = _settings.Flip;
            _lodUseGpu     = _settings.Lod == PbSettings.LodMode.Gpu;
            _lodEnabled    = _settings.Lod != PbSettings.LodMode.Off;

            ApplyLayout(); // FOV / ops / start + 縮圖倍率 → SetLayout

            if (_live != null)
            {
                _live.ThumbSelectedColor = _settings.ThumbBorder;   // 選取框唯一來源 = sdk ThumbView，色由設定層選
                _live.MergeAll      = _mergeAllMode;
                _live.SetMergeMode(_mergeMode);
                _live.MergeStrategy = (MergeOverlap)_mergeStrategy;
                _live.FlipVertical  = _flipDisplay;
                if (!_lodEnabled) _live.DisableLod();
                else _live.EnableLod(_lodUseGpu ? (GrayResize)LodResizeGpu : LodResizeCpu); // resize 委派二選一（GPU/CPU 跑測比較）
            }
            // 翻轉也套到 MIL 模式的相機（與舊 chkFlipVertical handler 行為一致）
            if (_cams != null)
                foreach (var cam in _cams)
                    if (cam != null) cam.FlipVertical = _flipDisplay;
        }

        /// <summary>座標來源統一：FOV（fovMm/影像寬，全相機等 ops）優先；沒 FOV 用 PropertyGrid 的 ops。
        /// start 取 PropertyGrid。換算成 LiveDisplayView 的 SetLayout（單張/合圖/overlay 同一條路）。</summary>
        private void ApplyLayout()
        {
            if (_live == null) return;
            double[] start = _settings.Start.ToArray();
            double[] ops = _settings.Ops.ToArray();

            double fovMm = _settings.FovMm;
            int fw = SelectedFrameWidth();
            if (fovMm > 0 && fw > 0)
            {
                double fovOpsUm = fovMm / fw * 1000.0; // mm/px → µm（全相機統一）
                for (int i = 0; i < ops.Length; i++) ops[i] = fovOpsUm;
            }
            _live.SetLayout(start, ops, _resizeScale, 0);
            _live.MergeStrategy = (MergeOverlap)_mergeStrategy;
        }

        /// <summary>選中相機的全解析度影像寬（FOV→ops 用）；無則 0。</summary>
        private int SelectedFrameWidth()
        {
            return (_selectedCam >= 0 && _cams != null && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                ? _cams[_selectedCam].FrameWidth : 0;
        }

        // ── 合圖參數（PropertyGrid 綁定物件；Cam1~8 可展開，仿主程式 MachineLayoutConfig 風格）──
        [TypeConverter(typeof(ExpandableObjectConverter))]
        private sealed class CamRow8
        {
            [DisplayName("Cam 1")] public double Cam1 { get; set; }
            [DisplayName("Cam 2")] public double Cam2 { get; set; }
            [DisplayName("Cam 3")] public double Cam3 { get; set; }
            [DisplayName("Cam 4")] public double Cam4 { get; set; }
            [DisplayName("Cam 5")] public double Cam5 { get; set; }
            [DisplayName("Cam 6")] public double Cam6 { get; set; }
            [DisplayName("Cam 7")] public double Cam7 { get; set; }
            [DisplayName("Cam 8")] public double Cam8 { get; set; }

            public double[] ToArray() => new[] { Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7, Cam8 };
            public void Set(double[] v)
            {
                if (v == null) return;
                if (v.Length > 0) Cam1 = v[0]; if (v.Length > 1) Cam2 = v[1]; if (v.Length > 2) Cam3 = v[2]; if (v.Length > 3) Cam4 = v[3];
                if (v.Length > 4) Cam5 = v[4]; if (v.Length > 5) Cam6 = v[5]; if (v.Length > 6) Cam7 = v[6]; if (v.Length > 7) Cam8 = v[7];
            }
            public override string ToString() => "";
        }

        // 設定面板綁定物件（仿主程式 InspectionSettings：互斥選項 enum、可展開 CamRow8）。
        // PropertyValueChanged → ApplyPbSettings 一次套用，view 不擁有邏輯（SSoT）。
        private sealed class PbSettings
        {
            [TypeConverter(typeof(EnumDescConverter))]
            public enum DisplayMode { [Description("單張")] Single, [Description("合圖")] Merge, [Description("合圖全部")] MergeAll }
            [TypeConverter(typeof(EnumDescConverter))]
            public enum LodMode { [Description("關閉")] Off, [Description("GPU")] Gpu, [Description("CPU")] Cpu }
            [TypeConverter(typeof(EnumDescConverter))]
            public enum OverlapMode { [Description("中線分界")] Midline, [Description("右覆蓋左")] RightOverLeft, [Description("左覆蓋右")] LeftOverRight }
            [TypeConverter(typeof(EnumDescConverter))]
            public enum InitMode { [Description("PictureBox")] PictureBox, [Description("MIL 直繪")] Mil }

            [Category("顯示")][DisplayName("繪圖模式")][Description("PictureBox（LiveDisplayView CPU 繪，本範例主路徑）/ MIL 直繪（MIL 直接畫 panel）。初始化後鎖定，需釋放才可改")]
            public InitMode Init { get; set; } = InitMode.PictureBox;
            [Category("顯示")][DisplayName("合圖方式")][Description("單張 / 合圖 / 合圖全部（含無畫面相機黑占位）")]
            public DisplayMode Display { get; set; } = DisplayMode.Single;
            [Category("顯示")][DisplayName("動態LOD")][Description("關閉 / GPU(TanukiCv) / CPU(GrayResizeCpu)。CPU 通常較順（無 H2D/D2H 來回）")]
            public LodMode Lod { get; set; } = LodMode.Off;
            [Category("顯示")][DisplayName("重疊策略")][Description("合圖重疊區分界：中線 / 右覆蓋左 / 左覆蓋右")]
            public OverlapMode Overlap { get; set; } = OverlapMode.Midline;
            [Category("顯示")][DisplayName("上下翻轉")][Description("線掃相機由下往上拍 → 顯示需上下翻")]
            public bool Flip { get; set; }
            [Category("顯示")][DisplayName("FOV (mm)")][Description("選中相機視野寬（>0 時全相機等 ops 由此算；0=改用下方 OPS）")]
            public int FovMm { get; set; } = 400;
            [Category("顯示")][DisplayName("縮圖倍率")][Description("餵入降採樣倍率（1=全解析度；>1 先 CPU stride 降採樣再餵）")]
            public int ResizeScale { get; set; } = 1;
            [Category("顯示")][DisplayName("選取框色")][Description("縮圖選取高亮框色（sdk ThumbView 唯一來源；各 UI 自選）")]
            public Color ThumbBorder { get; set; } = Color.Orange;

            [Category("合圖佈局")][DisplayName("OPS (µm)")]  public CamRow8 Ops   { get; } = new CamRow8();
            [Category("合圖佈局")][DisplayName("Start (mm)")] public CamRow8 Start { get; } = new CamRow8();
        }

        /// <summary>PropertyGrid enum 顯示 [Description] 中文（值的內部名稱仍英文）。</summary>
        private sealed class EnumDescConverter : EnumConverter
        {
            public EnumDescConverter(Type type) : base(type) { }
            public override object ConvertTo(ITypeDescriptorContext c, System.Globalization.CultureInfo cu, object value, Type destType)
            {
                if (destType == typeof(string) && value != null)
                {
                    var fi = EnumType.GetField(value.ToString());
                    var attr = fi?.GetCustomAttributes(typeof(DescriptionAttribute), false);
                    if (attr != null && attr.Length > 0) return ((DescriptionAttribute)attr[0]).Description;
                }
                return base.ConvertTo(c, cu, value, destType);
            }
            public override object ConvertFrom(ITypeDescriptorContext c, System.Globalization.CultureInfo cu, object value)
            {
                if (value is string s)
                    foreach (var f in EnumType.GetFields())
                    {
                        var attr = f.GetCustomAttributes(typeof(DescriptionAttribute), false);
                        if (attr.Length > 0 && ((DescriptionAttribute)attr[0]).Description == s) return Enum.Parse(EnumType, f.Name);
                    }
                return base.ConvertFrom(c, cu, value);
            }
        }

        /// <summary>依「初始化前選的模式」建立顯示控制項（btnInit 開頭呼叫）。
        /// MIL 模式：每容器疊 Panel（MIL 直繪目標）+ 主畫面疊 Panel；PictureBox 模式：用現成 LiveDisplayView。</summary>
        private void CreateDisplaysForMode()
        {
            _lodReleased = false; // 重新 init → 允許 LOD provider 再配置 pinned
            _milMode = _settings.Init == PbSettings.InitMode.Mil;
            _lockedInitMode = _settings.Init; _modeLocked = true; // 初始化後鎖定繪圖模式（PropertyGrid 軟鎖）
            propertyGridMerge.Refresh();
            if (!_milMode) return;

            for (int i = 0; i < SubPanelCount; i++)
            {
                var p = new Panel { Dock = DockStyle.Fill, BackColor = Color.Black };
                int idx = i;
                p.MouseClick += (s, e) => SelectCamera(idx);
                _camContainers[i].Controls.Add(p);
                p.BringToFront(); // 蓋住 LiveDisplayView 的縮圖
                _displayPanels[i] = p;
            }
            _mainPanelMil = new Panel { Dock = DockStyle.Fill, BackColor = Color.Black };
            panelMain.Controls.Add(_mainPanelMil);
            _mainPanelMil.BringToFront(); // 蓋住 SmartCanvas
        }

        // 縮圖倍率 / FOV / LOD GPU↔CPU 的切換已融入 PropertyGrid（PbSettings）→ ApplyPbSettings 統一套用。

        // ── LOD resize 委派（LiveDisplayView 在背景執行緒呼叫；只縮「可見區」一塊） ──
        /// <summary>GPU 版：裁好的可見區 → pinned → TanukiCv_Resize_GPU → 灰階 bytes。計時記 _lodGpuTimer。</summary>
        private byte[] LodResizeGpu(byte[] src, int sw, int sh, int dw, int dh)
        {
            int srcPix = sw * sh, dstPix = dw * dh;
            byte[] dst;
            lock (_lodBufLock)
            {
                if (_lodReleased) return null;
                if (_lodSrcCap < srcPix)
                {
                    if (_lodSrcPinned != IntPtr.Zero) NativeResize.TanukiCv_FreePinned(_lodSrcPinned);
                    _lodSrcPinned = NativeResize.TanukiCv_AllocPinned((ulong)srcPix); _lodSrcCap = srcPix;
                }
                if (_lodDstCap < dstPix)
                {
                    if (_lodDstPinned != IntPtr.Zero) NativeResize.TanukiCv_FreePinned(_lodDstPinned);
                    _lodDstPinned = NativeResize.TanukiCv_AllocPinned((ulong)dstPix); _lodDstCap = dstPix;
                }
                if (_lodSrcPinned == IntPtr.Zero || _lodDstPinned == IntPtr.Zero) return null;

                _lodGpuTimer.Start();                                  // ── 真 improcess（H2D+縮+D2H）──
                Marshal.Copy(src, 0, _lodSrcPinned, srcPix);
                NativeResize.TanukiCv_Resize_GPU(_lodSrcPinned, sw, sh, _lodDstPinned, dw, dh);
                dst = new byte[dstPix];
                Marshal.Copy(_lodDstPinned, dst, 0, dstPix);
                _lodGpuTimer.Stop();
            }
            return dst;
        }

        /// <summary>CPU 版：純 GrayResizeCpu（無 pinned/GPU）。計時記 _lodCpuTimer。</summary>
        private byte[] LodResizeCpu(byte[] src, int sw, int sh, int dw, int dh)
        {
            _lodCpuTimer.Start();
            byte[] dst = GrayResizeCpu.Resize(src, sw, sh, dw, dh);
            _lodCpuTimer.Stop();
            return dst;
        }

        /// <summary>每幀（MIL 回呼執行緒）：取 8-bit 灰階全解析度 → 餵 LiveDisplayView（不做 GPU resize / pinned）。
        /// numResize>1 時先 CPU stride 降採樣（便宜）→ feedScale 對應。</summary>
        private void OnCameraFrame(int idx, MilCamera cam, MIL_ID buffer)
        {
            if (_isReleasing) return;
            int fw = cam.FrameWidth, fh = cam.FrameHeight;
            if (fw <= 0 || fh <= 0) return;
            try
            {
                int n = fw * fh;
                if (_srcBytes[idx] == null || _srcBytes[idx].Length < n) _srcBytes[idx] = new byte[n];
                cam.GetFrameBytes(buffer, _srcBytes[idx]); // 8-bit 灰階原圖（全解析度）

                int scale = _resizeScale; if (scale < 1) scale = 1;
                if (scale == 1)
                {
                    _live?.PushFrame(idx + 1, _srcBytes[idx], fw, fh); // 全解析度（PushFrame 內會複製）
                }
                else
                {
                    int dw = Math.Max(1, fw / scale), dh = Math.Max(1, fh / scale);
                    var small = new byte[dw * dh];
                    byte[] s = _srcBytes[idx];
                    for (int y = 0; y < dh; y++)
                    {
                        int srow = (y * scale) * fw, drow = y * dw;
                        for (int x = 0; x < dw; x++) small[drow + x] = s[srow + x * scale];
                    }
                    _live?.PushFrame(idx + 1, small, dw, dh);
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[PbFrame] CAM idx{idx}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>更新計時 label（StatusTimer 每 500ms 呼叫）：比較 MIL 直繪 vs PictureBox 的顯示/LOD 成本。</summary>
        private void UpdateTimingLabel()
        {
            if (_lblTiming == null) return;
            double fps = (_selectedCam >= 0 && _cams != null && _selectedCam < _cams.Length && _cams[_selectedCam] != null)
                ? _cams[_selectedCam].CurrentFps : 0;
            if (_milMode)
            {
                _lblTiming.Text = $"模式: MIL 直繪\nFPS: {fps:F1}\n(縮圖/顯示由 MIL\n 硬體直繪，無 CPU 成本)";
                Trace.WriteLine($"[PbTiming] mode=MIL FPS={fps:F1}");
                return;
            }

            SmartCanvas c = _live?.Canvas;
            double paintMax  = c?.ResetMaxPaintMs() ?? 0;
            float  zoomRel   = c?.ZoomRelativeToFit ?? 1f;
            double physMag   = c?.PhysicalMagnification ?? 0;
            string merge     = _mergeAllMode ? "合圖全部" : (_mergeMode ? "合圖" : "單張");
            bool   lodOn     = c != null && c.LodActive;
            string lod       = lodOn ? (_lodUseGpu ? "GPU" : "CPU") : (_lodEnabled ? "待命" : "OFF");
            double lodGpuMax = _lodGpuTimer.ResetMax();   // 勾 GPU 時 >0、CPU=0；反之 → 一眼驗證哪條在跑
            double lodCpuMax = _lodCpuTimer.ResetMax();
            string physStr   = physMag > 0 ? $"{physMag:F2}x" : "-（需FOV）";
            _lblTiming.Text =
                $"模式: PictureBox\n畫面: {merge}\nLOD: {lod}\n  GPU縮放: {lodGpuMax:F1} ms\n  CPU縮放: {lodCpuMax:F1} ms\n餵入倍率: {_resizeScale}\nzoom×fit: {zoomRel:F2}\n實體倍率: {physStr}\n顯示(max): {paintMax:F1} ms\nFPS: {fps:F1}";
            Trace.WriteLine($"[PbTiming] mode=PB {merge} LOD={lod} GPU縮放={lodGpuMax:F1}ms CPU縮放={lodCpuMax:F1}ms feed={_resizeScale} zoom×fit={zoomRel:F2} 實體={physStr} 顯示max={paintMax:F1}ms FPS={fps:F1}");
        }

        /// <summary>釋放（ReleaseAll 呼叫，重 init 與關窗都會走）：MIL 疊圖 + LOD pinned。LiveDisplayView 存活到關窗。</summary>
        private void ReleasePictureBoxDisplays()
        {
            for (int i = 0; i < SubPanelCount; i++)
            {
                _srcBytes[i] = null;
                // MIL 模式疊上去的 Panel：移除 + dispose（下次 init 可重選模式）
                if (_displayPanels[i] != null)
                {
                    _camContainers[i]?.Controls.Remove(_displayPanels[i]);
                    _displayPanels[i].Dispose();
                    _displayPanels[i] = null;
                }
            }
            // LOD：停用 + 鎖內釋放 pinned（等背景 provider 用完再釋放，防 use-after-free）
            if (_live != null && _live.Canvas != null && _live.Canvas.LodActive) _live.DisableLod();
            lock (_lodBufLock)
            {
                _lodReleased = true;
                if (_lodSrcPinned != IntPtr.Zero) { NativeResize.TanukiCv_FreePinned(_lodSrcPinned); _lodSrcPinned = IntPtr.Zero; _lodSrcCap = 0; }
                if (_lodDstPinned != IntPtr.Zero) { NativeResize.TanukiCv_FreePinned(_lodDstPinned); _lodDstPinned = IntPtr.Zero; _lodDstCap = 0; }
            }

            if (_mainPanelMil != null)
            {
                panelMain?.Controls.Remove(_mainPanelMil);
                _mainPanelMil.Dispose();
                _mainPanelMil = null;
            }
            _modeLocked = false; // 釋放後可重選繪圖模式（PropertyGrid 解鎖）
            propertyGridMerge.Refresh();
        }
    }
}
