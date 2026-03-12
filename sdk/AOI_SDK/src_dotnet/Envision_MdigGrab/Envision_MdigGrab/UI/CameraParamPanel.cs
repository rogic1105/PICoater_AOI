using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 可收折的相機參數調整面板。
    /// 每個參數同時提供 TrackBar（拖動）與 NumericUpDown（鍵盤輸入），兩者互相同步。
    /// 拖動 TrackBar 放開後才送硬體指令；NumericUpDown 數值變更立即送出。
    /// Exposure 上限自動對應目前 LineRate 週期的 90%。
    /// </summary>
    public class CameraParamPanel : Panel
    {
        // TrackBars
        private readonly Dictionary<int, TrackBar> _expTracks      = new Dictionary<int, TrackBar>();
        private readonly Dictionary<int, TrackBar> _lineRateTracks = new Dictionary<int, TrackBar>();
        private readonly Dictionary<int, TrackBar> _heightTracks   = new Dictionary<int, TrackBar>();

        // NumericUpDowns
        private readonly Dictionary<int, NumericUpDown> _expInputs      = new Dictionary<int, NumericUpDown>();
        private readonly Dictionary<int, NumericUpDown> _lineRateInputs = new Dictionary<int, NumericUpDown>();
        private readonly Dictionary<int, NumericUpDown> _heightInputs   = new Dictionary<int, NumericUpDown>();

        /// <summary>目前正在被拖動的 TrackBar，SyncFromCamera 期間跳過。</summary>
        private readonly HashSet<TrackBar> _dragging = new HashSet<TrackBar>();

        /// <summary>程式碼同步期間設為 true，防止 TrackBar ↔ NumericUpDown 互相觸發。</summary>
        private bool _suppressSync = false;

        private const int RowHeight = 52;
        private const int PadV      = 8;

        // 固定範圍
        private const int ExpMin      =      1;
        private const int LineRateMin =      1;
        private const int LineRateMax = 10_000;
        private const int HeightMin   =      1;
        private const int HeightMax   = 10_000;

        /// <summary>Exposure 套用，參數：(camId, μs)</summary>
        public event Action<int, double> ExposureApply;

        /// <summary>LineRate 套用，參數：(camId, Hz)</summary>
        public event Action<int, double> LineRateApply;

        /// <summary>Height 套用，參數：(camId, px)</summary>
        public event Action<int, int> HeightApply;

        public CameraParamPanel(IList<CameraConfig> configs)
        {
            BackColor   = Color.FromArgb(240, 242, 248);
            BorderStyle = BorderStyle.FixedSingle;
            Padding     = new Padding(6, PadV / 2, 6, PadV / 2);
            Height      = configs.Count * RowHeight + PadV;
            BuildUI(configs);
        }

        // ================= UI 建構 =================

        private void BuildUI(IList<CameraConfig> configs)
        {
            var table = new TableLayoutPanel
            {
                Dock        = DockStyle.Fill,
                ColumnCount = 12,
                RowCount    = configs.Count,
            };

            //  0: CAM label   1: Exp label   2: Exp NUD   3: TrackBar Exp
            //  4: spacer       5: LR label    6: LR NUD    7: TrackBar LR
            //  8: spacer       9: H label    10: H NUD    11: TrackBar H
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  55));  //  0 CAM
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  40));  //  1 Exp label
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  80));  //  2 Exp NUD
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,  28f));  //  3 TrackBar Exp
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  12));  //  4 spacer
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  65));  //  5 LR label
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  85));  //  6 LR NUD
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,  36f));  //  7 TrackBar LR
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  12));  //  8 spacer
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  55));  //  9 H label
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,  80));  // 10 H NUD
            table.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,  36f));  // 11 TrackBar H

            for (int r = 0; r < configs.Count; r++)
                table.RowStyles.Add(new RowStyle(SizeType.Absolute, RowHeight));

            int row = 0;
            foreach (var cfg in configs)
            {
                int camId = cfg.Id;

                // ── Camera label ──────────────────────────────────────────
                table.Controls.Add(MakeLabel($"CAM {camId}", ContentAlignment.MiddleCenter, bold: true), 0, row);

                // ── Exposure ───────────────────────────────────────────────
                int initExpMax = CalcExpMax(LineRateMax);
                int initExp    = Math.Max(ExpMin, Math.Min(initExpMax, (int)cfg.ExposureUs));

                table.Controls.Add(MakeLabel("Exp(μs)", ContentAlignment.MiddleRight), 1, row);

                var expNud = MakeNud(ExpMin, initExpMax, initExp, decimalPlaces: 0, increment: 1);
                _expInputs[camId] = expNud;
                table.Controls.Add(expNud, 2, row);

                var expTrack = MakeTrackBar(ExpMin, initExpMax, initExp, smallChange: 1, largeChange: 10);
                _expTracks[camId] = expTrack;
                RegisterDragTracking(expTrack);
                expTrack.Scroll  += (s, e) => OnTrackScroll(_expTracks, _expInputs, camId);
                expTrack.MouseUp += (s, e) => ExposureApply?.Invoke(camId, _expTracks[camId].Value);
                expTrack.KeyUp   += (s, e) => ExposureApply?.Invoke(camId, _expTracks[camId].Value);
                table.Controls.Add(expTrack, 3, row);

                expNud.ValueChanged += (s, e) => OnNudChanged(_expNudToTrack(camId), expNud,
                    v => ExposureApply?.Invoke(camId, v));

                // ── LineRate ───────────────────────────────────────────────
                table.Controls.Add(new Label(), 4, row);
                table.Controls.Add(MakeLabel("LineRate(Hz)", ContentAlignment.MiddleRight), 5, row);

                var lrNud = MakeNud(LineRateMin, LineRateMax, LineRateMax, decimalPlaces: 0, increment: 100);
                _lineRateInputs[camId] = lrNud;
                table.Controls.Add(lrNud, 6, row);

                var lrTrack = MakeTrackBar(LineRateMin, LineRateMax, LineRateMax, smallChange: 100, largeChange: 500);
                _lineRateTracks[camId] = lrTrack;
                RegisterDragTracking(lrTrack);
                lrTrack.Scroll  += (s, e) => OnTrackScroll(_lineRateTracks, _lineRateInputs, camId);
                lrTrack.MouseUp += (s, e) => OnLineRateApply(camId);
                lrTrack.KeyUp   += (s, e) => OnLineRateApply(camId);
                table.Controls.Add(lrTrack, 7, row);

                lrNud.ValueChanged += (s, e) => OnNudChanged(_lrNudToTrack(camId), lrNud,
                    v => { UpdateExpMaxByLineRate(camId, (int)v); LineRateApply?.Invoke(camId, v); });

                // ── Height ─────────────────────────────────────────────────
                table.Controls.Add(new Label(), 8, row);
                table.Controls.Add(MakeLabel("Height(px)", ContentAlignment.MiddleRight), 9, row);

                var hNud = MakeNud(HeightMin, HeightMax, 2_048, decimalPlaces: 0, increment: 64);
                _heightInputs[camId] = hNud;
                table.Controls.Add(hNud, 10, row);

                var hTrack = MakeTrackBar(HeightMin, HeightMax, 2_048, smallChange: 64, largeChange: 512);
                _heightTracks[camId] = hTrack;
                RegisterDragTracking(hTrack);
                hTrack.Scroll  += (s, e) => OnTrackScroll(_heightTracks, _heightInputs, camId);
                hTrack.MouseUp += (s, e) => HeightApply?.Invoke(camId, _heightTracks[camId].Value);
                hTrack.KeyUp   += (s, e) => HeightApply?.Invoke(camId, _heightTracks[camId].Value);
                table.Controls.Add(hTrack, 11, row);

                hNud.ValueChanged += (s, e) => OnNudChanged(_hNudToTrack(camId), hNud,
                    v => HeightApply?.Invoke(camId, (int)v));

                row++;
            }

            Controls.Add(table);
        }

        // ── 輔助：NUD → Track 對應（避免 lambda 捕獲問題）─────────────────
        private TrackBar _expNudToTrack(int camId)      => _expTracks.TryGetValue(camId, out var t)      ? t : null;
        private TrackBar _lrNudToTrack(int camId)       => _lineRateTracks.TryGetValue(camId, out var t) ? t : null;
        private TrackBar _hNudToTrack(int camId)        => _heightTracks.TryGetValue(camId, out var t)   ? t : null;

        // ================= 事件處理 =================

        /// <summary>TrackBar 拖動中：同步 NUD 數值（不送硬體）。</summary>
        private void OnTrackScroll(
            Dictionary<int, TrackBar> tracks,
            Dictionary<int, NumericUpDown> nuds, int camId)
        {
            if (!tracks.TryGetValue(camId, out var track)) return;
            if (!nuds.TryGetValue(camId, out var nud)) return;

            _suppressSync = true;
            nud.Value = Clamp(nud, track.Value);
            _suppressSync = false;
        }

        /// <summary>
        /// NUD 數值變更：同步 TrackBar，並送出硬體指令。
        /// _suppressSync 為 true 時代表是程式碼更新，跳過。
        /// </summary>
        private void OnNudChanged(TrackBar track, NumericUpDown nud, Action<double> apply)
        {
            if (_suppressSync) return;
            if (track == null) return;

            _suppressSync = true;
            track.Value = Clamp(track, (int)nud.Value);
            _suppressSync = false;

            apply((double)nud.Value);
        }

        /// <summary>LineRate TrackBar 放開：更新 Exposure 上限後送硬體指令。</summary>
        private void OnLineRateApply(int camId)
        {
            if (!_lineRateTracks.TryGetValue(camId, out var lrTrack)) return;
            int hz = lrTrack.Value;

            // 同步 NUD
            _suppressSync = true;
            if (_lineRateInputs.TryGetValue(camId, out var lrNud))
                lrNud.Value = Clamp(lrNud, hz);
            _suppressSync = false;

            UpdateExpMaxByLineRate(camId, hz);
            LineRateApply?.Invoke(camId, hz);
        }

        // ================= Exposure 上限計算 =================

        private static int CalcExpMax(int lineRateHz)
        {
            if (lineRateHz <= 0) return ExpMin;
            return Math.Max(ExpMin, (int)(1_000_000.0 / lineRateHz * 0.9));
        }

        /// <summary>
        /// 依 LineRate 更新 Exposure TrackBar 與 NUD 的上限，超出時自動夾緊。
        /// 夾緊時數值標為橘紅色。
        /// </summary>
        private void UpdateExpMaxByLineRate(int camId, int lineRateHz)
        {
            if (!_expTracks.TryGetValue(camId, out var expTrack)) return;
            if (!_expInputs.TryGetValue(camId, out var expNud)) return;

            int newMax = CalcExpMax(lineRateHz);

            expTrack.Maximum       = newMax;
            expTrack.TickFrequency = Math.Max(1, newMax / 10);

            _suppressSync = true;
            expNud.Maximum = newMax;
            _suppressSync = false;

            if (expTrack.Value > newMax)
            {
                _suppressSync = true;
                expTrack.Value = newMax;
                expNud.Value   = newMax;
                _suppressSync = false;

                expNud.ForeColor = Color.OrangeRed;
                ExposureApply?.Invoke(camId, newMax);
            }
            else
            {
                expNud.ForeColor = SystemColors.WindowText;
            }
        }

        // ================= 從硬體同步 =================

        /// <summary>
        /// Timer Tick 時將硬體讀回值同步到 TrackBar 和 NUD。
        /// 拖動中（_dragging）或差異 ≤ 5% 時不更新，避免覆蓋使用者操作。
        /// </summary>
        public void SyncFromCamera(int camId, double exposureUs, double lineRateHz)
        {
            SyncPair(_expTracks, _expInputs, camId, (int)exposureUs);
            SyncPair(_lineRateTracks, _lineRateInputs, camId, (int)lineRateHz);

            if ((int)lineRateHz > 0)
                UpdateExpMaxByLineRate(camId, (int)lineRateHz);
        }

        private void SyncPair(
            Dictionary<int, TrackBar>      tracks,
            Dictionary<int, NumericUpDown> nuds,
            int camId, int hwValue)
        {
            if (!tracks.TryGetValue(camId, out var track)) return;
            if (!nuds.TryGetValue(camId, out var nud)) return;
            if (hwValue <= 0) return;
            if (_dragging.Contains(track)) return;

            int clamped = Math.Max(track.Minimum, Math.Min(track.Maximum, hwValue));
            double diff = Math.Abs(clamped - track.Value) / (double)Math.Max(1, track.Value);
            if (diff <= 0.05) return;

            _suppressSync = true;
            track.Value = clamped;
            nud.Value   = Clamp(nud, clamped);
            _suppressSync = false;
        }

        // ================= 私有輔助 =================

        private void RegisterDragTracking(TrackBar track)
        {
            track.MouseDown += (s, e) => _dragging.Add(track);
            track.MouseUp   += (s, e) => _dragging.Remove(track);
        }

        private static TrackBar MakeTrackBar(int min, int max, int value, int smallChange, int largeChange)
        {
            return new TrackBar
            {
                Minimum       = min,
                Maximum       = max,
                Value         = Math.Max(min, Math.Min(max, value)),
                SmallChange   = smallChange,
                LargeChange   = largeChange,
                TickFrequency = Math.Max(1, (max - min) / 10),
                Dock          = DockStyle.Fill,
                AutoSize      = false,
            };
        }

        private static NumericUpDown MakeNud(int min, int max, int value, int decimalPlaces, int increment)
        {
            return new NumericUpDown
            {
                Minimum       = min,
                Maximum       = max,
                Value         = Math.Max(min, Math.Min(max, value)),
                DecimalPlaces = decimalPlaces,
                Increment     = increment,
                Dock          = DockStyle.Fill,
            };
        }

        private static Label MakeLabel(string text, ContentAlignment align, bool bold = false) => new Label
        {
            Text      = text,
            TextAlign = align,
            Dock      = DockStyle.Fill,
            AutoSize  = false,
            Font      = bold ? new Font(SystemFonts.DefaultFont, FontStyle.Bold)
                             : SystemFonts.DefaultFont,
        };

        private static int    Clamp(TrackBar      c, int     v) => Math.Max(c.Minimum, Math.Min(c.Maximum, v));
        private static decimal Clamp(NumericUpDown c, decimal v) => Math.Max(c.Minimum, Math.Min(c.Maximum, v));
    }
}
