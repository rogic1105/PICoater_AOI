using System;
using System.Windows.Forms;
using MilGrabber.Core;

namespace MilGrabber.Monitor
{
    // MilGrabberPbForm 的「參數 Tab 接線」分區（Designer 固定 8 相機）：
    // 每個 tab（曝光/線掃/高度）有「全部相機」列 + Cam1~8 列，控制項本體在 Designer.cs 宣告，本檔負責接線
    // （雙向同步 + 套用 _cams）與初值/enable。控制項陣列（_tbExposure…）由主檔 ctor 組裝。
    public partial class MilGrabberPbForm
    {
        private const int ExpMin = 1, ExpMaxCap = 10000;       // 曝光 μs（ExpMaxCap = 絕對上限；動態上限走 CalcExpMax）
        private const int LrMin = 100, LrMax = 100000;          // 線掃 Hz
        private const int HtMin = 1, HtMax = 10000;             // 高度 px

        private enum ParamKind { Exposure, LineRate, Height }

        /// <summary>
        /// 依線掃速率算曝光動態上限（仿主程式）：lrHz≤0 → 絕對上限 ExpMaxCap；
        /// 否則 clamp(floor(900000 / lrHz), ExpMin, ExpMaxCap)。
        /// </summary>
        private int CalcExpMax(int lrHz) => MilCameraParams.CalcExposureMaxUs(lrHz, ExpMin, ExpMaxCap); // 公式單一真相在 MilGrabber.Core

        /// <summary>
        /// 依新線掃值重算第 i 台曝光 slider/NUD 的 Maximum = CalcExpMax(lrHz)；
        /// 當前曝光值 &gt; 新 Maximum 時夾到 Maximum（並套用到相機）。包 _suppressParamEvents 防遞迴。
        /// </summary>
        private void ApplyExposureMax(int camIdx, int lrHz)
        {
            if (_tbExposure == null || camIdx < 0 || camIdx >= _tbExposure.Length) return;
            TrackBar tb = _tbExposure[camIdx];
            NumericUpDown nud = _nudExposure[camIdx];
            if (tb == null || nud == null) return;

            int expMax = Math.Max(ExpMin, CalcExpMax(lrHz));
            bool overflow = tb.Value > expMax;

            _suppressParamEvents = true;
            try
            {
                tb.Maximum = expMax;
                nud.Maximum = expMax;
                if (overflow)
                    SetRowValue(tb, nud, expMax); // 當前值超出新上限 → 夾到上限
            }
            finally { _suppressParamEvents = false; }

            if (overflow)
                ApplyParam(camIdx, ParamKind.Exposure, expMax); // 夾緊後實際套用到相機

            SyncExpAllMax(); // 全部相機曝光列上限 = 各台上限最小值（全部都不超過）
        }

        /// <summary>全部相機曝光列(trackBarExpAll/nudExpAll)上限 = 各 active 台曝光上限的最小值；當前值超出則夾緊。</summary>
        private void SyncExpAllMax()
        {
            if (_tbExpAll == null || _nudExpAll == null || _tbExposure == null) return;
            int m = ExpMaxCap; bool any = false;
            for (int i = 0; i < _tbExposure.Length; i++)
                if (_panelExp != null && i < _panelExp.Length && _panelExp[i] != null && _panelExp[i].Enabled)
                    { m = Math.Min(m, _tbExposure[i].Maximum); any = true; }
            m = any ? Math.Max(ExpMin, m) : ExpMaxCap;
            _suppressParamEvents = true;
            try
            {
                _tbExpAll.Maximum = m; if (_tbExpAll.Value > m) _tbExpAll.Value = m;
                _nudExpAll.Maximum = m; if (_nudExpAll.Value > m) _nudExpAll.Value = m;
            }
            finally { _suppressParamEvents = false; }
        }

        /// <summary>建構式呼叫一次：把每相機列與全部相機列的事件接到對應 setter（控制項由 Designer 宣告）。</summary>
        private void WireParamControls()
        {
            for (int i = 0; i < SubPanelCount; i++)
            {
                WireParamRow(_tbExposure[i], _nudExposure[i], i, ParamKind.Exposure);
                WireParamRow(_tbLineRate[i], _nudLineRate[i], i, ParamKind.LineRate);
                WireParamRow(_tbHeight[i], _nudHeight[i], i, ParamKind.Height);
            }

            WireAllRow(_tbExpAll, _nudExpAll, ParamKind.Exposure);
            WireAllRow(_tbLrAll, _nudLrAll, ParamKind.LineRate);
            WireAllRow(_tbHtAll, _nudHtAll, ParamKind.Height);
        }

        /// <summary>單一相機列接線：trackBar↔NUD 雙向同步（_suppressParamEvents 防遞迴），值變更套用 _cams[camIdx]。
        /// 拖曳中只更新 NUD 顯示 + 曝光上限視覺，放掉(MouseUp)才寫硬體；鍵盤/點軌道/NUD 即時寫（仿 AniloxRoll.Monitor）。</summary>
        private void WireParamRow(TrackBar tb, NumericUpDown nud, int camIdx, ParamKind kind)
        {
            tb.MouseDown += (s, e) => _dragging.Add(tb);
            tb.MouseUp += (s, e) =>
            {
                if (!_dragging.Remove(tb)) return;
                ApplyParam(camIdx, kind, tb.Value);                                   // 拖曳放掉 → 立即寫硬體
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, tb.Value);
            };

            tb.Scroll += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { nud.Value = ClampDecimal(tb.Value, nud.Minimum, nud.Maximum); } // NUD 顯示即時跟動
                finally { _suppressParamEvents = false; }
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, tb.Value);   // 曝光上限視覺即時跟動
                if (!_dragging.Contains(tb)) ApplyParam(camIdx, kind, tb.Value);      // 拖曳中不寫硬體；鍵盤/點軌道即時寫
            };

            nud.ValueChanged += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { tb.Value = ClampInt((int)nud.Value, tb.Minimum, tb.Maximum, tb.Minimum); }
                finally { _suppressParamEvents = false; }
                ApplyParam(camIdx, kind, (double)nud.Value);
                if (kind == ParamKind.LineRate) ApplyExposureMax(camIdx, (int)nud.Value); // 線掃變更 → 重算該台曝光上限
            };
        }

        /// <summary>全部相機列接線：套用到全部 _cams + 同步各 Cam 列顯示（_suppressParamEvents 防遞迴）。
        /// 拖曳中只更新 NUD 顯示 + 曝光上限視覺，放掉(MouseUp)才寫全部硬體；鍵盤/點軌道/NUD 即時寫（仿 AniloxRoll.Monitor）。</summary>
        private void WireAllRow(TrackBar tb, NumericUpDown nud, ParamKind kind)
        {
            tb.MouseDown += (s, e) => _dragging.Add(tb);
            tb.MouseUp += (s, e) =>
            {
                if (!_dragging.Remove(tb)) return;
                ApplyParamAll(kind, tb.Value);                                  // 拖曳放掉 → 立即寫全部硬體
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll(tb.Value);
            };

            tb.Scroll += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { nud.Value = ClampDecimal(tb.Value, nud.Minimum, nud.Maximum); } // NUD 顯示即時跟動
                finally { _suppressParamEvents = false; }
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll(tb.Value);   // 曝光上限視覺即時跟動
                if (!_dragging.Contains(tb)) ApplyParamAll(kind, tb.Value);      // 拖曳中不寫硬體；鍵盤/點軌道即時寫
            };

            nud.ValueChanged += (s, e) =>
            {
                if (_suppressParamEvents) return;
                _suppressParamEvents = true;
                try { tb.Value = ClampInt((int)nud.Value, tb.Minimum, tb.Maximum, tb.Minimum); }
                finally { _suppressParamEvents = false; }
                ApplyParamAll(kind, (double)nud.Value);
                if (kind == ParamKind.LineRate) ApplyExposureMaxAll((int)nud.Value); // 全部相機線掃 → 全部重算曝光上限
            };
        }

        /// <summary>全部相機列線掃變更：對所有相機重算曝光上限（仿主程式）。</summary>
        private void ApplyExposureMaxAll(int lrHz)
        {
            if (_cams == null) return;
            for (int i = 0; i < _cams.Length; i++)
            {
                if (_cams[i] == null) continue;
                ApplyExposureMax(i, lrHz);
            }
        }

        // 固定初值（btnInit 即套到硬體）：曝光 100 μs、線掃 3000 Hz、高度 3000 px。
        private const int ExpInitDefault = 100, LrInitDefault = 3000, HtInitDefault = 3000;

        /// <summary>
        /// btnInit 建相機後：依實際相機數填固定初值 + enable/disable。
        /// 第 i 列若 i&lt;camCount 且 _cams[i]!=null → enable + 套固定預設到硬體（曝光 100、線掃 3000、高度 3000）；
        /// 曝光 slider/NUD Maximum = CalcExpMax(線掃預設)，value = 曝光預設。否則整列 disable（灰掉）。全部相機列：有相機才 enable。
        /// </summary>
        private void InitParamControls(int camCount)
        {
            _suppressParamEvents = true;   // 設初值期間抑制事件回灌相機
            try
            {
                for (int i = 0; i < SubPanelCount; i++)
                {
                    MilCamera cam = (_cams != null && i < _cams.Length) ? _cams[i] : null;
                    bool active = i < camCount && cam != null;

                    if (active)
                    {
                        int camId = (_devices != null && i < _devices.Count) ? _devices[i].Id : i + 1;
                        _lblExpLabel(i).Text = $"CAM{camId}";
                        _lblLrLabel(i).Text = $"CAM{camId}";
                        _lblHtLabel(i).Text = $"CAM{camId}";

                        // 固定預設值（不再讀相機回值）
                        int expInit = ExpInitDefault;
                        int lrInit = LrInitDefault;
                        int htInit = HtInitDefault;

                        // 曝光動態上限 = CalcExpMax(線掃預設)（= 300）；value = 曝光預設（= 100）
                        int expMax = Math.Max(ExpMin, CalcExpMax(lrInit));
                        _tbExposure[i].Maximum = expMax;
                        _nudExposure[i].Maximum = expMax;

                        SetRowValue(_tbExposure[i], _nudExposure[i], expInit);
                        SetRowValue(_tbLineRate[i], _nudLineRate[i], lrInit);
                        SetRowValue(_tbHeight[i], _nudHeight[i], htInit);

                        // init 即把固定預設套到硬體
                        cam.SetExposureUs(expInit);
                        cam.SetLineRateHz(lrInit);
                        cam.SetGrabHeight(htInit);
                    }

                    _panelExp[i].Enabled = active;
                    _panelLr[i].Enabled = active;
                    _panelHt[i].Enabled = active;
                }

                bool any = camCount > 0;
                panelExpAll.Enabled = any;
                panelLrAll.Enabled = any;
                panelHtAll.Enabled = any;
            }
            finally { _suppressParamEvents = false; }

            SyncExpAllMax(); // 全部相機曝光列上限 = 各 active 台上限最小值
        }

        /// <summary>釋放後：所有參數控制項列 disable（Designer 控制項保留，不 Dispose）。</summary>
        private void DisableAllParamControls()
        {
            if (_panelExp == null) return;
            for (int i = 0; i < SubPanelCount; i++)
            {
                _panelExp[i].Enabled = false;
                _panelLr[i].Enabled = false;
                _panelHt[i].Enabled = false;
            }
            panelExpAll.Enabled = false;
            panelLrAll.Enabled = false;
            panelHtAll.Enabled = false;
        }

        // 取各 tab 第 i 列的 CAM 標籤（Designer 具名，索引 0..7）。
        private Label _lblExpLabel(int i) => new[] { lblExpCam1, lblExpCam2, lblExpCam3, lblExpCam4, lblExpCam5, lblExpCam6, lblExpCam7, lblExpCam8 }[i];
        private Label _lblLrLabel(int i) => new[] { lblLrCam1, lblLrCam2, lblLrCam3, lblLrCam4, lblLrCam5, lblLrCam6, lblLrCam7, lblLrCam8 }[i];
        private Label _lblHtLabel(int i) => new[] { lblHtCam1, lblHtCam2, lblHtCam3, lblHtCam4, lblHtCam5, lblHtCam6, lblHtCam7, lblHtCam8 }[i];

        /// <summary>同步設 trackBar + NUD 值（夾在各自範圍內）；呼叫端需先設 _suppressParamEvents。</summary>
        private static void SetRowValue(TrackBar tb, NumericUpDown nud, int value)
        {
            tb.Value = ClampInt(value, tb.Minimum, tb.Maximum, tb.Minimum);
            nud.Value = ClampDecimal(value, nud.Minimum, nud.Maximum);
        }

        /// <summary>「全部相機」套用：對每台相機 ApplyParam，並同步該 kind 對應的每相機列控制項顯示。</summary>
        private void ApplyParamAll(ParamKind kind, double value)
        {
            if (_cams == null) return;

            // 依 kind 選對應的每相機列控制項陣列
            TrackBar[] tbs;
            NumericUpDown[] nuds;
            switch (kind)
            {
                case ParamKind.Exposure: tbs = _tbExposure; nuds = _nudExposure; break;
                case ParamKind.LineRate: tbs = _tbLineRate; nuds = _nudLineRate; break;
                case ParamKind.Height: tbs = _tbHeight; nuds = _nudHeight; break;
                default: return;
            }

            for (int i = 0; i < _cams.Length; i++)
            {
                ApplyParam(i, kind, value);

                // 同步該相機列的 trackBar/NUD 顯示（包 _suppressParamEvents 防遞迴觸發各列事件）。
                _suppressParamEvents = true;
                try
                {
                    if (tbs != null && i < tbs.Length && tbs[i] != null)
                        tbs[i].Value = ClampInt((int)Math.Round(value), tbs[i].Minimum, tbs[i].Maximum, tbs[i].Minimum);
                    if (nuds != null && i < nuds.Length && nuds[i] != null)
                        nuds[i].Value = ClampDecimal((int)Math.Round(value), nuds[i].Minimum, nuds[i].Maximum);
                }
                finally { _suppressParamEvents = false; }
            }
        }

        /// <summary>把參數值套用到對應相機（依 ParamKind 呼 SetExposureUs / SetLineRateHz / SetGrabHeight）。</summary>
        private void ApplyParam(int camIdx, ParamKind kind, double value)
        {
            MilCamera cam = (_cams != null && camIdx >= 0 && camIdx < _cams.Length) ? _cams[camIdx] : null;
            if (cam == null) return;
            switch (kind)
            {
                case ParamKind.Exposure: cam.SetExposureUs(value); break;
                case ParamKind.LineRate: cam.SetLineRateHz(value); break;
                case ParamKind.Height: cam.SetGrabHeight((int)Math.Round(value)); break;
            }
        }

        private static int ClampInt(int value, int min, int max, int fallbackWhenNonPositive)
        {
            if (value <= 0) value = fallbackWhenNonPositive;
            if (value < min) value = min;
            if (value > max) value = max;
            return value;
        }

        private static decimal ClampDecimal(int value, decimal min, decimal max)
        {
            decimal d = value;
            if (d < min) d = min;
            if (d > max) d = max;
            return d;
        }
    }
}
