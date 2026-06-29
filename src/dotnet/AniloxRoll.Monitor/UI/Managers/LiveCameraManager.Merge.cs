using System;
using System.Collections.Generic;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using AniloxRoll.Monitor.Core.Camera;
    // partial：全域合圖編排 + forwarder。
    // 「秀」整個生命週期（MIL display / timer / 滑鼠 hook / 視野範圍 / merged zoom-pan-1x）已提取到
    // GlobalMergeCoordinator（2026-06-26 重構）。本檔只留：①編排（建 MilCamera 清單、決定 SmartCanvas vs
    // MIL 直繪、SmartCanvas/Waterfall 佈局）②對外公開方法的 forwarder ③視野中心選中相機 callback。

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // ==================== Global Merge（編排 + forwarder） ====================

        /// <summary>啟用即時全域合圖：建底層相機清單 + 解除各台 secondary display → 委派 coordinator 啟動合圖
        /// （SmartCanvas 模式不綁 MIL display，避免 M_MOUSE_USE 攔截滾輪）→ SmartCanvas 模式再 CPU 拼。</summary>
        public void EnableGlobalMerge(double[] opsUm, double[] startPosMm)
        {
            if (_globalMerge.IsActive || _cameras.Count == 0) return;

            // 「拼」委派工頭：傳入底層 MilCamera 清單
            var mils = new List<MilCamera>(_cameras.Count);
            foreach (var cam in _cameras) mils.Add(cam.Mil);
            MIL_ID sysId = _cameras[0].OwnerSystemId;

            // 解除所有相機的 secondary display（合併 display / SmartCanvas 接管主畫面）
            foreach (var cam in _cameras)
                cam.SetSecondaryDisplay(IntPtr.Zero);

            // showMilDisplay：SmartCanvas 模式合圖由 LiveDisplayView CPU 拼，不綁 MIL display
            if (!_globalMerge.Enable(mils, sysId, opsUm, startPosMm, showMilDisplay: !_display.SmartCanvasMode))
            {
                _display.SwitchMainDisplay(_display.UserSelectedMainCameraId);   // 啟用失敗 → 復原 secondary display
                return;
            }

            // SmartCanvas 合圖：用工頭佈局(各台 start/ops) CPU 拼（feedScale=1：主程式餵全解析度）
            _display.OnGlobalMergeEnabled(opsUm, startPosMm);
        }

        /// <summary>停用即時全域合圖：coordinator 釋放 display + 工頭釋放合併 buffer → SmartCanvas 回單相機 → 復原選定相機顯示。</summary>
        public void DisableGlobalMerge()
        {
            if (!_globalMerge.IsActive) return;

            _globalMerge.Disable();
            _display.OnGlobalMergeDisabled(); // SmartCanvas 回單相機 + 復原使用者明確點選的相機
        }

        /// <summary>OPS/Start 變更時，重新計算全域合圖佈局（下一幀生效）。MIL display 重綁委派 coordinator；SmartCanvas/Waterfall 佈局同步留本類別。</summary>
        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm)
        {
            if (!_globalMerge.IsActive || _cameras.Count == 0) return;
            _globalMerge.RefreshLayout(opsUm, startPosMm);

            // SmartCanvas 合圖佈局同步（feedScale=1：主程式餵全解析度顯示 bytes）
            // Waterfall 合圖佈局同步（對齊全幅合圖；refOpsMm=mm/px 基準像素尺寸）
            _display.RefreshGlobalMergeLayout(opsUm, startPosMm, _globalMerge.RefOpsMm);
        }

        // ==================== View Range forwarder（供 AniloxRollForm overview / 曲線聯動） ====================

        /// <summary>取得合併 display 的 X 視野範圍（mm），供 overview chart 聯動。</summary>
        public bool TryGetMergedViewRange(out double leftMm, out double rightMm)
            => _globalMerge.TryGetViewRange(out leftMm, out rightMm);

        /// <summary>取得合併 display 的 Y 視野範圍（pixel），供法向曲線圖聯動。</summary>
        public bool TryGetMergedViewRangeY(out double topPixel, out double botPixel)
            => _globalMerge.TryGetViewRangeY(out topPixel, out botPixel);
    }
}
