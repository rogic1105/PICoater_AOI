using System.Collections.Generic;
using MilGrabber.Core;
    // partial：全域合圖編排 + forwarder。工頭生命週期在 GlobalMergeCoordinator；
    // 「秀」一律 CPU（顯示鐵則：即時=ImageDisplayView、瀑布=WaterfallView，佈局讀工頭）。

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // ==================== Global Merge（編排 + forwarder） ====================

        /// <summary>啟用即時全域合圖：建底層相機清單 → 委派 coordinator 啟動工頭
        /// （佈局 + 合併 buffer + 每台 merge target）→ 顯示層同步合圖佈局（CPU 拼）。</summary>
        public void EnableGlobalMerge(double[] opsUm, double[] startPosMm)
        {
            if (_globalMerge.IsActive || _cameras.Count == 0) return;

            // 「拼」委派工頭：傳入底層 MilCamera 清單
            var mils = new List<MilCamera>(_cameras.Count);
            foreach (var cam in _cameras) mils.Add(cam.Mil);
            if (!_globalMerge.Enable(mils, opsUm, startPosMm)) return;

            // ImageCanvas 合圖：用工頭佈局(各台 start/ops) CPU 拼（feedScale=1：主程式餵全解析度）
            _display.OnGlobalMergeEnabled(opsUm, startPosMm);
            AniloxRoll.Monitor.Core.Services.FlowTrace.Log($"EnableGlobalMerge（slots={startPosMm?.Length ?? 0}）");
        }

        /// <summary>停用即時全域合圖：工頭釋放合併 buffer → 顯示層回單相機 + 復原選定相機。</summary>
        public void DisableGlobalMerge()
        {
            if (!_globalMerge.IsActive) return;
            AniloxRoll.Monitor.Core.Services.FlowTrace.Log("DisableGlobalMerge");
            _globalMerge.Disable();
            _display.OnGlobalMergeDisabled(); // ImageCanvas 回單相機 + 復原使用者明確點選的相機
        }

        /// <summary>OPS/Start 變更時，重新計算全域合圖佈局（下一幀生效）；ImageCanvas/Waterfall 佈局同步。</summary>
        public void RefreshGlobalMergeLayout(double[] opsUm, double[] startPosMm)
        {
            if (!_globalMerge.IsActive || _cameras.Count == 0) return;
            _globalMerge.RefreshLayout(opsUm, startPosMm);

            // ImageCanvas 合圖佈局同步（feedScale=1：主程式餵全解析度顯示 bytes）
            // Waterfall 合圖佈局同步（對齊全幅合圖；refOpsMm=mm/px 基準像素尺寸）
            _display.RefreshGlobalMergeLayout(opsUm, startPosMm, _globalMerge.RefOpsMm);
        }
    }
}
