using System.Collections.Generic;
using MilGrabber.Core;
using AniloxRoll.Monitor.Core.Services;  // InspectionEngineConfig.MaxWidth

namespace AniloxRoll.Monitor.UI.Managers
{
    /// <summary>
    /// 即時全域合圖協調者：擁有 <see cref="MultiCameraMerger"/> 工頭的生命週期
    /// （「拼」＝算佈局 + 分配合併 buffer + 每台 merge target）。
    /// 「秀」一律 CPU（顯示鐵則：即時=ImageDisplayView、瀑布=WaterfallView），
    /// 顯示層要佈局時讀 <see cref="Merger"/>（SlotStartsMm / RefOpsMm）。
    /// </summary>
    internal sealed class GlobalMergeCoordinator
    {
        private MultiCameraMerger _merger;

        /// <summary>合圖是否啟用中。</summary>
        public bool IsActive { get; private set; }

        /// <summary>工頭（合圖佈局來源）；未啟用為 null。</summary>
        public MultiCameraMerger Merger => _merger;

        /// <summary>合併像素尺寸（mm/px），供 Waterfall 佈局同步。</summary>
        public double RefOpsMm => _merger?.RefOpsMm ?? 0;

        /// <summary>啟用全域合圖：建工頭算佈局 + 分配合併 buffer + 每台 merge target。
        /// 失敗（工頭啟用失敗）回 false 並自行清理。</summary>
        public bool Enable(IReadOnlyList<MilCamera> mils, double[] opsUm, double[] startPosMm)
        {
            if (IsActive) return false;
            if (mils == null || mils.Count == 0) return false;

            // 「拼」委派工頭：傳入底層 MilCamera 清單（空缺槽以 MaxWidth 作為標準寬度算全域範圍）
            _merger = new MultiCameraMerger(mils);
            if (!_merger.EnableMerge(opsUm, startPosMm, InspectionEngineConfig.MaxWidth))
            {
                _merger = null;
                return false;
            }
            IsActive = true;
            return true;
        }

        /// <summary>停用全域合圖：工頭釋放合併 buffer + 清各相機 merge target。
        /// 先翻 IsActive 旗標，讓仍在跑的 grab hook 早退。</summary>
        public void Disable()
        {
            if (!IsActive) return;
            IsActive = false;
            _merger?.DisableMerge();
            _merger = null;
        }

        /// <summary>OPS/Start 變更時重算佈局（下一幀生效）。運算委派工頭；回傳是否重新分配了合併 buffer。</summary>
        public bool RefreshLayout(double[] opsUm, double[] startPosMm)
        {
            if (!IsActive || _merger == null) return false;
            return _merger.RefreshLayout(opsUm, startPosMm, InspectionEngineConfig.MaxWidth);
        }
    }
}
