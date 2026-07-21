using System;
using System.Collections.Generic;
using Matrox.MatroxImagingLibrary;
using TanukiCv.Core; // MergeLayout（所有合圖 consumer 的佈局唯一來源）

namespace MilGrabber.Core
{
    /// <summary>
    /// 多相機「工頭」：把一組 <see cref="MilCamera"/> 的即時影像拼成單一全域合圖（純 MIL 範圍）。
    ///
    /// 職責分工：
    ///  - 工頭負責「拼」：算佈局（全域範圍、每台 xOffset、重疊中點分界）+ 分配 MIL 合併 buffer
    ///    + 設定每台相機把 grab 幀貼到 buffer。
    ///  - 上層負責消費 <see cref="MergedBuffer"/>：產品可轉成 CPU frame，SDK 範例也可選擇 MIL display；
    ///    座標換算使用工頭提供的 <see cref="MinStartMm"/> 等佈局資訊。
    ///
    /// 本類別刻意 0 依賴 System.Windows.Forms：不碰 panel / Label / Timer / Control / Handle。
    /// 顯示與 UI 全留上層。工頭不負責建立/釋放相機（相機生命週期由呼叫端管理）。
    ///
    /// 佈局/重疊分界算術走 **TanukiCv.Core.MergeLayout 單一來源**（與 LiveDisplayView/瀑布/曲線同一份；
    /// 黑槽=沒影像的相機亦參與中線分界，影像與曲線才對齊）。2026-06-26 收斂：原「MIL-only 自含一份」已移除
    /// —— MergeLayout 是零依賴純算術（不拉 WinForms），從拋棄層引用它依賴方向正確（throwaway→durable），不違反分層。
    /// 槽數＝呼叫端餵的「設定槽數」（startPosMm/opsUm 長度，可變），沒相機的槽用 defaultSlotWidthPx 占位。
    /// </summary>
    public class MultiCameraMerger
    {
        private readonly List<MilCamera> _cameras;
        private int _defaultSlotWidthPx;   // 沒相機的槽位占位寬（px）；ComputeLayout 存、ApplyMergeTargets 用

        // ==================== 合併資源 ====================
        private MIL_ID _mergedBuffer = MIL.M_NULL;

        /// <summary>合併 buffer 的 MIL_ID handle，供上層讀取或顯示。未啟用時為 M_NULL。</summary>
        public MIL_ID MergedBuffer => _mergedBuffer;

        /// <summary>合圖是否啟用中。</summary>
        public bool IsActive { get; private set; }

        // ==================== 座標資訊（供上層顯示與座標換算） ====================
        /// <summary>合併座標系原點（mm）：所有相機中最小的 start 位置。</summary>
        public double MinStartMm { get; private set; }

        /// <summary>合併像素尺寸（mm/px）：以第一台相機（opsUm[0]）為基準。</summary>
        public double RefOpsMm { get; private set; }

        /// <summary>合併 buffer 寬度（px）。</summary>
        public int TotalW { get; private set; }

        /// <summary>合併 buffer 高度（px）。</summary>
        public int TotalH { get; private set; }

        /// <summary>各槽位（含離線空缺）起始 mm；index = cameraId - 1。</summary>
        public double[] SlotStartsMm { get; private set; }

        /// <summary>各槽位（含離線空缺）結束 mm；index = cameraId - 1。</summary>
        public double[] SlotEndsMm { get; private set; }

        /// <summary>
        /// 建立工頭。相機清單由呼叫端提供（工頭不負責建立/釋放）。
        /// </summary>
        public MultiCameraMerger(IEnumerable<MilCamera> cameras)
        {
            if (cameras == null) throw new ArgumentNullException(nameof(cameras));
            _cameras = new List<MilCamera>(cameras);
        }

        /// <summary>
        /// 啟用全域合圖：算佈局 → 分配合併 buffer → 每台相機 SetMergeTarget。
        /// </summary>
        /// <param name="opsUm">各相機像素尺寸（μm/px），index = cameraId - 1。</param>
        /// <param name="startPosMm">各相機起始位置（mm），index = cameraId - 1。</param>
        /// <param name="defaultSlotWidthPx">離線（不在 _cameras 中）槽位的標準寬度（px），用於計算全域範圍。</param>
        /// <returns>成功啟用回傳 true；已啟用、無相機或尺寸無效回傳 false。</returns>
        public bool EnableMerge(double[] opsUm, double[] startPosMm, int defaultSlotWidthPx)
        {
            if (IsActive || _cameras.Count == 0) return false;
            if (opsUm == null || opsUm.Length == 0) return false;

            if (!ComputeLayout(opsUm, startPosMm, defaultSlotWidthPx,
                    out double minStart, out int totalW, out int maxH))
                return false;

            // 在第一台相機的 System 上分配合併 buffer
            MIL_ID sysId = _cameras[0].OwnerSystemId;
            if (sysId == MIL.M_NULL) return false;

            MIL.MbufAlloc2d(sysId, totalW, maxH, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _mergedBuffer);
            MIL.MbufClear(_mergedBuffer, 0);

            ApplyLayout(opsUm, startPosMm, minStart, totalW, maxH);

            IsActive = true;
            return true;
        }

        /// <summary>
        /// OPS/Start 變更時重算佈局（buffer/target 部分；不含任何顯示）。
        /// buffer 尺寸改變才重新分配；否則只清空 buffer + 重設 target。
        /// </summary>
        /// <returns>
        /// 若 buffer 因尺寸改變而重新分配，回傳 true（上層需重新取得 <see cref="MergedBuffer"/> handle）；
        /// 否則回傳 false（buffer handle 不變）。
        /// </returns>
        public bool RefreshLayout(double[] opsUm, double[] startPosMm, int defaultSlotWidthPx)
        {
            if (!IsActive || _cameras.Count == 0) return false;
            if (opsUm == null || opsUm.Length == 0) return false;

            // ① 暫停所有相機的合併複製（grab hook 中 mergedTargetBuffer == M_NULL → 跳過）
            foreach (var cam in _cameras)
                cam.ClearMergeTarget();

            // ② 重算座標系
            if (!ComputeLayout(opsUm, startPosMm, defaultSlotWidthPx,
                    out double minStart, out int totalW, out int maxH))
                return false;

            bool reallocated = false;

            // ③ buffer 大小改變 → 重新分配（上層需重新取得 handle）
            if (totalW != TotalW || maxH != TotalH)
            {
                MIL_ID sysId = _cameras[0].OwnerSystemId;
                if (sysId == MIL.M_NULL) return false;

                if (_mergedBuffer != MIL.M_NULL)
                {
                    MIL.MbufFree(_mergedBuffer);
                    _mergedBuffer = MIL.M_NULL;
                }
                MIL.MbufAlloc2d(sysId, totalW, maxH, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _mergedBuffer);
                MIL.MbufClear(_mergedBuffer, 0);
                reallocated = true;
            }
            else
            {
                MIL.MbufClear(_mergedBuffer, 0);
            }

            // ④ 更新座標系 + 各相機 clip，恢復合併複製
            ApplyLayout(opsUm, startPosMm, minStart, totalW, maxH);
            return reallocated;
        }

        /// <summary>停用全域合圖：清每台 merge target + 釋放合併 buffer。</summary>
        public void DisableMerge()
        {
            if (!IsActive) return;

            // 先清除各相機合併目標（停止 grab hook 內的 MbufCopyClip），再釋放 buffer
            foreach (var cam in _cameras)
                cam.ClearMergeTarget();

            if (_mergedBuffer != MIL.M_NULL)
            {
                MIL.MbufFree(_mergedBuffer);
                _mergedBuffer = MIL.M_NULL;
            }

            IsActive     = false;
            SlotStartsMm = null;
            SlotEndsMm   = null;
        }

        // ==================== 內部：佈局算術（從 LiveCameraManager.EnableGlobalMerge 搬入；暫不抽共用） ====================

        /// <summary>
        /// 唯一來源：設定座標系欄位（MinStartMm/RefOpsMm/TotalW/TotalH）後，算每台 xOffset + 裁切並 SetMergeTarget。
        /// 順序鎖死 —— 座標必須在 ApplyMergeTargets 之前（後者用 RefOpsMm 算 xOffset）。
        /// EnableMerge / RefreshLayout 共用，避免「設座標 + 算位置」各寫一份而順序分歧
        /// （曾因 EnableMerge 把 ApplyMergeTargets 寫在 RefOpsMm=0 時，切換 StitchMode 後 xOffset 變垃圾值 → 合圖全黑）。
        /// </summary>
        private void ApplyLayout(double[] opsUm, double[] startPosMm, double minStart, int totalW, int maxH)
        {
            MinStartMm = minStart;
            RefOpsMm   = opsUm[0] / 1000.0;
            TotalW     = totalW;
            TotalH     = maxH;
            ApplyMergeTargets(startPosMm, minStart);
        }

        /// <summary>
        /// 計算全域佈局：Pass 1 用 ALL slots（含離線空缺）算全域範圍 + 各槽位 mm 起訖；
        /// Pass 2 用在線相機取最大高度。回傳 minStart / totalW(px) / maxH(px)。
        /// </summary>
        private bool ComputeLayout(double[] opsUm, double[] startPosMm, int defaultSlotWidthPx,
            out double minStart, out int totalW, out int maxH)
        {
            minStart = double.MaxValue;
            totalW = 0;
            maxH = 0;
            _defaultSlotWidthPx = defaultSlotWidthPx;   // 供 ApplyMergeTargets 對沒相機的槽占位

            double refOpsMm = opsUm[0] / 1000.0;
            if (refOpsMm <= 0) return false;

            double maxEnd = double.MinValue;
            SlotStartsMm = new double[opsUm.Length];
            SlotEndsMm   = new double[opsUm.Length];
            var diag = new System.Text.StringBuilder(); // 診斷（合圖跑位）：每槽實際用的寬度（live FrameWidth vs 離線預設）
            for (int i = 0; i < opsUm.Length; i++)
            {
                double pos = (startPosMm != null && i < startPosMm.Length) ? startPosMm[i] : 0;
                double ops = opsUm[i];
                MilCamera liveCam = FindCamera(i + 1);
                double widthPx = (liveCam != null) ? liveCam.FrameWidth : defaultSlotWidthPx;
                double widthMm = widthPx * ops / 1000.0;
                if (pos < minStart) minStart = pos;
                if (pos + widthMm > maxEnd) maxEnd = pos + widthMm;
                SlotStartsMm[i] = pos;
                SlotEndsMm[i]   = pos + widthMm;
                diag.Append($" c{i + 1}:w={widthPx}{(liveCam != null ? "" : "(off)")}@{pos:F0}");
            }
            System.Diagnostics.Trace.WriteLine($"[Merger] ComputeLayout refOps={refOpsMm:F5}{diag}");

            foreach (var cam in _cameras)
                if (cam.FrameHeight > maxH) maxH = cam.FrameHeight;

            totalW = (int)Math.Ceiling((maxEnd - minStart) / refOpsMm);
            return totalW > 0 && maxH > 0;
        }

        /// <summary>
        /// 算每槽 xOffset + 重疊中點分界後對「有相機」的槽 SetMergeTarget。
        /// 佈局走 sdk 單一來源 <see cref="MergeLayout.Compute"/>（與 LiveDisplayView/瀑布/曲線同一份、Midline 策略）：
        /// 餵**設定的全槽**（沒相機的槽用 _defaultSlotWidthPx 占位）→ 黑槽亦參與中線分界（影像/曲線一致對齊）；
        /// 只對 FindCamera 找得到的槽 SetMergeTarget（沒相機的槽 = 留黑、不貼）。
        /// 順序鎖死：RefOpsMm 須在此之前由 ApplyLayout 設好。
        /// </summary>
        private void ApplyMergeTargets(double[] startPosMm, double minStart)
        {
            double refOpsMm = RefOpsMm;
            if (refOpsMm <= 0 || startPosMm == null) return;

            int n = startPosMm.Length;   // 設定的全槽數（可變）
            var cams = new List<MergeLayout.CamGeom>(n);
            for (int i = 0; i < n; i++)
            {
                var liveCam = FindCamera(i + 1);
                int widthPx = (liveCam != null) ? liveCam.FrameWidth : _defaultSlotWidthPx;
                cams.Add(new MergeLayout.CamGeom { CameraId = i + 1, StartMm = startPosMm[i], WidthPx = widthPx });
            }

            var placements = MergeLayout.Compute(cams, minStart, refOpsMm, 1, MergeOverlap.Midline, out _);
            foreach (var p in placements)
            {
                var cam = FindCamera(p.CameraId);
                if (cam != null && p.SrcWidth > 0)
                    cam.SetMergeTarget(_mergedBuffer, p.XOffset, p.SrcLeft, p.SrcWidth);
            }
        }

        private MilCamera FindCamera(int cameraId)
        {
            for (int i = 0; i < _cameras.Count; i++)
                if (_cameras[i].CameraId == cameraId) return _cameras[i];
            return null;
        }
    }
}
