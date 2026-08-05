using System;
using System.Collections.Generic;

namespace TanukiCv.Core
{
    /// <summary>
    /// 合圖重疊區歸屬策略（相鄰兩台相機重疊時，重疊像素歸給誰）。
    /// 三者只差「分界線 boundary 取在哪」，後續裁切算法完全相同。
    /// </summary>
    public enum MergeOverlap
    {
        /// <summary>中線分界：boundary = 重疊左緣 + 重疊寬/2（前相機畫到中線、後相機從中線起）。預設，與原 MIL 合圖一致。</summary>
        Midline,
        /// <summary>右覆蓋左：boundary = 重疊左緣（左相機讓出整個重疊區，由右相機填）。</summary>
        RightOverLeft,
        /// <summary>左覆蓋右：boundary = 重疊右緣（右相機讓出整個重疊區，由左相機填）。</summary>
        LeftOverRight,
    }

    /// <summary>一台相機在合圖中的擺放結果（顯示空間 px）。</summary>
    public struct CameraPlacement
    {
        /// <summary>相機 Id（1-based，對應輸入 <see cref="MergeLayout.CamGeom.CameraId"/>）。</summary>
        public int CameraId;
        /// <summary>相機左緣在全域合圖座標的 x 起點（px）。</summary>
        public int XOffset;
        /// <summary>來源影像裁切起點（px，重疊分界後）。</summary>
        public int SrcLeft;
        /// <summary>來源影像裁切寬度（px，重疊分界後；可能為 0=整台被相鄰覆蓋）。</summary>
        public int SrcWidth;
        /// <summary>共同顯示格點內的裁切起點；舊呼叫端未設定時等同 <see cref="SrcLeft"/>。</summary>
        public int DisplayLeft;
        /// <summary>共同顯示格點內的裁切寬度；0 代表與 <see cref="SrcWidth"/> 相同。</summary>
        public int DisplayWidth;
        /// <summary>裁切區在全域合圖的 x 起點。</summary>
        public int DestX => XOffset + (DisplayWidth > 0 ? DisplayLeft : SrcLeft);
        /// <summary>裁切區在全域合圖的顯示寬度。</summary>
        public int DestWidth => DisplayWidth > 0 ? DisplayWidth : SrcWidth;
    }

    /// <summary>
    /// 合圖佈局演算法（純算術，0 依賴 MIL / WinForms）：
    /// 給各相機 start(mm) / 寬度(px) + 基準像素尺寸 → 算每台 xOffset + 重疊分界裁切（依策略）。
    ///
    /// 放在 TanukiCv（durable、跨產品共用的影像 SDK）而非 MIL 區：MIL 區是「換 grabber/相機就整包
    /// 換掉」的拋棄層，可重用 IP 不該困在那。本演算法服務 ImageCanvas / CPU 合圖路徑（範例 +
    /// 主程式 LiveDisplayView / GrabImageStitcher），scale=縮圖倍率、strategy 可選三種。
    /// （MIL 即時合圖 MultiCameraMerger 另有自含的 MIL-only 中線實作，隨硬體丟，刻意不共用此份。）
    /// </summary>
    public static class MergeLayout
    {
        /// <summary>單台相機的合圖幾何輸入（顯示空間）。</summary>
        public struct CamGeom
        {
            /// <summary>相機 Id（1-based）。</summary>
            public int CameraId;
            /// <summary>相機影像左緣的實體位置（mm）。</summary>
            public double StartMm;
            /// <summary>相機影像在「顯示空間」的像素寬（全解析度合圖=FrameWidth；縮圖合圖=FrameWidth/scale）。</summary>
            public int WidthPx;
            /// <summary>映射到共同實體格點後的顯示寬度；0 代表與 <see cref="WidthPx"/> 相同。</summary>
            public int DisplayWidthPx;
        }

        /// <summary>
        /// 算各相機擺放 + 重疊分界。
        /// xOffset = round((StartMm − minStartMm) / refOpsMm / scale)，依 xOffset 排序後逐對處理重疊。
        /// </summary>
        /// <param name="cams">在線相機幾何（顯示空間寬度）。</param>
        /// <param name="minStartMm">全域原點（mm）= 最小 start。</param>
        /// <param name="refOpsMm">基準像素尺寸（mm/px，全解析度）。</param>
        /// <param name="scale">顯示降採樣倍率（全解析度=1；縮圖=resizeScale）。</param>
        /// <param name="strategy">重疊歸屬策略。</param>
        /// <param name="totalW">回傳合圖總寬（px）= max(xOffset + WidthPx)。</param>
        /// <returns>各相機擺放（依 CameraId 對齊原輸入順序回傳；內部以 xOffset 排序處理重疊）。</returns>
        public static List<CameraPlacement> Compute(
            IList<CamGeom> cams, double minStartMm, double refOpsMm, int scale,
            MergeOverlap strategy, out int totalW)
        {
            totalW = 0;
            var result = new List<CameraPlacement>();
            if (cams == null || cams.Count == 0 || refOpsMm <= 0) return result;
            if (scale < 1) scale = 1;

            int n = cams.Count;
            var xOff = new int[n];
            var sourceWidth = new int[n];
            var displayWidth = new int[n];
            for (int i = 0; i < n; i++)
            {
                xOff[i] = (int)Math.Round((cams[i].StartMm - minStartMm) / refOpsMm / scale);
                sourceWidth[i] = Math.Max(0, cams[i].WidthPx);
                displayWidth[i] = cams[i].DisplayWidthPx > 0
                    ? cams[i].DisplayWidthPx
                    : sourceWidth[i];
            }

            // 依 xOffset 排序（不動原陣列；order[k] = 第 k 個由左到右的相機 index）
            var order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            Array.Sort(order, (a, b) => xOff[a].CompareTo(xOff[b]));

            var drawLeft = new int[n];
            var drawRight = new int[n];
            for (int i = 0; i < n; i++) { drawLeft[i] = 0; drawRight[i] = displayWidth[i]; }

            // 逐對（左、右相鄰）處理重疊：三策略只差 boundary 取哪。
            for (int k = 0; k < n - 1; k++)
            {
                int a = order[k];      // 左
                int b = order[k + 1];  // 右
                int rightEdge = xOff[a] + displayWidth[a];
                int leftEdge  = xOff[b];
                int overlap   = rightEdge - leftEdge;
                if (overlap <= 0) continue;

                int boundary;
                switch (strategy)
                {
                    case MergeOverlap.RightOverLeft: boundary = leftEdge; break;              // 左讓出整個重疊
                    case MergeOverlap.LeftOverRight: boundary = rightEdge; break;             // 右讓出整個重疊
                    default:                         boundary = leftEdge + overlap / 2; break; // 中線
                }
                // 前相機畫到 boundary、後相機從 boundary 起（全域座標 → 各自 src 座標）
                drawRight[a] = Math.Min(drawRight[a], boundary - xOff[a]);
                drawLeft[b]  = Math.Max(drawLeft[b],  boundary - xOff[b]);
            }

            int maxEnd = 0;
            for (int i = 0; i < n; i++)
            {
                int displayLeft = Math.Max(0, drawLeft[i]);
                int displayRight = Math.Min(displayWidth[i], drawRight[i]);
                int drawWidth = Math.Max(0, displayRight - displayLeft);
                int sourceLeft = ScaleCoordinate(displayLeft, displayWidth[i], sourceWidth[i]);
                int sourceRight = ScaleCoordinate(displayRight, displayWidth[i], sourceWidth[i]);
                result.Add(new CameraPlacement
                {
                    CameraId = cams[i].CameraId,
                    XOffset = xOff[i],
                    SrcLeft = sourceLeft,
                    SrcWidth = Math.Max(0, sourceRight - sourceLeft),
                    DisplayLeft = displayLeft,
                    DisplayWidth = drawWidth
                });
                int end = xOff[i] + displayWidth[i];
                if (end > maxEnd) maxEnd = end;
            }
            totalW = maxEnd;
            return result;
        }

        private static int ScaleCoordinate(int value, int fromWidth, int toWidth)
        {
            if (value <= 0 || fromWidth <= 0 || toWidth <= 0) return 0;
            if (value >= fromWidth) return toWidth;
            return (int)Math.Round(value * (double)toWidth / fromWidth);
        }
    }
}
