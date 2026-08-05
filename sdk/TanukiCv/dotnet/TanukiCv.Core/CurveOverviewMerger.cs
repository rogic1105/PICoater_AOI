using System;
using System.Collections.Generic;

namespace TanukiCv.Core
{
    /// <summary>
    /// 多相機「欄全覽曲線」合併演算法（純算術，0 依賴 WinForms / MIL）：
    /// 把各相機的 mura 曲線依機台布局位置（Start mm）合到單一全域曲線。
    ///
    /// 重疊區依合圖方式（<see cref="MergeOverlap"/>）的 boundary **唯一歸屬** —— 與影像合圖共用
    /// 同一份 <see cref="MergeLayout"/> 分界邏輯，重疊區每個全域 bin 只取「該欄歸屬相機」的曲線值，
    /// 不再平均 → 曲線與影像 pixel 對齊。間空（無曲線）相機**參與分界但不貼資料**（黑占位，
    /// 同影像合圖 ALL-slots 佈局；在線相機曲線在與黑占位的中線被切，黑布區段留 0）。
    ///
    /// 把「秀」（畫到 chart、帶入視野）留給呼叫端（app / 範例各自接自己的曲線圖）。
    /// </summary>
    public static class CurveOverviewMerger
    {
        /// <summary>合併結果（純資料；呼叫端拿去畫 chart）。</summary>
        public struct Result
        {
            /// <summary>合併後 Mean 曲線（長度 <see cref="Length"/>）。</summary>
            public float[] Mean;
            /// <summary>合併後 Max 曲線。</summary>
            public float[] Max;
            /// <summary>全域左緣（mm）= 最小相機 Start。</summary>
            public double GlobalMinMm;
            /// <summary>格點間距（mm）= 合併曲線每點對應的實體寬。</summary>
            public double GridMm;
            /// <summary>點數。</summary>
            public int Length;
            /// <summary>
            /// Camera index owning each merged point after overlap clipping; -1 means no source data.
            /// </summary>
            public int[] OwnerCameraIndices;
            /// <summary>是否有有效輸出（false = 輸入不足/退化，呼叫端應略過）。</summary>
            public bool Valid;
        }

        /// <summary>
        /// 合併多台相機的欄曲線。
        /// </summary>
        /// <param name="allMean">各相機 Mean 曲線（null/空 = 該相機間空，不參與）。</param>
        /// <param name="allMax">各相機 Max 曲線（可 null）。</param>
        /// <param name="opsArr">各相機像素尺寸 OPS（um/px）。</param>
        /// <param name="posArr">各相機影像左緣實體位置 Start（mm）。</param>
        /// <param name="cameraCount">相機數。</param>
        /// <param name="overlap">重疊歸屬策略（與影像合圖一致；app 用 Midline）。</param>
        /// <param name="maxPoints">全覽點數上限（降採樣，預設 2000）。</param>
        public static Result Merge(
            float[][] allMean, float[][] allMax,
            double[] opsArr, double[] posArr, int cameraCount,
            MergeOverlap overlap, int maxPoints = 2000)
        {
            var result = new Result { Valid = false };
            if (allMean == null || opsArr == null || posArr == null || cameraCount <= 0) return result;

            // 全域範圍：涵蓋全部相機位置，缺圖用現有影像寬度平均類推（只影響邊界、不填假資料）
            double sumWidthMm = 0;
            int widthCount = 0;
            double minOpsUm = double.MaxValue;
            for (int i = 0; i < cameraCount; i++)
            {
                if (opsArr[i] > 0 && opsArr[i] < minOpsUm) minOpsUm = opsArr[i];
                var curve = allMean[i];
                if (curve != null && curve.Length > 0)
                {
                    sumWidthMm += curve.Length * (opsArr[i] / 1000.0);
                    widthCount++;
                }
            }
            if (minOpsUm <= 0 || minOpsUm == double.MaxValue) minOpsUm = 33.0;
            double avgWidthMm = widthCount > 0 ? sumWidthMm / widthCount : 400.0;

            double globalMin = double.MaxValue, globalMax = double.MinValue;
            for (int i = 0; i < cameraCount; i++)
            {
                double camStart = posArr[i];
                var curve = allMean[i];
                double camEnd = (curve != null && curve.Length > 0)
                    ? camStart + curve.Length * (opsArr[i] / 1000.0)
                    : camStart + avgWidthMm;
                if (camStart < globalMin) globalMin = camStart;
                if (camEnd > globalMax) globalMax = camEnd;
            }
            if (globalMin >= globalMax) return result;

            // 格點間距：至少 OPS 精度，但上限 maxPoints 點
            double gridMm = Math.Max(minOpsUm / 1000.0, (globalMax - globalMin) / maxPoints);
            int totalLen = (int)Math.Ceiling((globalMax - globalMin) / gridMm);
            if (totalLen <= 0 || totalLen > maxPoints + 1) return result;

            // 重疊歸屬（與影像合圖共用唯一來源 MergeLayout）：以 gridMm 當基準像素、curve 長度換成 grid 寬，
            // 算出每台在「全域 grid」的擁有區間 [ownStart, ownEnd)。
            // 間空相機**參與分界但不貼資料**（黑占位，同影像合圖 ALL-slots 佈局）：
            //   影像合圖把缺席相機當黑占位算 boundary（在線相機影像在中線被切、之後顯示黑），
            //   曲線必須同一套切法，否則最後一台在線相機的曲線會跑進黑布區（尾端沒被切、跟影像錯位）。
            //   間空的寬度用 avgWidthMm 估（同全域範圍的估法）；其擁有區段沒資料可貼 → 留 0。
            // MergeLayout 的 boundary 對相鄰兩台剛好 tile（a.ownEnd == b.ownStart），故無縫且不重疊。
            var ownStart = new int[cameraCount];
            var ownEnd   = new int[cameraCount];
            var geom = new List<MergeLayout.CamGeom>();
            for (int i = 0; i < cameraCount; i++)
            {
                var c = allMean[i];
                int wGrid = (c != null && c.Length > 0)
                    ? (int)Math.Round(c.Length * (opsArr[i] / 1000.0) / gridMm)
                    : (int)Math.Round(avgWidthMm / gridMm);        // 間空：黑占位寬，參與分界
                geom.Add(new MergeLayout.CamGeom { CameraId = i, StartMm = posArr[i], WidthPx = Math.Max(1, wGrid) });
            }
            foreach (var p in MergeLayout.Compute(geom, globalMin, gridMm, 1, overlap, out _))
            {
                ownStart[p.CameraId] = p.XOffset + p.SrcLeft;
                ownEnd[p.CameraId]   = p.XOffset + p.SrcLeft + p.SrcWidth;
            }

            // 每台只把「自己擁有的 grid 區段」的曲線貼進全域 bin；同一台多點落同一 bin → 取 max 保峰值。
            var mergedMean = new float[totalLen];
            var mergedMax  = new float[totalLen];
            var hit        = new bool[totalLen];
            var ownerCameraIndices = new int[totalLen];
            for (int i = 0; i < ownerCameraIndices.Length; i++)
                ownerCameraIndices[i] = -1;

            for (int i = 0; i < cameraCount; i++)
            {
                var curveMean = allMean[i];
                if (curveMean == null || curveMean.Length == 0) continue;
                var curveMax = (allMax != null && i < allMax.Length) ? allMax[i] : null;
                int os = ownStart[i], oe = ownEnd[i];

                double camOpsMm = opsArr[i] / 1000.0;
                double camStart = posArr[i];

                for (int j = 0; j < curveMean.Length; j++)
                {
                    int idx = (int)((camStart + j * camOpsMm - globalMin) / gridMm);
                    if (idx < 0 || idx >= totalLen) continue;
                    if (idx < os || idx >= oe) continue;            // 重疊區不屬本台 → 歸鄰台

                    if (!hit[idx] || curveMean[j] > mergedMean[idx]) mergedMean[idx] = curveMean[j];
                    float mv = (curveMax != null && j < curveMax.Length) ? curveMax[j] : 0;
                    if (!hit[idx] || mv > mergedMax[idx]) mergedMax[idx] = mv;
                    hit[idx] = true;
                    ownerCameraIndices[idx] = i;
                }
            }

            result.Mean = mergedMean;
            result.Max = mergedMax;
            result.GlobalMinMm = globalMin;
            result.GridMm = gridMm;
            result.Length = totalLen;
            result.OwnerCameraIndices = ownerCameraIndices;
            result.Valid = true;
            return result;
        }
    }
}
