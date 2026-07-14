using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using TanukiCv.Core;     // CurveOverviewMerger / MergeLayout / MergeOverlap（純算法 唯一來源）
using TanukiCv.Controls; // ColumnCurveChartHelper（曲線圖）

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 全覽圖合併演算法與 .bin 曲線讀取：
    /// 將 7 台相機的 mura 曲線依機台布局位置合併到全覽圖。
    /// </summary>
    public static class CurveMergeHelper
    {
        /// <summary>
        /// 將多台相機曲線合併到單一全覽圖並畫到 chart。
        /// 合併數學委派 sdk 唯一來源 <see cref="CurveOverviewMerger"/>（依合圖方式 boundary 唯一歸屬，
        /// 與影像合圖 pixel 對齊；間空相機留 0）；本方法只負責「秀」：設 ops/閾值 + 帶入合圖視野。
        /// </summary>
        public static void UpdateOverviewChart(
            float[][] allMean, float[][] allMax,
            double[] opsArr, double[] posArr,
            float errMean, float errMax,
            ColumnCurveChartHelper target,
            int cameraCount,
            StitchMode stitchMode,
            Func<int, bool, double, double> viewRangeProvider,
            MergeOverlap overlap = MergeOverlap.Midline,
            float valueScale = 1f)
        {
            if (target == null) return;
            if (allMean == null)
            {
                target.Clear();
                return;
            }

            var r = CurveOverviewMerger.Merge(allMean, allMax, opsArr, posArr, cameraCount, overlap);
            if (!r.Valid)
            {
                target.Clear();
                return;
            }
            ScaleMergedCurveValues(r.Mean, r.Max, valueScale);

            target.SetOps(r.GridMm * 1000.0);
            target.SetThresholds(errMean, errMax);

            // 合圖模式：帶入 canvas 當前視野
            double viewLeft = double.NaN, viewRight = double.NaN;
            if (stitchMode != StitchMode.Vertical && viewRangeProvider != null)
            {
                viewLeft = viewRangeProvider(0, true, double.NaN);
                viewRight = viewRangeProvider(0, false, double.NaN);
            }
            target.UpdateDataAndView(r.Mean, r.Max, r.GlobalMinMm, viewLeft, viewRight);
        }

        internal static void ScaleMergedCurveValues(float[] mean, float[] max, float valueScale)
        {
            if (HessianRescaleHelper.IsNoOp(valueScale)) return;
            ScaleInPlace(mean, valueScale);
            ScaleInPlace(max, valueScale);
        }

        private static void ScaleInPlace(float[] values, float scale)
        {
            if (values == null) return;
            for (int i = 0; i < values.Length; i++) values[i] *= scale;
        }

        /// <summary>
        /// 載入多張影像的 .bin 曲線，Mean 取平均、Max 取最大值。
        /// </summary>
        public static void MergeCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax)
        {
            MergeCurves(
                imagePaths, out mergedMean, out mergedMax, out _, CancellationToken.None);
        }

        public static void MergeCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax, CancellationToken cancellationToken)
        {
            MergeCurves(
                imagePaths, out mergedMean, out mergedMax, out _, cancellationToken);
        }

        public static void MergeCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax, out int mergedCaptureCount,
            CancellationToken cancellationToken)
        {
            mergedMean = null;
            mergedMax = null;
            mergedCaptureCount = 0;

            int curveLen = 0;
            int[] meanCounts = null;

            foreach (string path in imagePaths)
            {
                cancellationToken.ThrowIfCancellationRequested();
                string basePath = GetCurveBasePath(path);
                var mean = CurveBinFile.Load(CaptureFileNaming.ResolveMeanC(basePath));
                var max = CurveBinFile.Load(CaptureFileNaming.ResolveMaxC(basePath));
                if (mean != null && max != null && mean.Length > 0)
                {
                    mergedCaptureCount++;
                    if (curveLen == 0)
                    {
                        curveLen = mean.Length;
                        mergedMean = new float[curveLen];
                        mergedMax = new float[curveLen];
                        meanCounts = new int[curveLen];
                        for (int i = 0; i < curveLen; i++) mergedMax[i] = float.MinValue;
                    }

                    int meanLength = Math.Min(curveLen, mean.Length);
                    for (int i = 0; i < meanLength; i++)
                    {
                        mergedMean[i] += mean[i];
                        meanCounts[i]++;
                    }

                    int maxLength = Math.Min(curveLen, max.Length);
                    for (int i = 0; i < maxLength; i++)
                        if (max[i] > mergedMax[i]) mergedMax[i] = max[i];
                }
            }

            if (curveLen == 0) return;

            for (int x = 0; x < curveLen; x++)
            {
                mergedMean[x] = meanCounts[x] > 0 ? mergedMean[x] / meanCounts[x] : 0;
                if (mergedMax[x] == float.MinValue) mergedMax[x] = 0;
            }
        }

        /// <summary>
        /// Row 曲線合併：多張影像的 row curves 依時間順序串接。
        /// </summary>
        public static void MergeRowCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax = null;

            // slot-based：imagePaths 可含 null（呼叫端對齊參考時間軸後缺幀=null）→ 該格補「0 曲線」一格，
            // 與影像黑布占位（GrabImageStitcher）對齊；no-null 時行為與舊版一致。
            var allMean = new List<float[]>();   // null = 缺幀（0 占位）
            var allMax = new List<float[]>();
            int frameLen = 0;                     // 每幀列曲線長度（取自第一個有資料的幀）

            foreach (string path in imagePaths)
            {
                if (string.IsNullOrEmpty(path)) { allMean.Add(null); allMax.Add(null); continue; }
                string basePath = GetCurveBasePath(path);
                var mean = CurveBinFile.Load(CaptureFileNaming.ResolveMeanR(basePath));
                var max = CurveBinFile.Load(CaptureFileNaming.ResolveMaxR(basePath));
                if (mean != null && max != null && mean.Length > 0)
                {
                    allMean.Add(mean);
                    allMax.Add(max);
                    if (frameLen == 0) frameLen = mean.Length;
                }
                else { allMean.Add(null); allMax.Add(null); }   // bin 也缺 → 同樣 0 占位
            }

            if (frameLen == 0) return;   // 沒有任何真資料

            int totalLen = 0;
            foreach (var a in allMean) totalLen += (a?.Length ?? frameLen);

            mergedMean = new float[totalLen];   // 預設全 0 → 缺幀那格自然是 0 曲線
            mergedMax = new float[totalLen];
            int offset = 0;
            for (int j = 0; j < allMean.Count; j++)
            {
                if (allMean[j] != null)
                {
                    Array.Copy(allMean[j], 0, mergedMean, offset, allMean[j].Length);
                    Array.Copy(allMax[j], 0, mergedMax, offset, allMax[j].Length);
                    offset += allMean[j].Length;
                }
                else offset += frameLen;   // 缺幀：跳過一格（已是 0）= 黑布那格的 0 曲線
            }
        }

        /// <summary>
        /// 多台相機 Row 曲線重疊合併（Global 模式用）：
        /// 所有曲線靠上對齊（index 0），取最大長度保留全部資料。
        /// 重疊區 Mean 取平均、Max 取最大值；非重疊區保留有資料的相機值。
        /// </summary>
        public static void MergeRowCurvesOverlap(float[][] allMean, float[][] allMax,
            int cameraCount, out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax = null;

            int maxLen = 0;
            int validCount = 0;
            for (int i = 0; i < cameraCount; i++)
            {
                if (allMean != null && i < allMean.Length && allMean[i] != null && allMean[i].Length > 0)
                {
                    if (allMean[i].Length > maxLen) maxLen = allMean[i].Length;
                    validCount++;
                }
            }
            if (validCount == 0 || maxLen <= 0) return;

            mergedMean = new float[maxLen];
            mergedMax = new float[maxLen];
            var count = new int[maxLen];

            for (int i = 0; i < cameraCount; i++)
            {
                if (allMean == null || i >= allMean.Length || allMean[i] == null) continue;
                var curveMean = allMean[i];
                var curveMax = (allMax != null && i < allMax.Length) ? allMax[i] : null;
                int len = curveMean.Length;

                for (int j = 0; j < len; j++)
                {
                    mergedMean[j] += curveMean[j];
                    count[j]++;
                    float mv = (curveMax != null && j < curveMax.Length) ? curveMax[j] : 0;
                    if (mv > mergedMax[j]) mergedMax[j] = mv;
                }
            }

            for (int j = 0; j < maxLen; j++)
                if (count[j] > 1) mergedMean[j] /= count[j];
        }

        /// <summary>
        /// 從影像路徑取得 .bin 曲線的 basePath：
        /// _raw.jpg → strip suffix；其餘 → Path without extension。
        /// </summary>
        public static string GetCurveBasePath(string imagePath)
            => CaptureFileNaming.BaseFromImagePath(imagePath);
    }
}
