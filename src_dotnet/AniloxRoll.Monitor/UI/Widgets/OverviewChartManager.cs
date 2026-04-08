using System;
using System.Collections.Generic;
using System.IO;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Widgets
{
    /// <summary>
    /// 全覽圖合併演算法與 .bin 曲線讀取：
    /// 將 7 台相機的 mura 曲線依機台布局位置合併到全覽圖。
    /// </summary>
    public static class OverviewChartManager
    {
        private const int MaxOverviewPoints = 2000;

        /// <summary>
        /// 將多台相機曲線合併到單一全覽圖。
        /// 重疊區域：Mean 取平均、Max 取最大值。
        /// </summary>
        public static void UpdateOverviewChart(
            float[][] allMean, float[][] allMax,
            double[] opsArr, double[] posArr,
            float errMean, float errMax,
            MuraChartHelper target,
            int cameraCount,
            StitchMode stitchMode,
            Func<int, bool, double, double> viewRangeProvider)
        {
            if (target == null || allMean == null) return;

            // 全域範圍：涵蓋全部相機位置，缺圖用現有影像寬度平均類推
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
            if (globalMin >= globalMax) return;

            // 格點間距：至少 OPS 精度，但上限 MaxOverviewPoints 點
            double gridMm = Math.Max(minOpsUm / 1000.0, (globalMax - globalMin) / MaxOverviewPoints);

            int totalLen = (int)Math.Ceiling((globalMax - globalMin) / gridMm);
            if (totalLen <= 0 || totalLen > MaxOverviewPoints + 1) return;

            // 兩層合併：
            // 1) bin 內降解析（同一台相機多點 → 1 bin）→ max-window 保峰值
            // 2) 相機重疊（多台相機同一 bin）→ Mean 取平均、Max 取最大值
            var mergedMean = new float[totalLen];
            var mergedMax = new float[totalLen];
            var overlapCount = new int[totalLen];

            for (int i = 0; i < cameraCount; i++)
            {
                var curveMean = allMean[i];
                if (curveMean == null || curveMean.Length == 0) continue;
                var curveMax = (allMax != null && i < allMax.Length) ? allMax[i] : null;

                double camOpsMm = opsArr[i] / 1000.0;
                double camStart = posArr[i];

                var camBinMean = new float[totalLen];
                var camBinMax = new float[totalLen];
                var camBinHit = new bool[totalLen];

                for (int j = 0; j < curveMean.Length; j++)
                {
                    int idx = (int)((camStart + j * camOpsMm - globalMin) / gridMm);
                    if (idx < 0 || idx >= totalLen) continue;

                    if (!camBinHit[idx] || curveMean[j] > camBinMean[idx])
                        camBinMean[idx] = curveMean[j];

                    float mv = (curveMax != null && j < curveMax.Length) ? curveMax[j] : 0;
                    if (!camBinHit[idx] || mv > camBinMax[idx])
                        camBinMax[idx] = mv;

                    camBinHit[idx] = true;
                }

                for (int k = 0; k < totalLen; k++)
                {
                    if (!camBinHit[k]) continue;
                    mergedMean[k] += camBinMean[k];
                    overlapCount[k] += 1;
                    if (camBinMax[k] > mergedMax[k]) mergedMax[k] = camBinMax[k];
                }
            }

            // 重疊區域 Mean 取平均
            for (int i = 0; i < totalLen; i++)
                if (overlapCount[i] > 1) mergedMean[i] /= overlapCount[i];

            target.SetOps(gridMm * 1000.0);
            target.SetThresholds(errMean, errMax);

            // 合圖模式：帶入 canvas 當前視野
            double viewLeft = double.NaN, viewRight = double.NaN;
            if (stitchMode != StitchMode.Vertical && viewRangeProvider != null)
            {
                viewLeft = viewRangeProvider(0, true, double.NaN);
                viewRight = viewRangeProvider(0, false, double.NaN);
            }
            target.UpdateDataAndView(mergedMean, mergedMax, globalMin, viewLeft, viewRight);
        }

        /// <summary>
        /// 載入多張影像的 .bin 曲線，Mean 取平均、Max 取最大值。
        /// </summary>
        public static void MergeCurves(IList<string> imagePaths,
            out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax = null;

            var allMean = new List<float[]>();
            var allMax = new List<float[]>();
            int curveLen = 0;

            foreach (string path in imagePaths)
            {
                string basePath = GetCurveBasePath(path);
                var mean = InspectionEngine.LoadCurveBin(basePath + "_mean_v.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_mean.bin");
                var max = InspectionEngine.LoadCurveBin(basePath + "_max_v.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_max.bin");
                if (mean != null && max != null && mean.Length > 0)
                {
                    allMean.Add(mean);
                    allMax.Add(max);
                    if (curveLen == 0) curveLen = mean.Length;
                }
            }

            if (allMean.Count == 0 || curveLen == 0) return;

            mergedMean = new float[curveLen];
            mergedMax = new float[curveLen];
            for (int x = 0; x < curveLen; x++)
            {
                float sumMean = 0;
                float maxVal = float.MinValue;
                int count = 0;
                for (int j = 0; j < allMean.Count; j++)
                {
                    if (x < allMean[j].Length) { sumMean += allMean[j][x]; count++; }
                    if (x < allMax[j].Length && allMax[j][x] > maxVal) maxVal = allMax[j][x];
                }
                mergedMean[x] = count > 0 ? sumMean / count : 0;
                mergedMax[x] = maxVal > float.MinValue ? maxVal : 0;
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

            var allMean = new List<float[]>();
            var allMax = new List<float[]>();

            foreach (string path in imagePaths)
            {
                string basePath = GetCurveBasePath(path);
                var mean = InspectionEngine.LoadCurveBin(basePath + "_mean_h.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_row_mean.bin");
                var max = InspectionEngine.LoadCurveBin(basePath + "_max_h.bin")
                        ?? InspectionEngine.LoadCurveBin(basePath + "_row_max.bin");
                if (mean != null && max != null && mean.Length > 0)
                {
                    allMean.Add(mean);
                    allMax.Add(max);
                }
            }

            if (allMean.Count == 0) return;

            int totalLen = 0;
            foreach (var a in allMean) totalLen += a.Length;

            mergedMean = new float[totalLen];
            mergedMax = new float[totalLen];
            int offset = 0;
            for (int j = 0; j < allMean.Count; j++)
            {
                Array.Copy(allMean[j], 0, mergedMean, offset, allMean[j].Length);
                Array.Copy(allMax[j], 0, mergedMax, offset, allMax[j].Length);
                offset += allMean[j].Length;
            }
        }

        /// <summary>
        /// 從影像路徑取得 .bin 曲線的 basePath：
        /// _raw.jpg → strip suffix；其餘 → Path without extension。
        /// </summary>
        public static string GetCurveBasePath(string imagePath)
        {
            if (imagePath.EndsWith("_raw.jpg", StringComparison.OrdinalIgnoreCase))
                return imagePath.Substring(0, imagePath.Length - "_raw.jpg".Length);
            return Path.Combine(
                Path.GetDirectoryName(imagePath),
                Path.GetFileNameWithoutExtension(imagePath));
        }
    }
}
