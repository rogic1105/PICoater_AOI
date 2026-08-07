using System;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 統一處理 view-time 正規值 rescale 公式：
    /// display = (bin/255) × HM_capture × HM_current。
    /// Native curve bins contain neutral Hessian × 255 / HM_capture; multiplying by the
    /// capture factor first restores the neutral value, then HM_current is the display gain.
    ///
    /// 為什麼集中：rescale 邏輯原本散在 4 處（DataStatisticsPresenter / ReviewChartPresenter
    /// 各 2 份），公式雖一致但易漂移（未來改 ratio 計算邏輯必須 4 處同改）。
    /// </summary>
    public static class HessianRescaleHelper
    {
        /// <summary>
        /// 計算 view scale = HM_capture × HM_current；任一為 0 時回傳 1f。
        /// 因此目前正規值加倍時，顯示 Curve 也線性加倍，並與 neutral Hessian map 一致。
        /// </summary>
        public static float RawCurveToDisplayScale(float captureHm, float currentHm)
        {
            if (captureHm <= 0f || currentHm <= 0f) return 1f;
            return captureHm * currentHm;
        }

        /// <summary>
        /// Scales a value that was already normalized with the capture-time HM.
        /// CSV and #CURVE summaries use this representation.
        /// </summary>
        public static float NormalizedValueToDisplayScale(float captureHm, float currentHm)
        {
            if (captureHm <= 0f || currentHm <= 0f) return 1f;
            return currentHm / captureHm;
        }

        /// <summary>ratio 是否需要實際做 scaling（避免無謂的 O(N) 迴圈）。</summary>
        public static bool IsNoOp(float ratio) => Math.Abs(ratio - 1f) < 0.0001f;

        /// <summary>對 7 台相機的 2D 曲線就地套用 ratio。</summary>
        public static void RescaleInPlace2D(float[][] allMean, float[][] allMax,
            float captureHm, float currentHm)
        {
            float ratio = RawCurveToDisplayScale(captureHm, currentHm);
            if (IsNoOp(ratio)) return;
            ApplyRatio2D(allMean, ratio);
            ApplyRatio2D(allMax,  ratio);
        }

        /// <summary>對 1D 曲線就地套用 ratio。</summary>
        public static void RescaleInPlace1D(float[] data, float captureHm, float currentHm)
        {
            if (data == null) return;
            float ratio = RawCurveToDisplayScale(captureHm, currentHm);
            if (IsNoOp(ratio)) return;
            for (int i = 0; i < data.Length; i++) data[i] *= ratio;
        }

        /// <summary>複製 2D 曲線陣列並套 ratio（保留原快取不變）。</summary>
        public static float[][] CloneAndRescale2D(float[][] src, float captureHm, float currentHm)
        {
            if (src == null) return null;
            float ratio = RawCurveToDisplayScale(captureHm, currentHm);
            bool noOp = IsNoOp(ratio);
            var dst = new float[src.Length][];
            for (int i = 0; i < src.Length; i++)
            {
                if (src[i] == null) continue;
                dst[i] = new float[src[i].Length];
                if (noOp) Array.Copy(src[i], dst[i], src[i].Length);
                else for (int j = 0; j < src[i].Length; j++) dst[i][j] = src[i][j] * ratio;
            }
            return dst;
        }

        /// <summary>複製 1D 曲線並套 ratio。</summary>
        public static float[] CloneAndRescale1D(float[] src, float captureHm, float currentHm)
        {
            if (src == null) return null;
            float ratio = RawCurveToDisplayScale(captureHm, currentHm);
            bool noOp = IsNoOp(ratio);
            var dst = new float[src.Length];
            if (noOp) { Array.Copy(src, dst, src.Length); return dst; }
            for (int i = 0; i < src.Length; i++) dst[i] = src[i] * ratio;
            return dst;
        }

        private static void ApplyRatio2D(float[][] arr, float ratio)
        {
            if (arr == null) return;
            for (int i = 0; i < arr.Length; i++)
            {
                var row = arr[i];
                if (row == null) continue;
                for (int j = 0; j < row.Length; j++) row[j] *= ratio;
            }
        }
    }
}
