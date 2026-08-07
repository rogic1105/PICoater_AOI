using System;
using TanukiCv.Controls;

namespace AniloxRoll.Monitor.Core.Services
{
    internal sealed class LivePixelCurveAxisSample
    {
        public float SourceImageMax { get; set; }
        public float SourceCurveMeanMax { get; set; }
        public float SourceCurveMax { get; set; }
        public float SourceImagePeak { get; set; }
        public float SourceCurveMeanPeak { get; set; }
        public float SourceCurveMaxPeak { get; set; }
        public float DisplayImagePeak { get; set; }
        public float DisplayCurveMeanPeak { get; set; }
        public float DisplayCurveMaxPeak { get; set; }
        public float MaxDelta { get; set; }
        public bool QuantizedToZero { get; set; }
        public bool MaxMatches { get; set; }
    }

    internal sealed class LivePixelCurveProbeResult
    {
        public LivePixelCurveAxisSample Column { get; set; }
        public LivePixelCurveAxisSample Row { get; set; }
    }

    /// <summary>
    /// DVT-only measurement of the image and curve products emitted by one native frame.
    /// The maximum curve describes the same global peak as the corresponding Hessian image.
    /// The mean curve is recorded for diagnosis but is not compared for equality.
    /// </summary>
    internal static class LivePixelCurveProbe
    {
        // The image path is quantized to one byte while the curve remains float.
        internal const float MatchTolerance = 1f / 255f;

        public static LivePixelCurveProbeResult Measure(
            byte[] columnImage,
            byte[] rowImage,
            float[] columnMeanCurve,
            float[] columnMaxCurve,
            float[] rowMeanCurve,
            float[] rowMaxCurve,
            float captureHm,
            float currentColumnHm,
            float currentRowHm,
            float columnSourceGain,
            float rowSourceGain)
        {
            return new LivePixelCurveProbeResult
            {
                Column = MeasureAxis(
                    columnImage, columnMeanCurve, columnMaxCurve,
                    DisplayScale(currentColumnHm, columnSourceGain),
                    HessianRescaleHelper.RawCurveToDisplayScale(
                        captureHm, currentColumnHm)),
                Row = MeasureAxis(
                    rowImage, rowMeanCurve, rowMaxCurve,
                    DisplayScale(currentRowHm, rowSourceGain),
                    HessianRescaleHelper.RawCurveToDisplayScale(
                        captureHm, currentRowHm))
            };
        }

        private static float DisplayScale(float currentGain, float sourceGain)
            => currentGain > 0f && sourceGain > 0f ? currentGain / sourceGain : 1f;

        private static LivePixelCurveAxisSample MeasureAxis(
            byte[] image,
            float[] meanCurve,
            float[] maxCurve,
            float imageGain,
            float curveGain)
        {
            byte imageMax = Max(image);
            float meanMax = Max(meanCurve);
            float curveMax = Max(maxCurve);
            float sourceImagePeak = imageMax / 255f;
            float sourceMeanPeak = meanMax / 255f;
            float sourceCurveMaxPeak = curveMax / 255f;
            float displayImagePeak = GrayIntensity.Scale(imageMax, imageGain) / 255f;
            float displayMeanPeak = Math.Max(0f, sourceMeanPeak * curveGain);
            float displayCurveMaxPeak = Math.Max(0f, sourceCurveMaxPeak * curveGain);
            float comparableCurvePeak = Math.Min(1f, displayCurveMaxPeak);
            float delta = Math.Abs(displayImagePeak - comparableCurvePeak);
            bool quantizedToZero = imageMax == 0 && curveMax > 0f;

            return new LivePixelCurveAxisSample
            {
                SourceImageMax = imageMax,
                SourceCurveMeanMax = meanMax,
                SourceCurveMax = curveMax,
                SourceImagePeak = sourceImagePeak,
                SourceCurveMeanPeak = sourceMeanPeak,
                SourceCurveMaxPeak = sourceCurveMaxPeak,
                DisplayImagePeak = displayImagePeak,
                DisplayCurveMeanPeak = displayMeanPeak,
                DisplayCurveMaxPeak = displayCurveMaxPeak,
                MaxDelta = delta,
                QuantizedToZero = quantizedToZero,
                MaxMatches = !quantizedToZero && delta <= MatchTolerance
            };
        }

        private static byte Max(byte[] values)
        {
            if (values == null || values.Length == 0) return 0;
            byte max = 0;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > max) max = values[i];
                if (max == byte.MaxValue) break;
            }
            return max;
        }

        private static float Max(float[] values)
        {
            if (values == null || values.Length == 0) return 0f;
            float max = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                float value = values[i];
                if (!float.IsNaN(value) && value > max) max = value;
            }
            return max;
        }
    }
}
