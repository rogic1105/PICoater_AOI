using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.DvtRunner
{
    internal static class MonitorFunctionalEvidenceVerifier
    {
        private static readonly Regex PixelCurvePattern = new Regex(
            @"live pixel-curve probe cam(?<camera>\d+) tick=\d+ axis=(?<axis>C|R) .* " +
            @"imagePeak=(?<image>\d+(?:\.\d+)?) .*curveMaxPeak=(?<curve>\d+(?:\.\d+)?) " +
            @"delta=(?<delta>\d+(?:\.\d+)?) .*verdict=(?<verdict>match|mismatch)",
            RegexOptions.CultureInvariant);

        private static readonly Regex StimulusPattern = new Regex(
            @"live inspection stimulus brightness=\d+ direction=(?<axis>col|row) " +
            @"mean=(?<mean>\d+(?:\.\d+)?) max=(?<max>\d+(?:\.\d+)?) " +
            @"threshold=(?<meanThreshold>\d+(?:\.\d+)?)/(?<maxThreshold>\d+(?:\.\d+)?) " +
            @"mode=(?<mode>Mean|Max|Both) verdict=(?<verdict>O|X) ",
            RegexOptions.CultureInvariant);

        private static readonly Regex IcStatePattern = new Regex(
            @"IC state viewX \S+ viewY (?<top>-?\d+(?:\.\d+)?)~(?<bottom>-?\d+(?:\.\d+)?)",
            RegexOptions.CultureInvariant);

        public static string Verify(
            IReadOnlyList<string> lines,
            string specification)
        {
            Dictionary<string, string> expected = ParseSpecification(specification);
            string mode = Required(expected, "mode");
            string direction = Required(expected, "direction");
            string layer = Required(expected, "layer");
            string[] stimulusAxes = expected.TryGetValue("stimulus", out string axes)
                ? axes.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries)
                : new string[0];

            var failures = new List<string>();
            if (!lines.Any(line => line.Contains("capture gate open cams=")))
                failures.Add("capture gate did not open");
            if (!lines.Any(line => line.Contains(
                    "rowCurve present after=mainImage") &&
                    line.Contains("mode=" + mode)))
                failures.Add("row Curve was not presented after the main image in " + mode);

            string expectedEnhance = layer == "raw" ? "False" : "True";
            if (!lines.Any(line => line.Contains(
                    "live enhance enabled=" + expectedEnhance +
                    " direction=" + layer + " ")))
                failures.Add("main display layer did not settle to " + layer);

            string activeAxis = layer == "row" ? "R" : "C";
            List<Match> pixelSamples = lines
                .Select(line => PixelCurvePattern.Match(line))
                .Where(match => match.Success)
                .ToList();
            Match activeSample = pixelSamples.LastOrDefault(match =>
                match.Groups["axis"].Value == activeAxis &&
                match.Groups["verdict"].Value == "match");
            if (activeSample == null)
                failures.Add("main image/" + activeAxis + " Curve numeric match is missing");
            if (pixelSamples.Any(match =>
                    match.Groups["verdict"].Value == "mismatch"))
                failures.Add("a pixel/Curve mismatch was observed");
            foreach (string axis in new[] { "C", "R" })
            {
                if (!pixelSamples.Any(match =>
                        match.Groups["axis"].Value == axis))
                    failures.Add("pixel/Curve probe missing axis=" + axis);
            }

            if (!lines.Any(line => line.Contains(
                    "LC row rowChart dir=" + direction)))
                failures.Add("row chart direction did not settle to " + direction);

            if (string.Equals(mode, "WF", StringComparison.Ordinal))
            {
                string anchor = direction == "BottomToTop" ? "底" : "頂";
                if (!lines.Any(line =>
                        line.Contains("WF state ") &&
                        line.Contains("畫面端=" + anchor)))
                    failures.Add("waterfall content anchor was not " + anchor);
            }
            else
            {
                Match state = lines
                    .Select(line => IcStatePattern.Match(line))
                    .LastOrDefault(match => match.Success);
                if (state == null)
                {
                    failures.Add("IC view state is missing");
                }
                else
                {
                    double top = ParseDouble(state.Groups["top"].Value);
                    double bottom = ParseDouble(state.Groups["bottom"].Value);
                    bool correct = direction == "BottomToTop"
                        ? top > bottom
                        : top < bottom;
                    if (!correct)
                        failures.Add(
                            "IC view direction is inconsistent: " +
                            top.ToString("F3", CultureInfo.InvariantCulture) + "~" +
                            bottom.ToString("F3", CultureInfo.InvariantCulture));
                }
            }

            List<Match> stimuli = lines
                .Select(line => StimulusPattern.Match(line))
                .Where(match => match.Success)
                .ToList();
            foreach (string stimulusAxis in stimulusAxes)
            {
                Match sample = stimuli.LastOrDefault(match =>
                    match.Groups["axis"].Value == stimulusAxis);
                if (sample == null)
                {
                    failures.Add("inspection stimulus missing axis=" + stimulusAxis);
                    continue;
                }
                string formulaFailure = ValidateStimulusFormula(sample);
                if (formulaFailure != null)
                    failures.Add(formulaFailure);
            }

            string metrics =
                "mode=" + mode +
                " direction=" + direction +
                " layer=" + layer +
                " pixelSamples=" + pixelSamples.Count +
                " stimuli=" + stimuli.Count;
            if (activeSample != null)
            {
                metrics +=
                    " activeImagePeak=" + activeSample.Groups["image"].Value +
                    " activeCurvePeak=" + activeSample.Groups["curve"].Value +
                    " delta=" + activeSample.Groups["delta"].Value;
            }
            if (failures.Count > 0)
                throw new InvalidOperationException(
                    "Monitor functional DVT failed: " +
                    string.Join(", ", failures) + "; " + metrics);
            return metrics;
        }

        private static string ValidateStimulusFormula(Match sample)
        {
            double mean = ParseDouble(sample.Groups["mean"].Value);
            double maximum = ParseDouble(sample.Groups["max"].Value);
            double meanThreshold = ParseDouble(sample.Groups["meanThreshold"].Value);
            double maxThreshold = ParseDouble(sample.Groups["maxThreshold"].Value);
            string mode = sample.Groups["mode"].Value;
            bool expected =
                (mode != "Max" && mean > meanThreshold) ||
                (mode != "Mean" && maximum > maxThreshold);
            bool actual = sample.Groups["verdict"].Value == "X";
            if (expected == actual) return null;
            return string.Format(
                CultureInfo.InvariantCulture,
                "inspection formula mismatch axis={0} mean={1:F4}/{2:F4} " +
                "max={3:F4}/{4:F4} mode={5} actual={6}",
                sample.Groups["axis"].Value,
                mean,
                meanThreshold,
                maximum,
                maxThreshold,
                mode,
                actual ? "X" : "O");
        }

        private static Dictionary<string, string> ParseSpecification(string value)
        {
            var result = new Dictionary<string, string>(
                StringComparer.OrdinalIgnoreCase);
            foreach (string token in (value ?? string.Empty).Split(';'))
            {
                int separator = token.IndexOf('=');
                if (separator <= 0) continue;
                result[token.Substring(0, separator).Trim()] =
                    token.Substring(separator + 1).Trim();
            }
            return result;
        }

        private static string Required(
            Dictionary<string, string> values,
            string key)
        {
            if (!values.TryGetValue(key, out string value) ||
                string.IsNullOrWhiteSpace(value))
                throw new InvalidOperationException(
                    "Monitor functional DVT specification requires " + key + ".");
            return value;
        }

        private static double ParseDouble(string value)
        {
            return double.Parse(value, CultureInfo.InvariantCulture);
        }
    }
}
