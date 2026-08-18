using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.DvtRunner
{
    internal static class LiveColumnRangeEvidenceVerifier
    {
        private const double TargetToleranceMm = 0.05;
        private const double ViewDriftToleranceMm = 0.50;
        private const double PlotDriftTolerancePercent = 0.05;

        private static readonly Regex MainPattern = new Regex(
            @"LC mainRange viewX=(?<left>-?\d+(?:\.\d+)?)~(?<right>-?\d+(?:\.\d+)?)",
            RegexOptions.CultureInvariant);

        private static readonly Regex ColumnPattern = new Regex(
            @"LC colRange source=(?<source>view|data) " +
            @"target=(?<left>-?\d+(?:\.\d+)?)~(?<right>-?\d+(?:\.\d+)?) " +
            @"axis=(?<axisLeft>-?\d+(?:\.\d+)?)~(?<axisRight>-?\d+(?:\.\d+)?)/" +
            @"view=(?<viewLeft>-?\d+(?:\.\d+)?)~(?<viewRight>-?\d+(?:\.\d+)?) " +
            @"plot=(?<plotLeft>-?\d+(?:\.\d+)?)~(?<plotRight>-?\d+(?:\.\d+)?)",
            RegexOptions.CultureInvariant);

        public static string Verify(IReadOnlyList<string> lines)
        {
            var samples = new List<Sample>();
            var dataGroups = new Dictionary<string, List<Sample>>(StringComparer.Ordinal);
            var latestViewByTarget = new Dictionary<string, Sample>(StringComparer.Ordinal);
            double mainLeft = double.NaN;
            double mainRight = double.NaN;
            var failures = new List<string>();
            int pairedDataSamples = 0;

            foreach (string line in lines)
            {
                Match main = MainPattern.Match(line);
                if (main.Success)
                {
                    mainLeft = Parse(main.Groups["left"].Value);
                    mainRight = Parse(main.Groups["right"].Value);
                    continue;
                }

                Match column = ColumnPattern.Match(line);
                if (!column.Success) continue;
                var sample = new Sample(column);
                samples.Add(sample);
                if (!double.IsNaN(mainLeft) &&
                    (Math.Abs(sample.TargetLeft - mainLeft) > TargetToleranceMm ||
                     Math.Abs(sample.TargetRight - mainRight) > TargetToleranceMm))
                {
                    failures.Add(string.Format(
                        CultureInfo.InvariantCulture,
                        "column target {0:F2}~{1:F2} != main {2:F2}~{3:F2}",
                        sample.TargetLeft, sample.TargetRight, mainLeft, mainRight));
                }

                string targetKey = sample.TargetKey;
                if (sample.Source == "view")
                {
                    latestViewByTarget[targetKey] = sample;
                    continue;
                }

                List<Sample> group;
                if (!dataGroups.TryGetValue(targetKey, out group))
                {
                    group = new List<Sample>();
                    dataGroups.Add(targetKey, group);
                }
                group.Add(sample);

                Sample viewSample;
                if (latestViewByTarget.TryGetValue(targetKey, out viewSample))
                {
                    pairedDataSamples++;
                    CheckDifference(sample.ViewLeft, viewSample.ViewLeft,
                        ViewDriftToleranceMm, "view-left", targetKey, failures);
                    CheckDifference(sample.ViewRight, viewSample.ViewRight,
                        ViewDriftToleranceMm, "view-right", targetKey, failures);
                    CheckDifference(sample.PlotLeft, viewSample.PlotLeft,
                        PlotDriftTolerancePercent, "plot-left", targetKey, failures);
                    CheckDifference(sample.PlotRight, viewSample.PlotRight,
                        PlotDriftTolerancePercent, "plot-right", targetKey, failures);
                }
            }

            List<Sample> dataSamples = dataGroups.Values.SelectMany(group => group).ToList();
            if (dataSamples.Count < 3)
                failures.Add("fewer than three column data redraw samples were observed");
            if (pairedDataSamples == 0)
                failures.Add("no data redraw followed a view update for the same target");

            int stableTargets = 0;
            double worstViewDrift = 0.0;
            double worstPlotDrift = 0.0;
            foreach (KeyValuePair<string, List<Sample>> entry in dataGroups)
            {
                if (entry.Value.Count < 2) continue;
                stableTargets++;
                List<Sample> steady = entry.Value.Skip(1).ToList();
                double viewDrift = Math.Max(
                    Spread(steady.Select(sample => sample.ViewLeft)),
                    Spread(steady.Select(sample => sample.ViewRight)));
                double plotDrift = Math.Max(
                    Spread(steady.Select(sample => sample.PlotLeft)),
                    Spread(steady.Select(sample => sample.PlotRight)));
                worstViewDrift = Math.Max(worstViewDrift, viewDrift);
                worstPlotDrift = Math.Max(worstPlotDrift, plotDrift);
                if (viewDrift > ViewDriftToleranceMm)
                    failures.Add(string.Format(CultureInfo.InvariantCulture,
                        "target={0} view drift {1:F2} exceeds {2:F2}",
                        entry.Key, viewDrift, ViewDriftToleranceMm));
                if (plotDrift > PlotDriftTolerancePercent)
                    failures.Add(string.Format(CultureInfo.InvariantCulture,
                        "target={0} plot drift {1:F2} exceeds {2:F2}",
                        entry.Key, plotDrift, PlotDriftTolerancePercent));
            }

            Sample firstData = dataSamples.FirstOrDefault();
            string rangeReference = double.IsNaN(mainLeft) && firstData != null
                ? string.Format(
                    CultureInfo.InvariantCulture,
                    "target={0:F2}~{1:F2}",
                    firstData.TargetLeft,
                    firstData.TargetRight)
                : string.Format(
                    CultureInfo.InvariantCulture,
                    "main={0:F2}~{1:F2}",
                    mainLeft,
                    mainRight);
            string metrics = string.Format(
                CultureInfo.InvariantCulture,
                "samples={0} data={1} paired={2} targets={3} {4} " +
                "viewDriftMax={5:F2}mm plotDriftMax={6:F2}%",
                samples.Count,
                dataSamples.Count,
                pairedDataSamples,
                stableTargets,
                rangeReference,
                worstViewDrift,
                worstPlotDrift);
            if (failures.Count > 0)
                throw new InvalidOperationException(
                    "Live column range DVT failed: " +
                    string.Join("; ", failures.Distinct()) + "; " + metrics);
            return metrics;
        }

        private static void CheckDifference(
            double actual,
            double expected,
            double tolerance,
            string name,
            string target,
            ICollection<string> failures)
        {
            double difference = Math.Abs(actual - expected);
            if (difference <= tolerance) return;
            failures.Add(string.Format(
                CultureInfo.InvariantCulture,
                "target={0} data {1} differs from view by {2:F2} (actual={3:F2}, expected={4:F2})",
                target, name, difference, actual, expected));
        }

        private static double Spread(IEnumerable<double> values)
        {
            double[] materialized = values.ToArray();
            return materialized.Length == 0
                ? 0.0
                : materialized.Max() - materialized.Min();
        }

        private static double Parse(string value)
            => double.Parse(value, CultureInfo.InvariantCulture);

        private sealed class Sample
        {
            public Sample(Match match)
            {
                Source = match.Groups["source"].Value;
                TargetLeft = Parse(match.Groups["left"].Value);
                TargetRight = Parse(match.Groups["right"].Value);
                ViewLeft = Parse(match.Groups["viewLeft"].Value);
                ViewRight = Parse(match.Groups["viewRight"].Value);
                PlotLeft = Parse(match.Groups["plotLeft"].Value);
                PlotRight = Parse(match.Groups["plotRight"].Value);
            }

            public string Source { get; }
            public double TargetLeft { get; }
            public double TargetRight { get; }
            public double ViewLeft { get; }
            public double ViewRight { get; }
            public double PlotLeft { get; }
            public double PlotRight { get; }
            public string TargetKey => string.Format(
                CultureInfo.InvariantCulture,
                "{0:F2}~{1:F2}",
                TargetLeft,
                TargetRight);
        }
    }
}
