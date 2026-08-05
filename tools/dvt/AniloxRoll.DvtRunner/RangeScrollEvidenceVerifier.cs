using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.DvtRunner
{
    internal static class RangeScrollEvidenceVerifier
    {
        private static readonly Regex ListPattern = new Regex(
            @"DT range list preview gen=(\d+).* ms=(\d+) source=index",
            RegexOptions.CultureInvariant);

        private static readonly Regex CurvePattern = new Regex(
            @"DT range preview apply gen=(\d+) latest=(\d+).* loadMs=(\d+) drawMs=(\d+)",
            RegexOptions.CultureInvariant);

        private static readonly Regex StallPattern = new Regex(
            @"\[UiStall\] (\d+)ms",
            RegexOptions.CultureInvariant);

        public static string Verify(IReadOnlyList<string> lines)
        {
            int intents = lines.Count(line =>
                line.Contains("ui:【序號範圍-"));
            var lists = MatchRows(lines, ListPattern, 2);
            var curves = MatchRows(lines, CurvePattern, 4);
            int fallbacks = lines.Count(line =>
                line.Contains("DT list virtual fallback"));
            int fatal = lines.Count(line =>
                line.Contains("[Fatal]") || line.Contains("Fatal exception"));
            int worstStall = lines
                .Select(line => StallPattern.Match(line))
                .Where(match => match.Success)
                .Select(match => int.Parse(match.Groups[1].Value))
                .DefaultIfEmpty(0)
                .Max();

            int[] listGenerations = lists.Select(row => row[0]).ToArray();
            int[] curveGenerations = curves.Select(row => row[0]).ToArray();
            int lagged = curves.Count(row => row[0] < row[1]);
            bool listMonotonic = IsStrictlyIncreasing(listGenerations);
            bool curveMonotonic = IsStrictlyIncreasing(curveGenerations);
            bool finalCaughtUp =
                listGenerations.Length > 0 &&
                curveGenerations.Length > 0 &&
                curveGenerations[curveGenerations.Length - 1] >=
                    listGenerations[listGenerations.Length - 1];
            int worstListMs = lists.Select(row => row[1]).DefaultIfEmpty(0).Max();
            int worstCurveMs = curves
                .Select(row => row[2] + row[3])
                .DefaultIfEmpty(0)
                .Max();
            int coldCurveMs = curves.Count > 0 ? curves[0][2] + curves[0][3] : 0;
            int worstWarmCurveMs = curves
                .Skip(1)
                .Select(row => row[2] + row[3])
                .DefaultIfEmpty(0)
                .Max();

            var failures = new List<string>();
            if (intents < 100) failures.Add("intent<100");
            if (lists.Count < 2 || lists.Count >= intents)
                failures.Add("list throttle not visible");
            if (curves.Count < 2 || curves.Count >= intents)
                failures.Add("curve throttle not visible");
            if (lagged < 1) failures.Add("no intermediate jump");
            if (!listMonotonic) failures.Add("list generation regressed");
            if (!curveMonotonic) failures.Add("curve generation regressed");
            if (!finalCaughtUp) failures.Add("final curve did not catch up");
            if (worstListMs > 100) failures.Add("list preview >100ms");
            if (coldCurveMs > 3000) failures.Add("cold curve preview >3000ms");
            if (worstWarmCurveMs > 500) failures.Add("warm curve preview >500ms");
            if (worstStall > 1000) failures.Add("UI stall >1000ms");
            if (fallbacks > 0) failures.Add("virtual list fallback");
            if (fatal > 0) failures.Add("fatal exception");

            string metrics =
                $"intent={intents} list={lists.Count} curve={curves.Count} " +
                $"jumps={lagged} final={finalCaughtUp} " +
                $"listWorst={worstListMs}ms curveCold={coldCurveMs}ms " +
                $"curveWarmWorst={worstWarmCurveMs}ms curveWorst={worstCurveMs}ms " +
                $"uiStallWorst={worstStall}ms fallback={fallbacks} fatal={fatal}";
            if (failures.Count > 0)
                throw new InvalidOperationException(
                    "Range scroll DVT failed: " +
                    string.Join(", ", failures) + "; " + metrics);
            return metrics;
        }

        private static List<int[]> MatchRows(
            IEnumerable<string> lines,
            Regex pattern,
            int groupCount)
        {
            var rows = new List<int[]>();
            foreach (string line in lines)
            {
                Match match = pattern.Match(line);
                if (!match.Success) continue;
                var row = new int[groupCount];
                for (int index = 0; index < groupCount; index++)
                    row[index] = int.Parse(match.Groups[index + 1].Value);
                rows.Add(row);
            }
            return rows;
        }

        private static bool IsStrictlyIncreasing(int[] values)
        {
            for (int index = 1; index < values.Length; index++)
            {
                if (values[index] <= values[index - 1]) return false;
            }
            return true;
        }
    }
}
