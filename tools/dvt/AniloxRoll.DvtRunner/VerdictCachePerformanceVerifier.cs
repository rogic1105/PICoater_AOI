using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace AniloxRoll.DvtRunner
{
    internal static class VerdictCachePerformanceVerifier
    {
        private static readonly Regex CachePattern = new Regex(
            @"DT verdict cache gen=(\d+) hits=(\d+)/(\d+) days=(\d+) loadMs=(\d+)",
            RegexOptions.CultureInvariant);
        private static readonly Regex PartialPattern = new Regex(
            @"DT verdict index apply=partial gen=(\d+).* ms=(\d+)",
            RegexOptions.CultureInvariant);
        private static readonly Regex CompletePattern = new Regex(
            @"DT verdict index apply=ok gen=(\d+).* ms=(\d+)",
            RegexOptions.CultureInvariant);

        public static string Verify(IReadOnlyList<string> lines)
        {
            List<CacheRun> runs = lines
                .Select(line => CachePattern.Match(line))
                .Where(match => match.Success)
                .Select(match => new CacheRun
                {
                    Generation = Parse(match, 1),
                    Hits = Parse(match, 2),
                    Requested = Parse(match, 3),
                    Days = Parse(match, 4),
                    CacheLoadMs = Parse(match, 5)
                })
                .ToList();
            CacheRun cold = runs.FirstOrDefault(run =>
                run.Requested > 0 && run.Hits == 0);
            CacheRun warm = runs.LastOrDefault(run =>
                run.Requested > 0 && run.Hits == run.Requested);
            if (cold == null)
                throw new InvalidOperationException(
                    "Verdict cache DVT did not observe a cold read.");
            if (warm == null || warm.Generation <= cold.Generation)
                throw new InvalidOperationException(
                    "Verdict cache DVT did not observe a later full warm read.");
            if (warm.Requested != cold.Requested)
                throw new InvalidOperationException(
                    "Cold and warm reads used different report populations.");
            if (warm.Days <= 0)
                throw new InvalidOperationException(
                    "Warm read reported no daily index files.");

            Dictionary<int, int> partialMs = ReadTimings(lines, PartialPattern);
            Dictionary<int, int> completeMs = ReadTimings(lines, CompletePattern);
            if (!partialMs.TryGetValue(cold.Generation, out int coldPartialMs) ||
                !partialMs.TryGetValue(warm.Generation, out int warmPartialMs) ||
                !completeMs.TryGetValue(cold.Generation, out int coldCompleteMs) ||
                !completeMs.TryGetValue(warm.Generation, out int warmCompleteMs))
            {
                throw new InvalidOperationException(
                    "Verdict cache DVT is missing timing evidence.");
            }
            if (warmPartialMs > 1000)
                throw new InvalidOperationException(
                    "Warm verdict display exceeded 1000ms: " + warmPartialMs + "ms.");
            if (warmCompleteMs > coldCompleteMs)
                throw new InvalidOperationException(
                    "Warm verdict completion was slower than cold completion.");

            return string.Format(
                CultureInfo.InvariantCulture,
                "grabs={0} days={1} coldPartial={2}ms coldComplete={3}ms " +
                "warmPartial={4}ms warmComplete={5}ms cacheLoad={6}ms speedup={7:F1}x",
                warm.Requested,
                warm.Days,
                coldPartialMs,
                coldCompleteMs,
                warmPartialMs,
                warmCompleteMs,
                warm.CacheLoadMs,
                warmCompleteMs == 0
                    ? coldCompleteMs
                    : (double)coldCompleteMs / warmCompleteMs);
        }

        public static string VerifyWarmFirst(IReadOnlyList<string> lines)
        {
            const int MaxListMs = 100;
            const int MaxCacheLoadMs = 250;
            const int MaxVerdictMs = 500;

            CacheRun warm = lines
                .Select(line => CachePattern.Match(line))
                .Where(match => match.Success)
                .Select(match => new CacheRun
                {
                    Generation = Parse(match, 1),
                    Hits = Parse(match, 2),
                    Requested = Parse(match, 3),
                    Days = Parse(match, 4),
                    CacheLoadMs = Parse(match, 5)
                })
                .LastOrDefault(run =>
                    run.Requested > 0 && run.Hits == run.Requested);
            if (warm == null)
                throw new InvalidOperationException(
                    "First report read did not fully hit the daily verdict index.");

            Dictionary<int, int> partialMs = ReadTimings(lines, PartialPattern);
            Dictionary<int, int> completeMs = ReadTimings(lines, CompletePattern);
            if (!partialMs.TryGetValue(warm.Generation, out int firstVerdictMs) ||
                !completeMs.TryGetValue(warm.Generation, out int completeVerdictMs))
            {
                throw new InvalidOperationException(
                    "First report read is missing verdict timing evidence.");
            }

            var listPattern = new Regex(
                @"DT list reload range=\S+ rows=(\d+) ms=(\d+) source=index",
                RegexOptions.CultureInvariant);
            Match list = lines
                .Select(line => listPattern.Match(line))
                .Where(match => match.Success && Parse(match, 1) == warm.Requested)
                .OrderByDescending(match => Parse(match, 2))
                .FirstOrDefault();
            if (list == null)
                throw new InvalidOperationException(
                    "First report read is missing list timing evidence.");

            int rows = Parse(list, 1);
            int listMs = Parse(list, 2);
            if (listMs > MaxListMs)
                throw new InvalidOperationException(
                    "First report list exceeded " + MaxListMs + "ms: " + listMs + "ms.");
            if (warm.CacheLoadMs > MaxCacheLoadMs)
                throw new InvalidOperationException(
                    "Daily verdict index load exceeded " + MaxCacheLoadMs + "ms: " +
                    warm.CacheLoadMs + "ms.");
            if (firstVerdictMs > MaxVerdictMs || completeVerdictMs > MaxVerdictMs)
                throw new InvalidOperationException(
                    "First report verdict display exceeded " + MaxVerdictMs + "ms: first=" +
                    firstVerdictMs + "ms complete=" + completeVerdictMs + "ms.");

            return string.Format(
                CultureInfo.InvariantCulture,
                "rows={0} days={1} list={2}ms cacheLoad={3}ms " +
                "firstVerdict={4}ms completeVerdict={5}ms",
                rows,
                warm.Days,
                listMs,
                warm.CacheLoadMs,
                firstVerdictMs,
                completeVerdictMs);
        }

        private static Dictionary<int, int> ReadTimings(
            IEnumerable<string> lines,
            Regex pattern)
        {
            var result = new Dictionary<int, int>();
            foreach (string line in lines)
            {
                Match match = pattern.Match(line);
                if (match.Success)
                    result[Parse(match, 1)] = Parse(match, 2);
            }
            return result;
        }

        private static int Parse(Match match, int group)
        {
            return int.Parse(
                match.Groups[group].Value,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture);
        }

        private sealed class CacheRun
        {
            public int Generation { get; set; }
            public int Hits { get; set; }
            public int Requested { get; set; }
            public int Days { get; set; }
            public int CacheLoadMs { get; set; }
        }
    }
}
