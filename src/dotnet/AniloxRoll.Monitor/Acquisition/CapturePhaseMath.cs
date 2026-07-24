using System.Collections.Generic;
using System.Linq;

namespace AniloxRoll.Monitor.Core.Camera
{
    internal static class CapturePhaseMath
    {
        public static bool TryGetCircularSpreadTicks(
            IEnumerable<long> frameTicks,
            long periodTicks,
            out long spreadTicks)
        {
            spreadTicks = 0;
            if (frameTicks == null || periodTicks <= 0)
                return false;

            long[] positions = frameTicks
                .Select(tick => PositiveModulo(tick, periodTicks))
                .OrderBy(tick => tick)
                .ToArray();
            if (positions.Length == 0)
                return false;
            if (positions.Length == 1)
                return true;

            long largestGap =
                periodTicks - positions[positions.Length - 1] + positions[0];
            for (int i = 1; i < positions.Length; i++)
            {
                long gap = positions[i] - positions[i - 1];
                if (gap > largestGap)
                    largestGap = gap;
            }

            spreadTicks = periodTicks - largestGap;
            return true;
        }

        private static long PositiveModulo(long value, long divisor)
        {
            long result = value % divisor;
            return result < 0 ? result + divisor : result;
        }
    }
}
