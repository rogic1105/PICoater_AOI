using System;
using System.Collections.Concurrent;

namespace AniloxRoll.Monitor.Core.Camera
{
    /// <summary>
    /// 同步相同 Line Rate 的相機存檔時間戳。
    /// 同一 rate group 中，第一台 callback 觸發時建立時間戳，
    /// 後續同組相機在 tolerance 內到達時共用同一時間戳。
    /// 不同 rate 的相機各自獨立。
    /// </summary>
    public class CaptureTimestampCoordinator
    {
        private class RateGroup
        {
            public DateTime Timestamp;
            public readonly object Lock = new object();
        }

        private readonly ConcurrentDictionary<int, RateGroup> _groups
            = new ConcurrentDictionary<int, RateGroup>();

        /// <summary>同組容許時間差（ms）。預設 100ms，足以涵蓋 MIL 多台 callback 的排程延遲。</summary>
        private const double DefaultToleranceMs = 100.0;

        /// <summary>
        /// 取得此次存檔應使用的時間戳。
        /// 同一 lineRateHz 的相機，若在前一台觸發後 toleranceMs 內到達，共用同一時間戳。
        /// </summary>
        public DateTime Coordinate(int lineRateHz, DateTime now)
        {
            var group = _groups.GetOrAdd(lineRateHz, _ => new RateGroup());
            lock (group.Lock)
            {
                double diff = (now - group.Timestamp).TotalMilliseconds;
                if (diff >= 0 && diff <= DefaultToleranceMs)
                    return group.Timestamp;

                group.Timestamp = now;
                return now;
            }
        }
    }
}
