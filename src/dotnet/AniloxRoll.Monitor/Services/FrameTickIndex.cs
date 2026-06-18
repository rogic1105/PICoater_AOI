using System;
using System.Collections.Generic;
using System.IO;
using AniloxRoll.Monitor.Core.Camera;   // CameraFrameSaver.TickSidecarName

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 多相機回顧合圖的「跨相機幀對齊」唯一來源。
    ///
    /// 背景：每台相機會「各自獨立」掉不同的幀（實測同一段 17s：cam1 掉 8、cam2 掉 5，位置不同），
    /// 所以「第幾張」(seq) 會歪、檔名軟體協調戳也不知道每台「實際掉哪一幀」→ 用它們對齊會補錯位置。
    ///
    /// 唯一可靠的鑰匙 = 硬體 frame-start tick（Data Latch，同板 125MHz 同 epoch）：
    /// 物理同一瞬間拍的兩幀 tick 只差 &lt;0.5ms，而一個幀週期 ≥幾十 ms → 用 tick 就近聚類即可精準對位，
    /// 某台在某時間槽沒有幀 = 它在那裡掉幀 → 該格補黑（StitchCamera 的 null slot）。
    ///
    /// tick 由存檔側車 _ticks.csv（<see cref="CameraFrameSaver.AppendTickSidecar"/>）提供；
    /// 舊資料無側車時 <see cref="BuildAlignedByTick"/> 回 null，呼叫端 fallback <see cref="BuildAlignedByStitchKey"/>。
    /// </summary>
    public static class FrameTickIndex
    {
        /// <summary>讀取一組影像所在資料夾的 _ticks.csv → baseName("{ts}-{camId}") → tick。</summary>
        public static Dictionary<string, long> LoadTickMap(IEnumerable<string> imagePaths)
        {
            var map = new Dictionary<string, long>(StringComparer.Ordinal);
            var dirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var p in imagePaths)
            {
                if (string.IsNullOrEmpty(p)) continue;
                string d = Path.GetDirectoryName(p);
                if (!string.IsNullOrEmpty(d)) dirs.Add(d);
            }
            foreach (var d in dirs)
            {
                string sc = Path.Combine(d, CameraFrameSaver.TickSidecarName);
                if (!File.Exists(sc)) continue;
                try
                {
                    foreach (var line in File.ReadAllLines(sc))
                    {
                        int c = line.LastIndexOf(',');
                        if (c <= 0) continue;
                        if (long.TryParse(line.Substring(c + 1), out long t))
                            map[line.Substring(0, c)] = t;
                    }
                }
                catch { /* 側車讀失敗 → 當沒有，caller fallback */ }
            }
            return map;
        }

        /// <summary>影像路徑 → 側車 key（"{ts}-{camId}"，= 檔名去掉 _raw.jpg 等 suffix）。</summary>
        private static string TickKey(string path)
            => Path.GetFileName(CaptureFileNaming.BaseFromImagePath(path));

        /// <summary>
        /// 用硬體 tick 就近聚類建跨相機對齊時間軸。
        /// 回傳 camId → 對齊後 path 清單（缺幀位置=null，長度=時間槽數，各台一致）。
        /// 任一影像缺 tick（舊資料/latch 失敗）→ 回 null，請 caller fallback 檔名法。
        /// </summary>
        public static Dictionary<int, List<string>> BuildAlignedByTick(
            IDictionary<int, List<string>> grouped, Dictionary<string, long> pathToTick)
        {
            if (grouped == null || pathToTick == null || pathToTick.Count == 0) return null;

            var entries = new List<(int cam, string path, long tick)>();
            foreach (var kv in grouped)
                foreach (var p in kv.Value)
                {
                    if (!pathToTick.TryGetValue(TickKey(p), out long t) || t <= 0) return null; // 缺 tick → fallback
                    entries.Add((kv.Key, p, t));
                }
            if (entries.Count == 0) return null;

            long period = EstimatePeriod(grouped, pathToTick);
            long thr = period > 0 ? period / 2 : long.MaxValue / 4;  // 同槽容差＝半個週期

            entries.Sort((a, b) => a.tick.CompareTo(b.tick));

            // 聚類：相鄰 tick 差 > 容差 → 切新時間槽（同瞬間多相機 tick 很近會落同槽）
            var clusters = new List<List<(int cam, string path, long tick)>>();
            var cur = new List<(int, string, long)> { entries[0] };
            for (int i = 1; i < entries.Count; i++)
            {
                if (entries[i].tick - entries[i - 1].tick > thr)
                {
                    clusters.Add(cur);
                    cur = new List<(int, string, long)>();
                }
                cur.Add(entries[i]);
            }
            clusters.Add(cur);

            var result = new Dictionary<int, List<string>>();
            foreach (var camId in grouped.Keys)
            {
                var lst = new List<string>(clusters.Count);
                foreach (var cl in clusters)
                {
                    string p = null;
                    foreach (var e in cl) if (e.cam == camId) { p = e.path; break; }
                    lst.Add(p);   // null = 此相機在這個時間槽掉幀 → 補黑
                }
                result[camId] = lst;
            }
            return result;
        }

        /// <summary>各相機「相鄰 tick 差」的中位數＝真實幀週期（中位數對掉幀造成的 2×/3× 大 gap 免疫）。</summary>
        private static long EstimatePeriod(IDictionary<int, List<string>> grouped, Dictionary<string, long> pathToTick)
        {
            var diffs = new List<long>();
            foreach (var kv in grouped)
            {
                var ticks = new List<long>();
                foreach (var p in kv.Value)
                    if (pathToTick.TryGetValue(TickKey(p), out long t) && t > 0) ticks.Add(t);
                ticks.Sort();
                for (int i = 1; i < ticks.Count; i++) diffs.Add(ticks[i] - ticks[i - 1]);
            }
            if (diffs.Count == 0) return 0;
            diffs.Sort();
            return diffs[diffs.Count / 2];
        }

        /// <summary>
        /// 舊資料 fallback：用檔名共用時間戳 key（"{ts}-{camId}" 去掉 -camId）建聯集參考軸。
        /// 同次擷取多相機共用軟體協調戳時可對齊；但各台獨立掉幀時不如 tick 精準（會補錯位置）。
        /// </summary>
        public static Dictionary<int, List<string>> BuildAlignedByStitchKey(IDictionary<int, List<string>> grouped)
        {
            var allKeys = new SortedSet<string>(StringComparer.Ordinal);
            var camKeyToPath = new Dictionary<int, Dictionary<string, string>>();
            foreach (var kv in grouped)
            {
                var map = new Dictionary<string, string>();
                foreach (var p in kv.Value) { string k = StitchKey(p); map[k] = p; allKeys.Add(k); }
                camKeyToPath[kv.Key] = map;
            }
            var refKeys = new List<string>(allKeys);

            var result = new Dictionary<int, List<string>>();
            foreach (var kv in grouped)
            {
                var lst = new List<string>(refKeys.Count);
                var map = camKeyToPath[kv.Key];
                foreach (var k in refKeys) lst.Add(map.TryGetValue(k, out var ap) ? ap : null);
                result[kv.Key] = lst;
            }
            return result;
        }

        /// <summary>檔名共用時間戳 key＝"{ts}-{camId}" 去掉 "-{camId}"。</summary>
        private static string StitchKey(string path)
        {
            string baseName = Path.GetFileName(CaptureFileNaming.BaseFromImagePath(path)); // "{ts}-{camId}"
            int dash = baseName.LastIndexOf('-');
            return dash > 0 ? baseName.Substring(0, dash) : baseName;
        }
    }
}
