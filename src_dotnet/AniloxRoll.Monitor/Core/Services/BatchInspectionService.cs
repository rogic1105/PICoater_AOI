// PICoater_AOI\src_dotnet\AniloxRoll.Monitor\Core\Services\BatchInspectionService.cs

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// 負責管理多個 PICoaterProcessor 實例與並行運算
    /// </summary>
    public class BatchInspectionService : IDisposable
    {
        private readonly PICoaterProcessor[] _processors;
        private readonly string[] _currentFilePaths;

        public BatchInspectionService(int cameraCount = 7)
        {
            _processors = new PICoaterProcessor[cameraCount];
            _currentFilePaths = new string[cameraCount];

            for (int i = 0; i < cameraCount; i++)
            {
                _processors[i] = new PICoaterProcessor();
            }
        }

        public string GetFilePath(int index)
        {
            if (index < 0 || index >= _currentFilePaths.Length) return null;
            return _currentFilePaths[index];
        }

        /// <summary>
        /// 執行並行讀取或檢測
        /// </summary>
        public (TimedResult<Bitmap>[] results, ConcurrentQueue<string> logs) ProcessBatch(
            Dictionary<int, string> filesMap,
            bool enableProcessing)
        {
            var results = new TimedResult<Bitmap>[_processors.Length];
            var logs = new ConcurrentQueue<string>();

            var pOptions = new ParallelOptions { MaxDegreeOfParallelism = -1 }; // -1 代表不限制

            Parallel.For(0, _processors.Length, pOptions, i =>
            {
                int camId = i + 1;
                string path = filesMap.ContainsKey(camId) ? filesMap[camId] : null;
                _currentFilePaths[i] = path;

                if (!string.IsNullOrEmpty(path))
                {
                    // 根據模式選擇執行完整檢測或是純讀取
                    results[i] = enableProcessing
                        ? _processors[i].ProcessImage(path, 1000)
                        : _processors[i].LoadThumbnailOnly(path, 1000);

                    if (results[i] != null)
                    {
                        logs.Enqueue($"Cam {camId}: IO={results[i].IoDurationMs}ms, GPU={results[i].ComputeDurationMs}ms");
                    }
                }
                else
                {
                    results[i] = null;
                }
            });

            return (results, logs);
        }

        public Bitmap RunInspectionFullRes(int index)
        {
            var path = GetFilePath(index);
            if (string.IsNullOrEmpty(path)) return null;
            return _processors[index].RunInspectionFullRes(path);
        }

        public void Dispose()
        {
            foreach (var p in _processors) p?.Dispose();
        }
    }
}