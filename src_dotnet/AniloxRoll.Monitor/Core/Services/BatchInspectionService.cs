using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using AOI.SDK.Core.Models;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// [Service] 批次檢測服務，負責並行處理邏輯。
    /// </summary>
    public class BatchInspectionService : IDisposable
    {
        private readonly InspectionEngine[] _processors;
        private readonly string[] _currentFilePaths;

        public BatchInspectionService(int cameraCount = 7)
        {
            _processors = new InspectionEngine[cameraCount];
            _currentFilePaths = new string[cameraCount];

            for (int i = 0; i < cameraCount; i++)
            {
                _processors[i] = new InspectionEngine();
            }
        }

        public string GetFilePath(int index)
        {
            if (index < 0 || index >= _currentFilePaths.Length) return null;
            return _currentFilePaths[index];
        }

        public (TimedResult<Bitmap>[] results, ConcurrentQueue<string> logs) ProcessBatch(
            Dictionary<int, string> filesMap,
            bool enableProcessing)
        {
            var results = new TimedResult<Bitmap>[_processors.Length];
            var logs = new ConcurrentQueue<string>();

            // [修正] 強制平行度為相機數量 (7)，避免 ThreadPool 爬升延遲，確保最快速度
            var pOptions = new ParallelOptions { MaxDegreeOfParallelism = _processors.Length };

            Parallel.For(0, _processors.Length, pOptions, i =>
            {
                int camId = i + 1;
                string path = filesMap.ContainsKey(camId) ? filesMap[camId] : null;
                _currentFilePaths[i] = path;

                if (!string.IsNullOrEmpty(path))
                {
                    results[i] = enableProcessing
                        ? _processors[i].ProcessImage(path, 1000)
                        : _processors[i].LoadThumbnailOnly(path, 1000);

                    if (results[i] != null)
                    {
                        logs.Enqueue($"Cam {camId}: IO={results[i].IoDurationMs}ms, GPU={results[i].ComputeDurationMs}ms, BMP={results[i].BitmapDurationMs}ms");

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
            // 這裡會執行檢測運算，回傳大圖
            return _processors[index].RunInspectionFullRes(path);
        }

        public void Dispose()
        {
            foreach (var p in _processors) p?.Dispose();
        }
    }
}