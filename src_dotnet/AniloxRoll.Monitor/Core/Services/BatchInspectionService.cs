using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using AOI.SDK.Core.Models;
using AniloxRoll.Monitor.Core.Data; // 引用 InspectionData

namespace AniloxRoll.Monitor.Core.Services
{
    public class BatchInspectionService : IDisposable
    {
        private readonly InspectionEngine[] _processors;
        private readonly string[] _currentFilePaths;

        public BatchInspectionService(int cameraCount = 7)
        {
            _processors = new InspectionEngine[cameraCount];
            _currentFilePaths = new string[cameraCount];
            for (int i = 0; i < cameraCount; i++) _processors[i] = new InspectionEngine();
        }

        public string GetFilePath(int index)
        {
            if (index < 0 || index >= _currentFilePaths.Length) return null;
            return _currentFilePaths[index];
        }

        // [修正] 回傳型別改為 InspectionData
        public (TimedResult<InspectionData>[] results, ConcurrentQueue<string> logs) ProcessBatch(
            Dictionary<int, string> filesMap,
            bool enableProcessing)
        {
            // [修正] 陣列型別改為 InspectionData
            var results = new TimedResult<InspectionData>[_processors.Length];
            var logs = new ConcurrentQueue<string>();

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
            return _processors[index].RunInspectionFullRes(path);
        }

        public void Dispose()
        {
            foreach (var p in _processors) p?.Dispose();
        }
    }
}