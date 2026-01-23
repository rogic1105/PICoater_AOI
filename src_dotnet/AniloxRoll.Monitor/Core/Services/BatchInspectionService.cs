using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using AOI.SDK.Core.Models;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    public class BatchInspectionService : IDisposable
    {
        private readonly InspectionEngine _sharedProcessor;
        private readonly string[] _currentFilePaths;
        private readonly int _cameraCount;
        private float _hessianMaxFactor = 5.0f;
        private float _errorMean = 1.0f;
        private float _errorMax = 2.0f;

        public void UpdateAlgorithmParams(float hessianFactor, float errMean, float errMax)
        {
            _hessianMaxFactor = hessianFactor;
            _errorMean = errMean;
            _errorMax = errMax;
        }

        // [新增] 記錄當前檢視模式 (true=Processed/Normal, false=Original/UpsideDown)
        private bool _isProcessedMode = false;

        public BatchInspectionService(int cameraCount = 7)
        {
            _cameraCount = cameraCount;
            _sharedProcessor = new InspectionEngine();
            _currentFilePaths = new string[cameraCount];
        }

        public void WarmUp() => _sharedProcessor.WarmUp();

        public string GetFilePath(int index)
        {
            if (index < 0 || index >= _currentFilePaths.Length) return null;
            return _currentFilePaths[index];
        }

        public (TimedResult<InspectionData>[] results, ConcurrentQueue<string> logs) ProcessBatch(
            Dictionary<int, string> filesMap,
            bool enableProcessing)
        {
            // [更新狀態] 記錄這次 Batch 是哪種模式
            _isProcessedMode = enableProcessing;

            var results = new TimedResult<InspectionData>[_cameraCount];
            var logs = new ConcurrentQueue<string>();

            for (int i = 0; i < _cameraCount; i++)
            {
                int camId = i + 1;
                string path = filesMap.ContainsKey(camId) ? filesMap[camId] : null;
                _currentFilePaths[i] = path;

                if (!string.IsNullOrEmpty(path))
                {
                    // 根據 enableProcessing 呼叫不同方法
                    // ProcessImage -> 會執行翻轉 (變正常)
                    // LoadThumbnailOnly -> 不執行翻轉 (保持顛倒)
                    results[i] = enableProcessing
                        ? _sharedProcessor.ProcessImage(path, 1000, _hessianMaxFactor)
                        : _sharedProcessor.LoadThumbnailOnly(path, 1000);

                    if (results[i] != null)
                    {
                        logs.Enqueue($"Cam {camId}: IO={results[i].IoDurationMs}ms, GPU={results[i].ComputeDurationMs}ms, BMP={results[i].BitmapDurationMs}ms");
                    }
                }
                else
                {
                    results[i] = null;
                }
            }

            return (results, logs);
        }

        // 回傳 InspectionData 包含圖片與曲線 
        public InspectionData RunInspectionFullRes(int index)
        {
            var path = GetFilePath(index);
            if (string.IsNullOrEmpty(path)) return null;

            return _sharedProcessor.RunInspectionFullRes(path, _isProcessedMode, _hessianMaxFactor); // <--- 確認這裡
        }

        public void Dispose()
        {
            _sharedProcessor?.Dispose();
        }
    }
}