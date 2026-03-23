using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Threading.Tasks;
using AOI.SDK.Core.Models;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Acquisition.Inspection;

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

            if (enableProcessing)
            {
                for (int i = 0; i < _cameraCount; i++)
                {
                    ProcessSingleCamera(i, filesMap, results, logs, enableProcessing: true);
                }
            }
            else
            {
                // 純 IO 縮圖讀取路徑並行化，縮短多相機總讀取時間。
                Parallel.For(0, _cameraCount, i =>
                {
                    ProcessSingleCamera(i, filesMap, results, logs, enableProcessing: false);
                });
            }

            return (results, logs);
        }


        private void ProcessSingleCamera(
            int index,
            Dictionary<int, string> filesMap,
            TimedResult<InspectionData>[] results,
            ConcurrentQueue<string> logs,
            bool enableProcessing)
        {
            int camId = index + 1;
            filesMap.TryGetValue(camId, out string path);
            _currentFilePaths[index] = path;

            if (string.IsNullOrEmpty(path))
            {
                results[index] = null;
                return;
            }

            // 根據 enableProcessing 呼叫不同方法
            // ProcessImage -> 會執行翻轉 (變正常)
            // LoadThumbnailOnly -> 不執行翻轉 (保持顛倒)
            results[index] = enableProcessing
                ? _sharedProcessor.ProcessImage(path, 1000, _hessianMaxFactor)
                : _sharedProcessor.LoadThumbnailOnly(path, 1000);

            if (results[index] != null)
            {
                logs.Enqueue($"Cam {camId}: IO={results[index].IoDurationMs}ms, GPU={results[index].ComputeDurationMs}ms, BMP={results[index].BitmapDurationMs}ms");
            }
        }

        // 回傳 InspectionData 包含圖片與曲線 
        public InspectionData RunInspectionFullRes(int index)
        {
            var path = GetFilePath(index);
            if (string.IsNullOrEmpty(path)) return null;

            return _sharedProcessor.RunInspectionFullRes(path, _isProcessedMode, _hessianMaxFactor); // <--- 確認這裡
        }

        /// <summary>
        /// BMP 拼接專用：CoreCV_FastReadBMP + GPU resize 縮 scale 倍，回傳 Bitmap。
        /// 比 GDI+ new Bitmap(path) 快約 10x。
        /// </summary>
        public Bitmap LoadBmpAtScale(string path, int scale) =>
            _sharedProcessor.LoadBitmapAtScale(path, scale);

        public Bitmap ProcessBmpAtScale(string path, int scale,
            out float[] curveMean, out float[] curveMax) =>
            _sharedProcessor.ProcessBmpAtScale(path, scale, _hessianMaxFactor,
                out curveMean, out curveMax);

        public void Dispose()
        {
            _sharedProcessor?.Dispose();
        }
    }
}