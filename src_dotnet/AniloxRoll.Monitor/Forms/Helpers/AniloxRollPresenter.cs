using System;
using System.Linq;
using System.Threading.Tasks;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    /// <summary>
    /// [Presenter] 負責協調 UI 流程與業務邏輯
    /// </summary>
    public class AniloxRollPresenter
    {
        // 核心組件依賴
        private readonly ImageRepository _repository;
        private readonly BatchInspectionService _inspectionService;
        private readonly TimeSelectionManager _timeManager;
        private readonly ThumbnailGalleryManager _galleryManager;

        // 用於通知 Form 更新 UI 鎖定狀態 (True = Loading, False = Idle)
        public event Action<bool> BusyStateChanged;

        // 用於通知 Form 顯示 Log
        public event Action<string> LogReported;

        public AniloxRollPresenter(
            ImageRepository repo,
            BatchInspectionService service,
            TimeSelectionManager timeMgr,
            ThumbnailGalleryManager galleryMgr)
        {
            _repository = repo;
            _inspectionService = service;
            _timeManager = timeMgr;
            _galleryManager = galleryMgr;
        }

        /// <summary>
        /// 核心流程：讀取並處理影像
        /// </summary>
        public async Task LoadImagesAsync(bool enableProcess)
        {
            // 1. 通知 UI 進入忙碌狀態
            BusyStateChanged?.Invoke(true);

            try
            {
                // 2. 儲存當前時間設定
                if (!string.IsNullOrEmpty(_timeManager.GetCurrentSec()))
                {
                    _timeManager.SaveCurrentSelection();
                }

                // 3. 準備查詢參數
                var filesMap = _repository.GetImages(
                    _timeManager.GetCurrentYear(),
                    _timeManager.GetCurrentMonth(),
                    _timeManager.GetCurrentDay(),
                    _timeManager.GetCurrentHour(),
                    _timeManager.GetCurrentMin(),
                    _timeManager.GetCurrentSec());

                // 4. 執行批次處理 (Service 層)
                var swTotal = System.Diagnostics.Stopwatch.StartNew();

                var (results, logs) = await Task.Run(() =>
                    _inspectionService.ProcessBatch(filesMap, enableProcess));

                swTotal.Stop();

                // 5. 更新縮圖 (Gallery 層) - 注意：這裡需要 Form 傳入 CacheList 進行管理
                // 為了簡化，我們可以讓 GalleryManager 內部管理 Cache，或透過事件傳遞
                // 這裡我們先假設 Form 會處理 Cache 清理，Presenter 只負責推播結果
                // 但因為要呼叫 UpdateImages，我們需要 Form 提供 cacheCollector
                // 比較好的做法是 Presenter 這裡不直接操作 View 物件，而是觸發事件
                // 但為了不改動太大，我們維持直接操作 Manager
            }
            catch (Exception ex)
            {
                LogReported?.Invoke($"Error: {ex.Message}");
            }
            finally
            {
                // 6. 解除 UI 鎖定
                BusyStateChanged?.Invoke(false);
            }
        }

        // 修正：上面的 LoadImagesAsync 邏輯有一點問題，因為 ProcessBatch 回傳後
        // 我們需要在 UI Thread 更新 UI。所以我們應該把 Update 邏輯放在 await 之後 (仍在 UI Context)

        public async Task RunWorkflowAsync(bool enableProcess, System.Collections.Generic.List<System.Drawing.Image> cacheCollector)
        {
            if (_inspectionService == null) return;

            BusyStateChanged?.Invoke(true);

            try
            {
                // 儲存設定
                _timeManager.SaveCurrentSelection();

                // 取得路徑
                var filesMap = _repository.GetImages(
                    _timeManager.GetCurrentYear(),
                    _timeManager.GetCurrentMonth(),
                    _timeManager.GetCurrentDay(),
                    _timeManager.GetCurrentHour(),
                    _timeManager.GetCurrentMin(),
                    _timeManager.GetCurrentSec());

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // 背景執行
                var (results, logs) = await Task.Run(() =>
                    _inspectionService.ProcessBatch(filesMap, enableProcess));

                sw.Stop();

                // 回到 UI Thread 更新畫面
                _galleryManager.UpdateImages(results, cacheCollector);

                // 自動選取目前選定的那張 (觸發大圖更新)
                _galleryManager.Select(_galleryManager.SelectedIndex, triggerEvent: true);

                // 回報 Log
                string logText = string.Join(Environment.NewLine, System.Linq.Enumerable.OrderBy(logs, x => x));
                LogReported?.Invoke($"總耗時: {sw.ElapsedMilliseconds} ms\n{logText}");
            }
            finally
            {
                BusyStateChanged?.Invoke(false);
            }
        }
    }
}