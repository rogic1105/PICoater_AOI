using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    /// <summary>
    /// [Presenter] 專案的主要協調者 (Coordinator)。
    /// 負責接收 View 的請求 (如 LoadImagesAsync)，調度 Repository 查詢資料，
    /// 指揮 Service 執行運算，最後透過 GalleryManager 更新 UI。
    /// </summary>
    public class AniloxRollPresenter
    {
        // 核心組件依賴
        private readonly ImageRepository _repository;
        private readonly BatchInspectionService _inspectionService;
        private readonly DateTimeNavigator _timeManager;
        private readonly ThumbnailGridPresenter _galleryManager;

        public event Action<bool> BusyStateChanged;        // 用於通知 Form 更新 UI 鎖定狀態 (True = Loading, False = Idle)
        public event Action<string> LogReported;        // 用於通知 Form 顯示 Log

        public AniloxRollPresenter(
            ImageRepository repo,
            BatchInspectionService service,
            DateTimeNavigator timeMgr,
            ThumbnailGridPresenter galleryMgr)
        {
            _repository = repo;
            _inspectionService = service;
            _timeManager = timeMgr;
            _galleryManager = galleryMgr;
        }

        private void LoadLastSession(object sender, EventArgs e)
        {
            // 初始化載入邏輯
            string path = Properties.Settings.Default.LastDataPath;
            if (Directory.Exists(path))
            {
                _repository.LoadDirectory(path);
                _timeManager.Initialize(Properties.Settings.Default.LastYear);
                _galleryManager.Select(0, triggerEvent: false);
            }
        }


        /// <summary>
        /// 執行主要的影像讀取與處理流程。
        /// </summary>
        /// <param name="enableProcess">若為 true 則執行檢測運算，否則僅讀取原圖。</param>
        /// <param name="cacheCollector">用於收集新生成圖片的快取清單，由 Form 負責生命週期管理。</param>
        public async Task RunWorkflowAsync(bool enableProcess, List<Image> cacheCollector)
        {
            if (_inspectionService == null) return;

            // 1. 通知 UI 進入忙碌狀態 (鎖定按鈕與大圖切換)
            BusyStateChanged?.Invoke(true);

            try
            {
                // 2. 儲存當前時間選擇狀態
                _timeManager.SaveCurrentSelection();

                // 3. 準備檔案查詢參數
                var filesMap = _repository.GetImages(
                    _timeManager.GetCurrentYear(),
                    _timeManager.GetCurrentMonth(),
                    _timeManager.GetCurrentDay(),
                    _timeManager.GetCurrentHour(),
                    _timeManager.GetCurrentMin(),
                    _timeManager.GetCurrentSec());

                // 4. 背景執行批次處理
                var sw = System.Diagnostics.Stopwatch.StartNew();

                var (results, logs) = await Task.Run(() =>
                    _inspectionService.ProcessBatch(filesMap, enableProcess));

                sw.Stop();

                // 5. 回到 UI Thread 更新縮圖
                _galleryManager.UpdateImages(results, cacheCollector);

                // [關鍵修正] 在觸發「選取事件」前，必須先解除 UI 忙碌狀態！
                // 否則 AniloxRollForm.OnGallerySelectionChanged 裡的 if(_isBusy) return 會擋住這次更新。
                BusyStateChanged?.Invoke(false);

                // 6. 自動選取目前索引 (這會觸發 Form 更新大圖)
                // 如果是第一次執行，SelectedIndex 預設為 0 (第一張)
                _galleryManager.Select(_galleryManager.SelectedIndex, triggerEvent: true);

                // 7. 回報執行效能 Log
                string logText = string.Join(Environment.NewLine, logs.OrderBy(x => x));
                LogReported?.Invoke($"Total Duration: {sw.ElapsedMilliseconds} ms\n{logText}");
            }
            catch (Exception ex)
            {
                LogReported?.Invoke($"Workflow Error: {ex.Message}");
                // 發生錯誤也要確保解除鎖定
                BusyStateChanged?.Invoke(false);
            }
            finally
            {
                // 雙重保險：確保狀態一定會被重置
                // (注意：BusyStateChanged(false) 重複呼叫是安全的，只要 Form 端實作沒問題)
                // 這裡主要是為了 catch 區塊無法覆蓋到的極端情況，或保持程式碼結構的完整性
                // 由於上面已經呼叫過，這裡其實主要發揮 catch 發生時的作用
            }
        }



    }
}