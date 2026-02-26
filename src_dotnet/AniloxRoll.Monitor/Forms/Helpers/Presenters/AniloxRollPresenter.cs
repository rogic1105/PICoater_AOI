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
    public class AniloxRollPresenter
    {
        private readonly ImageRepository _repository;
        private readonly BatchInspectionService _inspectionService;
        private readonly DateTimeNavigator _timeManager;
        private readonly ThumbnailGridPresenter _galleryManager;

        public event Action<bool> BusyStateChanged;
        public event Action<string> LogReported;

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

            // 啟動時 WarmUp
            Task.Run(() =>
            {
                try { _inspectionService.WarmUp(); }
                catch { /* ignore */ }
            });
        }

        public async Task RunWorkflowAsync(bool enableProcess, List<Image> cacheCollector)
        {
            if (_inspectionService == null) return;

            BusyStateChanged?.Invoke(true);

            try
            {
                _timeManager.SaveCurrentSelection();

                var filesMap = _repository.GetImages(
                    _timeManager.GetCurrentYear(),
                    _timeManager.GetCurrentMonth(),
                    _timeManager.GetCurrentDay(),
                    _timeManager.GetCurrentHour(),
                    _timeManager.GetCurrentMin(),
                    _timeManager.GetCurrentSec());

                var sw = System.Diagnostics.Stopwatch.StartNew();

                var (results, logs) = await Task.Run(() =>
                    _inspectionService.ProcessBatch(filesMap, enableProcess));

                sw.Stop();

                _galleryManager.UpdateImages(results, cacheCollector);

                BusyStateChanged?.Invoke(false);

                // 自動選取第一張，觸發大圖檢視
                _galleryManager.Select(_galleryManager.SelectedIndex, triggerEvent: true);

                string logText = string.Join(Environment.NewLine, logs.OrderBy(x => x));
                LogReported?.Invoke($"Total Duration: {sw.ElapsedMilliseconds} ms\n{logText}");
            }
            catch (Exception ex)
            {
                LogReported?.Invoke($"Workflow Error: {ex.Message}");
                BusyStateChanged?.Invoke(false);
            }
        }
    }
}