using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.UI.Presenters
{
    public class AniloxRollPresenter
    {
        private readonly ImageRepository _repository;
        private readonly BatchInspectionService _inspectionService;
        private readonly DateTimeNavigator _timeManager;
        private readonly ThumbnailGridPresenter _galleryManager;
        private int _periodNavigationBusyCount = 0;

        public event Action<bool> BusyStateChanged;
        public event Action<string> LogReported;
        public event Action<bool, bool> PeriodNavigationStateChanged;

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
                try { _inspectionService?.WarmUp(); }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[AniloxRollPresenter] WarmUp failed: {ex.GetType().Name}: {ex.Message}");
                }
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

                // 更新縮圖選取框（不觸發事件，由呼叫端根據 StitchMode 決定顯示方式）
                _galleryManager.Select(_galleryManager.SelectedIndex, triggerEvent: false);

                string logText = string.Join(Environment.NewLine, logs.OrderBy(x => x));
                LogReported?.Invoke($"Total Duration: {sw.ElapsedMilliseconds} ms\n{logText}");
            }
            catch (Exception ex)
            {
                LogReported?.Invoke($"Workflow Error: {ex.Message}");
                BusyStateChanged?.Invoke(false);
            }
        }

        public async Task LoadImagesWithPeriodLockAsync(bool isProcessedMode, Func<bool, Task> loadImagesAsync)
        {
            BeginPeriodNavigationBusy();
            try
            {
                await loadImagesAsync(isProcessedMode);
            }
            finally
            {
                EndPeriodNavigationBusy();
            }
        }

        public async Task MovePeriodAsync(int step, bool isProcessedMode, Func<bool, Task> loadImagesAsync)
        {
            if (GetIsPeriodNavigationBusy()) return;

            var periods = _repository.GetAvailablePeriods();
            if (periods.Count == 0)
            {
                UpdatePeriodNavigationState();
                return;
            }

            DateTime current = _timeManager.GetCurrentPeriodOrDefault(periods[0]);
            int idx = periods.FindIndex(x => x == current);
            if (idx < 0)
            {
                idx = periods.FindLastIndex(x => x <= current);
                if (idx < 0) idx = 0;
            }

            int target = Math.Max(0, Math.Min(periods.Count - 1, idx + step));
            if (target == idx)
            {
                UpdatePeriodNavigationState();
                return;
            }

            _timeManager.SetPeriodToCombo(periods[target]);
            await LoadImagesWithPeriodLockAsync(isProcessedMode, loadImagesAsync);
        }

        public void BeginPeriodNavigationBusy()
        {
            Interlocked.Increment(ref _periodNavigationBusyCount);
            UpdatePeriodNavigationState();
        }

        public void EndPeriodNavigationBusy()
        {
            int next = Interlocked.Decrement(ref _periodNavigationBusyCount);
            if (next < 0)
            {
                Interlocked.Exchange(ref _periodNavigationBusyCount, 0);
            }

            UpdatePeriodNavigationState();
        }

        private bool GetIsPeriodNavigationBusy()
            => Interlocked.CompareExchange(ref _periodNavigationBusyCount, 0, 0) > 0;

        public void UpdatePeriodNavigationState()
        {
            var periods = _repository.GetAvailablePeriods();
            if (periods.Count == 0)
            {
                PeriodNavigationStateChanged?.Invoke(false, false);
                return;
            }

            DateTime current = _timeManager.GetCurrentPeriodOrDefault(periods[0]);
            int idx = periods.FindIndex(x => x == current);
            if (idx < 0)
            {
                idx = periods.FindLastIndex(x => x <= current);
                if (idx < 0) idx = 0;
            }

            bool canOperate = !GetIsPeriodNavigationBusy();
            PeriodNavigationStateChanged?.Invoke(canOperate && idx > 0, canOperate && idx < periods.Count - 1);
        }
    }
}
