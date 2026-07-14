using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Services;
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
        private readonly ImageCacheService _imageCache;
        private int _periodNavigationBusyCount = 0;

        public event Action<bool> BusyStateChanged;
        public event Action<string> LogReported;
        public event Action<bool, bool> PeriodNavigationStateChanged;

        public AniloxRollPresenter(
            ImageRepository repo,
            BatchInspectionService service,
            DateTimeNavigator timeMgr,
            ImageCacheService imageCache)
        {
            _repository = repo;
            _inspectionService = service;
            _timeManager = timeMgr;
            _imageCache = imageCache ?? throw new ArgumentNullException(nameof(imageCache));

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

        public Task RunWorkflowAsync(bool enableProcess)
            => RunWorkflowCoreAsync(enableProcess, null);

        public Task RunWorkflowForPeriodAsync(bool enableProcess, DateTime period)
            => RunWorkflowCoreAsync(enableProcess, period);

        private async Task RunWorkflowCoreAsync(bool enableProcess, DateTime? period)
        {
            _imageCache.Clear();
            if (_inspectionService == null) return;

            BusyStateChanged?.Invoke(true);

            try
            {
                Dictionary<int, string> filesMap;
                if (period.HasValue)
                {
                    filesMap = _repository.GetImages(period.Value);
                }
                else
                {
                    _timeManager.SaveCurrentSelection();
                    filesMap = _repository.GetImages(
                        _timeManager.GetCurrentYear(),
                        _timeManager.GetCurrentMonth(),
                        _timeManager.GetCurrentDay(),
                        _timeManager.GetCurrentHour(),
                        _timeManager.GetCurrentMin(),
                        _timeManager.GetCurrentSec());
                }

                var sw = System.Diagnostics.Stopwatch.StartNew();

                var (results, logs) = await Task.Run(() =>
                    _inspectionService.ProcessBatch(filesMap, enableProcess));

                sw.Stop();

                // 2b-ii-B：縮圖顯示由 ImageDisplayView 承接（ThumbnailGridPresenter 已刪）。
                //   ProcessBatch 產出的影像由 ImageCacheService 統一管理生命週期（防洩漏）。
                foreach (var r in results)
                    _imageCache.Track(r?.Data?.Image);

                BusyStateChanged?.Invoke(false);

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
            int idx = FindPeriodIndex(periods, current);

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
            int idx = FindPeriodIndex(periods, current);

            bool canOperate = !GetIsPeriodNavigationBusy();
            PeriodNavigationStateChanged?.Invoke(canOperate && idx > 0, canOperate && idx < periods.Count - 1);
        }

        private static int FindPeriodIndex(IReadOnlyList<DateTime> periods, DateTime current)
        {
            int low = 0;
            int high = periods.Count - 1;
            while (low <= high)
            {
                int middle = low + ((high - low) / 2);
                int comparison = periods[middle].CompareTo(current);
                if (comparison == 0) return middle;
                if (comparison < 0) low = middle + 1;
                else high = middle - 1;
            }

            return Math.Max(0, high);
        }
    }
}
