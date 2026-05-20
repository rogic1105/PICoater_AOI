using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.Monitor.Core.Services
{
    /// <summary>
    /// Storage PC 專用：每 10 秒自主檢查磁碟空間並觸發清理；
    /// 同時輪詢 cleanup-request.flag，偵測到旗標時立即觸發（不等下一個週期）。
    /// 啟動時立即執行一次，處理程式重啟前遺留的旗標與空間不足狀況。
    /// </summary>
    public sealed class CleanupFlagWatcher : IDisposable
    {
        private const string FlagFileName = "cleanup-request.flag";

        private readonly Func<string> _getWatchFolder;
        private readonly StorageRetentionService _retentionService;
        private readonly int _pollIntervalMs;

        private Thread _thread;
        private volatile bool _stopping;

        public CleanupFlagWatcher(
            Func<string> getWatchFolder,
            StorageRetentionService retentionService,
            int pollIntervalMs = 10_000)
        {
            _getWatchFolder   = getWatchFolder;
            _retentionService = retentionService;
            _pollIntervalMs   = pollIntervalMs;
        }

        public void Start()
        {
            // 啟動時立即執行一次（處理殘留旗標 + 空間不足）
            Task.Run(() => Tick());

            _stopping = false;
            _thread = new Thread(PollLoop) { IsBackground = true, Name = "CleanupFlagWatcher" };
            _thread.Start();
        }

        public void Stop()
        {
            _stopping = true;
            _thread?.Interrupt();
        }

        private void PollLoop()
        {
            while (!_stopping)
            {
                try { Thread.Sleep(_pollIntervalMs); }
                catch (ThreadInterruptedException) { break; }

                if (_stopping) break;
                Tick();
            }
        }

        /// <summary>每個週期執行：先處理旗標，再無條件呼叫 RunCleanup（空間夠時內部即返回）。</summary>
        private void Tick()
        {
            try
            {
                ProcessFlagIfExists();
                _retentionService?.RunCleanup();
            }
            catch (Exception ex)
            {
                Trace.TraceWarning($"[CleanupFlagWatcher] {ex.GetType().Name}: {ex.Message}");
            }
        }

        private void ProcessFlagIfExists()
        {
            string folder = _getWatchFolder?.Invoke();
            if (string.IsNullOrWhiteSpace(folder)) return;

            string flagPath = Path.Combine(folder, FlagFileName);
            if (!File.Exists(flagPath)) return;

            Trace.TraceInformation("[CleanupFlagWatcher] 收到清理旗標");
            try { File.Delete(flagPath); } catch { /* best effort */ }
        }

        public void Dispose() => Stop();
    }
}
