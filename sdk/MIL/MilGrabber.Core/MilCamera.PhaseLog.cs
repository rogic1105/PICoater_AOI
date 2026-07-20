using System;
using System.IO;
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    // MilCamera 的「多相機相位量測」分區（診斷用）。
    //
    // 機制＝Radient eV-CL Data Latch（板專屬硬體時戳，零 hook dispatch jitter；參考 BoardSpecific/DataLatch 範例）：
    //   在 M_GRAB_FRAME_START 訊號點由硬體 latch 一個 M_TIME_STAMP（板載時鐘 ticks）→ hook 內 MdigGetHookInfo 讀。
    //   走板載時鐘 → 同一 System(板) 的相機 ticks 可直接相減＝相位差；跨板(System0↔1)時鐘不同、不可直接比。
    //   取代失敗的 MbufInquire(M_GRAB_TIME_STAMP_NS)（上機回 0）。
    public partial class MilCamera
    {
        private long _dataLatchClockFreqHz;   // ticks→秒：ticks / freq
        private long _lastFrameStartTicks;    // 最後一幀 frame-start 的硬體時戳（ticks）
        private long _phaseSeq;               // 本相機 hook 幀序（跨相機對齊用：tick 就近配對）

        /// <summary>最後一幀 frame-start 硬體時戳（Data Latch ticks）；0＝尚未取得 / latch 未啟用。</summary>
        public long LastFrameStartTicks => _lastFrameStartTicks;
        /// <summary>Data Latch 時鐘頻率（Hz）；ticks / 此值 = 秒。</summary>
        public long DataLatchClockFreqHz => _dataLatchClockFreqHz;

        /// <summary>相位 log 檔路徑（static，多相機共寫）。設了就每幀 append；null＝不記。診斷用。
        /// 欄位：wallclock,cam,seq,ticks,freqHz（離線：tick 就近配對相減 /freq = 相位秒，×線掃率 = 差幾條線）。</summary>
        public static string PhaseLogPath;
        private static readonly object _phaseLogLock = new object();

        /// <summary>啟用 frame-start 硬體時戳 latch（LATCH0）。在 Initialize（digitizer alloc 後）呼叫。</summary>
        public void EnableFrameStartTimestampLatch(long latchIndex = MIL.M_LATCH0)
        {
            if (_milDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControl(_milDigitizer, MIL.M_DATA_LATCH_TRIGGER_SOURCE + latchIndex, MIL.M_GRAB_FRAME_START);
                MIL.MdigControl(_milDigitizer, MIL.M_DATA_LATCH_TYPE  + latchIndex, MIL.M_TIME_STAMP);
                MIL.MdigControl(_milDigitizer, MIL.M_DATA_LATCH_STATE + latchIndex, MIL.M_ENABLE);
                MIL_INT freq = 0;
                MIL.MdigInquire(_milDigitizer, MIL.M_DATA_LATCH_CLOCK_FREQUENCY, ref freq);
                _dataLatchClockFreqHz = (long)freq;
                System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] frame-start 時戳 latch 啟用（clock {_dataLatchClockFreqHz} Hz）");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] EnableFrameStartTimestampLatch 失敗：{ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>grab hook 內呼叫：排乾 frame-start latch FIFO（每個 latched 時戳記一列）+ 記 phase log。
        /// 排乾＝每幀都拿到乾淨硬體時戳、不重用、不殘留。對齊用「tick 值就近配對」（兩台共用板載時鐘 epoch）。</summary>
        internal void CaptureFrameStartLatch(MIL_ID eventId, long latchIndex = MIL.M_LATCH0)
        {
            // The first hot-standby frame reads the latch for readiness, but idle frames must not
            // grow the diagnostic phase log indefinitely.
            string path = _userWantsGrab ? PhaseLogPath : null;
            try
            {
                MIL_INT cnt = 0;
                MIL.MdigGetHookInfo(eventId, MIL.M_DATA_LATCH_VALUE_COUNT + latchIndex, ref cnt);
                for (MIL_INT i = 0; i < cnt; i++)
                {
                    MIL_INT v = 0;
                    MIL.MdigGetHookInfo(eventId, MIL.M_DATA_LATCH_VALUE + latchIndex, ref v);   // pop 最舊一筆
                    long ticks = (long)v;
                    _lastFrameStartTicks = ticks;
                    long seq = ++_phaseSeq;
                    if (path != null)
                    {
                        string line = $"{DateTime.Now:HH:mm:ss.fff},{CameraId},{seq},{ticks},{_dataLatchClockFreqHz}\r\n";
                        lock (_phaseLogLock) { File.AppendAllText(path, line); }
                    }
                }
            }
            catch { }
        }
    }
}
