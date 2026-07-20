using System;
using System.Threading.Tasks;
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    // MilCamera 的「CLProtocol（GenICam Camera Link）」分區：背景啟用 + 套用曝光/線掃。
    // CLProtocol 狀態欄位（_clProtocolEnabled / _clProtocolInitDone…）留主檔，多 partial 共用；本檔只放方法。
    public partial class MilCamera
    {
        // ==================== CLProtocol ====================

        /// <summary>背景啟用 CLProtocol（2-5s）+ 套用曝光/線掃。應在「相機分配完成後、第一次 grab 之前」呼叫
        /// （不與 MbufAlloc/MdispAlloc 競爭 MIL 內部鎖，也不在 grab 期間執行）。idempotent（_clProtocolInitStarted 守門）。
        /// 實際工作完成（成功或失敗）前 <see cref="IsHwParamsStable"/>=false，供上層把「開始抓取」鈕維持灰色。
        /// 逾時只記錄診斷，不得把仍在執行的參數寫入誤報為已完成。</summary>
        public void BeginCLProtocolInit()
        {
            if (_clProtocolInitStarted) return;
            _clProtocolInitStarted = true;

            var initTask = Task.Run((Action)TryEnableCLProtocol);
            var timeoutTask = Task.Delay(TimeSpan.FromSeconds(10));

            Task.WhenAny(initTask, timeoutTask).ContinueWith(_ =>
            {
                if (!initTask.IsCompleted)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] CLProtocol 初始化超過 10s，工作仍在執行；參數就緒 gate 保持關閉。");
                }
            });
        }

        /// <summary>斷線重連後重跑 CLProtocol（啟動時相機不在線、之後才連上的情境 → 重新啟用 CLProtocol 才讀得到
        /// 曝光/線掃參數）。僅在「已知在線（<see cref="IsConnected"/>）+ 尚未啟用 + 非 grab 中 + 上次 init 已結束」時
        /// 背景重試一次。判準理由：①不在線就 enable 會卡 MIL 鎖；②grab 中 enable+重套線掃會掉幀；③in-flight 防重複。</summary>
        public void RetryCLProtocolOnReconnect()
        {
            if (_isReleased || _milDigitizer == MIL.M_NULL) return;
            if (_clProtocolEnabled) return;                              // 已啟用 → 不必重試
            if (!IsConnected) return;                                    // 不在線 → 不對斷線相機 enable（防卡鎖）
            if (IsLive) return;                                          // grab 中不重套 CLProtocol（會掉幀）
            if (_clProtocolInitStarted && !_clProtocolInitDone) return;  // 上次 init 還在跑 → 等
            _clProtocolInitStarted = false;                             // 重置守門 → 允許 BeginCLProtocolInit 重跑
            _clProtocolInitDone = false;
            BeginCLProtocolInit();
        }

        private void TryEnableCLProtocol()
        {
            if (_milDigitizer == MIL.M_NULL)
            {
                _clProtocolInitDone = true;
                return;
            }

            lock (_clProtocolInitLock)
            {
                if (_isReleased || _clProtocolEnabled)
                {
                    _clProtocolInitDone = true;
                    return;
                }
                try
                {
                    MIL_INT numDevIds = 0;
                    MIL.MdigInquire(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID_NUM, ref numDevIds);
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: {numDevIds} device ID(s) available.");

                    if (numDevIds > 0)
                    {
                        var devId = new System.Text.StringBuilder(512);
                        MIL.MdigInquire(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, devId);
                        string selectedId = devId.ToString();
                        System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: using device ID = \"{selectedId}\"");
                        MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, selectedId);
                    }
                    else
                    {
                        System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol: no enumerated IDs, falling back to M_DEFAULT.");
                        MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
                    }

                    MIL.MdigControl(_milDigitizer, MIL.M_GC_CLPROTOCOL, MIL.M_ENABLE);
                    _clProtocolEnabled = true;
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol enabled successfully.");

                    if (!_isReleased)
                    {
                        if (_appliedExposureUs > 0) SetExposureUs(_appliedExposureUs);
                        if (_appliedLineRateHz > 0) SetLineRateHz(_appliedLineRateHz);
                    }
                }
                catch (Exception ex)
                {
                    _clProtocolEnabled = false;
                    System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] CLProtocol init failed: {ex.GetType().Name}: {ex.Message}");
                }
                finally
                {
                    // This is the only normal completion edge. A diagnostic timeout must never
                    // publish stable while CLProtocol or its parameter writes are still running.
                    _clProtocolInitDone = true;
                }
            }
        }
    }
}
