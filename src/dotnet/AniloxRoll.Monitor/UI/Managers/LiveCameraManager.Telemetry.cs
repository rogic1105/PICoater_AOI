using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;
using TanukiCv.Core; // PixelMmMapper（已收進 sdk 唯一來源）
using TanukiCv.Controls; // ImageDisplayView（共用多相機監控顯示元件）
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.Core.Interop; // NativeMethods（LOD GPU resize；P/Invoke 宣告唯一點）
    // partial：相機狀態 timer / hw-ready；游標資訊由 ImageCanvas overlay 自行顯示。

namespace AniloxRoll.Monitor.UI.Managers
{
    public partial class LiveCameraManager
    {
        // Stall 偵測職責已提取到 CameraStallDetector（純邏輯、可單獨測；判據＝M_PROCESS_FRAME_COUNT 前進、非 FPS 門檻）。
        private readonly CameraStallDetector _stallDetector = new CameraStallDetector();
        private readonly Dictionary<int, bool> _lastPresence = new Dictionary<int, bool>();    // 上次 tick 在線狀態（斷線→連線邊緣偵測 → 重跑 CLProtocol）
        private readonly HashSet<int> _parameterReadyLogged = new HashSet<int>();
        private readonly HashSet<int> _warmReadyLogged = new HashSet<int>();
        private bool _wasHwReady;                                                              // 上次 tick AreCamerasHwReady（偵 false→true 轉變 → 強制刷 count label/解鎖鈕）

        // ==================== Status Timer ====================

        /// <summary>
        /// 每 500ms 輪詢相機連線狀態並自動重啟抓圖，同 CameraSession.UpdatePresence()。
        /// IsReleasing = true 時提早返回，防止存取已釋放的相機資源。
        /// 使用快照（ToArray）避免 background FreeCameras 呼叫 _cameras.Clear() 時導致 InvalidOperationException。
        /// </summary>
        /// <summary>狀態 tick 單飛旗標（背景 MIL 讀取進行中不再疊發，防堆積）。</summary>
        private volatile bool _statusTickInFlight;
        /// <summary>UI 執行緒 SynchronizationContext（LCM 在 UI 執行緒建構時捕捉；背景結果 Post 回 UI 套用）。</summary>
        private readonly System.Threading.SynchronizationContext _uiSync = System.Threading.SynchronizationContext.Current;

        private void CameraStatusTimer_Tick(object sender, EventArgs e)
        {
            // MIL 讀取（CheckPresence/GetFrameCount＝MdigInquire）全部移背景執行緒（2026-07-07 [UiStack]
            // 定罪：空通道 MdigInquire 特別慢 → 每 500ms 卡 UI 63~109ms＝拖曳節奏性跳框主因）。
            // UI 只做旗標 gate + 套結果（label/事件），與 telemetry 16 欄背景化同一模式。
            if (IsReleasing || _statusTickInFlight) return;

            // CLProtocol 背景初始化期間（分配後 ~2-10s）不碰 MIL（與背景 enable 搶 MIL 內部鎖會凍 UI）。
            // AreCamerasHwReady 只讀旗標。gate-off 恢復時 hwJustReady=true → 強制刷 count label + 重發 OnHwReady。
            if (!AreCameraParametersReady)
            {
                _parameterReadyLogged.Clear();
                _wasHwReady = false;
                _hwReadyRaised = false;
                return;
            }
            bool hwReady = AreCamerasHwReady;
            bool hwJustReady = hwReady && !_wasHwReady;
            if (!hwReady) _hwReadyRaised = false;
            _wasHwReady = hwReady;

            // 快照：防 ReleaseAsync 清 _cameras 時 foreach 炸 InvalidOperationException
            AniloxCamera[] snapshot;
            try { snapshot = _cameras.ToArray(); }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.CameraStatusTimer] {ex.GetType().Name}: {ex.Message}"); return; }

            _statusTickInFlight = true;
            Task.Run(async () =>
            {
                try
                {
                    var statuses = new List<(int camId, string text, Color color, bool clearFrame)>();
                    foreach (var cam in snapshot)
                    {
                        if (IsReleasing) return; // 釋放流程已開始，立即中止（finally 仍會清旗標）

                        bool isConnected = cam.CheckPresence();

                        // 斷線→連線「邊緣」自動重跑 CLProtocol（RetryCLProtocolOnReconnect 內含守門：
                        // 已啟用/不在線/grab 中/in-flight 皆跳過）。_lastPresence/_stallDetector 只被本
                        // 單飛路徑觸碰 → 背景使用安全。
                        bool wasConnected = _lastPresence.TryGetValue(cam.CameraId, out var pv) && pv;
                        _lastPresence[cam.CameraId] = isConnected;
                        if (isConnected != wasConnected)
                        {
                            InvalidateCapturePhase(
                                "presence-cam" + cam.CameraId +
                                (isConnected ? "-connected" : "-disconnected"));
                        }
                        if (isConnected && !wasConnected)
                            cam.RetryCLProtocolOnReconnect();

                        // Reconnect may have just started a new CLProtocol handshake. Do not arm
                        // MdigProcess until that handshake and its parameter writes have settled.
                        if (isConnected && cam.IsHwParamsStable)
                        {
                            if (_parameterReadyLogged.Add(cam.CameraId))
                            {
                                FlowTrace.Log(
                                    $"acquisition parameters ready cam{cam.CameraId} " +
                                    $"cl={cam.IsClProtocolEnabled} lineRate={cam.AppliedLineRateHz:F0}");
                            }

                            bool started = cam.EnableHotStandby();
                            if (started)
                                FlowTrace.Log($"acquisition standby start cam{cam.CameraId}");
                        }

                        if (isConnected && cam.IsAcquisitionWarm)
                        {
                            if (_warmReadyLogged.Add(cam.CameraId))
                            {
                                FlowTrace.Log(
                                    $"acquisition standby ready cam{cam.CameraId} " +
                                    $"tick={cam.LastFrameStartTicks}");
                            }
                        }
                        else
                        {
                            _warmReadyLogged.Remove(cam.CameraId);
                        }

                        string statusText; Color color; bool clearFrame = false;
                        if (!isConnected)
                        {
                            statusText = "斷線"; color = Color.Pink;
                            _stallDetector.Reset(cam.CameraId);
                            clearFrame = true;
                        }
                        else if (!cam.IsLive || !cam.IsAcquisitionWarm)
                        {
                            statusText = "暖機中"; color = Color.LightGray;
                            _stallDetector.Reset(cam.CameraId);
                        }
                        else
                        {
                            // stall 偵測委派 CameraStallDetector：判據＝M_PROCESS_FRAME_COUNT 有沒有前進
                            double expFps = (cam.FrameHeight > 0 && cam.AppliedLineRateHz > 0)
                                ? cam.AppliedLineRateHz / cam.FrameHeight : 0;
                            bool stalled = _stallDetector.Update(cam.CameraId, cam.GetFrameCount(), expFps);
                            if (stalled)
                            {
                                InvalidateCapturePhase("stall-cam" + cam.CameraId);
                                statusText = "STALL";
                                color = Color.Red;
                            }
                            else if (!cam.UserWantsGrab)
                            {
                                statusText = "就緒"; color = Color.LightGreen;
                            }
                            else
                            {
                                statusText = $"FPS: {cam.CurrentFps:F1}"; color = Color.LightGreen;
                            }
                        }
                        statuses.Add((cam.CameraId, statusText, color, clearFrame));
                    }

                    string capturePhaseFault = null;
                    if (IsLiveGrabbing && _captureGateOpen)
                    {
                        AniloxCamera[] captureTargets = snapshot
                            .Where(cam => cam != null && cam.IsConnected)
                            .ToArray();
                        string phaseReason;
                        if (captureTargets.Length > 0 &&
                            !TryValidateStandbyPhase(
                                captureTargets,
                                out phaseReason,
                                logAligned: false,
                                logPrefix: "capture runtime phase"))
                        {
                            // A transient callback/tick skew must not truncate the machine's
                            // fixed High window. Keep this capture open, invalidate only the next
                            // start, and let idle preparation realign after the normal Stop flow.
                            if (System.Threading.Interlocked.CompareExchange(
                                ref _captureRuntimePhaseFaultRaised, 1, 0) == 0)
                            {
                                InvalidateCapturePhase("runtime-" + phaseReason);
                                capturePhaseFault = phaseReason;
                                FlowTrace.Log(
                                    $"capture phase drift deferred reason={phaseReason} " +
                                    "gate=open next=resync");
                            }
                        }
                    }

                    int connected = 0;
                    foreach (var cam in snapshot)
                        if (cam.IsConnected) connected++;

                    // 套用回 UI（label/清幀/事件都是 UI 物件）；IsReleasing 再驗一次
                    _uiSync?.Post(_ =>
                    {
                        if (IsReleasing) return;
                        foreach (var s in statuses)
                        {
                            if (s.clearFrame) ClearCameraFrame(s.camId);
                            _display.UpdateSingleCameraStatus(s.camId, s.text, s.color);
                        }
                        if (connected != ConnectedCameraCount || hwJustReady)
                        {
                            ConnectedCameraCount = connected;
                            OnCameraCountChanged?.Invoke(connected, ExpectedCameraCount);
                        }
                        // CLProtocol 全就緒 → 解鎖「開始抓取」鈕（gate-off 時重置 → 重連恢復後會重發）
                        if (!_hwReadyRaised && AreCamerasHwReady)
                        {
                            _hwReadyRaised = true;
                            OnHwReady?.Invoke();
                        }
                        if (capturePhaseFault != null)
                            OnCapturePhaseFault?.Invoke(capturePhaseFault);
                    }, null);

                    // A machine IO High is a fixed capture window. Prepare phase while IO is Low
                    // so an edge never spends several seconds inside full synchronization.
                    if (!IsLiveGrabbing && capturePhaseFault == null)
                        await PrepareIdleCaptureStandbyAsync(snapshot);
                }
                catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[LiveCameraManager.CameraStatusTick(bg)] {ex.GetType().Name}: {ex.Message}"); }
                finally { _statusTickInFlight = false; }
            });
        }
    }
}
