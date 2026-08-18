using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Management;
using System.Windows.Forms;
using StorageBridge.Core;
using LightBridge.Core;
using MilGrabber.Core;
using TanukiCv.Controls;
using TanukiCv.Utils;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Interop;
using AniloxRoll.Monitor.Core.Services;
using AniloxRoll.Monitor.UI.Coordinators;
using AniloxRoll.Monitor.UI.State;
using AniloxRoll.Monitor.UI.Managers;
using AniloxRoll.Monitor.UI.Navigators;
using AniloxRoll.Monitor.UI.Presenters;
using AniloxRoll.Monitor.UI.Widgets;

namespace AniloxRoll.Monitor.Forms
{
    /// <summary>AniloxRollForm 即時監控（grab 流程 / 即時曲線 / Mura 判定 / 強化切換）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- 相機按鈕事件 ---
        // ==========================================

        private async void btnLiveGrab_Click(object sender, EventArgs e)
        {
            await ToggleLiveGrabAsync("ui:【開始抓取】鈕");
        }

        /// <summary>開始/停止抓取的共用命令；按鈕、IO 接續與抓取上限都必須走這條收尾鏈。</summary>
        private async Task<bool> ToggleLiveGrabAsync(
            string intentLine,
            bool ioControlled = false,
            bool drainIoTail = false,
            Func<bool> captureStartStillValid = null)
        {
            FlowTrace.Log(intentLine);

            // 背景預覽中按 Grab → 清除預覽即可（共用顯示路：清幀＋回設定模式；舊 Free 重配已退場）
            if (IsBgPreviewActive)
                ClearBackgroundPreview();

            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;
            if (wasGrabbing && drainIoTail)
                await _liveCameraManager.DrainIoTailAsync();

            // CLProtocol 尚未就緒不可開始抓取（grab 進行中才 enable + 重套線掃會掉幀，cam1 最明顯）。
            // 手動鈕在就緒前已是灰色；此處主要擋 IO 觸發路徑（IoStartGrab 直接呼叫本方法繞過按鈕狀態）。
            if (!wasGrabbing && _liveCameraManager.IsAllocated && !_liveCameraManager.AreCamerasHwReady)
            {
                Trace.WriteLine("[Grab] CLProtocol 尚未就緒，忽略開始抓取請求。");
                return false;
            }

            // 啟動路徑：開燈命令完成後再開始 grab；實機已驗證不需要固定暖機等待。
            if (!wasGrabbing)
            {
                await Task.Run(() => LightTurnOn());
                _muraExceedLatch[0] = _muraExceedLatch[1] = false;   // 每輪 grab 重新邊緣觸發超標留痕
                _outputHealthService?.Resolve("MuraExceed.v");
                _outputHealthService?.Resolve("MuraExceed.h");
                if (_settings?.MuraDetectPaused != true) UpdateMuraLed(false);   // 新一輪檢測：警告閂鎖歸零
                _ = CurrentIoController?.ClearMura();   // 硬體 DO 閂鎖同步歸零（手動流程與 FSM EnterIdle 對齊）
            }

            bool grabStateChanged;
            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    grabStateChanged = await _liveCameraManager.EnsureAllocatedAndToggleGrabAsync(
                        _settings.EnableMuraEnhance, deferCaptureGate: true);
                    if (!_liveCameraManager.IsAllocated)
                    {
                        LightTurnOff();
                        return false;
                    }
                    LoadBackgroundBins();

                    // 初次分配即為 Global 模式 → 立即啟用即時合圖
                    if (_settings.StitchMode == StitchMode.Global)
                        _liveCameraManager.EnableGlobalMerge(
                            _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
                }
                catch (Exception ex)
                {
                    LightTurnOff();
                    MessageBox.Show($"相機配置失敗: {ex.Message}", "錯誤", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
            }
            else
            {
                grabStateChanged = await _liveCameraManager.ToggleGrabAsync(
                    deferCaptureGate: true,
                    requireVerifiedStandby: ioControlled);
            }

            if (!grabStateChanged)
            {
                if (!wasGrabbing)
                    LightTurnOff();
                UpdateGrabButton(_liveCameraManager.IsLiveGrabbing);
                return false;
            }

            if (!wasGrabbing && IsStandardBgSubEnabled && !IsBgBinReady())
            {
                FlowTrace.Log(
                    "capture start blocked reason=standard-background-not-ready");
                _liveCameraManager.StopGrab();
                LightTurnOff();
                UpdateGrabButton(false);
                return false;
            }

            // 剛從「未抓取」→「抓取中」：分配新的抓圖編號（燈已在上方開啟）
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
            {
                if (captureStartStillValid != null && !captureStartStillValid())
                {
                    FlowTrace.Log("capture start cancelled before gate reason=io-request-invalid");
                    _liveCameraManager.StopGrab();
                    LightTurnOff();
                    UpdateGrabButton(false);
                    return false;
                }

                _currentGrabId = _inspectionLogService.NextGrabId();
                DateTime captureDate = DateTime.Now;
                _currentGrabCaptureDate = captureDate;
                CaptureLayoutSnapshot initialLayout =
                    CaptureLayoutSnapshot.FromSettings(_currentGrabId, _settings, captureDate);
                if (initialLayout != null)
                    _captureLayouts[_currentGrabId] = initialLayout;
                _captureLayoutPending = false;
                _captureLayoutDeferredRenderPending = false;
                _liveCameraManager.BeginCaptureOutput(_currentGrabId, captureDate);
                string captureRoot = _settings?.CaptureRootPath ?? string.Empty;
                string imageDir = string.IsNullOrWhiteSpace(captureRoot)
                    ? "(empty)" : CaptureStoragePaths.DateImageDir(captureRoot, captureDate);
                string csvPath = string.IsNullOrWhiteSpace(captureRoot)
                    ? "(empty)" : CaptureStoragePaths.DailyCsv(captureRoot, captureDate);
                FlowTrace.Log($"capture plan grab={_currentGrabId} root={captureRoot} imageDir={imageDir} csv={csvPath} " +
                    $"archive={_currentGrabId}{CaptureArchiveStore.Extension} " +
                    $"assets=raw|proc_c|proc_r|hessian_c|hessian_r|mean_c|max_c|mean_r|max_r " +
                    $"preview=1920x1080x3 scale={InspectionEngineConfig.DefaultSaveResizeScale} " +
                    $"hessianScale={InspectionEngineConfig.DefaultHessianStandardMapScale}");

                int configuredLimitSeconds = Math.Max(
                    1,
                    _settings?.GrabLimitSeconds ?? InspectionDefaults.GrabLimitSeconds);
                CaptureStopCondition stopCondition = ioControlled
                    ? (_settings?.CaptureStopCondition ?? InspectionDefaults.DefaultCaptureStopCondition)
                    : CaptureStopCondition.Time;
                int heightLimitRows = Math.Max(
                    1,
                    _settings?.ImageView?.WaterfallTotalHeight ??
                    InspectionDefaults.WaterfallTotalHeight);
                int boundaryGraceSeconds =
                    _liveCameraManager.GetCaptureBoundaryGraceSeconds();
                bool activateTimeAfterFirstSet = _captureStopCoordinator.Arm(
                    stopCondition,
                    ioControlled,
                    configuredLimitSeconds,
                    boundaryGraceSeconds,
                    heightLimitRows,
                    _currentGrabId);

                if (!_liveCameraManager.OpenCaptureGate())
                {
                    _captureStopCoordinator.Cancel();
                    _liveCameraManager.StopGrab();
                    LightTurnOff();
                    UpdateGrabButton(false);
                    return false;
                }
                UpdateGrabButton(true);

                int firstSetTimeoutMs =
                    _liveCameraManager.GetCaptureFirstSetTimeoutMs();
                bool firstSetReady =
                    await _liveCameraManager.WaitForCaptureFirstSetReadyAsync(
                        firstSetTimeoutMs);
                if (!firstSetReady || !_liveCameraManager.IsLiveGrabbing)
                {
                    _captureStopCoordinator.FailFirstSet();
                    FlowTrace.Log(
                        $"capture start failed condition={stopCondition} " +
                        $"reason=first-set-not-ready timeoutMs={firstSetTimeoutMs} " +
                        $"grab={_currentGrabId}");
                    if (_liveCameraManager.IsLiveGrabbing)
                    {
                        await ToggleLiveGrabAsync(
                            $"auto:抓取取消 condition={stopCondition} " +
                            $"reason=first-set-not-ready grab={_currentGrabId}",
                            ioControlled: ioControlled);
                    }
                    else
                    {
                        LightTurnOff();
                        UpdateGrabButton(false);
                    }
                    return false;
                }

                if (activateTimeAfterFirstSet)
                    _captureStopCoordinator.ActivateTimeAfterFirstSet();
            }

            // 剛從「抓取中」→「停止」：關燈 + 觸發循環儲存 + 通知儲存機清理
            if (wasGrabbing && !_liveCameraManager.IsLiveGrabbing)
            {
                _captureStopCoordinator?.CompleteStop();
                _ = Task.Run(() => LightTurnOff());   // 序列埠寫入不佔 UI（[UiStack] 抓到停止時卡在 SerialStream.Write）
                string completedGrabId = _currentGrabId;
                DateTime completedCaptureDate = _currentGrabCaptureDate;
                CaptureLayoutSnapshot finalLayout =
                    CaptureLayoutSnapshot.FromSettings(
                        completedGrabId, _settings, DateTime.Now);
                if (finalLayout != null)
                {
                    _captureLayouts[completedGrabId] = finalLayout;
                    _inspectionLogService?.WriteFinalLayout(
                        finalLayout, completedCaptureDate);
                }
                ApplyFinalCaptureLayout(finalLayout);
                if (_settings?.EnableAutoCapture == true)
                    _ = FinalizeCaptureOutputsAsync(completedGrabId, completedCaptureDate);
                else
                {
                    _captureLayouts.TryRemove(completedGrabId, out _);
                    TriggerRetentionAndFlagAsync();
                }
                _muraExceedLatch[0] = _muraExceedLatch[1] = false;
                _outputHealthService?.Resolve("MuraExceed.v");
                _outputHealthService?.Resolve("MuraExceed.h");
                // 檢測結束＝MURA 警告閂鎖清除時機（與 DO latch/FSM 回 Idle 同語意；無 IO 時的等價點）
                if (_settings?.MuraDetectPaused != true) UpdateMuraLed(false);
                // 硬體 DO 閂鎖也要清：手動 grab 不經 FSM，不清則 DO_MURA 永遠掛著（Nakan 誤報 +
                // IO 暫停→恢復後 snapshot 讀回殘留 latch、燈「自己亮」——2026-07-07 盲測輪3抓到）。
                _ = CurrentIoController?.ClearMura();
            }

            UpdateGrabButton(_liveCameraManager.IsLiveGrabbing);
            return wasGrabbing != _liveCameraManager.IsLiveGrabbing;
        }

        /// <summary>狀態機只送 terminal intent；實際停止仍走共用命令，保留燈號、retention、MURA 與 IO 收尾。</summary>
        private async void HandleCaptureStopRequested(
            CaptureStopRequest request)
        {
            if (request == null) return;
            if (_liveCameraManager?.IsLiveGrabbing != true)
            {
                _captureStopCoordinator?.CompleteStop();
                return;
            }

            bool stopped = await ToggleLiveGrabAsync(
                request.CreateIntentLine());
            IoGrabController ioController = CurrentIoController;
            if (stopped && ioController != null)
            {
                try
                {
                    if (request.NotifyFixedGrabCompleted)
                        await ioController.NotifyFixedGrabCompleted();
                    else
                        await ioController.NotifyGrabStopped();
                }
                catch (Exception ex) { Trace.WriteLine($"[GrabLimit.NotifyStopped] {ex.GetType().Name}: {ex.Message}"); }
            }
        }

        private void OnCaptureCommonRowsCompleted(int commonRows)
        {
            _captureStopCoordinator?.ObserveCommonRows(commonRows);
        }

        /// <summary>
        /// 相機存檔後回呼（MIL 執行緒，非 UI 執行緒）。
        /// EnableAutoCapture=true 且抓取中時才會觸發。
        /// </summary>
        private void OnCameraInspectionResult(
            string grabId, int camId, string fileNameNoExt, float meanPeak, float maxPeak,
            float maxCMean, float meanRPeak, float maxRPeak)
        {
            if (string.IsNullOrEmpty(grabId)) return;
            DateTime captureDate;
            if (!DateTime.TryParseExact(
                grabId,
                "yyMMdd-HHmmss",
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out captureDate))
                captureDate = _currentGrabCaptureDate;
            int idx = camId - 1;
            if (_inspectionLogService != null)
            {
                // OnCameraInspectionResult 的 meanPeak/maxPeak 為 V 方向（pipeline 主處理方向），用 V 閾值記錄
                _inspectionLogService.AppendRecord(
                    grabId,
                    fileNameNoExt,
                    meanPeak,
                    maxPeak,
                    maxCMean,
                    meanRPeak,
                    maxRPeak,
                    _settings.ErrorValueMeanV,
                    _settings.ErrorValueMaxV,
                    idx >= 0 && idx < _settings.Acquisition.CameraGrabHeight.Length
                        ? _settings.Acquisition.CameraGrabHeight[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraLineRateHz.Length
                        ? _settings.Acquisition.CameraLineRateHz[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraExposureTimeUs.Length
                        ? _settings.Acquisition.CameraExposureTimeUs[idx] : 0,
                    BuildCaptureConfigSnapshot(grabId),
                    captureDate);

            }

            // 抓圖計數器 + watchdog 時間戳（Inspection 模式）
            if (_appMode?.Role != MachineRole.Storage)
            {
                _lastGrabEventTime = DateTime.UtcNow;
                int count = System.Threading.Interlocked.Increment(ref _completedGrabCount);
                if (count % 10 == 0)
                    TriggerRetentionAndFlagAsync();
            }
        }

        private async Task FinalizeCaptureOutputsAsync(string grabId, DateTime captureDate)
        {
            const int saveDrainTimeoutMs = 30000;
            const int previewMaxWidth = 1920;
            const int previewMaxHeight = 1080;

            try
            {
                bool drained = await _liveCameraManager.WaitForCaptureSavesAsync(
                    grabId, saveDrainTimeoutMs);
                if (!drained)
                    throw new TimeoutException(
                        $"等待 {grabId} 存檔完成超過 {saveDrainTimeoutMs}ms。");

                string captureRoot = _settings?.CaptureRootPath ?? string.Empty;
                string archivePath = CaptureStoragePaths.GrabArchive(
                    captureRoot, captureDate, grabId);
                if (!File.Exists(archivePath))
                    throw new FileNotFoundException("Grab 封裝檔未建立。", archivePath);

                var columnMean = new float[CameraCount][];
                var columnMax = new float[CameraCount][];
                var rowMean = new float[CameraCount][];
                var rowMax = new float[CameraCount][];
                CsvConfigSnapshot captureConfig = BuildCaptureConfigSnapshot(grabId);
                SingleGrabCurveSummary captureSummary = null;
                GrabIdInfo captureInfo = null;
                string alignmentMode = "none";
                int captureCount = 0;
                int mergedCaptureCount = 0;
                CaptureArchivePreviewAtlasResult preview = await Task.Run(() =>
                {
                    CaptureArchivePreviewAtlasResult result =
                        CapturePreviewAtlasCodec.AddToArchive(
                            archivePath,
                            previewMaxWidth,
                            previewMaxHeight,
                            replaceExisting: true,
                            progress: null);

                    var grouped = new Dictionary<int, List<string>>();
                    for (int camId = 1; camId <= CameraCount; camId++)
                    {
                        List<string> framePaths = CaptureArchiveStore.ListVirtualRawPaths(
                            archivePath, camId);
                        if (framePaths.Count > 0)
                            grouped[camId] = framePaths;
                        CurveMergeHelper.MergeCurves(
                            framePaths,
                            out columnMean[camId - 1],
                            out columnMax[camId - 1],
                            out int mergedForCamera,
                            System.Threading.CancellationToken.None);
                        mergedCaptureCount += mergedForCamera;
                    }

                    FrameAlignmentResult alignment = FrameTickIndex.ResolveAlignment(grouped);
                    alignmentMode = alignment.Mode;
                    captureCount = alignment.AllPaths.Count;
                    for (int camId = 1; camId <= CameraCount; camId++)
                    {
                        if (!grouped.TryGetValue(camId, out List<string> framePaths))
                            continue;
                        List<string> aligned = alignment.ByCamera.TryGetValue(
                            camId, out List<string> alignedPaths)
                            ? alignedPaths
                            : framePaths;
                        CurveMergeHelper.MergeRowCurves(
                            aligned, out rowMean[camId - 1], out rowMax[camId - 1]);
                    }
                    CurveMergeHelper.MergeRowCurvesOverlap(
                        rowMean, rowMax, CameraCount,
                        out float[] mergedRowMean, out float[] mergedRowMax);
                    if (captureCount > 0 && mergedCaptureCount == captureCount)
                    {
                        captureInfo = BuildCaptureGrabInfo(
                            grabId, alignment.AllPaths, captureDate);
                        captureSummary = new SingleGrabCurveSummary(
                            columnMean, columnMax,
                            mergedRowMean, mergedRowMax,
                            captureCount);
                    }
                    return result;
                });
                if (preview.FailedArchiveCount != 0 || preview.AtlasCount != 3)
                    throw new InvalidDataException(
                        $"預覽圖集不完整：atlas={preview.AtlasCount}/3 failed={preview.FailedArchiveCount}。");

                var cacheWatch = Stopwatch.StartNew();
                string summaryStatus = "skip-incomplete";
                string peakIndexStatus = "skip-incomplete";
                if (captureSummary != null && captureInfo != null)
                {
                    summaryStatus = SingleGrabCurveSummaryStore.QueueSave(
                        captureRoot, captureInfo, CameraCount, captureSummary)
                        ? "queued"
                        : "failed";
                    ColumnCurvePeakIndexResult cacheResult =
                        ColumnCurvePeakIndex.BuildAndStoreSummaryProjection(
                            captureRoot, captureInfo, captureConfig,
                            captureSummary, CameraCount);
                    peakIndexStatus = cacheResult.SummaryGrabCount == 1
                        ? "ok"
                        : "failed";
                }
                cacheWatch.Stop();
                FlowTrace.Log(
                    $"capture report cache grab={grabId} summary={summaryStatus} " +
                    $"peakIndex={peakIndexStatus} captures={captureCount} " +
                    $"merged={mergedCaptureCount} align={alignmentMode} " +
                    $"ms={cacheWatch.ElapsedMilliseconds}");

                _inspectionLogService?.AppendColumnCurveSummary(
                    grabId,
                    captureDate,
                    captureConfig?.HessianMaxFactorV ?? _settings.HessianMaxFactorV,
                    columnMean,
                    columnMax);

                string csvPath = CaptureStoragePaths.DailyCsv(captureRoot, captureDate);
                var completedFiles = new List<string> { archivePath };
                if (File.Exists(csvPath)) completedFiles.Add(csvPath);
                _remoteCopyService?.EnqueueFiles(completedFiles.ToArray());

                FlowTrace.Log(
                    $"capture finalize grab={grabId} archive={archivePath} " +
                    $"atlas={preview.AtlasCount} atlasBytes={preview.AtlasBytes} " +
                    $"remoteFiles={completedFiles.Count}");
                _outputHealthService?.Resolve("CaptureFinalizeFailure");
                TriggerRetentionAndFlagAsync();
            }
            catch (Exception ex)
            {
                string error = ex.GetType().Name + ": " + ex.Message;
                FlowTrace.Log($"capture finalize failed grab={grabId} error={error}");
                _outputHealthService?.Report(
                    "CaptureFinalizeFailure",
                    OutputHealthSeverity.OutputFault,
                    $"序號 {grabId} 封裝失敗：{error}");
            }
            finally
            {
                _captureLayouts.TryRemove(grabId, out _);
            }
        }

        private static GrabIdInfo BuildCaptureGrabInfo(
            string grabId, IEnumerable<string> paths, DateTime fallback)
        {
            DateTime earliest = DateTime.MaxValue;
            DateTime latest = DateTime.MinValue;
            if (paths != null)
            {
                foreach (string path in paths)
                {
                    string baseName = CaptureArchiveStore.IsVirtualPath(path)
                        ? CaptureArchiveStore.GetVirtualBaseName(path)
                        : Path.GetFileName(CaptureFileNaming.BaseFromImagePath(path));
                    if (!InspectionCsvReader.TryParseTimestamp(
                        baseName, out DateTime timestamp))
                        continue;
                    if (timestamp < earliest) earliest = timestamp;
                    if (timestamp > latest) latest = timestamp;
                }
            }
            if (earliest == DateTime.MaxValue || latest == DateTime.MinValue)
                earliest = latest = fallback;
            return new GrabIdInfo
            {
                GrabId = grabId,
                Earliest = earliest,
                Latest = latest
            };
        }

        private CsvConfigSnapshot BuildCaptureConfigSnapshot(string grabId)
        {
            CsvConfigSnapshot current = CsvConfigSnapshot.FromSettings(_settings);
            if (current == null || string.IsNullOrWhiteSpace(grabId))
                return current;
            return _captureLayouts.TryGetValue(grabId, out CaptureLayoutSnapshot layout)
                ? current.WithMachineLayout(layout)
                : current;
        }

        private void ApplyFinalCaptureLayout(CaptureLayoutSnapshot layout)
        {
            if (!_captureLayoutPending || layout == null) return;
            bool renderDeferredLayout = _captureLayoutDeferredRenderPending;
            _captureLayoutPending = false;
            _captureLayoutDeferredRenderPending = false;

            if (renderDeferredLayout)
            {
                UpdateRowChartPitch();
                if (_liveCameraManager?.IsGlobalMergeActive == true)
                    _liveCameraManager.RefreshGlobalMergeLayout(
                        layout.CamOps,
                        layout.CamPos);
                _liveCameraManager?.RefreshHorizontalDisplayCrop(
                    layout.TrimHeadMm,
                    layout.TrimTailMm);
            }
            FlowTrace.Log(
                $"capture layout applied grab={layout.GrabId} timing=stop " +
                $"{layout.ToFlowValues()} " +
                $"render={(renderDeferredLayout ? "once" : "already-applied")} " +
                "source=unchanged");
        }


        /// <summary>
        /// Live 曲線閾值判斷（callback 執行緒呼叫）。
        /// direction: "v"=欄, "h"=列；依 CheckLiveMura 設定的「檢測方向」決定是否觸發 DO1。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// </summary>
        // Mura 超標狀態（[0]=v,[1]=h；邊緣觸發 flow 留痕用，超標期間不洗版）
        private readonly bool[] _muraExceedLatch = new bool[2];
        private readonly object _liveInspectionStimulusLock = new object();
        private readonly bool[] _liveInspectionStimulusLogged = new bool[2];
        private int _liveInspectionStimulusBrightness = -1;
        private long _liveInspectionStimulusReadyAtTicks;
        private string _lastLiveRowScaleTrace;
        private string _lastLiveCurveAppliedTrace;

        private void ArmLiveInspectionStimulusProbe(int brightness)
        {
            if (!FlowTrace.DvtEnabled) return;
            lock (_liveInspectionStimulusLock)
            {
                _liveInspectionStimulusBrightness = brightness;
                _liveInspectionStimulusReadyAtTicks = DateTime.UtcNow.AddMilliseconds(500).Ticks;
                _liveInspectionStimulusLogged[0] = false;
                _liveInspectionStimulusLogged[1] = false;
            }
            FlowTrace.Dvt(
                $"live inspection stimulus armed brightness={brightness} settleMs=500 " +
                "purpose=inspection-standard-surrogate");
        }

        private void CheckLiveMura(float[] meanArr, float[] maxArr, string direction)
        {
            if (_settings == null || _settings.MuraDetectPaused) return;
            if (!_liveCameraManager.IsLiveGrabbing) return;

            var ridgeDir = _settings.RidgeDir;
            if (direction == "v" && ridgeDir == RidgeDirection.Horizontal) return;
            if (direction == "h" && ridgeDir == RidgeDirection.Vertical)   return;

            float meanPeak = 0f, maxPeak = 0f;
            if (meanArr != null) { for (int i = 0; i < meanArr.Length; i++) if (meanArr[i] > meanPeak) meanPeak = meanArr[i]; }
            if (maxArr  != null) { for (int i = 0; i < maxArr.Length;  i++) if (maxArr[i]  > maxPeak)  maxPeak  = maxArr[i];  }
            meanPeak /= 255f;
            maxPeak  /= 255f;

            // 依 direction 用對應方向閾值
            float thMean = direction == "h" ? _settings.ErrorValueMeanH : _settings.ErrorValueMeanV;
            float thMax  = direction == "h" ? _settings.ErrorValueMaxH  : _settings.ErrorValueMaxV;

            ColumnCurveDisplayMode curveMode = _settings.ColumnCurveMode;
            ColumnFailureCause cause = ThresholdContext.EvaluateColumnFailureCause(
                meanPeak, maxPeak, thMean, thMax, curveMode);
            bool exceed = cause != ColumnFailureCause.None;
            int di = direction == "h" ? 1 : 0;
            LogLiveInspectionStimulusSample(
                di, direction, meanPeak, maxPeak, thMean, thMax, curveMode, exceed);
            if (exceed != _muraExceedLatch[di])   // 邊緣觸發：進入/離開超標各記一行
            {
                _muraExceedLatch[di] = exceed;
                if (exceed)
                {
                    string ioState = CurrentIoController?.IsConnected != true ? "IO未連線→僅畫面警告"
                        : _isIoSuspended ? "IO暫停中→僅畫面警告" : "IO已連線";
                    FlowTrace.Log($"⚠ MURA 超標（{direction}）mean={meanPeak:F2}/max={maxPeak:F2}"
                        + $"（thr {thMean:F2}/{thMax:F2}，{ioState}）");
                    _outputHealthService?.Report(
                        "MuraExceed." + direction,
                        OutputHealthSeverity.Critical,
                        $"檢測異常（{(direction == "v" ? "欄" : "列")}）");
                }
                else
                {
                    FlowTrace.Log($"MURA 恢復（{direction}）");
                    _outputHealthService?.Resolve("MuraExceed." + direction);
                }
            }
            if (!exceed) return;

            // Pause can change while this callback is computing. Re-check at the
            // output boundary so stale work cannot re-assert DO1 after ClearMura.
            if (_settings.MuraDetectPaused) return;

            // 畫面警告與 IO 輸出解耦（2026-07-07 盲測抓到：無 IO 時操作員看不到任何警告＝設計缺陷）：
            // lblIoDoMura 視覺警告一律亮；DO 輸出（給 Nakan）看 IO 連線「且未被使用者暫停」（暫停=視同離線）。
            WarnMuraVisual();

            IoGrabController ioController = CurrentIoController;
            if (!_isIoSuspended && ioController?.IsConnected == true)
            {
                // fire-and-forget; 寫入失敗不應影響取像流程
                _ = ioController.NotifyMuraDetected().ContinueWith(
                    t => { /* swallow — PollTick 會偵測真正的 CommLost */ },
                    TaskContinuationOptions.OnlyOnFaulted);
            }
        }

        /// <summary>
        /// DVT 專用的檢測標準替代刺激樣本；光源亮暗只驗接線與公式，不代表真實 Mura。
        /// </summary>
        private void LogLiveInspectionStimulusSample(
            int directionIndex,
            string direction,
            float meanPeak,
            float maxPeak,
            float thresholdMean,
            float thresholdMax,
            ColumnCurveDisplayMode mode,
            bool exceed)
        {
            if (!FlowTrace.DvtEnabled) return;

            int brightness;
            lock (_liveInspectionStimulusLock)
            {
                if (_liveInspectionStimulusBrightness < 0 ||
                    DateTime.UtcNow.Ticks < _liveInspectionStimulusReadyAtTicks ||
                    _liveInspectionStimulusLogged[directionIndex])
                    return;

                _liveInspectionStimulusLogged[directionIndex] = true;
                brightness = _liveInspectionStimulusBrightness;
            }

            FlowTrace.Dvt(string.Format(
                CultureInfo.InvariantCulture,
                "live inspection stimulus brightness={0} direction={1} " +
                "mean={2:F4} max={3:F4} threshold={4:F4}/{5:F4} " +
                "mode={6} verdict={7} source=light-surrogate-not-mura",
                brightness,
                direction == "h" ? "row" : "col",
                meanPeak,
                maxPeak,
                thresholdMean,
                thresholdMax,
                mode,
                exceed ? "X" : "O"));
        }

        /// <summary>Mura 超標的畫面警告（callback 執行緒 → UI）：lblIoDoMura 亮並**閂鎖到該次檢測結束**
        /// （與 DO_MURA 同語意：latch 非脈衝；清除時機＝grab 停止＝FSM 回 Idle 的無 IO 等價）。
        /// 與 IO 輸出解耦——無 IO 硬體也看得到；IO 連線時 500ms snapshot 照 DO 實況刷（DO 同為 latch，一致）。</summary>
        private void WarnMuraVisual()
        {
            if (IsDisposed || Disposing) return;
            try
            {
                BeginInvoke(new Action(() =>
                {
                    if (IsDisposed || _settings?.MuraDetectPaused == true) return;
                    UpdateMuraLed(true);
                }));
            }
            catch (InvalidOperationException) { }
        }

        /// <summary>監控主畫面（ImageDisplayView）縮放/平移 → live 曲線圖 zoom 連動（bin↔主畫面對齊）。
        /// 欄(X)/overview(X) 用左右範圍、列(Y) 用上下範圍。UI 執行緒（ViewRangeMmChanged 來）。</summary>
        // 主畫面即時 X 可見範圍（mm）：ApplyLiveViewRange 存 → LiveViewRangeProvider 給 overview 的 500ms 更新沿用同值
        // （overview 立即跟隨 + 500ms 重畫沿用同範圍 → 不閃回原點）。NaN=非 ImageCanvas 即時狀態。
        private double _liveViewLeftMm = double.NaN, _liveViewRightMm = double.NaN;
        private double _liveViewTopMm = double.NaN, _liveViewBotMm = double.NaN;
        private string _liveLastMainRangeState;

        private bool ShouldFlipDisplayVertical()
            => GetVerticalDisplayDirection() == VerticalDisplayDirection.BottomToTop;

        private VerticalDisplayDirection GetVerticalDisplayDirection()
            => _settings?.VerticalDirection ?? InspectionDefaults.VerticalDirection;

        private void ApplyLiveViewRange(double leftMm, double rightMm, double topMm, double botMm)
        {
            if (IsDisposed) return;
            // ⚠ 勿節流此連動：曾試 100ms 節流 → 「圖表跟不上主畫面」立即被使用者退回（2026-07-07；
            // 加上先前兩次共三次教訓）。拖曳中曲線必須逐事件即時連動——優化只能降低單次成本，不能降頻。
            // [UiSlow] 卡頓歸因：拖曳中每次視野變更都走這（chart zoom 同步），chart 重畫慢＝拖曳跳框嫌疑
            var swVr = System.Diagnostics.Stopwatch.StartNew();
            bool wasReady = !double.IsNaN(_liveViewLeftMm) && _liveViewLeftMm < _liveViewRightMm;
            _liveViewLeftMm = leftMm; _liveViewRightMm = rightMm;     // 供 overview provider 沿用（不閃）
            _liveViewTopMm = topMm; _liveViewBotMm = botMm;
            LogLiveMainRange(leftMm, rightMm, topMm, botMm);
            _liveRowSync?.SetViewRange(topMm, botMm);

            bool nowReady = !double.IsNaN(leftMm) && leftMm < rightMm;
            if (!wasReady && nowReady)
            {
                // 首次 fit-to-screen 就緒（主畫面首幀 fit 後 RefireViewRange 發來）→ 立即原子畫一次：
                // 曲線第一筆就用 fit 範圍出現，不先閃全幅再跳到 fit。UI 執行緒，直接畫（非空 chart 補視野）。
                _liveOverviewDirty = true;
                LiveOverviewTimer_Tick(null, EventArgs.Empty);
            }
            else if (nowReady)
            {
                _liveOverviewHelper?.UpdateViewRange(leftMm, rightMm); // 已有資料 → 即時跟隨（500ms 重畫用同值不閃）
            }
            if (nowReady)
                LogLiveColumnRange("view");
            if (swVr.ElapsedMilliseconds > 50)
                FlowTrace.Log($"[UiSlow] LiveViewRangeSync {swVr.ElapsedMilliseconds}ms");
        }

        private void LogLiveMainRange(double leftMm, double rightMm, double topMm, double botMm)
        {
            string state = $"viewX={leftMm:F2}~{rightMm:F2} viewY={topMm:F2}~{botMm:F2}";
            if (string.Equals(_liveLastMainRangeState, state, StringComparison.Ordinal)) return;
            _liveLastMainRangeState = state;
            FlowTrace.Dvt($"LC mainRange {state}");
        }

        private void LogLiveColumnRange(string source)
        {
            if (chartLiveColumn == null || chartLiveColumn.IsDisposed ||
                chartLiveColumn.ChartAreas.Count == 0) return;
            var area = chartLiveColumn.ChartAreas[0];
            var axis = area.AxisX;
            FlowTrace.Dvt(
                $"LC colRange source={source} " +
                $"target={_liveViewLeftMm:F2}~{_liveViewRightMm:F2} " +
                $"axis={axis.Minimum:F2}~{axis.Maximum:F2}/" +
                $"view={axis.ScaleView.ViewMinimum:F2}~{axis.ScaleView.ViewMaximum:F2} " +
                $"plot={area.InnerPlotPosition.X:F2}~" +
                $"{area.InnerPlotPosition.Right:F2}");
        }

        private bool TryApplyLiveImageCanvasRowViewRange()
        {
            var mode = _settings?.he_MainDisplay;
            if (mode != MainDisplayMode.ImageCanvas && mode != MainDisplayMode.Waterfall) return false;
            return _liveRowSync?.TryApplyCurrentViewRange() ?? true;
        }

        private void SuspendLiveRowRangeUntilNextData()
        {
            _liveRowSync?.SuspendUntilNextData();
        }

        private bool UpdateLiveRowDataAndViewRange(float[] mean, float[] max)
        {
            var mode = _settings?.he_MainDisplay;
            bool requireViewRange = mode == MainDisplayMode.ImageCanvas || mode == MainDisplayMode.Waterfall;
            // 瀑布餵的是「顯示順序」band 緩衝（index 0=畫面最上列）；即時餵原始擷取順序 → 反向規則不同（adapter 內）
            if (_liveRowDisplay != null)
                _liveRowDisplay.DataIsDisplayOrdered = mode == MainDisplayMode.Waterfall;
            return _liveRowSync?.UpdateData(mean, max, requireViewRange) ?? true;
        }

        private void OnLiveCurveData(
            int camId,
            float[] meanArr,
            float[] maxArr,
            float frameHessianMaxFactor)
        {
            // 快取每台相機最新曲線（callback 執行緒，只是 ref 賦值）
            int cameraIndex = camId - 1;
            if (cameraIndex >= 0 && cameraIndex < CameraCount)
            {
                _liveCurveRawMean[cameraIndex] = meanArr;
                _liveCurveRawMax[cameraIndex] = maxArr;
                _liveCurveCaptureHm[cameraIndex] = frameHessianMaxFactor;
                float currentColumnFactor = (float)_settings.HessianMaxFactorV;
                _liveCurveMean[cameraIndex] = HessianRescaleHelper.CloneAndRescale1D(
                    meanArr, frameHessianMaxFactor, currentColumnFactor);
                _liveCurveMax[cameraIndex] = HessianRescaleHelper.CloneAndRescale1D(
                    maxArr, frameHessianMaxFactor, currentColumnFactor);
                // M8: memory barrier 確保 UI thread 透過 volatile _liveOverviewDirty 讀到 dirty=true 時，
                // array reference 寫入已完成（避免讀到舊指標）
                System.Threading.Interlocked.MemoryBarrier();
                _liveOverviewDirty = true;
            }

            // Live Mura 判斷（callback 執行緒，所有相機都檢查）
            CheckLiveMura(
                cameraIndex >= 0 && cameraIndex < CameraCount
                    ? _liveCurveMean[cameraIndex] : meanArr,
                cameraIndex >= 0 && cameraIndex < CameraCount
                    ? _liveCurveMax[cameraIndex] : maxArr,
                "v");
            // 單台欄 chart（chartLiveColumn 舊版）已刪除：全覽圖（接位後的 chartLiveColumn）
            // 由 _liveOverviewDirty + UpdateOverviewChart 路徑更新（boundary 唯一歸屬、與影像對齊）。
        }

        private void OnLiveRowCurveData(
            int camId,
            float[] meanArr,
            float[] maxArr,
            float frameHessianMaxFactor)
        {
            float currentRowFactor = (float)_settings.HessianMaxFactorH;
            lock (_pendingLiveRowCurveLock)
            {
                _liveRowRawMean[camId] = meanArr;
                _liveRowRawMax[camId] = maxArr;
                _liveRowCaptureHm[camId] = frameHessianMaxFactor;
            }
            float[] neutralMean = HessianRescaleHelper.CloneAndRescale1D(
                meanArr, frameHessianMaxFactor, 1f);
            float[] neutralMax = HessianRescaleHelper.CloneAndRescale1D(
                maxArr, frameHessianMaxFactor, 1f);
            float[] displayMean = HessianRescaleHelper.CloneAndRescale1D(
                meanArr, frameHessianMaxFactor, currentRowFactor);
            float[] displayMax = HessianRescaleHelper.CloneAndRescale1D(
                maxArr, frameHessianMaxFactor, currentRowFactor);
            string scaleTrace = string.Format(
                CultureInfo.InvariantCulture,
                "captureHm={0:F4} rowHm={1:F4} ratio={2:F4}",
                frameHessianMaxFactor,
                currentRowFactor,
                HessianRescaleHelper.RawCurveToDisplayScale(
                    frameHessianMaxFactor, currentRowFactor));
            if (FlowTrace.DvtEnabled && !string.Equals(
                _lastLiveRowScaleTrace, scaleTrace, StringComparison.Ordinal))
            {
                _lastLiveRowScaleTrace = scaleTrace;
                FlowTrace.Dvt("live row normalize " + scaleTrace);
            }

            // Live Mura 判斷（列方向）
            CheckLiveMura(displayMean, displayMax, "h");

            lock (_pendingLiveRowCurveLock)
            {
                _pendingLiveRowNeutralMean[camId] = neutralMean;
                _pendingLiveRowNeutralMax[camId] = neutralMax;
            }
        }

        private void PresentPendingLiveRowCurves()
        {
            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                SafeBeginInvoke(PresentPendingLiveRowCurves);
                return;
            }

            if (IsBgPreviewActive)
            {
                lock (_pendingLiveRowCurveLock)
                {
                    _pendingLiveRowNeutralMean.Clear();
                    _pendingLiveRowNeutralMax.Clear();
                }
                return;
            }

            Dictionary<int, float[]> readyMean;
            Dictionary<int, float[]> readyMax;
            lock (_pendingLiveRowCurveLock)
            {
                if (_pendingLiveRowNeutralMean.Count == 0) return;
                readyMean = new Dictionary<int, float[]>(_pendingLiveRowNeutralMean);
                readyMax = new Dictionary<int, float[]>(_pendingLiveRowNeutralMax);
                _pendingLiveRowNeutralMean.Clear();
                _pendingLiveRowNeutralMax.Clear();
            }

            _liveRowPresentationCameraCount = readyMean.Count;
            float rowFactor = (float)_settings.HessianMaxFactorH;
            var swRow = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                for (int camId = 1; camId <= CameraCount; camId++)
                {
                    if (!readyMean.TryGetValue(camId, out float[] neutralMean)) continue;
                    if (!readyMax.TryGetValue(camId, out float[] neutralMax)) continue;
                    float[] displayMean = HessianRescaleHelper.CloneAndRescale1D(
                        neutralMean, 1f, rowFactor);
                    float[] displayMax = HessianRescaleHelper.CloneAndRescale1D(
                        neutralMax, 1f, rowFactor);
                    OnLiveRowCurveDataUi(
                        camId, displayMean, displayMax, neutralMean, neutralMax);
                }
            }
            finally
            {
                if (swRow.ElapsedMilliseconds > 50)
                    FlowTrace.Log($"[UiSlow] RowChart {swRow.ElapsedMilliseconds}ms");
            }
        }

        private void OnLiveRowCurveAccepted()
        {
            FlowTrace.Log(
                $"rowCurve present after=mainImage cams={_liveRowPresentationCameraCount} " +
                $"mode={(_settings?.he_MainDisplay == MainDisplayMode.Waterfall ? "WF" : "IC")}");
        }

        private void OnLiveRowCurveDataUi(
            int camId,
            float[] displayMean,
            float[] displayMax,
            float[] neutralMean,
            float[] neutralMax)
        {
            if (_liveRowDisplay == null) return;

            if (_settings?.he_MainDisplay == MainDisplayMode.Waterfall)
            {
                UpdateLiveWaterfallRowChart(camId, neutralMean, neutralMax);
                return;
            }

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;

            // 視野同步走 ImageDisplayView.ViewRangeMmChanged → ApplyLiveViewRange（唯一路）
            if (isGlobal)
            {
                // 全域模式：快取每台相機資料，合併後更新（mean 取 mean, max 取 max）
                _liveRowMeanCache[camId] = displayMean;
                _liveRowMaxCache[camId]  = displayMax;
                if (!TryMergeLiveRowCurve(out float[] mergedMean, out float[] mergedMax)) return;
                UpdateLiveRowDataAndViewRange(mergedMean, mergedMax);
            }
            else
            {
                // 合圖未啟用（啟用失敗/尚未啟用）：只顯示選中相機
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                UpdateLiveRowDataAndViewRange(displayMean, displayMax);
            }
        }

        /// <summary>合併所有快取的 row curve 資料：mean 取平均、max 取最大值。</summary>
        private bool TryMergeLiveRowCurve(out float[] mergedMean, out float[] mergedMax)
        {
            mergedMean = null;
            mergedMax = null;
            if (_liveRowMeanCache.Count == 0) return false;

            // 取最短長度對齊
            int minLen = int.MaxValue;
            foreach (var arr in _liveRowMeanCache.Values)
                if (arr.Length < minLen) minLen = arr.Length;
            if (minLen <= 0 || minLen == int.MaxValue) return false;

            mergedMean = new float[minLen];
            mergedMax  = new float[minLen];

            int camCount = _liveRowMeanCache.Count;
            foreach (var arr in _liveRowMeanCache.Values)
                for (int i = 0; i < minLen; i++)
                    mergedMean[i] += arr[i];
            for (int i = 0; i < minLen; i++)
                mergedMean[i] /= camCount;

            foreach (var arr in _liveRowMaxCache.Values)
                for (int i = 0; i < minLen; i++)
                    if (arr[i] > mergedMax[i]) mergedMax[i] = arr[i];

            return true;
        }

        private void ResetLiveWaterfallRowChart()
        {
            SuspendLiveRowRangeUntilNextData();
            _liveRowSync?.ClearPending();
            _waterfallRowMeanPending.Clear();
            _waterfallRowMaxPending.Clear();
            _waterfallRowCurves.Reset();
        }

        private void ResetLiveChartsForDisplayTransition()
        {
            ResetLiveWaterfallRowChart();
            lock (_pendingLiveRowCurveLock)
            {
                _pendingLiveRowNeutralMean.Clear();
                _pendingLiveRowNeutralMax.Clear();
                _liveRowRawMean.Clear();
                _liveRowRawMax.Clear();
                _liveRowCaptureHm.Clear();
            }
            _liveRowMeanCache.Clear();
            _liveRowMaxCache.Clear();
            for (int i = 0; i < CameraCount; i++)
            {
                _liveCurveMean[i] = null;
                _liveCurveMax[i] = null;
                _liveCurveRawMean[i] = null;
                _liveCurveRawMax[i] = null;
                _liveCurveCaptureHm[i] = 0f;
            }
            _liveOverviewDirty = false;
            _liveOverviewHelper?.Clear();
            _liveViewLeftMm = _liveViewRightMm = double.NaN;
            _liveViewTopMm = _liveViewBotMm = double.NaN;
            _liveRowPresentationCameraCount = 0;
        }

        private void ApplyLiveInspectionSettings(string settingName)
        {
            _liveCameraManager?.SetHessianDisplayFactors(
                _settings.HessianMaxFactorV,
                _settings.HessianMaxFactorH);
            _liveOverviewHelper?.SetThresholds(
                _settings.ErrorValueMeanV, _settings.ErrorValueMaxV);
            _liveOverviewHelper?.SetVisibleMetrics(
                _settings.ShowCurveMean, _settings.ShowCurveMax);
            _liveRowDisplay?.SetThresholds(
                _settings.ErrorValueMeanH, _settings.ErrorValueMaxH);
            _liveRowDisplay?.SetVisibleMetrics(
                _settings.ShowCurveMean, _settings.ShowCurveMax);

            bool columnNormalizationChanged =
                settingName == nameof(InspectionSettings.dc_HessianMaxFactorV);
            bool rowNormalizationChanged =
                settingName == nameof(InspectionSettings.dd_HessianMaxFactorH);
            bool normalizationChanged =
                columnNormalizationChanged || rowNormalizationChanged;
            string action = normalizationChanged ? "normalization-latest" : "refresh";
            if (normalizationChanged)
                ScheduleLiveNormalizationRefresh(settingName);

            FlowTrace.Log(string.Format(
                CultureInfo.InvariantCulture,
                "live inspection apply setting={0} hm={1:F4}/{2:F4} " +
                "thresholdC={3:F4}/{4:F4} thresholdR={5:F4}/{6:F4} " +
                "mode={7} direction={8} action={9}",
                settingName,
                _settings.HessianMaxFactorV,
                _settings.HessianMaxFactorH,
                _settings.ErrorValueMeanV,
                _settings.ErrorValueMaxV,
                _settings.ErrorValueMeanH,
                _settings.ErrorValueMaxH,
                _settings.ColumnCurveMode,
                _settings.RidgeDir,
                action));
        }

        private void ScheduleLiveNormalizationRefresh(string settingName)
        {
            _pendingLiveNormalizationSetting = settingName;
            _liveNormalizationGeneration++;
            if (_liveNormalizationTimer == null)
            {
                _liveNormalizationTimer = new System.Windows.Forms.Timer { Interval = 80 };
                _liveNormalizationTimer.Tick += (sender, args) =>
                {
                    _liveNormalizationTimer.Stop();
                    ApplySettledLiveNormalization();
                };
            }
            _liveNormalizationTimer.Stop();
            _liveNormalizationTimer.Start();
        }

        private void ApplySettledLiveNormalization()
        {
            string settingName = _pendingLiveNormalizationSetting ?? string.Empty;
            int generation = _liveNormalizationGeneration;
            float columnFactor = (float)_settings.HessianMaxFactorV;
            float rowFactor = (float)_settings.HessianMaxFactorH;
            float rowMeanPeak = 0f;
            float rowMaxPeak = 0f;

            for (int i = 0; i < CameraCount; i++)
            {
                float[] rawMean = _liveCurveRawMean[i];
                float[] rawMax = _liveCurveRawMax[i];
                float captureHm = _liveCurveCaptureHm[i];
                if (rawMean == null || rawMax == null || captureHm <= 0f) continue;
                _liveCurveMean[i] = HessianRescaleHelper.CloneAndRescale1D(
                    rawMean, captureHm, columnFactor);
                _liveCurveMax[i] = HessianRescaleHelper.CloneAndRescale1D(
                    rawMax, captureHm, columnFactor);
            }
            _liveOverviewDirty = true;
            LiveOverviewTimer_Tick(null, EventArgs.Empty);

            string rowAction = "none";
            int rowWriteBefore = _waterfallRowCurves.WritePosition;
            if (!IsBgPreviewActive &&
                _settings?.he_MainDisplay == MainDisplayMode.Waterfall)
            {
                _waterfallRowCurves.Rescale(rowFactor);
                if (_waterfallRowCurves.HasData)
                {
                    UpdateLiveRowDataAndViewRange(
                        _waterfallRowCurves.Mean, _waterfallRowCurves.Max);
                    rowMeanPeak = FindCurvePeakNormalized(_waterfallRowCurves.Mean);
                    rowMaxPeak = FindCurvePeakNormalized(_waterfallRowCurves.Max);
                }
                rowAction = "rescale-current";
            }
            else if (!IsBgPreviewActive)
            {
                Dictionary<int, float[]> rawMeanByCamera;
                Dictionary<int, float[]> rawMaxByCamera;
                Dictionary<int, float> captureHmByCamera;
                lock (_pendingLiveRowCurveLock)
                {
                    rawMeanByCamera = new Dictionary<int, float[]>(_liveRowRawMean);
                    rawMaxByCamera = new Dictionary<int, float[]>(_liveRowRawMax);
                    captureHmByCamera = new Dictionary<int, float>(_liveRowCaptureHm);
                }

                foreach (KeyValuePair<int, float[]> item in rawMeanByCamera)
                {
                    int camId = item.Key;
                    if (!rawMaxByCamera.TryGetValue(camId, out float[] rawMax) ||
                        !captureHmByCamera.TryGetValue(camId, out float captureHm) ||
                        captureHm <= 0f)
                        continue;
                    float[] displayMean = HessianRescaleHelper.CloneAndRescale1D(
                        item.Value, captureHm, rowFactor);
                    float[] displayMax = HessianRescaleHelper.CloneAndRescale1D(
                        rawMax, captureHm, rowFactor);
                    OnLiveRowCurveDataUi(camId, displayMean, displayMax, null, null);
                    rowMeanPeak = Math.Max(rowMeanPeak, FindCurvePeakNormalized(displayMean));
                    rowMaxPeak = Math.Max(rowMaxPeak, FindCurvePeakNormalized(displayMax));
                }
                rowAction = "replace-current";
            }
            _lastLiveRowScaleTrace = null;
            int rowWriteAfter = _waterfallRowCurves.WritePosition;

            string actualTrace = string.Format(
                CultureInfo.InvariantCulture,
                "setting={0} generation={1} hm={2:F4}/{3:F4} " +
                "colMeanPeak={4:F4} colMaxPeak={5:F4} " +
                "rowMeanPeak={6:F4} rowMaxPeak={7:F4} " +
                "rowAction={8} rowWrite={9}->{10}",
                settingName,
                generation,
                columnFactor,
                rowFactor,
                FindCurvePeakNormalized(_liveCurveMean),
                FindCurvePeakNormalized(_liveCurveMax),
                rowMeanPeak,
                rowMaxPeak,
                rowAction,
                rowWriteBefore,
                rowWriteAfter);
            if (!string.Equals(_lastLiveCurveAppliedTrace, actualTrace, StringComparison.Ordinal))
            {
                _lastLiveCurveAppliedTrace = actualTrace;
                FlowTrace.Dvt("live curve applied " + actualTrace);
            }
        }

        private static float FindCurvePeakNormalized(IEnumerable<float[]> curves)
        {
            float peak = 0f;
            if (curves == null) return peak;
            foreach (float[] curve in curves)
            {
                if (curve == null) continue;
                for (int i = 0; i < curve.Length; i++)
                    if (curve[i] > peak) peak = curve[i];
            }
            return peak / 255f;
        }

        private static float FindCurvePeakNormalized(float[] curve)
        {
            float peak = 0f;
            if (curve == null) return peak;
            for (int i = 0; i < curve.Length; i++)
                if (curve[i] > peak) peak = curve[i];
            return peak / 255f;
        }

        private void ClearLiveRowChartForBackgroundPreview()
        {
            ResetLiveWaterfallRowChart();
            lock (_pendingLiveRowCurveLock)
            {
                _pendingLiveRowNeutralMean.Clear();
                _pendingLiveRowNeutralMax.Clear();
                _liveRowRawMean.Clear();
                _liveRowRawMax.Clear();
                _liveRowCaptureHm.Clear();
            }
            _liveRowMeanCache.Clear();
            _liveRowMaxCache.Clear();
            _liveRowPresentationCameraCount = 0;
            FlowTrace.Log("background preview rowChart clear");
        }

        private void UpdateLiveWaterfallRowChart(
            int camId,
            float[] neutralMean,
            float[] neutralMax)
        {
            if (neutralMean == null || neutralMean.Length == 0 || _liveRowDisplay == null) return;

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;
            if (!isGlobal)
            {
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                AppendLiveWaterfallRowBand(neutralMean, neutralMax);
                return;
            }

            _waterfallRowMeanPending[camId] = neutralMean;
            _waterfallRowMaxPending[camId] = neutralMax;

            int expected = Math.Max(1, _liveCameraManager?.ConnectedCameraCount ?? CameraCount);
            if (_waterfallRowMeanPending.Count < expected) return;

            var rowMean = new float[CameraCount][];
            var rowMax = new float[CameraCount][];
            foreach (var kv in _waterfallRowMeanPending)
                if (kv.Key >= 1 && kv.Key <= CameraCount) rowMean[kv.Key - 1] = kv.Value;
            foreach (var kv in _waterfallRowMaxPending)
                if (kv.Key >= 1 && kv.Key <= CameraCount) rowMax[kv.Key - 1] = kv.Value;

            CurveMergeHelper.MergeRowCurvesOverlap(rowMean, rowMax, CameraCount,
                out float[] mergedMean, out float[] mergedMax);
            if (mergedMean == null) return;

            _waterfallRowMeanPending.Clear();
            _waterfallRowMaxPending.Clear();
            AppendLiveWaterfallRowBand(mergedMean, mergedMax);
        }

        private void AppendLiveWaterfallRowBand(float[] neutralMeanBand, float[] neutralMaxBand)
        {
            int capacity = _settings?.ImageView?.WaterfallTotalHeight ?? InspectionDefaults.WaterfallTotalHeight;
            capacity = Math.Max(1000, capacity);
            bool ring = (_settings?.ImageView?.WaterfallFullMode ?? InspectionDefaults.WaterfallFullMode) == WaterfallFullMode.Ring;
            _waterfallRowCurves.Append(
                neutralMeanBand,
                neutralMaxBand,
                capacity,
                ring,
                (float)_settings.HessianMaxFactorH);

            if (UpdateLiveRowDataAndViewRange(
                _waterfallRowCurves.Mean, _waterfallRowCurves.Max)) return;
        }

        /// <summary>用 A輪速度 和選中相機的取樣頻率（Line Rate）更新列圖表座標。</summary>
        private void UpdateRowChartPitch()
        {
            if (_settings == null) return;
            double lineRateHz = _settings.Acquisition.CameraLineRateHz[0]; // CAM1 master
            _liveRowDisplay?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            _reviewRowDisplay?.SetRowPitchFromSpeed(
                _settings.AniloxRollSpeedMPerMin, lineRateHz);
            // 把 row pitch 餵給主畫面顯示 → SetLayout → 列曲線圖 Y 對齊（否則 ImageDisplayView 用 X ops 比例錯）
            if (_liveCameraManager != null)
                _liveCameraManager.RowPitchMm = _liveRowDisplay?.RowPitchMm ?? 0;
        }

        private void ApplyDisplayDirectionSetting()
        {
            _liveCameraManager?.ApplyDisplayDirection();
            _liveRowSync?.RefreshDirection();
            _reviewDisplayManager?.SetFlipVertical(ShouldFlipDisplayVertical());
            _reviewRowSync?.RefreshDirection();
            _stitchCoordinator?.RefreshChartsForSettingsChange();
            if (_stitchCoordinator?.IsStitchMode != true)
                _stitchCoordinator?.UpdateRowChartFromRepository();
        }


        private Panel[] GetLivePanels() => new[]
        {
            camLive1, camLive2, camLive3,
            camLive4, camLive5, camLive6, camLive7
        };

        /// <summary>
        /// 將 float[] column mean 擴展為 width×height 的 8bpp 灰階 bytes（共用顯示 PushStaticFrame 用）。
        /// 每列（row）相同：pixel[x] = clamp(colMean[x], 0, 255)。
        /// </summary>
        private static byte[] ExpandColMeanToGray(float[] colMean, int width, int height)
        {
            byte[] row = new byte[width];
            for (int x = 0; x < width; x++)
            {
                float v = colMean[x];
                row[x] = v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)(v + 0.5f);
            }

            byte[] pixels = new byte[width * height];
            for (int y = 0; y < height; y++)
                Buffer.BlockCopy(row, 0, pixels, y * width, width);

            return pixels;
        }

        private void UpdateGrabButton(bool isGrabbing)
        {
            btnLiveGrab.Text = isGrabbing ? "停止抓取" : "開始抓取";
            // 抓取中：凍結取得背景/預覽背景；停止後解鎖
            btnLiveGetBackground.Enabled = !isGrabbing;
            btnLiveViewBackground.Enabled = !isGrabbing;
            RefreshCameraParameterControlState(isGrabbing);
            if (!isGrabbing)
            {
                UpdateStandardBgSubLockState(); // 停止後依 bin 狀態重新檢查
            }
        }

        /// <summary>Live 顯示相關 setting（動態 LOD + OPS/Start 合圖佈局）→ 即時套到 LiveCameraManager。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入；dispatcher 只 fan-out。）</summary>
        private void HandleLiveLayoutSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.hf_LiveLod))
                _liveCameraManager?.SetLodMode(_settings.LiveLod);
            if (name == nameof(InspectionSettings.he_MainDisplay))
            {
                ResetLiveChartsForDisplayTransition();
                _liveCameraManager?.ApplyMainDisplayMode();   // 即時 / 瀑布 即時切換
            }
            if (name == nameof(InspectionSettings.hee_VerticalDirection))
                ApplyDisplayDirectionSetting();
            if (name == nameof(InspectionSettings.hg_WaterfallTotalHeight) || name == nameof(InspectionSettings.hh_WaterfallFullMode))
            {
                ResetLiveChartsForDisplayTransition();
                _liveCameraManager?.RefreshWaterfallDisplay(); // 瀑布總高/滿了行為變更 → 重建套新值
            }
            if (name == nameof(InspectionSettings.cb_CropHead) ||
                name == nameof(InspectionSettings.cc_CropTail))
            {
                _liveCameraManager?.RefreshHorizontalDisplayCrop();
            }
            if (OpsStartSettingNames.Contains(name) && _liveCameraManager?.IsGlobalMergeActive == true)
            {
                double[] opsUm = _settings.GetCameraOpsUmArray();
                double[] startsMm = _settings.GetCameraStartPositionMmArray();
                _liveCameraManager.RefreshGlobalMergeLayout(opsUm, startsMm);

                CaptureLayoutSnapshot layout = CaptureLayoutSnapshot.FromSettings(
                    _currentGrabId, _settings, DateTime.Now);
                FlowTrace.Log(
                    $"displayLayout applied setting={name} refGrid=cam1 " +
                    $"{layout.ToFlowValues()} scope=main+column-chart source=unchanged");
            }
        }

        /// <summary>強化 setting（監控 hc / 回顧 hd / 熱力圖 hda）→ 套用對應顯示。（Wave3 選項1：從 dispatcher 搬入。）</summary>
        private async Task HandleEnhanceSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.hc_EnableMuraEnhance))
                ApplyMuraEnhance(_settings.EnableMuraEnhance);
            if (name == nameof(InspectionSettings.hd_EnableReviewEnhance))
                await ApplyReviewEnhance(_settings.EnableReviewEnhance);
            if (name == nameof(InspectionSettings.hda_EnhanceHeatmap))
            {
                RefreshEnhanceColorMaps();
                IntensityColorMap liveMap = ResolveLiveColorMap();
                IntensityColorMap reviewMap = ResolveReviewColorMap();
                FlowTrace.Log(
                    $"enhance heatmap mode={_settings.EnhanceHeatmap} " +
                    $"live={ImageViewSettings.ColorMapFlowName(liveMap)} " +
                    $"review={ImageViewSettings.ColorMapFlowName(reviewMap)} " +
                    "scope=main-only data=unchanged");
            }
        }

        private void ApplyMuraEnhance(bool enabled)
        {
            _liveCameraManager?.SetLiveDisplayMode(enabled, _liveDisplayDirection);
            _liveCameraManager?.RefreshEnhanceColorMap();
            UpdateLiveDirectionVisual();
        }

        private void RefreshEnhanceColorMaps()
        {
            _liveCameraManager?.RefreshEnhanceColorMap();
            _reviewDisplayManager?.SetMainColorMap(ResolveReviewColorMap());
        }

        private IntensityColorMap ResolveLiveColorMap()
            => _settings?.ImageView?.ResolveColorMap(_settings.EnableMuraEnhance)
               ?? IntensityColorMap.Grayscale;

        private IntensityColorMap ResolveReviewColorMap()
            => _settings?.ImageView?.ResolveColorMap(_settings.EnableReviewEnhance)
               ?? IntensityColorMap.Grayscale;
    }
}
