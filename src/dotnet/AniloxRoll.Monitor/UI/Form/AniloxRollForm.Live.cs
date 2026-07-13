using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Diagnostics;
using System.Drawing;
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
            FlowTrace.Log("ui:【開始抓取】鈕");   // intent 行：之後的顯示變更行歸此動作管（孤兒判讀規則）

            // 背景預覽中按 Grab → 清除預覽即可（共用顯示路：清幀＋回設定模式；舊 Free 重配已退場）
            if (IsBgPreviewActive)
                ClearBackgroundPreview();

            bool wasGrabbing = _liveCameraManager.IsLiveGrabbing;

            // CLProtocol 尚未就緒不可開始抓取（grab 進行中才 enable + 重套線掃會掉幀，cam1 最明顯）。
            // 手動鈕在就緒前已是灰色；此處主要擋 IO 觸發路徑（IoStartGrab 直接呼叫本方法繞過按鈕狀態）。
            if (!wasGrabbing && _liveCameraManager.IsAllocated && !_liveCameraManager.AreCamerasHwReady)
            {
                Trace.WriteLine("[Grab] CLProtocol 尚未就緒，忽略開始抓取請求。");
                return;
            }

            // 啟動路徑：先亮燈 → 等光源穩定 → 再開始 grab
            if (!wasGrabbing)
            {
                await Task.Run(() => LightTurnOn());   // 序列埠寫入 ~百 ms，不佔 UI（順序仍保證：燈亮→暖機→grab）
                int warmup = _settings?.LightWarmupMs ?? 0;
                if (warmup > 0) await Task.Delay(warmup);
                ResetLiveChartsForDisplayTransition();
                _muraExceedLatch[0] = _muraExceedLatch[1] = false;   // 每輪 grab 重新邊緣觸發超標留痕
                if (_settings?.MuraDetectPaused != true) UpdateMuraLed(false);   // 新一輪檢測：警告閂鎖歸零
                _ = _ioGrabController?.ClearMura();   // 硬體 DO 閂鎖同步歸零（手動流程與 FSM EnterIdle 對齊）
            }

            if (!_liveCameraManager.IsAllocated)
            {
                try
                {
                    _liveCameraManager.EnsureAllocatedAndToggleGrab(_settings.EnableMuraEnhance);
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
                    return;
                }
            }
            else
            {
                _liveCameraManager.ToggleGrab();
            }

            // 剛從「未抓取」→「抓取中」：分配新的抓圖編號（燈已在上方開啟）
            if (!wasGrabbing && _liveCameraManager.IsLiveGrabbing)
            {
                _currentGrabId = _inspectionLogService.NextGrabId();
                DateTime captureDate = DateTime.Now;
                string captureRoot = _settings?.CaptureRootPath ?? string.Empty;
                string imageDir = string.IsNullOrWhiteSpace(captureRoot)
                    ? "(empty)" : CaptureStoragePaths.DateImageDir(captureRoot, captureDate);
                string csvPath = string.IsNullOrWhiteSpace(captureRoot)
                    ? "(empty)" : CaptureStoragePaths.DailyCsv(captureRoot, captureDate);
                FlowTrace.Log($"capture plan grab={_currentGrabId} root={captureRoot} imageDir={imageDir} csv={csvPath} " +
                    $"files=*{CaptureFileNaming.RawJpg}|*{CaptureFileNaming.ProcV}|*{CaptureFileNaming.ProcH}|*{CaptureFileNaming.MeanC}|*{CaptureFileNaming.MaxC}|*{CaptureFileNaming.MeanR}|*{CaptureFileNaming.MaxR} " +
                    $"scale={InspectionEngineConfig.DefaultSaveResizeScale}");
            }

            // 剛從「抓取中」→「停止」：關燈 + 觸發循環儲存 + 通知儲存機清理
            if (wasGrabbing && !_liveCameraManager.IsLiveGrabbing)
            {
                _ = Task.Run(() => LightTurnOff());   // 序列埠寫入不佔 UI（[UiStack] 抓到停止時卡在 SerialStream.Write）
                TriggerRetentionAndFlagAsync();
                // 檢測結束＝MURA 警告閂鎖清除時機（與 DO latch/FSM 回 Idle 同語意；無 IO 時的等價點）
                if (_settings?.MuraDetectPaused != true) UpdateMuraLed(false);
                // 硬體 DO 閂鎖也要清：手動 grab 不經 FSM，不清則 DO_MURA 永遠掛著（Nakan 誤報 +
                // IO 暫停→恢復後 snapshot 讀回殘留 latch、燈「自己亮」——2026-07-07 盲測輪3抓到）。
                _ = _ioGrabController?.ClearMura();
            }

            UpdateGrabButton(_liveCameraManager.IsLiveGrabbing);
        }

        /// <summary>
        /// 相機存檔後回呼（MIL 執行緒，非 UI 執行緒）。
        /// EnableAutoCapture=true 且抓取中時才會觸發。
        /// </summary>
        private void OnCameraInspectionResult(
            int camId, string fileNameNoExt, float meanPeak, float maxPeak, float maxCMean)
        {
            if (string.IsNullOrEmpty(_currentGrabId)) return;
            int idx = camId - 1;
            if (_inspectionLogService != null)
            {
                // OnCameraInspectionResult 的 meanPeak/maxPeak 為 V 方向（pipeline 主處理方向），用 V 閾值記錄
                _inspectionLogService.AppendRecord(
                    _currentGrabId,
                    fileNameNoExt,
                    meanPeak,
                    maxPeak,
                    maxCMean,
                    _settings.ErrorValueMeanV,
                    _settings.ErrorValueMaxV,
                    idx >= 0 && idx < _settings.Acquisition.CameraGrabHeight.Length
                        ? _settings.Acquisition.CameraGrabHeight[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraLineRateHz.Length
                        ? _settings.Acquisition.CameraLineRateHz[idx] : 0,
                    idx >= 0 && idx < _settings.Acquisition.CameraExposureTimeUs.Length
                        ? _settings.Acquisition.CameraExposureTimeUs[idx] : 0,
                    CsvConfigSnapshot.FromSettings(_settings));

                // CSV 寫完後排入遠端複製佇列（CSV 在 month 目錄，不在 OnFilesSaved 的 day 目錄）
                string csvPath = _inspectionLogService.LastCsvPath;
                if (!string.IsNullOrEmpty(csvPath))
                    _remoteCopyService?.EnqueueFile(csvPath);
            }

            // IO MURA 信號：任一相機超過閾值即通知（IO 暫停=視同離線，不發 DO）
            if (!_isIoSuspended && _ioGrabController?.IsConnected == true)
            {
                // meanPeak/maxPeak 為 V 方向，按 V 閾值判定
                bool isMura = meanPeak > _settings.ErrorValueMeanV || maxPeak > _settings.ErrorValueMaxV;
                if (isMura) _ = _ioGrabController.NotifyMuraDetected();
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


        /// <summary>
        /// Live 曲線閾值判斷（callback 執行緒呼叫）。
        /// direction: "v"=欄, "h"=列；依 CheckLiveMura 設定的「檢測方向」決定是否觸發 DO1。
        /// 陣列為 0-255，閾值為 0-1，取陣列 max 後除以 255 比較。
        /// </summary>
        // Mura 超標狀態（[0]=v,[1]=h；邊緣觸發 flow 留痕用，超標期間不洗版）
        private readonly bool[] _muraExceedLatch = new bool[2];

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

            bool exceed = meanPeak > thMean || maxPeak > thMax;
            int di = direction == "h" ? 1 : 0;
            if (exceed != _muraExceedLatch[di])   // 邊緣觸發：進入/離開超標各記一行
            {
                _muraExceedLatch[di] = exceed;
                if (exceed)
                {
                    string ioState = _ioGrabController?.IsConnected != true ? "IO未連線→僅畫面警告"
                        : _isIoSuspended ? "IO暫停中→僅畫面警告" : "IO已連線";
                    FlowTrace.Log($"⚠ MURA 超標（{direction}）mean={meanPeak:F2}/max={maxPeak:F2}"
                        + $"（thr {thMean:F2}/{thMax:F2}，{ioState}）");
                }
                else
                    FlowTrace.Log($"MURA 恢復（{direction}）");
            }
            if (!exceed) return;

            // 畫面警告與 IO 輸出解耦（2026-07-07 盲測抓到：無 IO 時操作員看不到任何警告＝設計缺陷）：
            // lblIoDoMura 視覺警告一律亮；DO 輸出（給 Nakan）看 IO 連線「且未被使用者暫停」（暫停=視同離線）。
            WarnMuraVisual();

            if (!_isIoSuspended && _ioGrabController?.IsConnected == true)
            {
                // fire-and-forget; 寫入失敗不應影響取像流程
                _ = _ioGrabController.NotifyMuraDetected().ContinueWith(
                    t => { /* swallow — PollTick 會偵測真正的 CommLost */ },
                    TaskContinuationOptions.OnlyOnFaulted);
            }
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
            if (swVr.ElapsedMilliseconds > 50)
                FlowTrace.Log($"[UiSlow] LiveViewRangeSync {swVr.ElapsedMilliseconds}ms");
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

        private void OnLiveCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // 快取每台相機最新曲線（callback 執行緒，只是 ref 賦值）
            int cameraIndex = camId - 1;
            if (cameraIndex >= 0 && cameraIndex < CameraCount)
            {
                _liveCurveMean[cameraIndex] = meanArr;
                _liveCurveMax[cameraIndex]  = maxArr;
                // M8: memory barrier 確保 UI thread 透過 volatile _liveOverviewDirty 讀到 dirty=true 時，
                // array reference 寫入已完成（避免讀到舊指標）
                System.Threading.Interlocked.MemoryBarrier();
                _liveOverviewDirty = true;
            }

            // Live Mura 判斷（callback 執行緒，所有相機都檢查）
            CheckLiveMura(meanArr, maxArr, "v");
            // 單台欄 chart（chartLiveColumn 舊版）已刪除：全覽圖（接位後的 chartLiveColumn）
            // 由 _liveOverviewDirty + UpdateOverviewChart 路徑更新（boundary 唯一歸屬、與影像對齊）。
        }

        private void OnLiveRowCurveData(int camId, float[] meanArr, float[] maxArr)
        {
            // Live Mura 判斷（列方向）
            CheckLiveMura(meanArr, maxArr, "h");

            if (InvokeRequired)
            {
                if (!IsHandleCreated || IsDisposed || Disposing) return;
                SafeBeginInvoke(() => OnLiveRowCurveData(camId, meanArr, maxArr));
                return;
            }

            // [UiSlow] 卡頓歸因：列 chart 每幀重建在 UI 執行緒（grab 中每幀一次）
            var swRow = System.Diagnostics.Stopwatch.StartNew();
            try { OnLiveRowCurveDataUi(camId, meanArr, maxArr); }
            finally
            {
                if (swRow.ElapsedMilliseconds > 50)
                    FlowTrace.Log($"[UiSlow] RowChart {swRow.ElapsedMilliseconds}ms");
            }
        }

        private void OnLiveRowCurveDataUi(int camId, float[] meanArr, float[] maxArr)
        {
            if (_liveRowDisplay == null) return;

            if (_settings?.he_MainDisplay == MainDisplayMode.Waterfall)
            {
                UpdateLiveWaterfallRowChart(camId, meanArr, maxArr);
                return;
            }

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;

            // 視野同步走 ImageDisplayView.ViewRangeMmChanged → ApplyLiveViewRange（唯一路）
            if (isGlobal)
            {
                // 全域模式：快取每台相機資料，合併後更新（mean 取 mean, max 取 max）
                _liveRowMeanCache[camId] = meanArr;
                _liveRowMaxCache[camId]  = maxArr;
                if (!TryMergeLiveRowCurve(out float[] mergedMean, out float[] mergedMax)) return;
                UpdateLiveRowDataAndViewRange(mergedMean, mergedMax);
            }
            else
            {
                // 合圖未啟用（啟用失敗/尚未啟用）：只顯示選中相機
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                UpdateLiveRowDataAndViewRange(meanArr, maxArr);
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
            _waterfallRowMean = null;
            _waterfallRowMax = null;
            _waterfallRowWrite = 0;
        }

        private void ResetLiveChartsForDisplayTransition()
        {
            ResetLiveWaterfallRowChart();
            _liveRowMeanCache.Clear();
            _liveRowMaxCache.Clear();
            for (int i = 0; i < CameraCount; i++)
            {
                _liveCurveMean[i] = null;
                _liveCurveMax[i] = null;
            }
            _liveOverviewDirty = false;
            _liveOverviewHelper?.Clear();
            _liveViewLeftMm = _liveViewRightMm = double.NaN;
            _liveViewTopMm = _liveViewBotMm = double.NaN;
        }

        private void UpdateLiveWaterfallRowChart(int camId, float[] meanArr, float[] maxArr)
        {
            if (meanArr == null || meanArr.Length == 0 || _liveRowDisplay == null) return;

            bool isGlobal = _liveCameraManager?.IsGlobalMergeActive == true;
            if (!isGlobal)
            {
                if (camId != _liveCameraManager.SelectedMainCameraId) return;
                AppendLiveWaterfallRowBand(meanArr, maxArr);
                return;
            }

            _waterfallRowMeanPending[camId] = meanArr;
            _waterfallRowMaxPending[camId] = maxArr;

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

        private void AppendLiveWaterfallRowBand(float[] meanBand, float[] maxBand)
        {
            int capacity = _settings?.ImageView?.WaterfallTotalHeight ?? InspectionDefaults.WaterfallTotalHeight;
            capacity = Math.Max(1000, capacity);
            if (_waterfallRowMean == null || _waterfallRowMean.Length != capacity)
            {
                _waterfallRowMean = new float[capacity];
                _waterfallRowMax = new float[capacity];
                _waterfallRowWrite = 0;
            }

            int bandLen = Math.Min(meanBand.Length, capacity);
            bool ring = (_settings?.ImageView?.WaterfallFullMode ?? InspectionDefaults.WaterfallFullMode) == WaterfallFullMode.Ring;
            if (!ring && _waterfallRowWrite + bandLen > capacity)
            {
                Array.Clear(_waterfallRowMean, 0, _waterfallRowMean.Length);
                Array.Clear(_waterfallRowMax, 0, _waterfallRowMax.Length);
                _waterfallRowWrite = 0;
            }

            for (int i = 0; i < bandLen; i++)
            {
                int dst = ring ? (_waterfallRowWrite + i) % capacity : _waterfallRowWrite + i;
                if (dst < 0 || dst >= capacity) break;
                _waterfallRowMean[dst] = meanBand[i];
                _waterfallRowMax[dst] = maxBand != null && i < maxBand.Length ? maxBand[i] : 0;
            }

            _waterfallRowWrite = ring
                ? (_waterfallRowWrite + bandLen) % capacity
                : Math.Min(capacity, _waterfallRowWrite + bandLen);

            if (UpdateLiveRowDataAndViewRange(_waterfallRowMean, _waterfallRowMax)) return;
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
            _stitchCoordinator?.RefreshCurrentCameraChartsForSettingsChange();
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
                FlowTrace.Log($"ui:設定[主畫面顯示] → {_settings.he_MainDisplay}");   // intent 行（孤兒判讀規則）
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
            if (OpsStartSettingNames.Contains(name) && _liveCameraManager?.IsGlobalMergeActive == true)
                _liveCameraManager.RefreshGlobalMergeLayout(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
        }

        /// <summary>強化 setting（監控 hc / 回顧 hd）→ 套用對應強化。（Wave3 選項1：從 dispatcher 搬入。）</summary>
        private async Task HandleEnhanceSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.hc_EnableMuraEnhance))
                ApplyMuraEnhance(_settings.EnableMuraEnhance);
            if (name == nameof(InspectionSettings.hd_EnableReviewEnhance))
                await ApplyReviewEnhance(_settings.EnableReviewEnhance);
        }

        private void ApplyMuraEnhance(bool enabled)
        {
            _liveCameraManager?.SetImageProcessingEnabled(enabled);
            _liveCameraManager?.SetLiveDisplayDirection(_liveDisplayDirection);
            UpdateLiveDirectionVisual();
        }
    }
}
