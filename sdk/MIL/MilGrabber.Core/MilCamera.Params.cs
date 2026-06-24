using System;
using System.Runtime.InteropServices;
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    // MilCamera 的「相機參數」分區：曝光 / 線掃速率 / Grab 高度（設定 + 套用硬體 + buffer 重分配）。
    // 與核心生命週期分檔；共用 _milDigitizer / _appliedExposureUs / _appliedLineRateHz / _milGrabBuffers 等私有欄位。
    public partial class MilCamera
    {
        // ==================== Exposure ====================

        /// <summary>
        /// 設定曝光（μs）。CLProtocol 啟用走 GenICam Feature（單位 μs），否則 legacy M_EXPOSURE_TIME（ns，自動 ×1000）。
        /// 上限：clamp(floor(900000 / lineRateHz), 1, 10000)。
        /// </summary>
        public void SetExposureUs(double exposureUs)
        {
            if (exposureUs <= 0) return;
            _appliedExposureUs = exposureUs;          // 先記住設定值（Initialize 前設也記得，待 Initialize() 套用）
            if (_milDigitizer == MIL.M_NULL) return;  // digitizer 未就緒：只記不套

            if (_appliedLineRateHz > 0)
            {
                // 曝光上限公式單一真相 → MilCameraParams.CalcExposureMaxUs（勿在此再抄 900000/線掃）。
                double maxUs = MilCameraParams.CalcExposureMaxUs(_appliedLineRateHz, 1.0, 10000.0);
                if (exposureUs > maxUs)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetExposureUs: {exposureUs} μs 超出上限 {maxUs} μs（LR={_appliedLineRateHz} Hz），已夾緊。");
                    exposureUs = maxUs;
                    _appliedExposureUs = exposureUs;  // 夾緊後更新設定值
                }
            }

            if (_clProtocolEnabled)
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_DOUBLE, ref exposureUs);
            else
                MIL.MdigControl(_milDigitizer, MIL.M_EXPOSURE_TIME, exposureUs * 1000.0);
        }

        /// <summary>回傳最後一次 SetExposureUs 的設定值（μs），不依賴硬體回讀。</summary>
        public double GetExposureUs() => _appliedExposureUs;

        /// <summary>從硬體讀回曝光（μs）。CLProtocol 未啟用時回傳 0（避免相機預設值覆寫設定）。</summary>
        public double GetMeasuredExposureUs()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            if (_clProtocolEnabled)
            {
                double valUs = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "ExposureTime", MIL.M_TYPE_DOUBLE, ref valUs);
                return valUs;
            }
            return 0;
        }

        // ==================== Line Rate ====================

        /// <summary>最後一次 SetLineRateHz 設定的線掃速率（Hz）；不依賴硬體回讀，CLProtocol 未就緒也有值。
        /// 供時間戳協調等需要「設定值」而非「即時硬體值」的場景（GetLineRateHz 在 CLProtocol 未就緒時回 0）。</summary>
        public double AppliedLineRateHz => _appliedLineRateHz;

        public double GetLineRateHz()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch { return 0; }
        }

        /// <summary>查 grabber/相機的取樣頻率(line rate)上限(Hz)。需 CLProtocol 啟用；未就緒回 0。</summary>
        public double GetLineRateMaxHz()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            try
            {
                double max = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_MAX, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref max);
                return max;
            }
            catch { return 0; }
        }

        /// <summary>設定線掃速率（Hz）。CLProtocol 未就緒時僅記錄，待啟用後自動重套。</summary>
        public void SetLineRateHz(double hz)
        {
            if (hz <= 0) return;
            _appliedLineRateHz = hz;
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return;
            try
            {
                MIL.MdigControlFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] SetLineRateHz({hz}) failed: {ex.GetType().Name}: {ex.Message}");
            }
        }

        /// <summary>查相機/grabber 回報的 grab 高度上限（px）。只讀（MdigInquireFeature，不寫 → 安全，
        /// 不同於曾搞壞的「寫 Height」）。line-scan 相機高度上限可能掛在不同 feature 名 → 多候選查、回第一個 >0。
        /// 0＝CLProtocol 未就緒 / 都查不到。診斷「高度拉太高就 stall」是否＝超過相機上限。</summary>
        public int GetGrabHeightMaxPx()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return 0;
            // 候選 GenICam feature（不同相機/廠商命名不一）：相機影像高度上限。
            string[] candidates = { "Height", "HeightMax", "SensorHeight" };
            foreach (var name in candidates)
            {
                try
                {
                    long max = 0;
                    MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_MAX, name, MIL.M_TYPE_INT64, ref max);
                    if (max > 0)
                    {
                        System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] GrabHeightMax via \"{name}\" = {max}");
                        // line-scan 相機常回 uint.MaxValue(4294967295)＝無限制 → 視為「無有效上限」回 0。
                        if (max >= uint.MaxValue) return 0;
                        return max > int.MaxValue ? int.MaxValue : (int)max;
                    }
                }
                catch { /* feature 不存在 → 試下一個 */ }
            }
            return 0;
        }

        /// <summary>診斷：log 相機 GenICam Height feature 的合法範圍 Min/Max/Increment（CLProtocol 就緒後呼叫）。
        /// source size 合法值＝Min + k×Increment ≤ Max；設格點外的值（如 8736）→ cam stall（Matrox doc）。
        /// 只讀查詢，且在就緒後呼叫（非 cam init 序列）→ 安全。</summary>
        public void LogHeightFeatureInfo()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return;
            try
            {
                long min = 0, max = 0, inc = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_MIN,       "Height", MIL.M_TYPE_INT64, ref min);
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_MAX,       "Height", MIL.M_TYPE_INT64, ref max);
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_INCREMENT, "Height", MIL.M_TYPE_INT64, ref inc);
                System.Diagnostics.Trace.WriteLine(
                    $"[HeightFeature] CAM{CameraId} Height Min={min} Max={max} Increment={inc}（合法值=Min+k×Inc）");
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine($"[HeightFeature] CAM{CameraId} 查 Height feature 失敗：{ex.Message}");
            }
        }

        // ==================== Grab Height ====================

        /// <summary>變更 Grab 高度並重新分配 MIL Buffer。失敗自動 rollback 至原高度。</summary>
        /// <summary>變更 grab 高度。<paramref name="onStoppedBeforeRestart"/>＝「grab 已停、buffer 已配新尺寸、
        /// 尚未重啟」時的回呼：上層在此重配自己的 buffer（如 native 檢測記憶體），保證**不與 grab callback 競爭**
        /// → 消除「邊抓邊換 buffer」的不一致窗（高度變更 AccessViolation 根治）。</summary>
        public void SetGrabHeight(int height, Action onStoppedBeforeRestart = null)
        {
            if (_milDigitizer == MIL.M_NULL || height <= 0) return;

            // 高度沒變且 buffer 已配 → 什麼都不用做，直接 return。**防呆關鍵**：啟動套設定時會對每台呼一次
            // SetGrabHeight(同值)，若不擋會做多餘的 free+realloc，撞上正在背景跑的 CLProtocol enable（並發 MIL）
            // → CAM1 stall（實測 12000→12000 realloc 插進 CAM1 CLProtocol 序列即中招）。同值不 realloc 即避開。
            if (height == CameraGrabHeight && _milGrabBuffers[0] != MIL.M_NULL)
            {
                System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}][HtRealloc] 高度未變({height})＋buffer已配 → 跳過 realloc");
                return;
            }

            bool wasLive = IsLive;
            int oldHeight = CameraGrabHeight;

            // ===== 階段二 flag：no-realloc 改高度 =====
            // buffer 已配 max 且新高度 <= 配置高度 → 只改 M_SOURCE_SIZE_Y、**不 free/realloc**（避開「realloc→re-arm
            // 累積」＝改高度 stall 的根因路徑）。仍 stop+drain 求穩。log M_SIZE_Y：上機看 ==height（max-buffer 可行）
            // 還是 ==max（doc 疑慮成立、幀變 max 高度 → 改走 auto-allocate）。
            if (UseMaxHeightBuffers && _milGrabBuffers[0] != MIL.M_NULL && height <= _grabBufAllocH)
            {
                if (wasLive)
                {
                    MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                        MIL.M_STOP + MIL.M_WAIT, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                    IsLive = false;
                }
                try { MIL.MdigControl(_milDigitizer, MIL.M_GRAB_ABORT, MIL.M_DEFAULT); } catch { }

                MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)height);
                MIL_INT sy = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
                FrameHeight = (int)sy;
                CameraGrabHeight = height;
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] 階段二 no-realloc 改高度：req={height} M_SIZE_Y={FrameHeight} buf={_grabBufAllocH}（若 M_SIZE_Y≠req＝max-buffer 不可行）");

                if (onStoppedBeforeRestart != null)
                {
                    try { onStoppedBeforeRestart(); }
                    catch (Exception ex) { System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] onStoppedBeforeRestart: {ex.Message}"); }
                }
                if (wasLive && _userWantsGrab) StartProcess();
                return;
            }

            // ===== 改高度診斷 log（realloc 路徑）。**不在此查 MsysInquire（GetMemoryFreeMB）**：會插進相機 MIL
            // 序列 → cam1 stall（實測：本診斷一度自己觸發 CAM1 stall，因 MsysInquire 與 CAM1 CLProtocol enable 並發）。
            // 板載記憶體改看背景執行緒寫的 resource-monitor CSV / telemetry 列表（安全）。
            System.Diagnostics.Trace.WriteLine(
                $"[CAM{CameraId}][HtRealloc] 改高度 {oldHeight}->{height} wasLive={wasLive}");

            // M_STOP + M_WAIT：等佇列中的 grab 全部跑完才返回（drain，非只取消）→ 之後沒有 FrameReady 在跑。
            if (wasLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP + MIL.M_WAIT, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            // 硬排空 digitizer grab queue/DMA（不論本路徑是否停的；協調路徑是上層先停的）。
            // Matrox doc：M_STOP 只取消佇列、M_GRAB_ABORT 才「立即中止 in-flight + 佇列」→ 防「優雅停止留殘留 →
            // 重複 realloc+re-arm 累積壞狀態 → 永久 stall」。eV-CL 支援；guard 防 .NET wrapper 不支援。
            try { MIL.MdigControl(_milDigitizer, MIL.M_GRAB_ABORT, MIL.M_DEFAULT); }
            catch (Exception ex) { System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] M_GRAB_ABORT 不支援/失敗（continue）：{ex.Message}"); }

            FreeGrabBuffers();

            bool ready = true;
            try
            {
                AllocateAndBind(height);   // 配 MIL buffer + 設新 FrameWidth/Height（不在此重啟）
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] SetGrabHeight({height}) failed: {ex.GetType().Name}: {ex.Message}. Rolling back to {oldHeight}px.");
                FreeGrabBuffers();
                try
                {
                    AllocateAndBind(oldHeight);
                }
                catch (Exception rex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetGrabHeight rollback to {oldHeight}px also failed: {rex.GetType().Name}: {rex.Message}. Camera disabled.");
                    _userWantsGrab = false;
                    ready = false;
                }
            }

            // grab 仍停著、FrameWidth/Height 已是最終尺寸 → 上層重配 native/host buffer（無 callback 競爭）
            if (ready) { try { onStoppedBeforeRestart?.Invoke(); } catch (Exception ex) { System.Diagnostics.Trace.WriteLine($"[CAM{CameraId}] onStoppedBeforeRestart: {ex.Message}"); } }

            // 所有 buffer 就緒後才重啟 grab
            if (ready && wasLive && _userWantsGrab)
            {
                StartProcess();
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}][HtRealloc] 改高度完成、grab 已重啟 M_START（h={height}）。若此後 FPS=0 不恢復＝digitizer re-arm 端 stall，非 buffer 配置。");
            }
        }

        private void StartProcess()
        {
            MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
            IsLive = true;
        }

        private void FreeGrabBuffers()
        {
            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                if (_milGrabBuffers[i] != MIL.M_NULL)
                {
                    MIL.MbufFree(_milGrabBuffers[i]);
                    _milGrabBuffers[i] = MIL.M_NULL;
                }
            }
            if (_milDisplayBuffer != MIL.M_NULL) { MIL.MbufFree(_milDisplayBuffer); _milDisplayBuffer = MIL.M_NULL; }
            _milLastGrabBuffer = MIL.M_NULL;
        }

        private void AllocateAndBind(int targetHeight)
        {
            // 註：曾試「同步寫相機 GenICam Height feature」→ 反而讓相機輸出尺寸錯亂、兩台都 stall + FPS 算錯。
            // 結論：此 line-scan 相機的 Height 不可被寫成 grab 高度，切幀只能靠 digitizer M_SOURCE_SIZE_Y。已移除。
            MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)targetHeight);
            CameraGrabHeight = targetHeight;

            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            FrameWidth = (int)sizeX;
            FrameHeight = (int)sizeY;

            // 診斷：M_SIZE_Y 是否＝req（≠req＝相機沒吃這高度）+ 每個 buffer 配置成功與否（M_NULL=失敗）。
            // 不查 MsysInquire（見上：會觸發 cam1 stall）。
            bool allocFail = false;
            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                if (_milGrabBuffers[i] == MIL.M_NULL)
                {
                    allocFail = true;
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}][HtRealloc] ★ MbufAlloc2d(grab[{i}]) 回 M_NULL＝配置失敗（記憶體不足）！");
                }
                else MIL.MbufClear(_milGrabBuffers[i], 0);
            }
            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            if (_milDisplayBuffer == MIL.M_NULL)
            {
                allocFail = true;
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}][HtRealloc] ★ MbufAlloc2d(display) 回 M_NULL＝配置失敗！");
            }
            else MIL.MbufClear(_milDisplayBuffer, 0);

            System.Diagnostics.Trace.WriteLine(
                $"[CAM{CameraId}][HtRealloc] AllocateAndBind req={targetHeight} M_SIZE_Y={(int)sizeY} allocFail={allocFail}");

            if (_milDisplayBuffer != MIL.M_NULL)
                MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
            MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);

            // settle：改完 M_SOURCE_SIZE_Y 後讓 digitizer/相機套用新尺寸「沉澱」再讓 grab re-arm。
            // 根因（實測）：每次「resize→M_START」都有小機率沒乾淨 re-arm → 重複做會累積 stall；
            // 而「resize 多次→start 一次」不會 stall ＝ 最後一次 resize 到 start 之間有 settle 時間。
            // 故在 re-arm 前補足 settle（此時 grab 已停、無 callback 競爭）。
            System.Threading.Thread.Sleep(HeightSettleMs);
            // 重啟 grab 不在此做：交給 SetGrabHeight 在「上層 buffer 也配好後」才呼 StartProcess（消除不一致窗）。
        }

        /// <summary>height 改完到 grab re-arm 之間的 settle（ms）。防「resize→start 重複累積 stall」。</summary>
        private const int HeightSettleMs = 250;
    }

    /// <summary>
    /// 相機參數計算公式（純函式、無狀態、無 UI／MIL 依賴）— 跨專案共用的單一真相。
    /// 主程式與範例都呼叫這裡，避免同一條物理公式抄多份而分歧。
    /// </summary>
    public static class MilCameraParams
    {
        /// <summary>曝光時間(μs) × 線掃速率(Hz) ≈ 此固定常數（硬體限制）；線掃越高，曝光上限越低。</summary>
        public const int ExposureLineRateProduct = 900000;

        /// <summary>
        /// 依線掃速率算曝光動態上限(μs) 核心（double）：lineRateHz ≤ 0 → expMaxCap（無線掃資訊時用絕對上限）；
        /// 否則 clamp(floor(ExposureLineRateProduct / lineRateHz), expMin, expMaxCap)。曝光相關「上限公式」唯一真相。
        /// </summary>
        public static double CalcExposureMaxUs(double lineRateHz, double expMin, double expMaxCap)
        {
            if (lineRateHz <= 0) return expMaxCap;
            double v = Math.Floor(ExposureLineRateProduct / lineRateHz);
            return Math.Max(expMin, Math.Min(expMaxCap, v));
        }

        /// <summary>int 版（曝光滑桿上限用）：委派 double 核心。正整數線掃下 floor 與整數除法結果一致，行為不變。</summary>
        public static int CalcExposureMaxUs(int lineRateHz, int expMin, int expMaxCap)
            => (int)CalcExposureMaxUs((double)lineRateHz, expMin, expMaxCap);
    }
}
