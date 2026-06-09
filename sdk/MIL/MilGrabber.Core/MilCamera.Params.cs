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

        // ==================== Grab Height ====================

        /// <summary>變更 Grab 高度並重新分配 MIL Buffer。失敗自動 rollback 至原高度。</summary>
        public void SetGrabHeight(int height)
        {
            if (_milDigitizer == MIL.M_NULL || height <= 0) return;

            bool wasLive = IsLive;
            int oldHeight = CameraGrabHeight;

            if (wasLive)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = false;
            }

            FreeGrabBuffers();

            try
            {
                AllocateAndBind(height, wasLive);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine(
                    $"[CAM{CameraId}] SetGrabHeight({height}) failed: {ex.GetType().Name}: {ex.Message}. Rolling back to {oldHeight}px.");
                FreeGrabBuffers();
                try
                {
                    AllocateAndBind(oldHeight, wasLive);
                }
                catch (Exception rex)
                {
                    System.Diagnostics.Trace.WriteLine(
                        $"[CAM{CameraId}] SetGrabHeight rollback to {oldHeight}px also failed: {rex.GetType().Name}: {rex.Message}. Camera disabled.");
                    _userWantsGrab = false;
                }
            }
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

        private void AllocateAndBind(int targetHeight, bool shouldRestart)
        {
            MIL.MdigControl(_milDigitizer, MIL.M_SOURCE_SIZE_Y, (MIL_INT)targetHeight);
            CameraGrabHeight = targetHeight;

            MIL_INT sizeX = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MdigInquire(_milDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);
            FrameWidth = (int)sizeX;
            FrameHeight = (int)sizeY;

            for (int i = 0; i < _milGrabBufferListSize; i++)
            {
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                MIL.MbufClear(_milGrabBuffers[i], 0);
            }
            MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
            MIL.MbufClear(_milDisplayBuffer, 0);

            MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
            MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);

            if (shouldRestart && _userWantsGrab)
            {
                MIL.MdigProcess(_milDigitizer, _milGrabBuffers, _milGrabBufferListSize,
                    MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                IsLive = true;
            }
        }
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
