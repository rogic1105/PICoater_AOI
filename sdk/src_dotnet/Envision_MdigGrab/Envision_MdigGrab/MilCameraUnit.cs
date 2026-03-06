using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Threading;
using Matrox.MatroxImagingLibrary;
using AOI.SDK.Core;

namespace Envision_MdigGrab
{
    public class MilCameraUnit
    {
        public MIL_ID MilDigitizer = MIL.M_NULL;
        public MIL_ID MilDisplay = MIL.M_NULL;
        private MIL_ID _milProcBuffer = MIL.M_NULL;

        // [新增] 紀錄這支相機所屬的 System ID
        private MIL_ID _ownerSystemId = MIL.M_NULL;

        // 雙緩衝區與顯示緩衝區
        private MIL_ID[] _milGrabBuffers = new MIL_ID[2];
        private MIL_ID _milDisplayBuffer = MIL.M_NULL;
        private MIL_INT _milGrabBufferListSize = 2;

        public bool IsLive { get; private set; } = false;
        public int CameraId { get; private set; }
        public bool IsConnected { get; private set; } = false;

        // 公開屬性
        public bool UserWantsGrab => _userWantsGrab;
        public bool EnableImageProcessing { get; set; } = true;
        public bool EnableHessian { get; set; } = true;
        public double BinarizeThreshold { get; set; } = 128.0;
        public double HessianSigma { get; set; } = 85;   
        public double HessianFixedMax { get; set; } = 1.0;

        private bool _userWantsGrab = false;
        private bool _isReleased = false;
        private MIL_INT _devNum;
        private string _dcfPath;
        private IntPtr _panelHandle;

        private int _frameWidth = 0;
        private int _frameHeight = 0;
        private byte[] _hostInputBuffer = null;
        private byte[] _hostOutputBuffer = null;
        private IntPtr _gpuInputBuffer = IntPtr.Zero;
        private IntPtr _gpuOutputBuffer = IntPtr.Zero;

        private long _fpsWindowStartTicks = 0;
        private int _fpsFrameCount = 0;
        private double _currentFps = 0;

        public double CurrentFps => Volatile.Read(ref _currentFps);

        // Delegates (防止被 GC 回收)
        private MIL_DIG_HOOK_FUNCTION_PTR _cameraStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        public event Action<int, int, int, int> OnMouseDataChanged;

        /// <summary>
        /// 建構子：必須傳入 systemId
        /// </summary>
        public MilCameraUnit(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            _ownerSystemId = systemId; // [重要] 保存 System ID
            CameraId = id;
            _devNum = devNum;
            _dcfPath = dcfPath;
            _panelHandle = panelHandle;
            EnableImageProcessing = enableImageProcessing;

            _cameraStatusDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(CameraStatusHandler);
            _mouseStatusDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseStatusHandler);
            _processingDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);
            _hUserData = GCHandle.Alloc(this);
        }

        public void Initialize()
        {
            if (_ownerSystemId == MIL.M_NULL) return;

            MIL.MdigAlloc(_ownerSystemId, _devNum, _dcfPath, MIL.M_DEFAULT, ref MilDigitizer);

            if (MilDigitizer != MIL.M_NULL)
            {
                // ... (Display Alloc 保持不變) ...
                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref MilDisplay);

                MIL_INT sizeX = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
                MIL_INT sizeY = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);

                _frameWidth = (int)sizeX;
                _frameHeight = (int)sizeY;
                _hostInputBuffer = new byte[_frameWidth * _frameHeight];
                _hostOutputBuffer = new byte[_frameWidth * _frameHeight];

                CoreCVWrapper.CoreCV_MallocGPU(out _gpuInputBuffer, _frameWidth, _frameHeight);
                CoreCVWrapper.CoreCV_MallocGPU(out _gpuOutputBuffer, _frameWidth, _frameHeight);

                // 1. Grab Buffers (保持不變)
                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                        MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                    MIL.MbufClear(_milGrabBuffers[i], 0);
                }

                // 2. Display Buffer (保持不變)
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
                MIL.MbufClear(_milDisplayBuffer, 0);

                // 3. [新增] 分配幕後處理 Buffer (Off-screen)
                // 屬性不需要 M_DISP，只要 M_IMAGE + M_PROC 即可
                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_PROC, ref _milProcBuffer);
                MIL.MbufClear(_milProcBuffer, 0);

                // ... (Display Select Window 與 Hooks 保持不變) ...
                MIL.MdispSelectWindow(MilDisplay, _milDisplayBuffer, _panelHandle);

                MIL.MdispControl(MilDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(MilDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(MilDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                // Hooks
                MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                MIL.MdigHookFunction(MilDigitizer, MIL.M_CAMERA_PRESENT, _cameraStatusDelegate, (IntPtr)CameraId);
            }
        }

        private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (userPtr == IntPtr.Zero) return MIL.M_NULL;

            GCHandle hObj = GCHandle.FromIntPtr(userPtr);
            var cam = hObj.Target as MilCameraUnit;
            if (cam == null || cam._isReleased) return MIL.M_NULL;

            MIL_ID modifiedBuffer = MIL.M_NULL;
            MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);

            // 確保所有 Buffer 都有效
            if (modifiedBuffer != MIL.M_NULL && cam._milProcBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                if (!cam.EnableImageProcessing)
                {
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                    cam.UpdateFps();
                    return MIL.M_NULL;
                }

                bool processedByCoreCv = cam.TryApplyThresholdGpu(modifiedBuffer, cam._milProcBuffer);

                if (processedByCoreCv)
                {
                    MIL.MbufCopy(cam._milProcBuffer, cam._milDisplayBuffer);
                }
                else
                {
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                }
            }

            cam.UpdateFps();
            return MIL.M_NULL;
        }

        public void SetUserGrabIntent(bool enable)
        {
            _userWantsGrab = enable;
            ApplyGrabState();
        }

        public void ApplyGrabState()
        {
            if (MilDigitizer == MIL.M_NULL) return;

            if (_userWantsGrab && !IsLive && CheckPresence())
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize, MIL.M_START, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                ResetFps();
                IsLive = true;
            }
            else if (!_userWantsGrab && IsLive)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize, MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                ResetFps();
                IsLive = false;
            }
        }

        public bool CheckPresence()
        {
            if (MilDigitizer == MIL.M_NULL) { IsConnected = false; return false; }
            MIL_INT presence = 0;
            MIL.MdigInquire(MilDigitizer, MIL.M_CAMERA_PRESENT, ref presence);
            IsConnected = (presence == MIL.M_YES);
            return IsConnected;
        }

        public void Free()
        {
            _isReleased = true;

            if (MilDigitizer != MIL.M_NULL)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize, MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                ResetFps();
                IsLive = false;

                MIL.MdigHookFunction(MilDigitizer, MIL.M_CAMERA_PRESENT + MIL.M_UNHOOK, _cameraStatusDelegate, IntPtr.Zero);
                if (MilDisplay != MIL.M_NULL)
                {
                    MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    MIL.MdispSelectWindow(MilDisplay, MIL.M_NULL, IntPtr.Zero);
                }

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    if (_milGrabBuffers[i] != MIL.M_NULL)
                    {
                        MIL.MbufFree(_milGrabBuffers[i]);
                        _milGrabBuffers[i] = MIL.M_NULL;
                    }
                }

                if (_milDisplayBuffer != MIL.M_NULL)
                {
                    MIL.MbufFree(_milDisplayBuffer);
                    _milDisplayBuffer = MIL.M_NULL;
                }
                if (_milProcBuffer != MIL.M_NULL)
                {
                    MIL.MbufFree(_milProcBuffer);
                    _milProcBuffer = MIL.M_NULL;
                }

                if (_gpuInputBuffer != IntPtr.Zero)
                {
                    CoreCVWrapper.CoreCV_FreeGPU(_gpuInputBuffer);
                    _gpuInputBuffer = IntPtr.Zero;
                }
                if (_gpuOutputBuffer != IntPtr.Zero)
                {
                    CoreCVWrapper.CoreCV_FreeGPU(_gpuOutputBuffer);
                    _gpuOutputBuffer = IntPtr.Zero;
                }
                _hostInputBuffer = null;
                _hostOutputBuffer = null;
                if (MilDisplay != MIL.M_NULL) 
                {
                    MIL.MdispFree(MilDisplay); MilDisplay = MIL.M_NULL; 
                }
                MIL.MdigFree(MilDigitizer);
                MilDigitizer = MIL.M_NULL;
            }

            if (_hUserData.IsAllocated) _hUserData.Free();

            // 注意：我們不在這裡釋放 System，因為 System 是由外部 (Form) 傳入並管理的
        }



        private bool TryApplyThresholdGpu(MIL_ID srcBuffer, MIL_ID dstBuffer)
        {
            if (srcBuffer == MIL.M_NULL || dstBuffer == MIL.M_NULL) return false;
            if (_frameWidth <= 0 || _frameHeight <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;
            if (_gpuInputBuffer == IntPtr.Zero || _gpuOutputBuffer == IntPtr.Zero) return false;

            try
            {
                MIL.MbufGet2d(srcBuffer, 0, 0, _frameWidth, _frameHeight, _hostInputBuffer);

                GCHandle hIn = GCHandle.Alloc(_hostInputBuffer, GCHandleType.Pinned);
                GCHandle hOut = GCHandle.Alloc(_hostOutputBuffer, GCHandleType.Pinned);

                try
                {
                    int uploadResult = CoreCVWrapper.CoreCV_Upload(hIn.AddrOfPinnedObject(), _gpuInputBuffer, _frameWidth, _frameHeight);
                    if (uploadResult != 0) return false;

                    byte threshold = (byte)Math.Max(0, Math.Min(255, (int)BinarizeThreshold));
                    int thresholdResult = CoreCVWrapper.CoreCV_Threshold_GPU(_gpuInputBuffer, _frameWidth, _frameHeight, threshold, _gpuOutputBuffer);
                    if (thresholdResult != 0) return false;

                    int downloadResult = CoreCVWrapper.CoreCV_Download(_gpuOutputBuffer, hOut.AddrOfPinnedObject(), _frameWidth, _frameHeight);
                    if (downloadResult != 0) return false;
                }
                finally
                {
                    if (hIn.IsAllocated) hIn.Free();
                    if (hOut.IsAllocated) hOut.Free();
                }

                MIL.MbufPut2d(dstBuffer, 0, 0, _frameWidth, _frameHeight, _hostOutputBuffer);
                return true;
            }
            catch
            {
                return false;
            }
        }

        private void ResetFps()
        {
            _fpsWindowStartTicks = 0;
            _fpsFrameCount = 0;
            Volatile.Write(ref _currentFps, 0);
        }

        private void UpdateFps()
        {
            long now = Stopwatch.GetTimestamp();
            if (_fpsWindowStartTicks == 0)
            {
                _fpsWindowStartTicks = now;
                _fpsFrameCount = 0;
            }

            _fpsFrameCount++;
            double elapsedSec = (double)(now - _fpsWindowStartTicks) / Stopwatch.Frequency;
            if (elapsedSec >= 1.0)
            {
                Volatile.Write(ref _currentFps, _fpsFrameCount / elapsedSec);
                _fpsWindowStartTicks = now;
                _fpsFrameCount = 0;
            }
        }

        private MIL_INT MouseStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_isReleased || _milDisplayBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

            int x = (int)posX;
            int y = (int)posY;
            int pixelValue = -1;

            MIL_INT sizeX = MIL.MbufInquire(_milDisplayBuffer, MIL.M_SIZE_X, MIL.M_NULL);
            MIL_INT sizeY = MIL.MbufInquire(_milDisplayBuffer, MIL.M_SIZE_Y, MIL.M_NULL);

            if (x >= 0 && x < sizeX && y >= 0 && y < sizeY)
            {
                byte[] data = new byte[1];
                MIL.MbufGet2d(_milDisplayBuffer, x, y, 1, 1, data);
                pixelValue = data[0];
            }

            OnMouseDataChanged?.Invoke(CameraId, x, y, pixelValue);
            return MIL.M_NULL;
        }

        private MIL_INT CameraStatusHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_isReleased) return MIL.M_NULL;
            bool present = CheckPresence();
            if (!present && IsLive)
            {
                MIL.MdigProcess(MilDigitizer, _milGrabBuffers, _milGrabBufferListSize, MIL.M_STOP, MIL.M_DEFAULT, _processingDelegate, GCHandle.ToIntPtr(_hUserData));
                ResetFps();
                IsLive = false;
            }
            return MIL.M_NULL;
        }
    }
}
