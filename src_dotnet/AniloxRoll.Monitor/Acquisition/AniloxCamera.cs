using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Threading;
using Matrox.MatroxImagingLibrary;
using AOI.SDK.Core;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Camera
{
    public class AniloxCamera
    {
        public MIL_ID MilDigitizer = MIL.M_NULL;
        public MIL_ID MilDisplay = MIL.M_NULL;
        public MIL_ID MilSecondaryDisplay = MIL.M_NULL; // [新增] 第二顯示區

        private MIL_ID _milProcBuffer = MIL.M_NULL;
        private MIL_ID _ownerSystemId = MIL.M_NULL;

        private MIL_ID[] _milGrabBuffers = new MIL_ID[2];
        private MIL_ID _milDisplayBuffer = MIL.M_NULL;
        private MIL_INT _milGrabBufferListSize = 2;

        public bool IsLive { get; private set; } = false;
        public int CameraId { get; private set; }
        public bool IsConnected { get; private set; } = false;

        public bool UserWantsGrab => _userWantsGrab;
        public bool EnableImageProcessing { get; set; } = true;
        public bool EnableHessian { get; set; } = true;
        public bool EnableAutoCapture { get; set; } = false;
        public string CaptureRootPath { get; set; } = string.Empty;
        public int CameraGrabHeight { get; set; } = 0;
        public double CameraExposureTimeUs { get; set; } = 0;
        public double BinarizeThreshold { get; set; } = 128.0;
        public double HessianSigma { get; set; } = 85;
        public double HessianFixedMax { get; set; } = 1.0;

        private bool _userWantsGrab = false;
        private bool _isReleased = false;
        private bool _isSecondaryHooked = false; // [新增] 紀錄第二顯示區是否已掛載滑鼠事件

        private MIL_INT _devNum;
        private string _dcfPath;
        private IntPtr _panelHandle;

        private int _frameWidth = 0;
        private int _frameHeight = 0;

        private static readonly string[] ExposureControlCandidates =
            { "M_EXPOSURE", "M_EXPOSURE_TIME", "M_CAMERA_EXPOSURE", "M_GRAB_EXPOSURE" };

        private static bool _exposureControlResolved = false;
        private static MIL_INT _exposureControlType = MIL.M_NULL;
        private byte[] _hostInputBuffer = null;
        private byte[] _hostOutputBuffer = null;

        private readonly object _picoaterLock = new object();
        private readonly AoiService _aoiService = new AoiService();
        private IntPtr _picoaterInputBuffer = IntPtr.Zero;
        private IntPtr _picoaterRidgeBuffer = IntPtr.Zero;
        private ulong _picoaterBufferSize = 0;

        private long _fpsWindowStartTicks = 0;
        private int _fpsFrameCount = 0;
        private double _currentFps = 0;
        private string _lastCaptureSecondKey = string.Empty;

        public double CurrentFps => Volatile.Read(ref _currentFps);

        // Delegates
        private MIL_DIG_HOOK_FUNCTION_PTR _cameraStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseStatusDelegate;
        private MIL_DISP_HOOK_FUNCTION_PTR _mouseClickDelegate; // [新增] 點擊事件委派
        private MIL_DIG_HOOK_FUNCTION_PTR _processingDelegate;
        private GCHandle _hUserData;

        // Events
        public event Action<int, int, int, int> OnMouseDataChanged;
        public event Action<int> OnCameraClicked; // [新增] 點擊事件

        public AniloxCamera(MIL_ID systemId, int id, MIL_INT devNum, string dcfPath, IntPtr panelHandle, bool enableImageProcessing = true)
        {
            _ownerSystemId = systemId;
            CameraId = id;
            _devNum = devNum;
            _dcfPath = dcfPath;
            _panelHandle = panelHandle;
            EnableImageProcessing = enableImageProcessing;

            _cameraStatusDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(CameraStatusHandler);
            _mouseStatusDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseStatusHandler);
            _mouseClickDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(MouseClickHandler); // [新增] 初始化點擊委派
            _processingDelegate = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);
            _hUserData = GCHandle.Alloc(this);
        }

        public void Initialize()
        {
            if (_ownerSystemId == MIL.M_NULL) return;

            MIL.MdigAlloc(_ownerSystemId, _devNum, _dcfPath, MIL.M_DEFAULT, ref MilDigitizer);

            if (MilDigitizer != MIL.M_NULL)
            {
                ApplyDigitizerSettings();
                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref MilDisplay);
                MIL.MdispAlloc(_ownerSystemId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref MilSecondaryDisplay); // [新增] 分配第二顯示區

                MIL_INT sizeX = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_X, MIL.M_NULL);
                MIL_INT sizeY = MIL.MdigInquire(MilDigitizer, MIL.M_SIZE_Y, MIL.M_NULL);

                _frameWidth = (int)sizeX;
                _frameHeight = (int)sizeY;
                _hostInputBuffer = new byte[_frameWidth * _frameHeight];
                _hostOutputBuffer = new byte[_frameWidth * _frameHeight];

                _aoiService.Initialize();
                _picoaterBufferSize = (ulong)(_frameWidth * _frameHeight);
                _picoaterInputBuffer = Marshal.AllocHGlobal((IntPtr)_picoaterBufferSize);
                _picoaterRidgeBuffer = Marshal.AllocHGlobal((IntPtr)_picoaterBufferSize);

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                        MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref _milGrabBuffers[i]);
                    MIL.MbufClear(_milGrabBuffers[i], 0);
                }

                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref _milDisplayBuffer);
                MIL.MbufClear(_milDisplayBuffer, 0);

                MIL.MbufAlloc2d(_ownerSystemId, sizeX, sizeY, 8 + MIL.M_UNSIGNED,
                    MIL.M_IMAGE + MIL.M_PROC, ref _milProcBuffer);
                MIL.MbufClear(_milProcBuffer, 0);

                MIL.MdispSelectWindow(MilDisplay, _milDisplayBuffer, _panelHandle);

                MIL.MdispControl(MilDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(MilDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(MilDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                // 掛載 Hooks
                MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_LEFT_BUTTON_DOWN, _mouseClickDelegate, (IntPtr)CameraId); // [新增] 掛載左鍵點擊
                MIL.MdigHookFunction(MilDigitizer, MIL.M_CAMERA_PRESENT, _cameraStatusDelegate, (IntPtr)CameraId);
            }
        }

        // [新增] 設定第二顯示區的方法
        public void SetSecondaryDisplay(IntPtr handle)
        {
            if (MilSecondaryDisplay == MIL.M_NULL) return;

            if (handle == IntPtr.Zero)
            {
                if (_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(MilSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    _isSecondaryHooked = false;
                }
                MIL.MdispSelectWindow(MilSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);
            }
            else
            {
                MIL.MdispSelectWindow(MilSecondaryDisplay, _milDisplayBuffer, handle);
                MIL.MdispControl(MilSecondaryDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(MilSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(MilSecondaryDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE); // 允許滑鼠互動(平移/縮放)

                if (!_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(MilSecondaryDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                    _isSecondaryHooked = true;
                }
            }
        }

        // [新增] 滑鼠點擊攔截事件
        private MIL_INT MouseClickHandler(MIL_INT HookType, MIL_ID EventId, IntPtr UserPtr)
        {
            if (_isReleased) return MIL.M_NULL;
            OnCameraClicked?.Invoke(CameraId); // 通知 UI 切換畫面
            return MIL.M_NULL;
        }

        private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (userPtr == IntPtr.Zero) return MIL.M_NULL;

            GCHandle hObj = GCHandle.FromIntPtr(userPtr);
            var cam = hObj.Target as AniloxCamera;
            if (cam == null || cam._isReleased) return MIL.M_NULL;

            MIL_ID modifiedBuffer = MIL.M_NULL;
            MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);

            if (modifiedBuffer != MIL.M_NULL && cam._milProcBuffer != MIL.M_NULL && cam._milDisplayBuffer != MIL.M_NULL)
            {
                if (!cam.EnableImageProcessing)
                {
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                    cam.TrySaveCapture(modifiedBuffer);
                    cam.UpdateFps();
                    return MIL.M_NULL;
                }

                bool processedByPicoater = cam.TryApplyPicoaterRidge(modifiedBuffer, cam._milProcBuffer);

                if (processedByPicoater)
                {
                    MIL.MbufCopy(cam._milProcBuffer, cam._milDisplayBuffer);
                }
                else
                {
                    MIL.MbufCopy(modifiedBuffer, cam._milDisplayBuffer);
                }

                cam.TrySaveCapture(modifiedBuffer);
            }

            cam.UpdateFps();
            return MIL.M_NULL;
        }

        private void ApplyDigitizerSettings()
        {
            if (MilDigitizer == MIL.M_NULL) return;

            if (CameraGrabHeight > 0)
            {
                MIL_INT height = (MIL_INT)CameraGrabHeight;
                MIL.MdigControl(MilDigitizer, MIL.M_SOURCE_SIZE_Y, height);
            }

            if (CameraExposureTimeUs > 0)
            {
                MIL_INT exposureControl = ResolveExposureControlType();
                if (exposureControl != MIL.M_NULL)
                {
                    MIL.MdigControl(MilDigitizer, exposureControl, CameraExposureTimeUs);
                }
                else
                {
                    TrySetExposureByFeature("ExposureTime", CameraExposureTimeUs);
                    TrySetExposureByFeature("ExposureTimeAbs", CameraExposureTimeUs);
                }
            }
        }

        private void TrySetExposureByFeature(string featureName, double value)
        {
            if (MilDigitizer == MIL.M_NULL) return;
            if (string.IsNullOrWhiteSpace(featureName)) return;

            try
            {
                MethodInfo[] methods = typeof(MIL).GetMethods(BindingFlags.Public | BindingFlags.Static);
                foreach (MethodInfo method in methods)
                {
                    if (!string.Equals(method.Name, "MdigControlFeature", StringComparison.Ordinal)) continue;

                    ParameterInfo[] ps = method.GetParameters();
                    if (ps.Length != 4) continue;

                    object[] args = new object[4];
                    args[0] = MilDigitizer;
                    args[1] = ResolveMilConstant("M_FEATURE_VALUE");
                    args[2] = featureName;
                    args[3] = value;
                    method.Invoke(null, args);
                    break;
                }
            }
            catch
            {
                // ignore if feature API is unavailable on this MIL version.
            }
        }

        private static object ResolveMilConstant(string name)
        {
            try
            {
                FieldInfo f = typeof(MIL).GetField(name, BindingFlags.Public | BindingFlags.Static);
                return f != null ? f.GetValue(null) : 0;
            }
            catch
            {
                return 0;
            }
        }

        private static MIL_INT ResolveExposureControlType()
        {
            if (_exposureControlResolved) return _exposureControlType;

            _exposureControlResolved = true;
            Type milType = typeof(MIL);
            foreach (string name in ExposureControlCandidates)
            {
                FieldInfo f = milType.GetField(name, BindingFlags.Public | BindingFlags.Static);
                if (f == null) continue;

                object raw = f.GetValue(null);
                if (raw == null) continue;

                try
                {
                    _exposureControlType = (MIL_INT)Convert.ToInt64(raw);
                    if (_exposureControlType != MIL.M_NULL) break;
                }
                catch
                {
                    // try next candidate
                }
            }

            return _exposureControlType;
        }

        public void ApplyAcquisitionSettings()
        {
            ApplyDigitizerSettings();
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
                ApplyDigitizerSettings();
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

                // [修改] 完整釋放主顯示區與 Hooks
                if (MilDisplay != MIL.M_NULL)
                {
                    MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    MIL.MdispHookFunction(MilDisplay, MIL.M_MOUSE_LEFT_BUTTON_DOWN + MIL.M_UNHOOK, _mouseClickDelegate, IntPtr.Zero);
                    MIL.MdispSelectWindow(MilDisplay, MIL.M_NULL, IntPtr.Zero);
                }

                // [新增] 完整釋放第二顯示區與 Hooks
                if (MilSecondaryDisplay != MIL.M_NULL)
                {
                    if (_isSecondaryHooked)
                    {
                        MIL.MdispHookFunction(MilSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                        _isSecondaryHooked = false;
                    }
                    MIL.MdispSelectWindow(MilSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);
                    MIL.MdispFree(MilSecondaryDisplay);
                    MilSecondaryDisplay = MIL.M_NULL;
                }

                for (int i = 0; i < _milGrabBufferListSize; i++)
                {
                    if (_milGrabBuffers[i] != MIL.M_NULL)
                    {
                        MIL.MbufFree(_milGrabBuffers[i]);
                        _milGrabBuffers[i] = MIL.M_NULL;
                    }
                }

                if (_milDisplayBuffer != MIL.M_NULL) { MIL.MbufFree(_milDisplayBuffer); _milDisplayBuffer = MIL.M_NULL; }
                if (_milProcBuffer != MIL.M_NULL) { MIL.MbufFree(_milProcBuffer); _milProcBuffer = MIL.M_NULL; }

                lock (_picoaterLock)
                {
                    if (_picoaterInputBuffer != IntPtr.Zero) { Marshal.FreeHGlobal(_picoaterInputBuffer); _picoaterInputBuffer = IntPtr.Zero; }
                    if (_picoaterRidgeBuffer != IntPtr.Zero) { Marshal.FreeHGlobal(_picoaterRidgeBuffer); _picoaterRidgeBuffer = IntPtr.Zero; }
                    _aoiService.Dispose();
                }

                _hostInputBuffer = null;
                _hostOutputBuffer = null;

                if (MilDisplay != MIL.M_NULL) { MIL.MdispFree(MilDisplay); MilDisplay = MIL.M_NULL; }
                MIL.MdigFree(MilDigitizer);
                MilDigitizer = MIL.M_NULL;
            }

            if (_hUserData.IsAllocated) _hUserData.Free();
        }

        private bool TryApplyPicoaterRidge(MIL_ID srcBuffer, MIL_ID dstBuffer)
        {
            if (srcBuffer == MIL.M_NULL || dstBuffer == MIL.M_NULL) return false;
            if (_frameWidth <= 0 || _frameHeight <= 0) return false;
            if (_hostInputBuffer == null || _hostOutputBuffer == null) return false;

            lock (_picoaterLock)
            {
                if (_picoaterInputBuffer == IntPtr.Zero || _picoaterRidgeBuffer == IntPtr.Zero)
                    return false;

                try
                {
                    MIL.MbufGet2d(srcBuffer, 0, 0, _frameWidth, _frameHeight, _hostInputBuffer);

                    Marshal.Copy(_hostInputBuffer, 0, _picoaterInputBuffer, _hostInputBuffer.Length);

                    _aoiService.ProcessImage(
                        _frameWidth,
                        _frameHeight,
                        _picoaterInputBuffer,
                        IntPtr.Zero,
                        IntPtr.Zero,
                        _picoaterRidgeBuffer,
                        IntPtr.Zero,
                        IntPtr.Zero,
                        2.0f,
                        (float)HessianSigma,
                        (float)HessianFixedMax,
                        "vertical",
                        IntPtr.Zero);

                    Marshal.Copy(_picoaterRidgeBuffer, _hostOutputBuffer, 0, _hostOutputBuffer.Length);
                    MIL.MbufPut2d(dstBuffer, 0, 0, _frameWidth, _frameHeight, _hostOutputBuffer);
                    return true;
                }
                catch
                {
                    return false;
                }
            }
        }

        private void TrySaveCapture(MIL_ID sourceBuffer)
        {
            if (!EnableAutoCapture) return;
            if (sourceBuffer == MIL.M_NULL) return;
            if (string.IsNullOrWhiteSpace(CaptureRootPath)) return;

            try
            {
                DateTime now = DateTime.Now;
                string secondKey = now.ToString("yyyyMMdd_HHmmss");
                if (string.Equals(_lastCaptureSecondKey, secondKey, StringComparison.Ordinal)) return;

                string yearFolder = now.ToString("yyyy");
                string yearMonthFolder = now.ToString("yyyyMM");
                string dayFolder = now.ToString("yyyyMMdd");
                string fileName = $"{now:yyyyMMdd_HHmmss}-{CameraId}.bmp";

                string saveDir = Path.Combine(CaptureRootPath, yearFolder, yearMonthFolder, dayFolder);
                Directory.CreateDirectory(saveDir);

                string fullPath = Path.Combine(saveDir, fileName);
                MIL.MbufExport(fullPath, MIL.M_BMP, sourceBuffer);

                _lastCaptureSecondKey = secondKey;
            }
            catch
            {
                // ignore save error
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
            else if (present && _userWantsGrab && !IsLive)
            {
                ApplyGrabState();
            }

            return MIL.M_NULL;
        }
    }
}