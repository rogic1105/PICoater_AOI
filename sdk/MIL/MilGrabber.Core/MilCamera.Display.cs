using System;
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    // MilCamera 的「顯示 / 緩衝 I/O」分區：host↔MIL buffer 複製、主/副顯示綁定與 zoom/pan、滑鼠 hook（射座標事件）。
    // 與核心生命週期分檔；共用 _milDisplay / _milSecondaryDisplay / _milDisplayBuffer / _panelHandle 等私有欄位與事件。
    public partial class MilCamera
    {
        // ==================== Buffer Helpers（給上層檢測 / 顯示用） ====================

        /// <summary>把指定 MIL buffer 的影像複製到 host byte[]（長度需 ≥ FrameWidth*FrameHeight）。</summary>
        public void GetFrameBytes(MIL_ID buffer, byte[] dst)
        {
            if (buffer == MIL.M_NULL || dst == null) return;
            MIL.MbufGet2d(buffer, 0, 0, FrameWidth, FrameHeight, dst);
        }

        /// <summary>把 host byte[] 寫入顯示 buffer（顯示處理結果）。</summary>
        public void PutDisplayBytes(byte[] src)
        {
            if (_milDisplayBuffer == MIL.M_NULL || src == null) return;
            MIL.MbufPut2d(_milDisplayBuffer, 0, 0, FrameWidth, FrameHeight, src);
        }

        /// <summary>把指定 MIL buffer 複製到顯示 buffer（顯示原圖或上層處理後的 MIL buffer）。</summary>
        public void CopyToDisplay(MIL_ID src)
        {
            if (src == MIL.M_NULL || _milDisplayBuffer == MIL.M_NULL) return;
            MIL.MbufCopy(src, _milDisplayBuffer);
        }

        /// <summary>清空顯示 buffer（填黑）。停 grab 後 displayBuffer 殘留最後一幀，重新綁定顯示前清掉避免顯示殘影。</summary>
        public void ClearDisplay()
        {
            if (_milDisplayBuffer != MIL.M_NULL) MIL.MbufClear(_milDisplayBuffer, 0);
        }

        // ==================== Primary Display ====================

        /// <summary>Detach / restore 主顯示（visible=false → MdispSelectWindow(M_NULL)）。</summary>
        public void SetPrimaryDisplayVisible(bool visible)
        {
            if (_milDisplay == MIL.M_NULL) return;
            if (visible)
            {
                if (_panelHandle != IntPtr.Zero && _milDisplayBuffer != MIL.M_NULL)
                {
                    MIL.MdispSelectWindow(_milDisplay, _milDisplayBuffer, _panelHandle);
                    MIL.MdispControl(_milDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                }
            }
            else
            {
                MIL.MdispSelectWindow(_milDisplay, MIL.M_NULL, IntPtr.Zero);
            }
        }

        // ==================== Secondary Display ====================

        public void SetSecondaryDisplay(IntPtr handle)
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;

            if (handle == IntPtr.Zero)
            {
                if (_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, _mouseStatusDelegate, IntPtr.Zero);
                    _isSecondaryHooked = false;
                }
                MIL.MdispSelectWindow(_milSecondaryDisplay, MIL.M_NULL, IntPtr.Zero);
            }
            else
            {
                MIL.MdispSelectWindow(_milSecondaryDisplay, _milDisplayBuffer, handle);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_MOUSE_USE, MIL.M_ENABLE);

                if (!_isSecondaryHooked)
                {
                    MIL.MdispHookFunction(_milSecondaryDisplay, MIL.M_MOUSE_MOVE, _mouseStatusDelegate, (IntPtr)CameraId);
                    _isSecondaryHooked = true;
                }
            }
        }

        /// <summary>查詢副顯示 zoom/pan 狀態（隨使用者滾輪改變）。</summary>
        public bool TryGetSecondaryDisplayGeometry(out double zoomX, out double zoomY, out double panX, out double panY)
        {
            zoomX = zoomY = panX = panY = 0;
            if (_milSecondaryDisplay == MIL.M_NULL) return false;
            try
            {
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_ZOOM_FACTOR_X, ref zoomX);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_PAN_OFFSET_X, ref panX);
                MIL.MdispInquire(_milSecondaryDisplay, MIL.M_PAN_OFFSET_Y, ref panY);
                return zoomX > 0 && zoomY > 0;
            }
            catch { return false; }
        }

        /// <summary>設定副顯示縮放/平移（M_UPDATE DISABLE/ENABLE 批次，避免閃爍）。</summary>
        public void SetSecondaryDisplayZoom(double zoom, double panX, double panY)
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;
            try
            {
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_UPDATE, MIL.M_DISABLE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_DISABLE);
                MIL.MdispZoom(_milSecondaryDisplay, zoom, zoom);
                MIL.MdispPan(_milSecondaryDisplay, panX, panY);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_UPDATE, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[MilCamera.SetSecondaryDisplayZoom] {ex.GetType().Name}: {ex.Message}"); }
        }

        /// <summary>重置副顯示縮放/平移為 fit-to-window。</summary>
        public void ResetSecondaryDisplayView()
        {
            if (_milSecondaryDisplay == MIL.M_NULL) return;
            try
            {
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);
                MIL.MdispControl(_milSecondaryDisplay, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);
            }
            catch (Exception ex) { System.Diagnostics.Trace.TraceWarning($"[MilCamera.ResetSecondaryDisplayView] {ex.GetType().Name}: {ex.Message}"); }
        }

        // ==================== Mouse Hooks（MIL 機制，射出座標事件給上層） ====================

        private MIL_INT MouseClickHandler(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (_isReleased) return MIL.M_NULL;
            OnCameraClicked?.Invoke(CameraId);
            return MIL.M_NULL;
        }

        private MIL_INT MouseStatusHandler(MIL_INT hookType, MIL_ID eventId, IntPtr userPtr)
        {
            if (_isReleased || _milDisplayBuffer == MIL.M_NULL) return MIL.M_NULL;

            double posX = 0, posY = 0;
            MIL.MdispGetHookInfo(eventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
            MIL.MdispGetHookInfo(eventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);

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
    }
}
