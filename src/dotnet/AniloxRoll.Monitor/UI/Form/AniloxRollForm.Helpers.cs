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
    /// <summary>AniloxRollForm 輔助方法（PG refresh / 相機參數同步 / Review 座標 / 通用）— 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- Hardware → UI 反向同步 ---
        // ==========================================

        /// <summary>
        /// 每次 Telemetry Timer Tick 呼叫。從相機硬體讀回曝光與線掃速率，
        /// 若差異超過 5% 且使用者未拖曳，則更新 TrackBar/NUD（帶 hysteresis 防抖）。
        /// </summary>



        /// <summary>
        /// 精準 refresh 單一 PropertyGrid cell — 利用 PG 內建「SelectedGridItem 改變時 force 重讀 value」的行為。
        /// 對比 Refresh() 整個 re-build grid items 造成閃爍，這個 trick 只動單 cell、不閃。
        /// 來源：使用者觀察「點監控強化標題會更新」揭露的 PG 內建 mechanism。
        /// </summary>
        private void RefreshGridItem(string propertyName)
        {
            if (string.IsNullOrEmpty(propertyName)) return;
            if (propertyGridSettings == null) return;
            GridItem root = propertyGridSettings.SelectedGridItem;
            while (root?.Parent != null) root = root.Parent;
            if (root == null) return;
            GridItem found = FindGridItemRecursive(root, propertyName);
            if (found == null) return;  // PG 不顯示此 property（Browsable false）— 無需 refresh

            // 保 scroll position：set SelectedGridItem 會 trigger PG auto-scroll 到該 cell（若不在 viewport 內）。
            // 找內部 PropertyGridView + VScrollBar，trick 完恢復原 scroll，避免使用者看到 PG 跳動。
            Control gridView = null;
            foreach (Control c in propertyGridSettings.Controls)
                if (c.GetType().Name == "PropertyGridView") { gridView = c; break; }
            System.Windows.Forms.ScrollBar scrollBar = null;
            if (gridView != null)
                foreach (Control c in gridView.Controls)
                    if (c is System.Windows.Forms.VScrollBar) { scrollBar = (System.Windows.Forms.VScrollBar)c; break; }
            int scrollPos = scrollBar?.Value ?? 0;

            var saved = propertyGridSettings.SelectedGridItem;
            _suppressGridSelChange = true;
            try
            {
                propertyGridSettings.SelectedGridItem = found;
                if (saved != null && saved != found)
                    propertyGridSettings.SelectedGridItem = saved;
                if (scrollBar != null)
                {
                    int max = Math.Max(0, scrollBar.Maximum - scrollBar.LargeChange + 1);
                    scrollBar.Value = Math.Max(0, Math.Min(scrollPos, max));
                }
            }
            finally { _suppressGridSelChange = false; }
        }

        private static GridItem FindGridItemRecursive(GridItem parent, string name)
        {
            if (parent == null) return null;
            foreach (GridItem c in parent.GridItems)
            {
                if (c.PropertyDescriptor?.Name == name) return c;
                var sub = FindGridItemRecursive(c, name);
                if (sub != null) return sub;
            }
            return null;
        }

        /// <summary>
        /// 全 PG refresh + 保 scroll — fallback 用，極少場景才呼（多 setting 同時變且無法精準定位）。
        /// 雙重 WM_SETREDRAW 凍結減少閃爍。
        /// </summary>
        private void RefreshPropertyGridKeepScroll()
        {
            const int WM_SETREDRAW = 0x000B;
            Control gridView = null;
            foreach (Control c in propertyGridSettings.Controls)
                if (c.GetType().Name == "PropertyGridView") { gridView = c; break; }
            if (gridView == null) { propertyGridSettings.Refresh(); return; }
            System.Windows.Forms.ScrollBar scrollBar = null;
            foreach (Control c in gridView.Controls)
                if (c is System.Windows.Forms.VScrollBar) { scrollBar = (System.Windows.Forms.VScrollBar)c; break; }
            int scrollPos = scrollBar?.Value ?? 0;
            propertyGridSettings.SuspendLayout();
            NativeMethods.SendMessage(propertyGridSettings.Handle, WM_SETREDRAW, IntPtr.Zero, IntPtr.Zero);
            NativeMethods.SendMessage(gridView.Handle, WM_SETREDRAW, IntPtr.Zero, IntPtr.Zero);
            try
            {
                propertyGridSettings.Refresh();
                if (scrollBar != null)
                {
                    int max = Math.Max(0, scrollBar.Maximum - scrollBar.LargeChange + 1);
                    scrollBar.Value = Math.Max(0, Math.Min(scrollPos, max));
                }
            }
            finally
            {
                NativeMethods.SendMessage(gridView.Handle, WM_SETREDRAW, new IntPtr(1), IntPtr.Zero);
                NativeMethods.SendMessage(propertyGridSettings.Handle, WM_SETREDRAW, new IntPtr(1), IntPtr.Zero);
                propertyGridSettings.ResumeLayout(false);
                propertyGridSettings.Invalidate(true);
            }
        }

        // ── TrackBar 滾輪：每格僅移動 1 ──────────────────────────────────
        private void RegisterWheelInterceptors(TrackBar[] bars)
        {
            foreach (var bar in bars)
                _wheelInterceptors.Add(new TrackBarWheelInterceptor(bar));
        }


        private void SyncCameraParamsFromHardware()
        {
            if (_expBars == null || _lrBars == null) return;

            var cameras = _liveCameraManager.Cameras;
            var acq     = _settings?.Acquisition;
            if (acq == null) return;

            for (int idx = 0; idx < CameraCount; idx++)
            {
                try
                {
                    var cam = FindCameraById(idx + 1);
                    if (cam == null) continue;
                    if (!cam.IsHwParamsStable) continue;

                    SyncHardwareParam(_expBars[idx], _expNums[idx],
                        cam.GetMeasuredExposureUs(), v => acq.CameraExposureTimeUs[idx] = v);

                    SyncHardwareParam(_lrBars[idx], _lrNums[idx],
                        cam.GetLineRateHz(), v => acq.CameraLineRateHz[idx] = v);
                }
                catch (Exception ex) { Debug.WriteLine($"[SyncHw] CAM{idx + 1}: {ex.Message}"); }
            }
        }

        // ── Helper Methods ──────────────────────────────────────────

        private void SyncHardwareParam(TrackBar bar, NumericUpDown nud, double hwValue, Action<int> saveSetting)
        {
            if (_dragging.Contains(bar) || hwValue <= 0) return;
            int clamped = Math.Max(bar.Minimum, Math.Min(bar.Maximum, (int)hwValue));
            double diff = Math.Abs(clamped - bar.Value) / (double)Math.Max(1, bar.Value);
            if (diff <= 0.05) return;
            _syncingFromHw = true;
            bar.Value = clamped;
            nud.Value = clamped;
            saveSetting(clamped);
            _syncingFromHw = false;
        }

        /// <summary>
        /// Live overview 用：Global 模式從合併 display 取視野，否則返回 NaN（Vertical 模式 X 軸固定）。
        /// </summary>
        private double LiveViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            if (_liveCameraManager?.IsGlobalMergeActive == true &&
                _liveCameraManager.TryGetMergedViewRange(out double left, out double right))
                return isLeft ? left : right;
            return defaultValue;
        }

        /// <summary>
        /// CurveMergeHelper 用的 viewRange 代理：將 TryComputeCurrentViewRange 包裝為 Func。
        /// </summary>
        private double ViewRangeProvider(int cameraIndex, bool isLeft, double defaultValue)
        {
            if (_interactionHelper == null) return defaultValue;
            if (!_interactionHelper.TryComputeCurrentViewRange(cameraIndex, out double left, out double right))
                return defaultValue;
            return isLeft ? left : right;
        }

        private AniloxCamera FindCameraById(int camId)
        {
            if (_liveCameraManager?.Cameras == null) return null;
            foreach (var c in _liveCameraManager.Cameras)
                if (c.CameraId == camId) return c;
            return null;
        }

        // ── 回顧縮圖↔主畫面雙向同步（Global 模式）──────────────────

        private double[] GetReviewOpsArray() =>
            _interactionHelper?.ReviewConfig?.CamOps ?? _settings?.GetCameraOpsUmArray() ?? new double[7];

        private double[] GetReviewPosArray() =>
            _interactionHelper?.ReviewConfig?.CamPos ?? _settings?.GetCameraStartPositionMmArray() ?? new double[7];

        private bool TryGetMergedReviewCoords(out double globalMinMm, out double refOpsMm)
        {
            globalMinMm = 0; refOpsMm = 0;
            var opsArr = GetReviewOpsArray();
            var posArr = GetReviewPosArray();
            if (opsArr == null || opsArr.Length == 0 || opsArr[0] <= 0) return false;
            globalMinMm = double.MaxValue;
            for (int i = 0; i < posArr.Length; i++)
                if (posArr[i] < globalMinMm) globalMinMm = posArr[i];
            if (globalMinMm == double.MaxValue) { globalMinMm = 0; }
            refOpsMm = opsArr[0] * InspectionEngineConfig.DefaultSaveResizeScale / 1000.0;
            return refOpsMm > 0;
        }

        private void PanCanvasToReviewCameraCenter(int camIdx)
        {
            if (!_stitchCoordinator.IsGlobalMerged && !_stitchCoordinator.IsPeriodMerged) return;
            if (!TryGetMergedReviewCoords(out double globalMinMm, out double refOpsMm)) return;
            var posArr = GetReviewPosArray();
            var opsArr = GetReviewOpsArray();
            if (camIdx < 0 || camIdx >= posArr.Length) return;

            double slotWidthMm = InspectionEngineConfig.MaxWidth * opsArr[camIdx] / 1000.0;
            double camCenterMm = posArr[camIdx] + slotWidthMm / 2.0;
            double camCenterPx = (camCenterMm - globalMinMm) / refOpsMm;
            float newPanX = canvasMain.Width / 2.0f - (float)(camCenterPx * canvasMain.Zoom);
            canvasMain.SetView(canvasMain.Zoom, new System.Drawing.PointF(newPanX, canvasMain.PanOffset.Y));
        }

        private void UpdateSelectedReviewCamFromViewCenter(CanvasInfo info)
        {
            if (_settings?.StitchMode != StitchMode.Global) return;
            if (!TryGetMergedReviewCoords(out double globalMinMm, out double refOpsMm)) return;
            var posArr = GetReviewPosArray();
            var opsArr = GetReviewOpsArray();

            double centerPx = (canvasMain.Width / 2.0f - info.PanOffset.X) / info.Zoom;
            double centerMm = globalMinMm + centerPx * refOpsMm;

            int bestIdx = 0;
            double bestDist = double.MaxValue;
            for (int i = 0; i < posArr.Length; i++)
            {
                double slotWidthMm = InspectionEngineConfig.MaxWidth * opsArr[i] / 1000.0;
                double slotCenterMm = posArr[i] + slotWidthMm / 2.0;
                double dist = Math.Abs(centerMm - slotCenterMm);
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }

            if (bestIdx == _galleryManager.SelectedIndex) return;
            _galleryManager.Select(bestIdx, triggerEvent: false);
        }

        // ── Helper Methods ──────────────────────────────────────────

        private static void AutoFitListViewColumns(ListView lv)
        {
            for (int i = 0; i < lv.Columns.Count; i++)
            {
                lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.ColumnContent);
                int contentWidth = lv.Columns[i].Width;
                lv.AutoResizeColumn(i, ColumnHeaderAutoResizeStyle.HeaderSize);
                if (contentWidth > lv.Columns[i].Width)
                    lv.Columns[i].Width = contentWidth;
            }
        }

        // ── Inner Classes ───────────────────────────────────────────

        /// <summary>
        /// Storage 模式 PropertyGrid 過濾器：隱藏 IO / 相機 / 光源三個大類。
        /// 使用 TypeDescriptor instance-level provider，不影響 Inspection 模式。
        /// </summary>
        private sealed class StorageModeSettingsFilter : TypeDescriptionProvider
        {
            private static readonly HashSet<string> Hidden = new HashSet<string>
            {
                "5. IO 模組設定",
                "6. 相機參數設定",
                "7. 光源設定"
            };

            public StorageModeSettingsFilter(TypeDescriptionProvider parent) : base(parent) { }

            public override ICustomTypeDescriptor GetTypeDescriptor(Type objectType, object instance)
                => new FilteredDescriptor(base.GetTypeDescriptor(objectType, instance));

            private sealed class FilteredDescriptor : CustomTypeDescriptor
            {
                public FilteredDescriptor(ICustomTypeDescriptor parent) : base(parent) { }

                public override PropertyDescriptorCollection GetProperties()
                    => Filter(base.GetProperties());
                public override PropertyDescriptorCollection GetProperties(Attribute[] attributes)
                    => Filter(base.GetProperties(attributes));

                private static PropertyDescriptorCollection Filter(PropertyDescriptorCollection all)
                {
                    var visible = all.Cast<PropertyDescriptor>()
                        .Where(p => !Hidden.Contains(p.Category))
                        .ToArray();
                    return new PropertyDescriptorCollection(visible);
                }
            }
        }

        private bool IsCanvasFitToScreen()
        {
            if (canvasMain.Image == null) return false;
            float ratioW = (float)canvasMain.Width / canvasMain.Image.Width;
            float ratioH = (float)canvasMain.Height / canvasMain.Image.Height;
            float fitZoom = Math.Min(ratioW, ratioH) * 0.95f;
            if (Math.Abs(canvasMain.Zoom - fitZoom) > 0.001f) return false;

            float drawW = canvasMain.Image.Width * fitZoom;
            float drawH = canvasMain.Image.Height * fitZoom;
            float fitPanX = (canvasMain.Width - drawW) / 2f;
            float fitPanY = (canvasMain.Height - drawH) / 2f;
            var pan = canvasMain.PanOffset;
            return Math.Abs(pan.X - fitPanX) < 1f && Math.Abs(pan.Y - fitPanY) < 1f;
        }
    }
}
