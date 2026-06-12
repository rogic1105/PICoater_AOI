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
    /// <summary>AniloxRollForm 方向 / ridge / 合圖模式切換相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        /// <summary>
        /// 切換 Live 顯示的 V/H 處理圖方向，點選 chartLiveVertical/HorizontalLive 觸發。
        /// 三態邏輯同 Review tab 的 SwitchRidgeDirection：
        /// 未勾選 → 自動勾選 + 設方向；同方向 → 取消勾選；不同方向 → 切換。
        /// </summary>
        private void SwitchLiveDisplayDirection(string dir)
        {
            // 未強化 → 開啟並設方向；強化中同方向 → 關閉；強化中不同方向 → 換方向（不改 setting）
            if (!_settings.EnableMuraEnhance)
            {
                _liveDisplayDirection = dir;
                _settingsHub.Set(s => s.hc_EnableMuraEnhance, true);   // event → ApplyMuraEnhance + UpdateLiveDirectionVisual
                return;
            }
            if (dir == _liveDisplayDirection)
            {
                _settingsHub.Set(s => s.hc_EnableMuraEnhance, false);
                return;
            }
            _liveDisplayDirection = dir;
            _liveCameraManager?.SetLiveDisplayDirection(dir);
            UpdateLiveDirectionVisual();
        }

        private void UpdateLiveDirectionVisual()
        {
            // 視覺規則（2026-06-12 改版）：藍底＝該方向強化圖顯示中；mode 底色 + 橘框已廢
            //（mode 雙 chart 切換器已隨舊單台切向 chart 刪除；StitchMode 走 PropertyGrid）。
            var enhanceBg = System.Drawing.Color.FromArgb(230, 240, 255);
            var normal    = System.Drawing.SystemColors.Control;
            string dir = (_settings?.EnableMuraEnhance == true) ? _liveDisplayDirection : null;

            chartLiveVertical.BackColor   = dir == "v" ? enhanceBg : normal;
            chartLiveHorizontal.BackColor = dir == "h" ? enhanceBg : normal;
            foreach (var c in new[] { chartLiveVertical, chartLiveHorizontal })
            {
                c.BorderlineColor = System.Drawing.Color.Transparent;
                c.BorderlineWidth = 1;
                c.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.NotSet;
            }
        }

        /// <summary>
        /// 切換 camReviewMain 的 V/H 處理圖方向，點選 chartReviewVertical/Horizontal 觸發。
        /// 未勾選強化圖時：自動勾選 + 設方向。
        /// 已勾選強化圖且點同方向：取消勾選（回原圖）。
        /// 已勾選強化圖且點不同方向：切換方向。
        /// </summary>
        private async void SwitchRidgeDirection(string dir)
        {
            try
            {
                // 未強化 → 開啟並設方向；強化中同方向 → 關閉；強化中不同方向 → 換方向（reload）
                if (!_stitchCoordinator.LastReviewProcessedMode)
                {
                    _stitchCoordinator.ActiveRidgeDirection = dir;
                    _interactionHelper.SetRidgeDirection(dir);
                    UpdateRidgeDirectionVisual(dir);
                    _settingsHub.Set(s => s.hd_EnableReviewEnhance, true);  // event → ApplyReviewEnhance(true)
                    return;
                }
                if (dir == _stitchCoordinator.ActiveRidgeDirection)
                {
                    UpdateRidgeDirectionVisual(null);
                    _settingsHub.Set(s => s.hd_EnableReviewEnhance, false); // event → ApplyReviewEnhance(false)
                    return;
                }
                // 不同方向：純 ridge dir 切換（沒有 setting 變更，直接 reload 處理圖）
                _stitchCoordinator.ActiveRidgeDirection = dir;
                _interactionHelper.SetRidgeDirection(dir);
                UpdateRidgeDirectionVisual(dir);
                _interactionHelper.SaveCanvasView();
                if (_stitchCoordinator.IsStitchMode)
                {
                    int idx = cbReviewId.SelectedIndex;
                    if (idx >= 0 && idx < _dataStatsPresenter.GrabIdInfos.Count)
                    {
                        var info = _dataStatsPresenter.GrabIdInfos[idx];
                        await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, true);
                    }
                }
                else
                {
                    _stitchCoordinator.ClearStitchedMode();
                    await _presenter.LoadImagesWithPeriodLockAsync(true, _interactionHelper.LoadImages);
                    ApplyPostLoadDisplay();
                }
            }
            catch (Exception ex) { Trace.WriteLine($"[SwitchRidgeDirection] {ex}"); }
        }

        private bool IsEnhanceDisplayActive =>
            _stitchCoordinator.IsStitchMode
                ? _settings.EnableReviewEnhance
                : _stitchCoordinator.LastReviewProcessedMode;

        /// <summary>
        /// Review tab：點對方 chart 切 StitchMode 時順便關 enhance。
        /// 對應 Live tab 的 SwitchStitchModeWithEnhanceSequence。
        /// 一次性從硬碟 reload 原圖 + 切 mode，避免「先用緩存 merge 顯示強化版、再 reload 顯示原圖」的兩段閃爍。
        /// </summary>
        private async Task SwitchReviewStitchModeAndDisableEnhance(StitchMode newMode)
        {
            if (_settings == null) return;
            bool wasStitchMode = _stitchCoordinator.IsStitchMode;

            // 之前用 camReviewMain.Visible=false 包 transition 想做 commit-on-end，但實測 reload 期間
            // canvas 整個消失（黑屏）幾百 ms 反而比中間幀更難看。改回直接 transition，由 ReloadCurrentStitchedView
            // 內 LoadGrabStitchedViewAsync 換 Image 時自然 paint 一次（短暫舊→新轉換可接受）。
            _settingsHub.SetBatch(s =>
            {
                s.EnableReviewEnhance = false;
                s.hb_StitchMode       = newMode;
            });
            _stitchCoordinator.LastReviewProcessedMode = false;
            UpdateRidgeDirectionVisual(null);
            RefreshGridItem(nameof(InspectionSettings.hd_EnableReviewEnhance));
            RefreshGridItem(nameof(InspectionSettings.hb_StitchMode));

            await OnStitchModeChangedAsync(skipStitchedImageRefresh: wasStitchMode);
            if (wasStitchMode && _stitchCoordinator.IsStitchMode)
            {
                await ReloadCurrentStitchedView(false);
                if (camReviewMain.Image != null) camReviewMain.FitToScreen();
            }
        }

        private async Task OnStitchModeChangedAsync(bool skipStitchedImageRefresh = false)
        {
            // 關程式時 fire-and-forget 的切換 async（_ = SwitchStitchMode...）可能在 Form/控制項
            // disposed 後續跑，碰 disposed 的 chart/canvas → NullReferenceException，故源頭早退。
            if (IsDisposed || Disposing) return;
            // Live tab：即時全域合圖
            if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsAllocated == true)
                _liveCameraManager.EnableGlobalMerge(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            else
            {
                _liveCameraManager?.DisableGlobalMerge();
                _liveRowMeanCache.Clear();
                _liveRowMaxCache.Clear();
            }

            if (_settings.StitchMode == StitchMode.Global)
            {
            }

            // 根據當前選中的回顧縮圖重新載入回顧主畫面
            if (_stitchCoordinator.IsStitchMode)
            {
                // skipStitchedImageRefresh=true：caller 自己會 LoadGrabStitchedViewAsync 重 load，
                // 跳過這裡的緩存 merge 避免「先顯示緩存再 reload」兩段閃爍。
                if (!skipStitchedImageRefresh)
                {
                    int idx = _galleryManager?.SelectedIndex ?? 0;
                    if (_settings.StitchMode == StitchMode.Global)
                        _stitchCoordinator.MergeAndShowFromStitchedImages();
                    else
                    {
                        _stitchCoordinator.DisposeGlobalMergedImage();
                        _stitchCoordinator.ShowStitchedCameraInCanvas(idx);
                    }
                }
            }
            else if (_imageRepository.FileCount > 0)
            {
                _stitchCoordinator.ClearStitchedMode();
                await _presenter.LoadImagesWithPeriodLockAsync(
                    _stitchCoordinator.LastReviewProcessedMode, _interactionHelper.LoadImages);
                ApplyPostLoadDisplay();
            }

            // 重繪縮圖外框（切換 StitchMode 後橘框隨之更新）
            foreach (var pb in _cameraPanels) pb.Invalidate();

            // 切換合圖方式後主畫面 fit to screen
            // skipStitchedImageRefresh=true：caller 自己會 reload 新原圖再 fit，這裡 fit 會作用在舊強化版 image 上，跳過。
            if (!skipStitchedImageRefresh && camReviewMain.Image != null)
                camReviewMain.FitToScreen();

            // 底色（藍）依 StitchMode；橘框依強化狀態
            UpdateRidgeDirectionVisual(
                IsEnhanceDisplayActive ? _stitchCoordinator.ActiveRidgeDirection : null);
            UpdateLiveDirectionVisual();
        }

        private void UpdateRidgeDirectionVisual(string dir)
        {
            // 視覺規則（2026-06-12 改版）：藍底＝該方向強化圖顯示中；mode 底色 + 橘框已廢（同 Live）。
            var enhanceBg = System.Drawing.Color.FromArgb(230, 240, 255);
            var normal    = System.Drawing.SystemColors.Control;

            chartReviewVertical.BackColor = dir == "v" ? enhanceBg : normal;
            if (chartReviewHorizontal != null)
                chartReviewHorizontal.BackColor = dir == "h" ? enhanceBg : normal;
            foreach (var c in new[] { chartReviewVertical, chartReviewHorizontal })
            {
                if (c == null) continue;
                c.BorderlineColor = System.Drawing.Color.Transparent;
                c.BorderlineWidth = 1;
                c.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.NotSet;
            }
        }
    }
}
