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
    /// <summary>AniloxRollForm 回顧（資料夾/時段載入、回顧強化）相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        private async void btnReviewSelectFolder_Click(object sender, EventArgs e)
        {
            try
            {
                _interactionHelper.SelectAndLoadFolder();
                _presenter.UpdatePeriodNavigationState();
                await ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: false);
            }
            catch (Exception ex) { Trace.WriteLine($"[btnReviewSelectFolder_Click] {ex}"); }
        }

        /// <summary>
        /// 載入 Anilox 資料夾後共用的 Review 重置 + 主畫面載入：
        /// state reset（合圖方式=全域、回顧強化=否）、Live merge sync + chart clear、
        /// DataPresenter 同步、Review 主畫面載入。
        /// btnReviewSelectFolder（Review tab）跟 OnDataFolderSelected（Data tab 觸發）共用。
        /// </summary>
        private async Task ResetAndLoadReviewAfterFolderChanged(bool dataPresenterAlreadySynced)
        {
            _stitchCoordinator.LastReviewProcessedMode = false;
            _settingsHub.SetBatch(s =>
            {
                s.EnableReviewEnhance = false;
                s.hb_StitchMode       = StitchMode.Global;
            });
            RefreshGridItem(nameof(InspectionSettings.hd_EnableReviewEnhance));
            RefreshGridItem(nameof(InspectionSettings.hb_StitchMode));

            // Live tab 副作用（SetBatch 沒 raise event，手動同步）
            if (_settings.StitchMode == StitchMode.Global && _liveCameraManager?.IsAllocated == true)
                _liveCameraManager.EnableGlobalMerge(
                    _settings.GetCameraOpsUmArray(), _settings.GetCameraStartPositionMmArray());
            else
                _liveCameraManager?.DisableGlobalMerge();
            if (_settings.StitchMode == StitchMode.Global)
            {
            }
            UpdateLiveDirectionVisual();

            // Data tab 已 LoadDataFolder 時跳過 SyncFromReviewFolder（避免 duplicate load）。
            // SyncGrabIdFromTime 兩條路徑都需要（保持 DataPresenter 內部 _grabIdInfos 對齊 navigator 當前 period）。
            if (_imageRepository.FileCount > 0)
            {
                if (!dataPresenterAlreadySynced)
                {
                    var reviewPath = UserSessionState.LastDataPath;
                    if (!string.IsNullOrWhiteSpace(reviewPath))
                        _dataStatsPresenter.SyncFromReviewFolder(reviewPath);
                }
                var current = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
                if (current != DateTime.MinValue)
                    _dataStatsPresenter.SyncGrabIdFromTime(current);
            }

            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(true);
            _dataStatsPresenter.SelectLatestInSingleSheetMode();

            // 預設 grpReviewGrabNav（單片序號模式）→ 直接 LoadGrabStitchedViewAsync
            int reviewIdx = cbReviewId.SelectedIndex;
            if (reviewIdx >= 0 && reviewIdx < _dataStatsPresenter.GrabIdInfos.Count)
            {
                var info = _dataStatsPresenter.GrabIdInfos[reviewIdx];
                await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest);
                _reviewDisplayManager?.RefireViewRange();   // 同上：載入完恢復曲線視野跟隨
                if (camReviewMain.Image != null) camReviewMain.FitToScreen();
                _reviewDirty = false;
            }
            else
            {
                // 資料夾無序號 → period 模式 fallback
                await _presenter.LoadImagesWithPeriodLockAsync(false, LoadImagesWithReviewConfig);
                ApplyPostLoadDisplay();
            }
        }

        private async Task ApplyReviewEnhance(bool enableProcess)
        {
            try
            {
            UpdateRidgeDirectionVisual(enableProcess ? _stitchCoordinator.ActiveRidgeDirection : null);
            if (_stitchCoordinator.IsStitchMode)
            {
                await ReloadCurrentStitchedView(enableProcess);
                return;
            }
            _stitchCoordinator.LastReviewProcessedMode = enableProcess;
            _stitchCoordinator.ClearStitchedMode();
            await _presenter.LoadImagesWithPeriodLockAsync(enableProcess, _interactionHelper.LoadImages);
            ApplyPostLoadDisplay();
            }
            catch (Exception ex) { Trace.WriteLine($"[ApplyReviewEnhance] {ex}"); }
        }

        private async Task ReloadCurrentStitchedView(bool enableProcess)
        {
            int idx = cbReviewId.SelectedIndex;
            if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
            _interactionHelper.SaveCanvasView();
            var info = _dataStatsPresenter.GrabIdInfos[idx];
            await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest, enableProcess);
            _reviewDisplayManager?.RefireViewRange();   // chart 重建會重設軸 → 補發當前視野（不用等滑鼠動）
        }

        /// <summary>
        /// 從當前 Period 日期的 CSV 載入 #CFG，更新 ReviewConfig。
        /// 應在每次 Period 切換或資料夾載入後呼叫。
        /// </summary>
        private void RefreshReviewConfigForCurrentPeriod()
        {
            string rootPath = UserSessionState.LastDataPath;
            if (string.IsNullOrWhiteSpace(rootPath)) { _interactionHelper.ReviewConfig = null; return; }

            var periodDate = _dateTimeNavigator.GetCurrentPeriodOrDefault(DateTime.MinValue);
            if (periodDate == DateTime.MinValue) { _interactionHelper.ReviewConfig = null; return; }

            var cfg = InspectionStatisticsService.LoadConfigForDate(rootPath, periodDate);
            _interactionHelper.ReviewConfig = cfg;
        }

        /// <summary>
        /// 包裝 LoadImages：先刷新 ReviewConfig（navigator 已指向新日期），再載入影像。
        /// 確保 OnGallerySelectionChanged 觸發時 ReviewConfig 已是正確的 CFG。
        /// </summary>
        private async Task LoadImagesWithReviewConfig(bool enableProcess)
        {
            RefreshReviewConfigForCurrentPeriod();
            await _interactionHelper.LoadImages(enableProcess);
        }

        /// <summary>
        /// 載入影像後，根據 StitchMode 決定顯示方式：
        /// Vertical → 觸發 gallery 選取顯示單台影像；Global → 合圖顯示。
        /// 最後更新全覽圖。
        /// </summary>
        private void ApplyPostLoadDisplay()
        {
            // 永遠 Global：時序路徑顯示走 LiveDisplayView（ApplyGlobalMergeIfNeeded 發 StitchedImagesReady）。
            _stitchCoordinator.ApplyGlobalMergeIfNeeded();
            _stitchCoordinator.UpdateOverviewChartFromRepository();
            _reviewDisplayManager?.RefireViewRange();   // chart 重建會重設軸 → 補發當前視野
        }



        /// <summary>cbReviewDate/cbReviewTime 手動滾動時載入對應圖片（同 btnReviewPeriodPrev/Next）。
        /// _dataStatsPresenter.GrabIdNavGuard 時跳過（由 OnReviewGrabIdChanged 等程式碼觸發的 NavigateToDateTime）。</summary>
        private async void OnPeriodComboChanged()
        {
            if (_dataStatsPresenter.GrabIdNavGuard.IsSet) return;
            if (_imageRepository.FileCount == 0) return;
            try
            {
            bool wasStitch = _stitchCoordinator.IsStitchMode;
            _interactionHelper.SaveCanvasView();
            _stitchCoordinator.ClearStitchedMode();
            _dataStatsPresenter.SetReviewGroupBoxes(false);
            await _presenter.LoadImagesWithPeriodLockAsync(_stitchCoordinator.LastReviewProcessedMode, LoadImagesWithReviewConfig);
            ApplyPostLoadDisplay();
            if (wasStitch && camReviewMain.Image != null) camReviewMain.FitToScreen();
            }
            catch (Exception ex) { Trace.WriteLine($"[OnPeriodComboChanged] {ex}"); }
        }
    }
}
