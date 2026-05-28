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
    /// <summary>AniloxRollForm 檢測數據 Tab 相關方法 — 由主檔拆出的 partial。</summary>
    public partial class AniloxRollForm
    {
        // ==========================================
        // --- 檢測數據 Tab ---
        // ==========================================

        private void SetupDataTab()
        {
            _dataStatsPresenter = new DataStatisticsPresenter(new DataStatisticsContext
            {
                CbStartDate = cbStartDate, CbStartTime = cbStartTime,
                CbEndDate = cbEndDate, CbEndTime = cbEndTime,
                CbGrabIdStart = cbGrabIdStart, CbGrabIdEnd = cbGrabIdEnd,
                CbDataGrabId = cbDataGrabId, CbReviewGrabId = cbReviewGrabId,
                BtnGrabIdPrev = btnGrabIdPrev, BtnGrabIdNext = btnGrabIdNext,
                BtnGrabIdDataPrev = btnGrabIdDataPrev, BtnGrabIdDataNext = btnGrabIdDataNext,
                BtnSelectDataFolder = btnSelectDataFolder, BtnShowFail = btnShowFail,
                GroupBoxGrabIdRange = groupBoxGrabIdRange, GrpDataSingleSheet = grpDataSingleSheet,
                GroupBoxTimeRange = groupBoxTimeRange,
                GrpReviewGrabNav = grpReviewGrabNav, GrpReviewTimePeriod = grpReviewTimePeriod,
                ListViewGrabDetail = listViewGrabDetail,
                PanelStatCams = new[] { panelStatCam1, panelStatCam2, panelStatCam3,
                                        panelStatCam4, panelStatCam5, panelStatCam6, panelStatCam7 },
                ChartMuraProfile = chartMuraProfile,
                ChartYearly = chartYearly, ChartMonthly = chartMonthly, ChartDaily = chartDaily,
                CbChartYear = cbChartYear, CbChartMonth = cbChartMonth, CbChartDay = cbChartDay,
                Settings = _settings, CameraCount = CameraCount,
            });
            _dataStatsPresenter.Initialize();

            // 延遲注入：_stitchCoordinator 在 InitUiLayer 初始化時 _dataStatsPresenter 尚未建立
            _stitchCoordinator.SetDataStatsPresenter(_dataStatsPresenter);

            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）——僅用於升序排列的 ComboBox
            foreach (var cb in new[] { cbChartYear, cbChartMonth, cbChartDay })
                _wheelInterceptors.Add(new ComboBoxWheelReverser(cb));

            // 跨 Tab 事件
            _dataStatsPresenter.GrabIdSelectedFromData += OnDataGrabIdSelected;
            _dataStatsPresenter.GrabIdSelectedFromReview += OnReviewGrabIdSelected;
            _dataStatsPresenter.PeriodComboManualChanged += OnPeriodComboChanged;
            _dataStatsPresenter.DataFolderSelected += OnDataFolderSelected;

            // Data → Review tab 切換時，若有 pending grabId 才載圖（避免 Data 操作中等待 IO）
            tabMain.SelectedIndexChanged += async (s, e) =>
            {
                if (tabMain.SelectedTab != tabPageReview || !_reviewDirty) return;
                int idx = cbReviewGrabId.SelectedIndex;
                if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
                _reviewDirty = false;
                var info = _dataStatsPresenter.GrabIdInfos[idx];
                try
                {
                    await _stitchCoordinator.LoadGrabStitchedViewAsync(info.GrabId, info.Earliest, info.Latest);
                    if (canvasMain.Image != null) canvasMain.FitToScreen();
                }
                catch (Exception ex) { Trace.WriteLine($"[tabMain → Review] {ex}"); }
            };
        }

        private void OnDataGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                using (_dataStatsPresenter.GrabIdCrossGuard.Enter())
                {
                    using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    {
                        cbReviewGrabId.SelectedIndex = idx;
                        _interactionHelper.NavigateToDateTime(earliest);
                    }
                    _presenter.UpdatePeriodNavigationState();
                    _dataStatsPresenter.UpdateGrabIdNavState();
                    _dataStatsPresenter.SetReviewGroupBoxes(true);
                    _reviewDirty = true;
                }
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataGrabIdSelected] {ex}"); }
        }

        private async void OnReviewGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                _interactionHelper.SaveCanvasView();
                using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    _interactionHelper.NavigateToDateTime(earliest);
                _presenter.UpdatePeriodNavigationState();

                await _stitchCoordinator.LoadGrabStitchedViewAsync(grabId, earliest, latest);
                if (canvasMain.Image != null) canvasMain.FitToScreen();
                _reviewDirty = false;

                // 同步 Data tab
                if (!_dataStatsPresenter.GrabIdCrossGuard.IsSet
                    && cbDataGrabId.Items.Count > 0 && idx < cbDataGrabId.Items.Count)
                {
                    var info = _dataStatsPresenter.GrabIdInfos[idx];
                    _dataStatsPresenter.SyncDataGrabIdFromReview(idx, info);
                }
            }
            catch (Exception ex) { Trace.WriteLine($"[OnReviewGrabIdSelected] {ex}"); }
        }

        private async void OnDataFolderSelected(string path)
        {
            try
            {
                // 同步 Review tab：先載入 ImageRepository + Navigator，再走共用 reset + 主畫面載入。
                UserSessionState.SetLastDataPath(path);
                UserSessionState.Save();
                _interactionHelper.LoadDirectoryAndInitNavigator(path);
                _presenter.UpdatePeriodNavigationState();
                // DataPresenter 已透過 LoadDataFolder 同步 _grabIdInfos，skip SyncFromReviewFolder
                await ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: true);
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataFolderSelected] {ex}"); }
        }
    }
}
