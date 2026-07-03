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

        /// <summary>檢測報表 Y 軸 setting（gb 模式 / gc/gd/ge 各週期 YMax）→ 套到 Data 統計 charts。
        /// （Wave3 選項1：從 OnSettingChanged dispatcher 搬入。）</summary>
        private void HandleChartScaleSettingsChanged(string name)
        {
            if (name == nameof(InspectionSettings.gb_ChartScaleMode))
                _dataStatsPresenter.ApplyChartScaleFromSettings();
            else if (name == nameof(InspectionSettings.gc_YearlyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Yearly", _settings.Chart.YearlyYMax);
            else if (name == nameof(InspectionSettings.gd_MonthlyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Monthly", _settings.Chart.MonthlyYMax);
            else if (name == nameof(InspectionSettings.ge_DailyYMax))
                _dataStatsPresenter.ApplyFixedScaleForChart("Daily", _settings.Chart.DailyYMax);
        }

        /// <summary>檢測參數變更 → 重畫 chartDataColumn 曲線 + 重算統計。
        /// **只在真正影響 Data 曲線/Pass-Fail 的參數才跑**（正規值 V/H、檢出方向、V/H 平均/最大閾值）；
        /// 其餘設定（IO/光源/儲存/DO_MURA 暫停/主畫面顯示/LOD/合圖…）不動 Data 曲線，避免無關設定
        /// 觸發 chartDataColumn reload+重綁造成「閃一下再復原」。細線濾除(de_RidgeSigma)是 capture-time、
        /// 不改已存 .bin，故不列入。（Wave3：Data 副作用從 OnSettingChanged 共用區搬入本 feature handler。）</summary>
        private void HandleDataStatsSettingsChanged(string name)
        {
            switch (name)
            {
                case nameof(InspectionSettings.dc_HessianMaxFactorV):
                case nameof(InspectionSettings.dd_HessianMaxFactorH):
                case nameof(InspectionSettings.eb_RidgeDir):
                case nameof(InspectionSettings.ec_ErrorValueMeanV):
                case nameof(InspectionSettings.ed_ErrorValueMaxV):
                case nameof(InspectionSettings.ee_ErrorValueMeanH):
                case nameof(InspectionSettings.ef_ErrorValueMaxH):
                    _dataStatsPresenter?.RefreshMuraProfileForSettingsChange();  // 立即重畫曲線（坡度/閾值線即時回饋）
                    ScheduleStatsRefresh();                                      // debounce 重算 Pass/Fail 統計 + 明細
                    break;
            }
        }

        private void SetupDataTab()
        {
            _dataStatsPresenter = new DataStatisticsPresenter(new DataStatisticsContext
            {
                CbGrabIdStart = cbDataIdStart, CbGrabIdEnd = cbDataIdEnd,
                CbDataGrabId = cbDataId, CbReviewGrabId = cbReviewId,
                BtnSelectDataFolder = btnDataSelectFolder, BtnShowFail = btnDataShowFail,
                GroupBoxGrabIdRange = groupBoxGrabIdRange, GrpDataSingleSheet = grpDataSingleSheet,
                GrpReviewGrabNav = grpReviewGrabNav, GrpReviewTimePeriod = grpReviewTimePeriod,
                ListViewGrabDetail = listViewGrabDetail,
                PanelStatCams = new[] { camData1, camData2, camData3,
                                        camData4, camData5, camData6, camData7 },
                ChartDataPatch = chartDataColumn,
                ChartDataYieldYearly = chartDataYieldYearly, ChartDataYieldMonthly = chartDataYieldMonthly, ChartDataYieldDaily = chartDataYieldDaily,
                CbChartYear = cbDataYieldYear, CbChartMonth = cbDataYieldMonth, CbChartDay = cbDataYieldDay,
                LblChartNavYear = lblChartNavYear, LblChartNavMonth = lblChartNavMonth, LblChartNavDay = lblChartNavDay,
                Settings = _settings, CameraCount = CameraCount,
            });

            // 年/月/日 label 做成「看起來可點」的浮雕小晶片（Fixed3D 外框 + 手指游標）；點擊行為由 navigator 接
            foreach (var lbl in new[] { lblChartNavYear, lblChartNavMonth, lblChartNavDay })
            {
                lbl.BorderStyle = BorderStyle.Fixed3D;
                lbl.Cursor = Cursors.Hand;
                lbl.Padding = new Padding(6, 2, 6, 2);
                lbl.TextAlign = ContentAlignment.MiddleCenter;
            }

            _dataStatsPresenter.Initialize();

            // 延遲注入：_stitchCoordinator 在 InitUiLayer 初始化時 _dataStatsPresenter 尚未建立
            _stitchCoordinator.SetDataStatsPresenter(_dataStatsPresenter);

            // 滾輪上滾 = 數值增加（反轉 ComboBox 預設行為）——僅用於升序排列的 ComboBox
            foreach (var cb in new[] { cbDataYieldYear, cbDataYieldMonth, cbDataYieldDay })
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
                int idx = cbReviewId.SelectedIndex;
                if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
                _reviewDirty = false;
                var info = _dataStatsPresenter.GrabIdInfos[idx];
                try
                {
                    await LoadGrabStitchedViewGuardRowRangeAsync(info.GrabId, info.Earliest, info.Latest);
                    // 2b-ii：fit 由 ImageDisplayView 首幀自動 fit 承接
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
                        cbReviewId.SelectedIndex = idx;
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
                using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    _interactionHelper.NavigateToDateTime(earliest);
                _presenter.UpdatePeriodNavigationState();

                await LoadGrabStitchedViewGuardRowRangeAsync(grabId, earliest, latest);
                // 2b-ii：SaveCanvasView/fit（讀已砍 canvas）移除；ImageDisplayView 自管視野
                _reviewDirty = false;

                // 同步 Data tab
                if (!_dataStatsPresenter.GrabIdCrossGuard.IsSet
                    && cbDataId.Items.Count > 0 && idx < cbDataId.Items.Count)
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
