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
using AniloxRoll.Monitor.UI.Binders;
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
                _dataStatsPresenter.ApplyChartScaleForChart("Yearly");
            else if (name == nameof(InspectionSettings.gd_MonthlyYMax))
                _dataStatsPresenter.ApplyChartScaleForChart("Monthly");
            else if (name == nameof(InspectionSettings.ge_DailyYMax))
                _dataStatsPresenter.ApplyChartScaleForChart("Daily");
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
                GrabDetailList = new GrabDetailListBinder(listViewGrabDetail, CameraCount),
                PanelStatCams = new[] { camData1, camData2, camData3,
                                        camData4, camData5, camData6, camData7 },
                PanelStatRow = camDataRow,
                ChartDataPatch = chartDataColumn,
                ChartDataRow = chartDataRow,
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

            // Data → Review tab 切換：pending selection 才載圖；既有內容一律補一次可見重繪。
            tabMain.SelectedIndexChanged += async (s, e) =>
            {
                if (tabMain.SelectedTab != tabPageReview || _suppressTabIntent) return;

                if (_reviewDirty && _hasPendingDataReviewSelection)
                {
                    var pending = _pendingDataReviewSelection;
                    int idx = pending.idx;
                    if (idx < 0 || idx >= _dataStatsPresenter.GrabIdInfos.Count ||
                        !string.Equals(_dataStatsPresenter.GrabIdInfos[idx].GrabId, pending.grabId, StringComparison.Ordinal))
                    {
                        idx = _dataStatsPresenter.GrabIdInfos.FindIndex(
                            candidate => string.Equals(candidate.GrabId, pending.grabId, StringComparison.Ordinal));
                    }
                    if (idx >= 0 && idx < _dataStatsPresenter.GrabIdInfos.Count)
                    {
                        var info = _dataStatsPresenter.GrabIdInfos[idx];
                        try
                        {
                            using (_dataStatsPresenter.GrabIdCrossGuard.Enter())
                            using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                            {
                                cbReviewId.SelectedIndex = idx;
                                _dateTimeNavigator.SetPeriodToCombo(info.Earliest);
                            }
                            _presenter.UpdatePeriodNavigationState();
                            _dataStatsPresenter.UpdateGrabIdNavState();
                            _dataStatsPresenter.SetReviewGroupBoxes(true);
                            _reviewDirty = false;
                            _hasPendingDataReviewSelection = false;
                            FlowTrace.Log($"DT review sync apply {info.GrabId}");
                            await LoadGrabStitchedViewGuardRowRangeAsync(info.GrabId, info.Earliest, info.Latest);
                            // 2b-ii：fit 由 ImageDisplayView 首幀自動 fit 承接
                        }
                        catch (Exception ex)
                        {
                            _reviewDirty = true;
                            Trace.WriteLine($"[tabMain → Review] {ex}");
                        }
                    }
                }

                // SelectedIndexChanged 發生時子控制項的可見 layout 尚未保證完成；延後一個 UI message，
                // 再以目前尺寸補 LOD tile + paint。只補顯示，不重讀檔或重設視野。
                if (IsHandleCreated && !IsDisposed && !Disposing)
                {
                    try
                    {
                        BeginInvoke(new Action(() =>
                        {
                            if (tabMain.SelectedTab == tabPageReview)
                                _reviewDisplayManager?.RefreshVisible();
                        }));
                    }
                    catch (InvalidOperationException) { /* Form 正在關閉 */ }
                }
            };
        }

        private (string grabId, int idx) _pendingDataReviewSelection;
        private bool _hasPendingDataReviewSelection;

        private void OnDataGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                _pendingDataReviewSelection = (grabId, idx);
                _hasPendingDataReviewSelection = true;
                _reviewDirty = true;
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataGrabIdSelected] {ex}"); }
        }

        // 快速滾序號的載入 debounce（250ms）：滾動中只做輕量日期同步與 latest-only 曲線；
        // 停頓才載入最後一張圖片、儲存 session、同步 Data tab。
        private Timer _reviewLoadDebounce;
        private (string grabId, DateTime earliest, DateTime latest, int idx, int sequence) _pendingReviewLoad;
        private int _reviewSelectionSeq;

        private void OnReviewGrabIdSelected(string grabId, DateTime earliest, DateTime latest, int idx)
        {
            try
            {
                int selectionSeq = ++_reviewSelectionSeq;
                _stitchCoordinator.InvalidateImageLoad();

                // 滾動中只同步畫面上的日期/時間，不寫 session、不重建完整日期清單。
                using (_dataStatsPresenter.GrabIdNavGuard.Enter())
                    _dateTimeNavigator.SetPeriodToCombo(earliest);
                _presenter.UpdatePeriodNavigationState();

                // 分層載入：曲線（輕）即時跟滾動 → 使用者快速掃 chart 找異常；影像（重）settle 才載。
                _ = _stitchCoordinator.LoadGrabCurvesOnlyAsync(grabId, earliest, latest);

                _pendingReviewLoad = (grabId, earliest, latest, idx, selectionSeq);
                if (_reviewLoadDebounce == null)
                {
                    _reviewLoadDebounce = new Timer { Interval = 250 };
                    _reviewLoadDebounce.Tick += async (s2, e2) =>
                    {
                        _reviewLoadDebounce.Stop();
                        var p = _pendingReviewLoad;
                        // session 只在使用者停下後落盤一次。
                        _dateTimeNavigator.SaveCurrentSelection();
                        try
                        {
                            await LoadGrabStitchedViewGuardRowRangeAsync(p.grabId, p.earliest, p.latest);
                            if (p.sequence != _reviewSelectionSeq) return;
                            _reviewDirty = false;
                        }
                        catch (Exception ex) { Trace.WriteLine($"[ReviewLoadDebounce] {ex}"); }
                        // Data tab 同步排在影像之後：統計全重算（掃目錄+CSV 解析+Mura 圖 bin IO）
                        // 是 UI 執行緒重活——settle 才做一次，且使用者先看到影像。
                        // （原在每格序號 inline 跑、註解自稱「便宜」→ 快撥 18 格＝18 輪重算
                        //   ＝UiStall 5.7s＋曲線快路全餓死，2026-07-10 log 定罪）
                        if (p.sequence == _reviewSelectionSeq)
                            SyncDataTabFromReviewSettled(p.idx);
                    };
                }
                _reviewLoadDebounce.Stop();
                _reviewLoadDebounce.Start();   // 重壓計時：每次選取重新等 250ms，停下才載入最終選取
            }
            catch (Exception ex) { Trace.WriteLine($"[OnReviewGrabIdSelected] {ex}"); }
        }

        /// <summary>Review 序號 settle 後同步 Data tab（統計/Mura 圖重算的唯一觸發點——不得回到逐格 inline）。</summary>
        private void SyncDataTabFromReviewSettled(int idx)
        {
            try
            {
                if (_dataStatsPresenter.GrabIdCrossGuard.IsSet) return;
                if (cbDataId.Items.Count == 0 || idx < 0 || idx >= cbDataId.Items.Count) return;
                if (idx >= _dataStatsPresenter.GrabIdInfos.Count) return;
                var info = _dataStatsPresenter.GrabIdInfos[idx];
                _dataStatsPresenter.SyncDataGrabIdFromReview(idx, info);
            }
            catch (Exception ex) { Trace.WriteLine($"[SyncDataTabFromReviewSettled] {ex}"); }
        }

        private async void OnDataFolderSelected(string path)
        {
            try
            {
                // 同步 Review tab：先載入 ImageRepository + Navigator，再走共用 reset + 主畫面載入。
                UserSessionState.SetLastDataPath(path);
                UserSessionState.Save();
                _reviewFolderCoordinator.LoadDirectoryAndInitNavigator(path);
                _presenter.UpdatePeriodNavigationState();
                // DataPresenter 已透過 LoadDataFolder 同步 _grabIdInfos，skip SyncFromReviewFolder
                await ResetAndLoadReviewAfterFolderChanged(dataPresenterAlreadySynced: true);
            }
            catch (Exception ex) { Trace.WriteLine($"[OnDataFolderSelected] {ex}"); }
        }
    }
}
