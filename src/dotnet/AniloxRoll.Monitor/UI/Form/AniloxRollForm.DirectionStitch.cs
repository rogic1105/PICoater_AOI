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
        /// 切換 Live 顯示的 V/H 處理圖方向，點選 chartLiveColumn/HorizontalLive 觸發。
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

            chartLiveColumn.BackColor   = dir == "v" ? enhanceBg : normal;
            chartLiveRow.BackColor = dir == "h" ? enhanceBg : normal;
            foreach (var c in new[] { chartLiveColumn, chartLiveRow })
            {
                c.BorderlineColor = System.Drawing.Color.Transparent;
                c.BorderlineWidth = 1;
                c.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.NotSet;
            }
        }

        /// <summary>
        /// 切換 camReviewMain 的 V/H 處理圖方向，點選 chartReviewColumn/Horizontal 觸發。
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
                // 2b-ii：SaveCanvasView（讀已砍 canvas）移除；ImageDisplayView 自管視野
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


        private void UpdateRidgeDirectionVisual(string dir)
        {
            // 視覺規則（2026-06-12 改版）：藍底＝該方向強化圖顯示中；mode 底色 + 橘框已廢（同 Live）。
            var enhanceBg = System.Drawing.Color.FromArgb(230, 240, 255);
            var normal    = System.Drawing.SystemColors.Control;

            chartReviewColumn.BackColor = dir == "v" ? enhanceBg : normal;
            if (chartReviewRow != null)
                chartReviewRow.BackColor = dir == "h" ? enhanceBg : normal;
            foreach (var c in new[] { chartReviewColumn, chartReviewRow })
            {
                if (c == null) continue;
                c.BorderlineColor = System.Drawing.Color.Transparent;
                c.BorderlineWidth = 1;
                c.BorderlineDashStyle = System.Windows.Forms.DataVisualization.Charting.ChartDashStyle.NotSet;
            }
        }
    }
}
