using System;
using System.Drawing;
using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;
using MilGrabber.Core;

namespace MilGrabber.Monitor
{
    // MilGrabberPbForm 的「Telemetry / ListView」分區：500ms timer 狀態刷新、lvCameras 17 欄、lvEngine 常數表。
    // Telemetry 專屬欄位也搬來此分區；UI 通用 helper（SetSubLabel/UpdateButtonsEnabled）留主檔。
    public partial class MilGrabberPbForm
    {
        // =========================================================================
        // Timer（500ms）：更新子畫面狀態 label + 選中相機 telemetry
        // tick 在 UI 執行緒，故可直接更新 UI
        // =========================================================================
        private int _statusTick;                 // StatusTimer tick 計數（決定慢欄位刷新時機）
        private const int SlowTelemetryEveryTicks = 8; // 慢欄位（溫度/PCIe/…）每 N tick 刷一次（500ms×8=4s）
        private double[] _maxLineRateCache;      // Max Line Rate 固定上限，抓到一次即快取（CLProtocol 查詢最慢）

        private void StatusTimer_Tick(object sender, EventArgs e)
        {
            if (_isReleasing || _cams == null) return;

            // 子畫面狀態：Online 綠 / Offline 紅
            for (int i = 0; i < _cams.Length; i++)
            {
                var cam = _cams[i];
                if (cam == null) continue;
                bool online = cam.CheckPresence();
                int camId = CamIdForIndex(i);
                if (online)
                    SetSubLabel(i, $"Cam {camId}  {cam.CurrentFps:F1} fps", Color.LightGreen, Color.Black);
                else
                    SetSubLabel(i, $"Cam {camId}: Offline", Color.Red, Color.White);
            }

            // 每 tick 更新所有相機列（快欄位每次、慢欄位每 4s；避免每 500ms 在 UI 執行緒做整套 MIL 查詢造成凍結）
            bool slow = (_statusTick++ % SlowTelemetryEveryTicks) == 0;
            UpdateCamerasListView(slow);
            // 選中相機那列（動態）
            UpdateEngineSelectedCam();
            // 計時比較（縮圖/顯示/FPS；MIL vs PictureBox）
            UpdateTimingLabel();
            // 三擊實體 1:1 校正（FOV/scale/模式/相機可能變，這裡保持最新）→ 重套佈局（含 FOV→ops 校正）
            ApplyLayout();
        }

        // =========================================================================
        // lvCameras：17 欄「所有相機資訊」表（每相機一列）
        // 欄位與資料來源照 LiveTelemetryPresenter；新增「Max Line Rate(Hz)」欄（CLProtocol 上限，只顯示不自動套用）。
        // =========================================================================

        /// <summary>
        /// 依實際建立的相機數新增初始列（每列 Tag = 相機 index，各欄填 "N/A"）。
        /// 未建立的相機不加列。在 btnInit 建相機後呼叫一次。
        /// </summary>
        private void InitCamerasListView(int camCount)
        {
            _maxLineRateCache = null; // 重新 init → 清快取，下個 tick 重抓 Max Line Rate
            lvCameras.BeginUpdate();
            try
            {
                lvCameras.Items.Clear();
                for (int i = 0; i < camCount; i++)
                {
                    // 顯示文字用 config 的 CameraId；Tag 用子畫面 index（= _cams 索引，UpdateCamerasListView 依此查相機）
                    var item = new ListViewItem($"CAM {CamIdForIndex(i)}");
                    for (int c = 1; c < 17; c++) item.SubItems.Add("N/A"); // 欄 1~16
                    item.Tag = i;
                    lvCameras.Items.Add(item);
                }
            }
            finally
            {
                lvCameras.EndUpdate();
            }
        }

        /// <summary>
        /// Timer（500ms）更新所有相機列。相機未建立/釋放後該列顯示 "-"。
        /// 17 欄索引：0 Camera / 1 FPS / 2 Target FPS / 3 Line Rate / 4 Max Line Rate /
        /// 5 Exp Set / 6 Exp Meas / 7 Frames / 8 Missed / 9 Grab Miss / 10 Resolution /
        /// 11 Scan Mode / 12 FPGA / 13 Cam Temp / 14 Mem Free / 15 PCIe Lanes / 16 PCIe Speed。
        /// Max Line Rate(欄 4)：cam.GetLineRateMaxHz()，&gt;0 顯示整數，否則 "-"（CLProtocol 未就緒即 "-"）。
        /// </summary>
        private void UpdateCamerasListView(bool slow)
        {
            if (lvCameras.Items.Count == 0) return;
            if (_maxLineRateCache == null || _maxLineRateCache.Length != (_cams?.Length ?? 0))
                _maxLineRateCache = new double[_cams?.Length ?? 0];

            lvCameras.BeginUpdate();
            try
            {
                foreach (ListViewItem item in lvCameras.Items)
                {
                    int idx = (int)item.Tag;
                    MilCamera cam = (_cams != null && idx >= 0 && idx < _cams.Length) ? _cams[idx] : null;

                    if (cam == null)
                    {
                        for (int c = 1; c <= 16; c++) item.SubItems[c].Text = "-";
                        continue;
                    }

                    // ── 快欄位（每 tick；C# 計數器 / 便宜查詢，快變要即時）──
                    item.SubItems[1].Text = $"{cam.CurrentFps:F2}";
                    item.SubItems[2].Text = $"{cam.GetSelectedFrameRate():F2}";

                    double lineRate = cam.GetLineRateHz();
                    item.SubItems[3].Text = lineRate > 0 ? $"{lineRate:F1}" : "-";

                    double expUs = cam.GetExposureUs();
                    item.SubItems[5].Text = expUs > 0 ? $"{expUs:F1}" : "-";

                    double measUs = cam.GetMeasuredExposureUs();
                    item.SubItems[6].Text = measUs > 0 ? $"{measUs:F1}" : "-";

                    item.SubItems[7].Text = $"{cam.GetFrameCount()}";
                    item.SubItems[8].Text = $"{cam.GetFrameMissed()}";
                    item.SubItems[9].Text = $"{cam.GetGrabFrameMissed()}";
                    item.SubItems[10].Text = $"{cam.FrameWidth}×{cam.FrameHeight}";

                    // Max Line Rate(Hz)：固定上限，抓到一次即快取（CLProtocol 查詢最慢，不必每次問）
                    if (idx < _maxLineRateCache.Length && _maxLineRateCache[idx] <= 0)
                    {
                        double m = cam.GetLineRateMaxHz();
                        if (m > 0) { _maxLineRateCache[idx] = m; item.SubItems[4].Text = $"{(long)Math.Floor(m)}"; }
                        else if (item.SubItems[4].Text != "-") item.SubItems[4].Text = "-";
                    }

                    if (!slow) continue; // 慢欄位（溫度/PCIe/掃描模式/記憶體）每 4s 才刷，避免 UI 凍結

                    item.SubItems[11].Text = cam.GetScanMode();

                    double fpgaTemp = cam.GetFpgaTemperature();
                    item.SubItems[12].Text = double.IsNaN(fpgaTemp) ? "-" : $"{fpgaTemp:F1}";

                    double camTemp = cam.GetCameraTemperature();
                    item.SubItems[13].Text = double.IsNaN(camTemp) ? "-" : $"{camTemp:F1}";

                    long memFree = cam.GetMemoryFreeMB();
                    item.SubItems[14].Text = memFree >= 0 ? $"{memFree}" : "-";

                    int lanes = cam.GetPcieNumberOfLanes();
                    item.SubItems[15].Text = lanes >= 0 ? $"{lanes}" : "-";

                    item.SubItems[16].Text = cam.GetPcieSpeed();
                }
            }
            finally
            {
                lvCameras.EndUpdate();
            }
        }

        /// <summary>所有相機列欄 1~16 重置為 "-"（相機釋放後呼叫）。</summary>
        private void ResetCamerasListView()
        {
            foreach (ListViewItem item in lvCameras.Items)
                for (int c = 1; c <= 16; c++) item.SubItems[c].Text = "-";
        }

        // 維持原 btnRelease 呼叫名稱：釋放後把相機資訊表全部重置為 "-"。
        private void ClearSysInfo()
        {
            ResetCamerasListView();
        }

        // =========================================================================
        // lvEngine：取像/系統常數（非檢測引擎；本範例無檢測）
        // 多為靜態，初始化填一次；「選中相機」那列由 SelectCamera/Timer 動態更新。
        // =========================================================================

        private const int EngineSelectedCamRow = 6; // 「選中相機」列索引（動態更新）

        private void InitEngineListView(int camCount)
        {
            lvEngine.BeginUpdate();
            try
            {
                lvEngine.Items.Clear();
                // Config 來源：讀到幾台 + 是否 fallback（DevNum 分布 SystemNum 0/1）
                string cfgSource = _usedFallbackConfig ? "fallback 預設" : "system-settings.json";
                AddEngineRow("Config", $"{_devices?.Count ?? 0} 台（{cfgSource}）");
                AddEngineRow("相機數", camCount.ToString());
                AddEngineRow("擷取卡(System)", _systems.Count.ToString());
                AddEngineRow("Grab Buffer", "2");          // MilCamera 內部雙 buffer（固定）
                AddEngineRow("Telemetry Interval", $"{_statusTimer.Interval} ms");
                AddEngineRow("MIL Version", GetMilVersion());
                AddEngineRow("選中相機", "-");             // index EngineSelectedCamRow，動態
            }
            finally
            {
                lvEngine.EndUpdate();
            }
        }

        private void AddEngineRow(string param, string value)
        {
            var item = new ListViewItem(param);
            item.SubItems.Add(value);
            lvEngine.Items.Add(item);
        }

        private void UpdateEngineSelectedCam()
        {
            if (lvEngine.Items.Count <= EngineSelectedCamRow) return;
            int camId = (_selectedCam >= 0 && _devices != null && _selectedCam < _devices.Count)
                ? _devices[_selectedCam].Id : _selectedCam;
            string text = (_selectedCam >= 0) ? $"CAM {camId}" : "-";
            lvEngine.Items[EngineSelectedCamRow].SubItems[1].Text = text;
        }

        /// <summary>取 MIL 版本（M_VERSION）；取不到回 "MIL.NET"。</summary>
        private static string GetMilVersion()
        {
            try
            {
                var sb = new System.Text.StringBuilder(256);
                MIL.MappInquire(MIL.M_DEFAULT, MIL.M_VERSION, sb);
                string v = sb.ToString();
                return string.IsNullOrWhiteSpace(v) ? "MIL.NET" : v;
            }
            catch { return "MIL.NET"; }
        }

        /// <summary>子畫面 index → config 的 CameraId（無 config 時退回 index）。</summary>
        private int CamIdForIndex(int index)
        {
            return (_devices != null && index >= 0 && index < _devices.Count) ? _devices[index].Id : index;
        }
    }
}
