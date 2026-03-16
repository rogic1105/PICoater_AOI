using System;
using System.Collections.Generic;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Camera;
using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>
    /// 負責 listViewCameras 的 16 欄 Telemetry 初始化與每 500ms 資料更新。
    /// 移植自 MilGrabSample.CameraListViewPresenter，將 MilCameraUnit 換成 AniloxCamera。
    /// </summary>
    public class LiveTelemetryPresenter
    {
        private readonly ListView _listView;

        public LiveTelemetryPresenter(ListView listView)
        {
            _listView = listView;
        }

        // ── 初始化欄位 ──────────────────────────────────────────────────

        /// <summary>
        /// 設定 ListView 為 Details 模式，建立 16 欄，並依 configs 新增初始列（各欄填 "N/A"）。
        /// 每個 ListViewItem.Tag 存 CameraHardwareConfig.Id，供 Update 對應。
        /// </summary>
        public void Initialize(IList<CameraHardwareConfig> configs)
        {
            _listView.View          = View.Details;
            _listView.FullRowSelect = true;
            _listView.GridLines     = true;
            _listView.Columns.Clear();

            // 欄位定義（索引 0–15）— 與 MilGrabSample 相同
            _listView.Columns.Add("Camera",        70);  // [0]  相機 ID
            _listView.Columns.Add("FPS",           70);  // [1]  實際 FPS
            _listView.Columns.Add("Target FPS",    80);  // [2]  DCF 目標 FPS
            _listView.Columns.Add("Line Rate(Hz)", 95);  // [3]  CLProtocol AcquisitionLineRate
            _listView.Columns.Add("Exp Set(μs)",  100);  // [4]  設定值（不回讀硬體）
            _listView.Columns.Add("Exp Meas(μs)", 100);  // [5]  InquireFeature 量測值
            _listView.Columns.Add("Frames",        80);  // [6]  累計 Frame 數
            _listView.Columns.Add("Missed",        70);  // [7]  Processing 遺漏 Frame
            _listView.Columns.Add("Grab Miss",     75);  // [8]  硬體 Grab 遺漏 Frame
            _listView.Columns.Add("Resolution",   110);  // [9]  影像解析度
            _listView.Columns.Add("Scan Mode",     90);  // [10] Line / Progressive
            _listView.Columns.Add("FPGA(°C)",      75);  // [11] 擷取卡 FPGA 溫度
            _listView.Columns.Add("Cam Temp(°C)",  85);  // [12] 相機本體溫度
            _listView.Columns.Add("Mem Free(MB)",  90);  // [13] 板卡可用記憶體
            _listView.Columns.Add("PCIe Lanes",    80);  // [14] PCIe 通道數
            _listView.Columns.Add("PCIe Speed",    75);  // [15] PCIe 速度

            _listView.Items.Clear();
            foreach (var cfg in configs)
            {
                var item = new ListViewItem($"CAM {cfg.Id}");
                for (int i = 0; i < 15; i++) item.SubItems.Add("N/A");
                item.Tag = cfg.Id;
                _listView.Items.Add(item);
            }
        }

        // ── 更新資料 ─────────────────────────────────────────────────────

        /// <summary>
        /// 從 cameras 讀取最新 Telemetry 並更新 ListView 所有欄位。
        /// 找不到對應相機時，該列全顯示 "N/A"。支援跨執行緒呼叫。
        /// </summary>
        public void Update(IReadOnlyList<AniloxCamera> cameras)
        {
            if (_listView == null) return;

            if (_listView.InvokeRequired)
            {
                _listView.BeginInvoke(new Action(() => Update(cameras)));
                return;
            }

            foreach (ListViewItem item in _listView.Items)
            {
                int camId = (int)item.Tag;
                var cam   = FindCamera(cameras, camId);

                if (cam == null)
                {
                    for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "N/A";
                }
                else
                {
                    item.SubItems[1].Text  = $"{cam.CurrentFps:F2}";
                    item.SubItems[2].Text  = $"{cam.GetSelectedFrameRate():F2}";

                    double lineRate = cam.GetLineRateHz();
                    item.SubItems[3].Text  = lineRate > 0 ? $"{lineRate:F1}" : "N/A";

                    double expUs = cam.GetExposureUs();
                    item.SubItems[4].Text  = expUs > 0 ? $"{expUs:F1}" : "N/A";

                    double measUs = cam.GetMeasuredExposureUs();
                    item.SubItems[5].Text  = measUs > 0 ? $"{measUs:F1}" : "N/A";

                    item.SubItems[6].Text  = $"{cam.GetFrameCount()}";
                    item.SubItems[7].Text  = $"{cam.GetFrameMissed()}";
                    item.SubItems[8].Text  = $"{cam.GetGrabFrameMissed()}";
                    item.SubItems[9].Text  = $"{cam.FrameWidth}×{cam.FrameHeight}";
                    item.SubItems[10].Text = cam.GetScanMode();

                    double fpgaTemp = cam.GetFpgaTemperature();
                    item.SubItems[11].Text = double.IsNaN(fpgaTemp) ? "N/A" : $"{fpgaTemp:F1}";

                    double camTemp = cam.GetCameraTemperature();
                    item.SubItems[12].Text = double.IsNaN(camTemp) ? "N/A" : $"{camTemp:F1}";

                    long memFree = cam.GetMemoryFreeMB();
                    item.SubItems[13].Text = memFree >= 0 ? $"{memFree}" : "N/A";

                    int lanes = cam.GetPcieNumberOfLanes();
                    item.SubItems[14].Text = lanes >= 0 ? $"{lanes}" : "N/A";

                    item.SubItems[15].Text = cam.GetPcieSpeed();
                }
            }
        }

        // ── 重置 ─────────────────────────────────────────────────────────

        /// <summary>所有欄位重置為 "N/A"（相機釋放後呼叫）。</summary>
        public void ResetAll()
        {
            if (_listView == null) return;
            foreach (ListViewItem item in _listView.Items)
                for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "N/A";
        }

        // ── 私有輔助 ─────────────────────────────────────────────────────

        private static AniloxCamera FindCamera(IReadOnlyList<AniloxCamera> cameras, int camId)
        {
            for (int i = 0; i < cameras.Count; i++)
                if (cameras[i].CameraId == camId) return cameras[i];
            return null;
        }
    }
}
