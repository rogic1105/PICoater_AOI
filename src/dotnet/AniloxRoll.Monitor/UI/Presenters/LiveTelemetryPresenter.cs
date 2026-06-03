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

        /// <summary>單台相機的 telemetry 快照（純字串，無 MIL）。Capture 在背景產生、Apply 在 UI 套用。</summary>
        public sealed class CamSnapshot
        {
            public int CamId;
            public double Fps;                       // 供 caller 算 maxFps（不必再查 MIL）
            public readonly string[] Cells = new string[16];  // [1..15] 對應 SubItems
        }

        /// <summary>★ 背景執行緒呼叫：對每台相機做 MIL 查詢，產生純字串快照（**不碰 ListView**）。
        /// 這些 MdigInquire/MsysInquire 約 ~100ms/tick，移到背景後就不再卡 UI 執行緒。</summary>
        public List<CamSnapshot> Capture(IReadOnlyList<AniloxCamera> cameras)
        {
            var list = new List<CamSnapshot>();
            if (cameras == null) return list;

            foreach (var cam in cameras)
            {
                if (cam == null) continue;
                var s = new CamSnapshot { CamId = cam.CameraId, Fps = cam.CurrentFps };
                var c = s.Cells;
                c[1]  = $"{s.Fps:F2}";
                c[2]  = $"{cam.GetSelectedFrameRate():F2}";
                double lineRate = cam.GetLineRateHz();   c[3]  = lineRate > 0 ? $"{lineRate:F1}" : "-";
                double expUs    = cam.GetExposureUs();    c[4]  = expUs > 0 ? $"{expUs:F1}" : "-";
                double measUs   = cam.GetMeasuredExposureUs(); c[5] = measUs > 0 ? $"{measUs:F1}" : "-";
                c[6]  = $"{cam.GetFrameCount()}";
                c[7]  = $"{cam.GetFrameMissed()}";
                c[8]  = $"{cam.GetGrabFrameMissed()}";
                c[9]  = $"{cam.FrameWidth}×{cam.FrameHeight}";
                c[10] = cam.GetScanMode();
                double fpgaTemp = cam.GetFpgaTemperature(); c[11] = double.IsNaN(fpgaTemp) ? "-" : $"{fpgaTemp:F1}";
                double camTemp  = cam.GetCameraTemperature(); c[12] = double.IsNaN(camTemp) ? "-" : $"{camTemp:F1}";
                long memFree = cam.GetMemoryFreeMB();      c[13] = memFree >= 0 ? $"{memFree}" : "-";
                int lanes    = cam.GetPcieNumberOfLanes(); c[14] = lanes >= 0 ? $"{lanes}" : "-";
                c[15] = cam.GetPcieSpeed();
                list.Add(s);
            }
            return list;
        }

        /// <summary>★ UI 執行緒呼叫：把背景產生的快照套進 ListView（純字串指派，**不碰 MIL**，極快）。</summary>
        public void Apply(List<CamSnapshot> snapshots)
        {
            if (_listView == null || _listView.IsDisposed) return;

            foreach (ListViewItem item in _listView.Items)
            {
                int camId = (int)item.Tag;
                CamSnapshot s = null;
                if (snapshots != null)
                    for (int i = 0; i < snapshots.Count; i++)
                        if (snapshots[i].CamId == camId) { s = snapshots[i]; break; }

                if (s == null)
                    for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "-";
                else
                    for (int i = 1; i <= 15; i++) item.SubItems[i].Text = s.Cells[i] ?? "-";
            }
        }

        /// <summary>同步版（背景 Capture + 此執行緒 Apply）。注意：會在呼叫端執行緒做 MIL 查詢，UI 執行緒勿用。</summary>
        public void Update(IReadOnlyList<AniloxCamera> cameras) => Apply(Capture(cameras));

        // ── 重置 ─────────────────────────────────────────────────────────

        /// <summary>所有欄位重置為 "-"（相機釋放後呼叫）。</summary>
        public void ResetAll()
        {
            if (_listView == null) return;
            foreach (ListViewItem item in _listView.Items)
                for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "-";
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
