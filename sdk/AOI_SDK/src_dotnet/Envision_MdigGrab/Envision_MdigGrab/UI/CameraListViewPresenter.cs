using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace Envision_MdigGrab
{
    /// <summary>
    /// 負責 ListView 欄位的初始化與每次 Timer Tick 的資料更新，以及 FPS StatusStrip label。
    /// 從 GrabForm 分離出來，讓 GrabForm 只需呼叫 Initialize() 和 Update()，
    /// 不需要知道欄位索引或格式細節。
    /// </summary>
    public class CameraListViewPresenter
    {
        private readonly ListView _listView;
        private readonly ToolStripStatusLabel _fpsLabel;

        /// <summary>
        /// 建構子，注入要管理的 ListView 與 FPS StatusStrip Label。
        /// </summary>
        /// <param name="listView">顯示相機資訊的 ListView 控制項</param>
        /// <param name="fpsLabel">顯示 FPS 摘要的 StatusStrip Label</param>
        public CameraListViewPresenter(ListView listView, ToolStripStatusLabel fpsLabel)
        {
            _listView = listView;
            _fpsLabel = fpsLabel;
        }

        // ================= 初始化欄位 =================

        /// <summary>
        /// 設定 ListView 為 Details 模式，建立所有欄位（共 15 欄），
        /// 並依 configs 逐一新增列，每列預先填入 "N/A"。
        /// 每個 ListViewItem.Tag 存放 CameraConfig.Id 供後續更新時對應。
        /// </summary>
        /// <param name="configs">相機設定清單，決定要建立幾列</param>
        public void Initialize(IList<CameraConfig> configs)
        {
            _listView.View = View.Details;
            _listView.FullRowSelect = true;
            _listView.GridLines = true;
            _listView.Columns.Clear();

            // 欄位定義（索引 0–15）
            _listView.Columns.Add("Camera",         70);  // [0]  相機 ID
            _listView.Columns.Add("FPS",            70);  // [1]  實際 FPS（MdigProcess 量測）
            _listView.Columns.Add("Target FPS",     80);  // [2]  DCF 設定目標 FPS
            _listView.Columns.Add("Line Rate(Hz)",  95);  // [3]  CLProtocol AcquisitionLineRate
            _listView.Columns.Add("Exp Set(μs)",   100);  // [4]  手動填入的曝光設定值
            _listView.Columns.Add("Exp Meas(μs)",  100);  // [5]  MdigInquireFeature 讀回的曝光量測值
            _listView.Columns.Add("Frames",         80);  // [6]  累計處理 Frame 數
            _listView.Columns.Add("Missed",         70);  // [7]  Processing 遺漏 Frame 數
            _listView.Columns.Add("Grab Miss",      75);  // [8]  硬體 Grab 遺漏 Frame 數
            _listView.Columns.Add("Resolution",    110);  // [9]  影像解析度（寬×高）
            _listView.Columns.Add("Scan Mode",      90);  // [10] 掃描模式（Line / Progressive）
            _listView.Columns.Add("FPGA(°C)",       75);  // [11] 擷取卡 FPGA 溫度（MsysInquire）
            _listView.Columns.Add("Cam Temp(°C)",   85);  // [12] 相機本體溫度（CLProtocol DeviceTemperature）
            _listView.Columns.Add("Mem Free(MB)",   90);  // [13] 板卡可用記憶體
            _listView.Columns.Add("PCIe Lanes",     80);  // [14] PCIe 通道數
            _listView.Columns.Add("PCIe Speed",     75);  // [15] PCIe 速度（Gen1/2/3）

            _listView.Items.Clear();
            foreach (var cfg in configs)
            {
                var item = new ListViewItem($"CAM {cfg.Id}");
                // 預先補齊 15 個子項目（SubItems[1]–[15]），避免後續存取時 IndexOutOfRange
                for (int i = 0; i < 15; i++) item.SubItems.Add("N/A");
                item.Tag = cfg.Id;
                _listView.Items.Add(item);
            }
        }

        // ================= 更新資料 =================

        /// <summary>
        /// 從 cameras 讀取最新數據，更新 ListView 每一列的所有欄位。
        /// 若找不到對應的 MilCameraUnit（相機未初始化），該列全部顯示 "N/A"。
        /// 支援跨執行緒呼叫（Timer 在 UI 執行緒觸發，通常不需要 Invoke，保留以防萬一）。
        /// </summary>
        /// <param name="cameras">目前已初始化的相機清單</param>
        public void Update(IReadOnlyList<MilCameraUnit> cameras)
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
                var cam = FindCamera(cameras, camId);

                if (cam == null)
                {
                    for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "N/A";
                }
                else
                {
                    item.SubItems[1].Text = $"{cam.CurrentFps:F2}";
                    item.SubItems[2].Text = $"{cam.GetSelectedFrameRate():F2}";

                    double lineRate = cam.GetLineRateHz();
                    item.SubItems[3].Text = lineRate > 0 ? $"{lineRate:F1}" : "N/A";

                    double expUs = cam.GetExposureUs();
                    item.SubItems[4].Text = expUs > 0 ? $"{expUs:F1}" : "N/A";

                    double measUs = cam.GetMeasuredExposureUs();
                    item.SubItems[5].Text = measUs > 0 ? $"{measUs:F1}" : "N/A";

                    item.SubItems[6].Text = $"{cam.GetFrameCount()}";
                    item.SubItems[7].Text = $"{cam.GetFrameMissed()}";
                    item.SubItems[8].Text = $"{cam.GetGrabFrameMissed()}";
                    item.SubItems[9].Text = cam.Resolution;
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

            UpdateFpsLabel(cameras);
        }

        /// <summary>
        /// 將所有欄位重置為 "N/A"，並將 FPS label 重置為 "FPS: N/A"。
        /// 在 Free MIL 完成後或系統尚未初始化時呼叫。
        /// </summary>
        public void ResetAll()
        {
            if (_listView == null) return;

            foreach (ListViewItem item in _listView.Items)
                for (int i = 1; i <= 15; i++) item.SubItems[i].Text = "N/A";

            if (_fpsLabel != null) _fpsLabel.Text = "FPS: N/A";
        }

        // ================= 私有輔助方法 =================

        /// <summary>
        /// 在 cameras 清單中依 camId 尋找對應的 MilCameraUnit。
        /// 找不到時回傳 null。
        /// </summary>
        private static MilCameraUnit FindCamera(IReadOnlyList<MilCameraUnit> cameras, int camId)
        {
            for (int i = 0; i < cameras.Count; i++)
                if (cameras[i].CameraId == camId) return cameras[i];
            return null;
        }

        /// <summary>
        /// 更新 StatusStrip 的 FPS label。
        /// 沒有相機時顯示 "FPS: N/A"；
        /// 有相機時顯示每台個別 FPS 及總和，格式：
        /// "FPS | CAM 1: 60.00 | CAM 2: 60.00 | Total: 120.00"
        /// </summary>
        private void UpdateFpsLabel(IReadOnlyList<MilCameraUnit> cameras)
        {
            if (_fpsLabel == null) return;

            if (cameras.Count == 0)
            {
                _fpsLabel.Text = "FPS: N/A";
                return;
            }

            var parts = new List<string>();
            double total = 0;
            foreach (var cam in cameras)
            {
                double fps = cam.CurrentFps;
                total += fps;
                parts.Add($"CAM {cam.CameraId}: {fps:F2}");
            }

            _fpsLabel.Text = $"FPS | {string.Join(" | ", parts)} | Total: {total:F2}";
        }
    }
}
