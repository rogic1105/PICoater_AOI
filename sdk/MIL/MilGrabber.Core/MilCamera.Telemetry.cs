using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Core
{
    // MilCamera 的「唯讀遙測」分區（純 MdigInquire / MsysInquire / CLProtocol Feature 查詢，不改狀態）。
    // 與核心生命週期分檔，避免一堆 getter clutter 主檔；共用 _milDigitizer / _ownerSystemId / _clProtocolEnabled。
    public partial class MilCamera
    {
        // ==================== Telemetry ====================

        /// <summary>目前實際量測 FPS（M_PROCESS_FRAME_RATE）。抓圖未啟動時回傳 0。</summary>
        public double CurrentFps
        {
            get
            {
                if (_milDigitizer == MIL.M_NULL) return 0;
                double fps = 0;
                MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_RATE, ref fps);
                return fps;
            }
        }

        /// <summary>相機本體溫度（°C，CLProtocol Feature DeviceTemperature）。未啟用回 NaN。</summary>
        public double GetCameraTemperature()
        {
            if (!_clProtocolEnabled || _milDigitizer == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MdigInquireFeature(_milDigitizer, MIL.M_FEATURE_VALUE, "DeviceTemperature", MIL.M_TYPE_DOUBLE, ref val);
                return val;
            }
            catch { return double.NaN; }
        }

        /// <summary>擷取卡 FPGA 溫度（°C，MsysInquire M_TEMPERATURE_FPGA）。</summary>
        public double GetFpgaTemperature()
        {
            if (_ownerSystemId == MIL.M_NULL) return double.NaN;
            try
            {
                double val = 0;
                MIL.MsysInquire(_ownerSystemId, MIL.M_TEMPERATURE_FPGA, ref val);
                return val;
            }
            catch { return double.NaN; }
        }

        /// <summary>板卡可用記憶體（MB）。</summary>
        public long GetMemoryFreeMB()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_MEMORY_FREE, ref val);
            return (long)val / (1024 * 1024);
        }

        /// <summary>PCIe 通道數。</summary>
        public int GetPcieNumberOfLanes()
        {
            if (_ownerSystemId == MIL.M_NULL) return -1;
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_NUMBER_OF_LANES, ref val);
            return (int)val;
        }

        /// <summary>PCIe 速度字串（Gen1 / Gen2 / Gen3）。</summary>
        public string GetPcieSpeed()
        {
            if (_ownerSystemId == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MsysInquire(_ownerSystemId, MIL.M_PCIE_SPEED, ref val);
            if (val == MIL.M_GEN1) return "Gen1";
            if (val == MIL.M_GEN2) return "Gen2";
            if (val == MIL.M_GEN3) return "Gen3";
            return $"0x{val:X}";
        }

        /// <summary>DCF 設定的目標 FPS（M_SELECTED_FRAME_RATE）。</summary>
        public double GetSelectedFrameRate()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            double val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SELECTED_FRAME_RATE, ref val);
            return val;
        }

        /// <summary>累計已處理 Frame 數（M_PROCESS_FRAME_COUNT）。</summary>
        public long GetFrameCount()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_COUNT, ref val);
            return (long)val;
        }

        /// <summary>Processing callback 遺漏 Frame 數（M_PROCESS_FRAME_MISSED）。</summary>
        public long GetFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_PROCESS_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>硬體 Grab 層遺漏 Frame 數（M_GRAB_FRAME_MISSED）。</summary>
        public long GetGrabFrameMissed()
        {
            if (_milDigitizer == MIL.M_NULL) return 0;
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_GRAB_FRAME_MISSED, ref val);
            return (long)val;
        }

        /// <summary>掃描模式（"Line" / "Progressive"）。</summary>
        public string GetScanMode()
        {
            if (_milDigitizer == MIL.M_NULL) return "N/A";
            MIL_INT val = 0;
            MIL.MdigInquire(_milDigitizer, MIL.M_SCAN_MODE, ref val);
            return (val == MIL.M_LINESCAN) ? "Line" : "Progressive";
        }
    }
}
