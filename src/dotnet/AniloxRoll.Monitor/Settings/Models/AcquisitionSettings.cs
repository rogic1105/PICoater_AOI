using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AcquisitionSettings
    {
        /// <summary>7 台相機的擷取高度（px），索引 0 = CAM1。</summary>
        public int[] CameraGrabHeight { get; set; } = AcquisitionDefaults.NewGrabHeightArray();

        /// <summary>7 台相機的曝光時間（μs），索引 0 = CAM1。</summary>
        public double[] CameraExposureTimeUs { get; set; } = AcquisitionDefaults.NewExposureTimeArray();

        /// <summary>7 台相機的線掃速率（Hz），索引 0 = CAM1。</summary>
        public double[] CameraLineRateHz { get; set; } = AcquisitionDefaults.NewLineRateArray();

        /// <summary>grab buffer 高度上限（px）。**0 = 自動依板載算**（跨 grabber 通用）；>0 = 手動覆寫。
        /// 防撞 grabber 板載記憶體 stall。算法見 AcquisitionDefaults.MaxGrabBufferHeightPx。</summary>
        public int MaxGrabBufferHeightPx { get; set; } = AcquisitionDefaults.MaxGrabBufferHeightPx;

        /// <summary>板載安全使用率（%）。自動 autoMax = 板載 × 此% ÷ 台數 ÷ 每台每行。預設 80。</summary>
        public int GrabBufferSafetyPercent { get; set; } = AcquisitionDefaults.GrabBufferSafetyPercent;

        /// <summary>每張板板載總量（MB）。autoMax 用（不查 MIL，避免第一台相機 stall）。換 grabber 改此值。</summary>
        public int BoardTotalMemMB { get; set; } = AcquisitionDefaults.BoardTotalMemMB;

        public void Validate()
        {
            if (CameraGrabHeight == null || CameraGrabHeight.Length != AcquisitionDefaults.CamCount)
                CameraGrabHeight = AcquisitionDefaults.NewGrabHeightArray();
            if (CameraExposureTimeUs == null || CameraExposureTimeUs.Length != AcquisitionDefaults.CamCount)
                CameraExposureTimeUs = AcquisitionDefaults.NewExposureTimeArray();
            if (CameraLineRateHz == null || CameraLineRateHz.Length != AcquisitionDefaults.CamCount)
                CameraLineRateHz = AcquisitionDefaults.NewLineRateArray();
            if (MaxGrabBufferHeightPx < 0) MaxGrabBufferHeightPx = 0;   // 0=自動（合法）；負數歸 0
            if (GrabBufferSafetyPercent <= 0 || GrabBufferSafetyPercent > 100)
                GrabBufferSafetyPercent = AcquisitionDefaults.GrabBufferSafetyPercent;
            if (BoardTotalMemMB <= 0) BoardTotalMemMB = AcquisitionDefaults.BoardTotalMemMB;

            for (int i = 0; i < AcquisitionDefaults.CamCount; i++)
            {
                if (CameraGrabHeight[i]     <= 0) CameraGrabHeight[i]     = AcquisitionDefaults.GrabHeight;
                if (CameraExposureTimeUs[i] <= 0) CameraExposureTimeUs[i] = AcquisitionDefaults.ExposureTimeUs;
                if (CameraLineRateHz[i]     <= 0) CameraLineRateHz[i]     = AcquisitionDefaults.LineRateHz;
            }
        }

        public override string ToString() => "Acquisition";
    }
}
