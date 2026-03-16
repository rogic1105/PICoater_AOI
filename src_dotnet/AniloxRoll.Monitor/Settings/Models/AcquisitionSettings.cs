using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AcquisitionSettings
    {
        private const int CamCount = 7;

        /// <summary>7 台相機的擷取高度（px），索引 0 = CAM1。預設 2048。</summary>
        public int[] CameraGrabHeight { get; set; } = new int[] { 2048, 2048, 2048, 2048, 2048, 2048, 2048 };

        /// <summary>7 台相機的曝光時間（μs），索引 0 = CAM1。預設 50。</summary>
        public double[] CameraExposureTimeUs { get; set; } = new double[] { 50, 50, 50, 50, 50, 50, 50 };

        /// <summary>7 台相機的線掃速率（Hz），索引 0 = CAM1。預設 5000。</summary>
        public double[] CameraLineRateHz { get; set; } = new double[] { 5000, 5000, 5000, 5000, 5000, 5000, 5000 };

        public void Validate()
        {
            if (CameraGrabHeight == null || CameraGrabHeight.Length != CamCount)
                CameraGrabHeight = new int[] { 2048, 2048, 2048, 2048, 2048, 2048, 2048 };

            if (CameraExposureTimeUs == null || CameraExposureTimeUs.Length != CamCount)
                CameraExposureTimeUs = new double[] { 50, 50, 50, 50, 50, 50, 50 };

            if (CameraLineRateHz == null || CameraLineRateHz.Length != CamCount)
                CameraLineRateHz = new double[] { 5000, 5000, 5000, 5000, 5000, 5000, 5000 };

            for (int i = 0; i < CamCount; i++)
            {
                if (CameraGrabHeight[i] <= 0)     CameraGrabHeight[i]     = 2048;
                if (CameraExposureTimeUs[i] <= 0) CameraExposureTimeUs[i] = 50;
                if (CameraLineRateHz[i] <= 0)     CameraLineRateHz[i]     = 5000;
            }
        }

        public override string ToString() => "Acquisition";
    }
}
