using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AcquisitionSettings
    {
        private const int CamCount = 7;

        /// <summary>7 台相機的擷取高度（px），索引 0 = CAM1。預設 3001。</summary>
        public int[] CameraGrabHeight { get; set; } = new int[] { 3001, 3001, 3001, 3001, 3001, 3001, 3001 };

        /// <summary>7 台相機的曝光時間（μs），索引 0 = CAM1。預設 149。</summary>
        public double[] CameraExposureTimeUs { get; set; } = new double[] { 149, 149, 149, 149, 149, 149, 149 };

        /// <summary>7 台相機的線掃速率（Hz），索引 0 = CAM1。預設 3001。</summary>
        public double[] CameraLineRateHz { get; set; } = new double[] { 3001, 3001, 3001, 3001, 3001, 3001, 3001 };

        public void Validate()
        {
            if (CameraGrabHeight == null || CameraGrabHeight.Length != CamCount)
                CameraGrabHeight = new int[] { 3001, 3001, 3001, 3001, 3001, 3001, 3001 };

            if (CameraExposureTimeUs == null || CameraExposureTimeUs.Length != CamCount)
                CameraExposureTimeUs = new double[] { 149, 149, 149, 149, 149, 149, 149 };

            if (CameraLineRateHz == null || CameraLineRateHz.Length != CamCount)
                CameraLineRateHz = new double[] { 3001, 3001, 3001, 3001, 3001, 3001, 3001 };

            for (int i = 0; i < CamCount; i++)
            {
                if (CameraGrabHeight[i] <= 0)     CameraGrabHeight[i]     = 3001;
                if (CameraExposureTimeUs[i] <= 0) CameraExposureTimeUs[i] = 149;
                if (CameraLineRateHz[i] <= 0)     CameraLineRateHz[i]     = 3001;
            }
        }

        public override string ToString() => "Acquisition";
    }
}
