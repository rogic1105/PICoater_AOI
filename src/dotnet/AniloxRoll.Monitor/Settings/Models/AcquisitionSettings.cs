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

        public void Validate()
        {
            if (CameraGrabHeight == null || CameraGrabHeight.Length != AcquisitionDefaults.CamCount)
                CameraGrabHeight = AcquisitionDefaults.NewGrabHeightArray();
            if (CameraExposureTimeUs == null || CameraExposureTimeUs.Length != AcquisitionDefaults.CamCount)
                CameraExposureTimeUs = AcquisitionDefaults.NewExposureTimeArray();
            if (CameraLineRateHz == null || CameraLineRateHz.Length != AcquisitionDefaults.CamCount)
                CameraLineRateHz = AcquisitionDefaults.NewLineRateArray();

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
