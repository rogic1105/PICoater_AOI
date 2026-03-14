using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AcquisitionSettings
    {
        [DisplayName("取像高度 (Pixel)")] public int CameraGrabHeight { get; set; } = 2048;
        [DisplayName("曝光時間 (us)")] public double CameraExposureTimeUs { get; set; } = 50;
        [DisplayName("線掃速率 (Hz)")] public double CameraLineRateHz { get; set; } = 5000;

        public void Validate()
        {
            if (CameraGrabHeight <= 0) CameraGrabHeight = 2048;
            if (CameraExposureTimeUs <= 0) CameraExposureTimeUs = 50;
            if (CameraLineRateHz <= 0) CameraLineRateHz = 5000;
        }

        public override string ToString() => "Acquisition";
    }
}
