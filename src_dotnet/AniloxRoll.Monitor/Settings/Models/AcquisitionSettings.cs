using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class AcquisitionSettings
    {
        [DisplayName("取像高度 (Pixel)")] public int CameraGrabHeight { get; set; } = 5000;
        [DisplayName("曝光時間 (us)")] public double CameraExposureTimeUs { get; set; } = 50;

        public void Validate()
        {
            if (CameraGrabHeight <= 0) CameraGrabHeight = 5000;
            if (CameraExposureTimeUs <= 0) CameraExposureTimeUs = 50;
        }

        public override string ToString() => "Acquisition";
    }
}
