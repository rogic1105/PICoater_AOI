using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CameraOpsConfig
    {
        [DisplayName("Cam 1")] public double Cam1 { get; set; } = 33.0;
        [DisplayName("Cam 2")] public double Cam2 { get; set; } = 33.0;
        [DisplayName("Cam 3")] public double Cam3 { get; set; } = 33.0;
        [DisplayName("Cam 4")] public double Cam4 { get; set; } = 33.0;
        [DisplayName("Cam 5")] public double Cam5 { get; set; } = 33.0;
        [DisplayName("Cam 6")] public double Cam6 { get; set; } = 33.0;
        [DisplayName("Cam 7")] public double Cam7 { get; set; } = 33.0;

        public void Validate()
        {
            if (Cam1 <= 0) Cam1 = 33.0; if (Cam2 <= 0) Cam2 = 33.0; if (Cam3 <= 0) Cam3 = 33.0;
            if (Cam4 <= 0) Cam4 = 33.0; if (Cam5 <= 0) Cam5 = 33.0; if (Cam6 <= 0) Cam6 = 33.0; if (Cam7 <= 0) Cam7 = 33.0;
        }

        public double[] ToArray() => new[] { Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7 };
        public override string ToString() => "";
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CameraStartPositionConfig
    {
        [DisplayName("Cam 1")] public double Cam1 { get; set; } = 0.0;
        [DisplayName("Cam 2")] public double Cam2 { get; set; } = 400.0;
        [DisplayName("Cam 3")] public double Cam3 { get; set; } = 800.0;
        [DisplayName("Cam 4")] public double Cam4 { get; set; } = 1200.0;
        [DisplayName("Cam 5")] public double Cam5 { get; set; } = 1600.0;
        [DisplayName("Cam 6")] public double Cam6 { get; set; } = 2000.0;
        [DisplayName("Cam 7")] public double Cam7 { get; set; } = 2400.0;

        public double[] ToArray() => new[] { Cam1, Cam2, Cam3, Cam4, Cam5, Cam6, Cam7 };
        public override string ToString() => "";
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class CameraCropConfig
    {
        [DisplayName("去頭 (mm)")] public double TrimHeadMm { get; set; } = 0.0;
        [DisplayName("去尾 (mm)")] public double TrimTailMm { get; set; } = 0.0;

        public void Validate()
        {
            if (TrimHeadMm < 0) TrimHeadMm = 0;
            if (TrimTailMm < 0) TrimTailMm = 0;
        }

        public override string ToString() => "";
    }

    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MachineLayoutConfig
    {
        public CameraOpsConfig Ops { get; set; } = new CameraOpsConfig();
        public CameraStartPositionConfig StartPosition { get; set; } = new CameraStartPositionConfig();
        public CameraCropConfig Crop { get; set; } = new CameraCropConfig();

        // 向後相容：舊 JSON 欄位名稱對應
        [Browsable(false)] public double Cam1_Ops { get => Ops.Cam1; set => Ops.Cam1 = value; }
        [Browsable(false)] public double Cam2_Ops { get => Ops.Cam2; set => Ops.Cam2 = value; }
        [Browsable(false)] public double Cam3_Ops { get => Ops.Cam3; set => Ops.Cam3 = value; }
        [Browsable(false)] public double Cam4_Ops { get => Ops.Cam4; set => Ops.Cam4 = value; }
        [Browsable(false)] public double Cam5_Ops { get => Ops.Cam5; set => Ops.Cam5 = value; }
        [Browsable(false)] public double Cam6_Ops { get => Ops.Cam6; set => Ops.Cam6 = value; }
        [Browsable(false)] public double Cam7_Ops { get => Ops.Cam7; set => Ops.Cam7 = value; }
        [Browsable(false)] public double Cam1_Pos { get => StartPosition.Cam1; set => StartPosition.Cam1 = value; }
        [Browsable(false)] public double Cam2_Pos { get => StartPosition.Cam2; set => StartPosition.Cam2 = value; }
        [Browsable(false)] public double Cam3_Pos { get => StartPosition.Cam3; set => StartPosition.Cam3 = value; }
        [Browsable(false)] public double Cam4_Pos { get => StartPosition.Cam4; set => StartPosition.Cam4 = value; }
        [Browsable(false)] public double Cam5_Pos { get => StartPosition.Cam5; set => StartPosition.Cam5 = value; }
        [Browsable(false)] public double Cam6_Pos { get => StartPosition.Cam6; set => StartPosition.Cam6 = value; }
        [Browsable(false)] public double Cam7_Pos { get => StartPosition.Cam7; set => StartPosition.Cam7 = value; }

        public void Validate()
        {
            if (Ops == null) Ops = new CameraOpsConfig();
            if (StartPosition == null) StartPosition = new CameraStartPositionConfig();
            if (Crop == null) Crop = new CameraCropConfig();
            Ops.Validate();
            Crop.Validate();
        }

        [Browsable(false)] public double TrimHeadMm { get => Crop.TrimHeadMm; set => Crop.TrimHeadMm = value; }
        [Browsable(false)] public double TrimTailMm { get => Crop.TrimTailMm; set => Crop.TrimTailMm = value; }

        public double[] GetCameraOpsUmArray() => Ops.ToArray();
        public double[] GetCameraStartPositionMmArray() => StartPosition.ToArray();
        public override string ToString() => "Layout (OPS/Position)";
    }
}
