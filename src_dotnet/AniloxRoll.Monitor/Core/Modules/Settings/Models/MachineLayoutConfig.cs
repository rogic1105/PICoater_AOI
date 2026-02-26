using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MachineLayoutConfig
    {
        [DisplayName("Cam 1 OPS (um)")] public double Cam1_Ops { get; set; } = 33.0;
        [DisplayName("Cam 2 OPS (um)")] public double Cam2_Ops { get; set; } = 33.0;
        [DisplayName("Cam 3 OPS (um)")] public double Cam3_Ops { get; set; } = 33.0;
        [DisplayName("Cam 4 OPS (um)")] public double Cam4_Ops { get; set; } = 33.0;
        [DisplayName("Cam 5 OPS (um)")] public double Cam5_Ops { get; set; } = 33.0;
        [DisplayName("Cam 6 OPS (um)")] public double Cam6_Ops { get; set; } = 33.0;
        [DisplayName("Cam 7 OPS (um)")] public double Cam7_Ops { get; set; } = 33.0;

        [DisplayName("Cam 1 Start (mm)")] public double Cam1_Pos { get; set; } = 0.0;
        [DisplayName("Cam 2 Start (mm)")] public double Cam2_Pos { get; set; } = 400.0;
        [DisplayName("Cam 3 Start (mm)")] public double Cam3_Pos { get; set; } = 800.0;
        [DisplayName("Cam 4 Start (mm)")] public double Cam4_Pos { get; set; } = 1200.0;
        [DisplayName("Cam 5 Start (mm)")] public double Cam5_Pos { get; set; } = 1600.0;
        [DisplayName("Cam 6 Start (mm)")] public double Cam6_Pos { get; set; } = 2000.0;
        [DisplayName("Cam 7 Start (mm)")] public double Cam7_Pos { get; set; } = 2400.0;

        public void Validate()
        {
            if (Cam1_Ops <= 0) Cam1_Ops = 33.0; if (Cam2_Ops <= 0) Cam2_Ops = 33.0; if (Cam3_Ops <= 0) Cam3_Ops = 33.0;
            if (Cam4_Ops <= 0) Cam4_Ops = 33.0; if (Cam5_Ops <= 0) Cam5_Ops = 33.0; if (Cam6_Ops <= 0) Cam6_Ops = 33.0; if (Cam7_Ops <= 0) Cam7_Ops = 33.0;
        }

        public double[] GetCameraOpsUmArray() => new[] { Cam1_Ops, Cam2_Ops, Cam3_Ops, Cam4_Ops, Cam5_Ops, Cam6_Ops, Cam7_Ops };
        public double[] GetCameraStartPositionMmArray() => new[] { Cam1_Pos, Cam2_Pos, Cam3_Pos, Cam4_Pos, Cam5_Pos, Cam6_Pos, Cam7_Pos };
        public override string ToString() => "Layout (OPS/Position)";
    }
}
