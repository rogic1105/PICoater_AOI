using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class InspectionSettings
    {
        [Category("1. 機台佈局")]
        [DisplayName("Machine Layout")]
        public MachineLayoutConfig MachineLayout { get; set; } = new MachineLayoutConfig();

        [Category("2. 取像設定")]
        [DisplayName("Acquisition")]
        public AcquisitionSettings Acquisition { get; set; } = new AcquisitionSettings();

        [Category("3. 檢測配方")]
        [DisplayName("Inspection Recipe")]
        public InspectionRecipe Recipe { get; set; } = new InspectionRecipe();

        [Category("4. 儲存設定")]
        [DisplayName("Storage")]
        public StorageSettings Storage { get; set; } = new StorageSettings();

        public void Validate()
        {
            if (MachineLayout == null) MachineLayout = new MachineLayoutConfig();
            if (Acquisition == null) Acquisition = new AcquisitionSettings();
            if (Recipe == null) Recipe = new InspectionRecipe();
            if (Storage == null) Storage = new StorageSettings();

            MachineLayout.Validate();
            Acquisition.Validate();
            Recipe.Validate();
            Storage.Validate();
        }

        public double[] GetCameraOpsUmArray() => MachineLayout.GetCameraOpsUmArray();
        public double[] GetCameraStartPositionMmArray() => MachineLayout.GetCameraStartPositionMmArray();

        // ===== Backward-compatible flattened accessors =====
        [Browsable(false)] public double Cam1_Ops { get => MachineLayout.Cam1_Ops; set => MachineLayout.Cam1_Ops = value; }
        [Browsable(false)] public double Cam2_Ops { get => MachineLayout.Cam2_Ops; set => MachineLayout.Cam2_Ops = value; }
        [Browsable(false)] public double Cam3_Ops { get => MachineLayout.Cam3_Ops; set => MachineLayout.Cam3_Ops = value; }
        [Browsable(false)] public double Cam4_Ops { get => MachineLayout.Cam4_Ops; set => MachineLayout.Cam4_Ops = value; }
        [Browsable(false)] public double Cam5_Ops { get => MachineLayout.Cam5_Ops; set => MachineLayout.Cam5_Ops = value; }
        [Browsable(false)] public double Cam6_Ops { get => MachineLayout.Cam6_Ops; set => MachineLayout.Cam6_Ops = value; }
        [Browsable(false)] public double Cam7_Ops { get => MachineLayout.Cam7_Ops; set => MachineLayout.Cam7_Ops = value; }
        [Browsable(false)] public double Cam1_Pos { get => MachineLayout.Cam1_Pos; set => MachineLayout.Cam1_Pos = value; }
        [Browsable(false)] public double Cam2_Pos { get => MachineLayout.Cam2_Pos; set => MachineLayout.Cam2_Pos = value; }
        [Browsable(false)] public double Cam3_Pos { get => MachineLayout.Cam3_Pos; set => MachineLayout.Cam3_Pos = value; }
        [Browsable(false)] public double Cam4_Pos { get => MachineLayout.Cam4_Pos; set => MachineLayout.Cam4_Pos = value; }
        [Browsable(false)] public double Cam5_Pos { get => MachineLayout.Cam5_Pos; set => MachineLayout.Cam5_Pos = value; }
        [Browsable(false)] public double Cam6_Pos { get => MachineLayout.Cam6_Pos; set => MachineLayout.Cam6_Pos = value; }
        [Browsable(false)] public double Cam7_Pos { get => MachineLayout.Cam7_Pos; set => MachineLayout.Cam7_Pos = value; }
        [Browsable(false)] public float HessianMaxFactor { get => Recipe.HessianMaxFactor; set => Recipe.HessianMaxFactor = value; }
        [Browsable(false)] public float ErrorValueMean { get => Recipe.ErrorValueMean; set => Recipe.ErrorValueMean = value; }
        [Browsable(false)] public float ErrorValueMax { get => Recipe.ErrorValueMax; set => Recipe.ErrorValueMax = value; }
        [Browsable(false)] public int CameraGrabHeight { get => Acquisition.CameraGrabHeight; set => Acquisition.CameraGrabHeight = value; }
        [Browsable(false)] public double CameraExposureTimeUs { get => Acquisition.CameraExposureTimeUs; set => Acquisition.CameraExposureTimeUs = value; }
        [Browsable(false)] public double CameraLineRateHz { get => Acquisition.CameraLineRateHz; set => Acquisition.CameraLineRateHz = value; }
        [Browsable(false)] public bool EnableAutoCapture { get => Storage.EnableAutoCapture; set => Storage.EnableAutoCapture = value; }
        [Browsable(false)] public string CaptureRootPath { get => Storage.CaptureRootPath; set => Storage.CaptureRootPath = value; }
    }
}
