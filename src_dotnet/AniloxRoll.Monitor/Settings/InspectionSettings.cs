using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class InspectionSettings
    {
        // 子物件隱藏於 PropertyGrid，屬性直接平鋪於各 Category（兩層顯示）
        [Browsable(false)] public MachineLayoutConfig MachineLayout { get; set; } = new MachineLayoutConfig();
        [Browsable(false)] public AcquisitionSettings Acquisition   { get; set; } = new AcquisitionSettings();
        [Browsable(false)] public InspectionRecipe    Recipe        { get; set; } = new InspectionRecipe();
        [Browsable(false)] public StorageSettings     Storage       { get; set; } = new StorageSettings();

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

        // ===== 1. 機台佈局 =====
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 1")]   public double Cam1_Ops { get => MachineLayout.Cam1_Ops; set => MachineLayout.Cam1_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 2")]   public double Cam2_Ops { get => MachineLayout.Cam2_Ops; set => MachineLayout.Cam2_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 3")]   public double Cam3_Ops { get => MachineLayout.Cam3_Ops; set => MachineLayout.Cam3_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 4")]   public double Cam4_Ops { get => MachineLayout.Cam4_Ops; set => MachineLayout.Cam4_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 5")]   public double Cam5_Ops { get => MachineLayout.Cam5_Ops; set => MachineLayout.Cam5_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 6")]   public double Cam6_Ops { get => MachineLayout.Cam6_Ops; set => MachineLayout.Cam6_Ops = value; }
        [Category("1. 機台佈局 / OPS (um)")][DisplayName("Cam 7")]   public double Cam7_Ops { get => MachineLayout.Cam7_Ops; set => MachineLayout.Cam7_Ops = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 1")] public double Cam1_Pos { get => MachineLayout.Cam1_Pos; set => MachineLayout.Cam1_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 2")] public double Cam2_Pos { get => MachineLayout.Cam2_Pos; set => MachineLayout.Cam2_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 3")] public double Cam3_Pos { get => MachineLayout.Cam3_Pos; set => MachineLayout.Cam3_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 4")] public double Cam4_Pos { get => MachineLayout.Cam4_Pos; set => MachineLayout.Cam4_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 5")] public double Cam5_Pos { get => MachineLayout.Cam5_Pos; set => MachineLayout.Cam5_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 6")] public double Cam6_Pos { get => MachineLayout.Cam6_Pos; set => MachineLayout.Cam6_Pos = value; }
        [Category("1. 機台佈局 / Start (mm)")][DisplayName("Cam 7")] public double Cam7_Pos { get => MachineLayout.Cam7_Pos; set => MachineLayout.Cam7_Pos = value; }

        // ===== 2. 檢測配方 =====
        [Category("2. 檢測配方")][DisplayName("平均閾值")] public float ErrorValueMean   { get => Recipe.ErrorValueMean;   set => Recipe.ErrorValueMean   = value; }
        [Category("2. 檢測配方")][DisplayName("最大閾值")] public float ErrorValueMax    { get => Recipe.ErrorValueMax;    set => Recipe.ErrorValueMax    = value; }
        [Category("2. 檢測配方")][DisplayName("正規值")] public float HessianMaxFactor { get => Recipe.HessianMaxFactor; set => Recipe.HessianMaxFactor = value; }


        // ===== 3. 儲存設定 =====
        [Category("3. 儲存設定")][DisplayName("存檔")]       public bool   EnableAutoCapture    { get => Storage.EnableAutoCapture;    set => Storage.EnableAutoCapture    = value; }
        [Category("3. 儲存設定")][DisplayName("壓縮")]       public bool   UseCompressedCapture { get => Storage.UseCompressedCapture; set => Storage.UseCompressedCapture = value; }
        [Category("3. 儲存設定")][DisplayName("存檔目錄")] public string CaptureRootPath { get => Storage.CaptureRootPath; set => Storage.CaptureRootPath = value; }

    }
}
