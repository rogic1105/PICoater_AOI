using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    // OPS 展開：Cam1~7 + A輪速度
    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class OpsDisplayConfig
    {
        private readonly CameraOpsConfig  _ops;
        private readonly InspectionRecipe _recipe;

        public OpsDisplayConfig(CameraOpsConfig ops, InspectionRecipe recipe)
        { _ops = ops; _recipe = recipe; }

        [DisplayName("Cam 1")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam1 { get => _ops.Cam1; set => _ops.Cam1 = value; }
        [DisplayName("Cam 2")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam2 { get => _ops.Cam2; set => _ops.Cam2 = value; }
        [DisplayName("Cam 3")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam3 { get => _ops.Cam3; set => _ops.Cam3 = value; }
        [DisplayName("Cam 4")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam4 { get => _ops.Cam4; set => _ops.Cam4 = value; }
        [DisplayName("Cam 5")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam5 { get => _ops.Cam5; set => _ops.Cam5 = value; }
        [DisplayName("Cam 6")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam6 { get => _ops.Cam6; set => _ops.Cam6 = value; }
        [DisplayName("Cam 7")][TypeConverter(typeof(LeftAlignNumericConverter))] public double Cam7 { get => _ops.Cam7; set => _ops.Cam7 = value; }

        [DisplayName("A輪速度 (m/min)")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public double AniloxRollSpeedMPerMin { get => _recipe.AniloxRollSpeedMPerMin; set => _recipe.AniloxRollSpeedMPerMin = value; }

        public override string ToString() => "";
    }

    // 演算法展開：去背演算法 + 正規值
    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class AlgorithmDisplayConfig
    {
        private readonly InspectionRecipe _recipe;
        public AlgorithmDisplayConfig(InspectionRecipe recipe) { _recipe = recipe; }

        [DisplayName("去背演算法")]
        public BackgroundAlgorithm Algorithm { get => _recipe.Algorithm; set => _recipe.Algorithm = value; }

        [DisplayName("正規值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float HessianMaxFactor { get => _recipe.HessianMaxFactor; set => _recipe.HessianMaxFactor = value; }

        public override string ToString() => "";
    }

    // 檢出標準展開：檢出方向 + 平均閾值 + 最大閾值
    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class DetectionStandardConfig
    {
        private readonly InspectionRecipe _recipe;
        public DetectionStandardConfig(InspectionRecipe recipe) { _recipe = recipe; }

        [DisplayName("檢出方向")]
        public RidgeDirection RidgeDir { get => _recipe.RidgeDir; set => _recipe.RidgeDir = value; }

        [DisplayName("平均閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ErrorValueMean { get => _recipe.ErrorValueMean; set => _recipe.ErrorValueMean = value; }

        [DisplayName("最大閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ErrorValueMax { get => _recipe.ErrorValueMax; set => _recipe.ErrorValueMax = value; }

        public override string ToString() => "";
    }

    // 背景校正展開：取樣秒數
    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class BackgroundCorrectionConfig
    {
        private readonly InspectionRecipe _recipe;
        public BackgroundCorrectionConfig(InspectionRecipe recipe) { _recipe = recipe; }

        [DisplayName("取樣秒數")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public int BackgroundSampleSeconds { get => _recipe.BackgroundSampleSeconds; set => _recipe.BackgroundSampleSeconds = value; }

        public override string ToString() => "";
    }
}
