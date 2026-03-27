using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// Mura 圖表閾值的 PropertyGrid 展開代理。
    /// 實際資料存在 InspectionRecipe，此類只做轉發。
    /// </summary>
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class MuraChartConfig
    {
        private readonly InspectionRecipe _recipe;

        public MuraChartConfig(InspectionRecipe recipe)
        {
            _recipe = recipe;
        }

        [DisplayName("平均閾值")] public float ErrorValueMean { get => _recipe.ErrorValueMean; set => _recipe.ErrorValueMean = value; }
        [DisplayName("最大閾值")] public float ErrorValueMax  { get => _recipe.ErrorValueMax;  set => _recipe.ErrorValueMax  = value; }

        public override string ToString() => "";
    }
}
