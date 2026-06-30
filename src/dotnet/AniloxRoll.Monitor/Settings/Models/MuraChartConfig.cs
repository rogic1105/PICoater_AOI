using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// Mura 圖表閾值的 PropertyGrid 展開代理。
    /// 實際資料存在 InspectionRecipe，此類只做轉發。
    /// </summary>
    [TypeConverter(typeof(ExpandableLeftAlignConverter))]
    public class MuraChartConfig
    {
        private readonly InspectionRecipe _recipe;

        public MuraChartConfig(InspectionRecipe recipe)
        {
            _recipe = recipe;
        }

        [DisplayName("欄平均閾值")][TypeConverter(typeof(LeftAlignNumericConverter))] public float ErrorValueMeanV { get => _recipe.ErrorValueMeanV; set => _recipe.ErrorValueMeanV = value; }
        [DisplayName("欄最大閾值")][TypeConverter(typeof(LeftAlignNumericConverter))] public float ErrorValueMaxV  { get => _recipe.ErrorValueMaxV;  set => _recipe.ErrorValueMaxV  = value; }
        [DisplayName("列平均閾值")][TypeConverter(typeof(LeftAlignNumericConverter))] public float ErrorValueMeanH { get => _recipe.ErrorValueMeanH; set => _recipe.ErrorValueMeanH = value; }
        [DisplayName("列最大閾值")][TypeConverter(typeof(LeftAlignNumericConverter))] public float ErrorValueMaxH  { get => _recipe.ErrorValueMaxH;  set => _recipe.ErrorValueMaxH  = value; }

        public override string ToString() => "";
    }
}
