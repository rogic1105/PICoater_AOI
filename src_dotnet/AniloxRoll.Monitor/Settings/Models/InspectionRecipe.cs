using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = 1.0f;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = 0.3f;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = 0.5f;

        /// <summary>存檔縮小倍率（預設 5）。原圖寬高各除以此值後存成 JPEG。</summary>
        [Browsable(false)] public int SaveResizeScale { get; set; } = 5;

        /// <summary>JPEG 存檔品質（1–100，預設 90）。</summary>
        [Browsable(false)] public int SaveJpgQuality  { get; set; } = 90;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = 1.0f;
            if (ErrorValueMean <= 0) ErrorValueMean = 0.3f;
            if (ErrorValueMax <= 0) ErrorValueMax = 0.5f;
            if (SaveResizeScale <= 0) SaveResizeScale = 5;
            if (SaveJpgQuality  < 1 || SaveJpgQuality > 100) SaveJpgQuality = 90;
        }

        public override string ToString() => "Inspection Recipe";
    }
}
