using System.ComponentModel;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = 1.0f;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = 0.3f;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = 0.5f;

        /// <summary>存檔縮小倍率。原圖寬高各除以此值後存成 JPEG。唯一預設值來源：InspectionEngineConfig.DefaultSaveResizeScale。</summary>
        [Browsable(false)] public int SaveResizeScale { get; set; } = InspectionEngineConfig.DefaultSaveResizeScale;

        /// <summary>JPEG 存檔品質（1–100）。唯一預設值來源：InspectionEngineConfig.DefaultSaveJpgQuality。</summary>
        [Browsable(false)] public int SaveJpgQuality  { get; set; } = InspectionEngineConfig.DefaultSaveJpgQuality;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = 1.0f;
            if (ErrorValueMean <= 0) ErrorValueMean = 0.3f;
            if (ErrorValueMax <= 0) ErrorValueMax = 0.5f;
            if (SaveResizeScale <= 0) SaveResizeScale = InspectionEngineConfig.DefaultSaveResizeScale;
            if (SaveJpgQuality  < 1 || SaveJpgQuality > 100) SaveJpgQuality = InspectionEngineConfig.DefaultSaveJpgQuality;
        }

        public override string ToString() => "Inspection Recipe";
    }
}
