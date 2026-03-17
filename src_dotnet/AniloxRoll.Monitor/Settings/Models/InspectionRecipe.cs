using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = 1.0f;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = 0.3f;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = 0.5f;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = 1.0f;
            if (ErrorValueMean <= 0) ErrorValueMean = 0.3f;
            if (ErrorValueMax <= 0) ErrorValueMax = 0.5f;
        }

        public override string ToString() => "Inspection Recipe";
    }
}
