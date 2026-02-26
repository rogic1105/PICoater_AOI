using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [TypeConverter(typeof(ExpandableObjectConverter))]
    public class InspectionRecipe
    {
        [DisplayName("Hessian Max Factor")] public float HessianMaxFactor { get; set; } = 5.0f;
        [DisplayName("Error Value Mean")] public float ErrorValueMean { get; set; } = 1.0f;
        [DisplayName("Error Value Max")] public float ErrorValueMax { get; set; } = 2.0f;

        public void Validate()
        {
            if (HessianMaxFactor <= 0) HessianMaxFactor = 5.0f;
            if (ErrorValueMean <= 0) ErrorValueMean = 1.0f;
            if (ErrorValueMax <= 0) ErrorValueMax = 2.0f;
        }

        public override string ToString() => "Inspection Recipe";
    }
}
