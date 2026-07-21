using AniloxRoll.Monitor.Core.Data;

namespace AniloxRoll.Monitor.Core.Services
{
    internal struct RowCurvePhysicalScale
    {
        public double SpeedMPerMin;
        public double LineRateHz;
    }

    /// <summary>
    /// Resolves the capture-time inputs used by both Review and Data row charts.
    /// Older CFG records omit speed, so they fall back to the current recipe.
    /// </summary>
    internal static class RowCurvePhysicalScaleResolver
    {
        public static RowCurvePhysicalScale Resolve(
            CsvConfigSnapshot config,
            InspectionSettings settings)
        {
            double speed = config?.AniloxRollSpeedMPerMin ?? 0;
            if (speed <= 0)
                speed = settings?.AniloxRollSpeedMPerMin ?? 0;

            double lineRate = 0;
            if (config?.CamLineRateHz != null && config.CamLineRateHz.Length > 0)
                lineRate = config.CamLineRateHz[0];
            if (lineRate <= 0 && settings?.Acquisition?.CameraLineRateHz != null &&
                settings.Acquisition.CameraLineRateHz.Length > 0)
                lineRate = settings.Acquisition.CameraLineRateHz[0];

            return new RowCurvePhysicalScale
            {
                SpeedMPerMin = speed,
                LineRateHz = lineRate
            };
        }
    }
}
