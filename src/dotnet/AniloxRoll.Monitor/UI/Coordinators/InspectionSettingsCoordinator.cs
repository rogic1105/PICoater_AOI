using AniloxRoll.Monitor.Core.Acquisition.Inspection;
using AniloxRoll.Monitor.Core.Data;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Coordinators
{
    /// <summary>
    /// Applies inspection settings to the processing service.
    /// </summary>
    public sealed class InspectionSettingsCoordinator
    {
        private readonly BatchInspectionService _inspectionService;
        private readonly InspectionSettings _settings;

        public InspectionSettingsCoordinator(
            BatchInspectionService inspectionService,
            InspectionSettings settings)
        {
            _inspectionService = inspectionService;
            _settings = settings;
        }

        public void ApplySettingsToService()
        {
            if (_inspectionService == null || _settings == null) return;
            _inspectionService.UpdateAlgorithmParams(
                _settings.HessianMaxFactorV,
                _settings.ErrorValueMeanV,
                _settings.ErrorValueMaxV,
                InspectionRecipe.RidgeDirectionToNative(_settings.RidgeDir));
        }

        public void SetRidgeDirection(string direction)
            => _inspectionService?.SetRidgeDirection(direction);

    }
}
