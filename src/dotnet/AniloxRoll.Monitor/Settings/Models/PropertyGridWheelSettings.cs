namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// PropertyGrid mouse-wheel increments. These values control editing feel only and are not
    /// part of the inspection recipe or capture CFG snapshot.
    /// </summary>
    public sealed class PropertyGridWheelSettings
    {
        public decimal CamOpsStep { get; set; } = UiInteractionDefaults.CamOpsStep;
        public decimal RollSpeedStep { get; set; } = UiInteractionDefaults.RollSpeedStep;
        public decimal CameraStartStep { get; set; } = UiInteractionDefaults.CameraStartStep;
        public decimal CropStep { get; set; } = UiInteractionDefaults.CropStep;
        public decimal BackgroundSampleSecondsStep { get; set; } = UiInteractionDefaults.BackgroundSampleSecondsStep;
        public decimal ColumnNormalizationStep { get; set; } = UiInteractionDefaults.ColumnNormalizationStep;
        public decimal RowNormalizationStep { get; set; } = UiInteractionDefaults.RowNormalizationStep;
        public decimal ThinLineRemovalStep { get; set; } = UiInteractionDefaults.ThinLineRemovalStep;
        public decimal ColumnMeanThresholdStep { get; set; } = UiInteractionDefaults.ColumnMeanThresholdStep;
        public decimal ColumnMaxThresholdStep { get; set; } = UiInteractionDefaults.ColumnMaxThresholdStep;
        public decimal RowMeanThresholdStep { get; set; } = UiInteractionDefaults.RowMeanThresholdStep;
        public decimal RowMaxThresholdStep { get; set; } = UiInteractionDefaults.RowMaxThresholdStep;
        public decimal CaptureDurationSecondsStep { get; set; } = UiInteractionDefaults.CaptureDurationSecondsStep;
        public decimal WaterfallHeightStep { get; set; } = UiInteractionDefaults.WaterfallHeightStep;
        public decimal YearlyYieldStep { get; set; } = UiInteractionDefaults.YearlyYieldStep;
        public decimal MonthlyYieldStep { get; set; } = UiInteractionDefaults.MonthlyYieldStep;
        public decimal DailyYieldStep { get; set; } = UiInteractionDefaults.DailyYieldStep;
        public decimal LocalFreeSpaceGbStep { get; set; } = UiInteractionDefaults.LocalFreeSpaceGbStep;
        public decimal LogRetentionHoursStep { get; set; } = UiInteractionDefaults.LogRetentionHoursStep;
        public decimal LightChannelStep { get; set; } = UiInteractionDefaults.LightChannelStep;
        public decimal LightBrightnessStep { get; set; } = UiInteractionDefaults.LightBrightnessStep;
        public decimal IoPortStep { get; set; } = UiInteractionDefaults.IoPortStep;

        internal void Validate()
        {
            CamOpsStep = PositiveOrDefault(CamOpsStep, UiInteractionDefaults.CamOpsStep);
            RollSpeedStep = PositiveOrDefault(RollSpeedStep, UiInteractionDefaults.RollSpeedStep);
            CameraStartStep = PositiveOrDefault(CameraStartStep, UiInteractionDefaults.CameraStartStep);
            CropStep = PositiveOrDefault(CropStep, UiInteractionDefaults.CropStep);
            BackgroundSampleSecondsStep = PositiveOrDefault(BackgroundSampleSecondsStep, UiInteractionDefaults.BackgroundSampleSecondsStep);
            ColumnNormalizationStep = PositiveOrDefault(ColumnNormalizationStep, UiInteractionDefaults.ColumnNormalizationStep);
            RowNormalizationStep = PositiveOrDefault(RowNormalizationStep, UiInteractionDefaults.RowNormalizationStep);
            ThinLineRemovalStep = PositiveOrDefault(ThinLineRemovalStep, UiInteractionDefaults.ThinLineRemovalStep);
            ColumnMeanThresholdStep = PositiveOrDefault(ColumnMeanThresholdStep, UiInteractionDefaults.ColumnMeanThresholdStep);
            ColumnMaxThresholdStep = PositiveOrDefault(ColumnMaxThresholdStep, UiInteractionDefaults.ColumnMaxThresholdStep);
            RowMeanThresholdStep = PositiveOrDefault(RowMeanThresholdStep, UiInteractionDefaults.RowMeanThresholdStep);
            RowMaxThresholdStep = PositiveOrDefault(RowMaxThresholdStep, UiInteractionDefaults.RowMaxThresholdStep);
            CaptureDurationSecondsStep = PositiveOrDefault(CaptureDurationSecondsStep, UiInteractionDefaults.CaptureDurationSecondsStep);
            WaterfallHeightStep = PositiveOrDefault(WaterfallHeightStep, UiInteractionDefaults.WaterfallHeightStep);
            YearlyYieldStep = PositiveOrDefault(YearlyYieldStep, UiInteractionDefaults.YearlyYieldStep);
            MonthlyYieldStep = PositiveOrDefault(MonthlyYieldStep, UiInteractionDefaults.MonthlyYieldStep);
            DailyYieldStep = PositiveOrDefault(DailyYieldStep, UiInteractionDefaults.DailyYieldStep);
            LocalFreeSpaceGbStep = PositiveOrDefault(LocalFreeSpaceGbStep, UiInteractionDefaults.LocalFreeSpaceGbStep);
            LogRetentionHoursStep = PositiveOrDefault(LogRetentionHoursStep, UiInteractionDefaults.LogRetentionHoursStep);
            LightChannelStep = PositiveOrDefault(LightChannelStep, UiInteractionDefaults.LightChannelStep);
            LightBrightnessStep = PositiveOrDefault(LightBrightnessStep, UiInteractionDefaults.LightBrightnessStep);
            IoPortStep = PositiveOrDefault(IoPortStep, UiInteractionDefaults.IoPortStep);
        }

        private static decimal PositiveOrDefault(decimal value, decimal fallback)
        {
            return value > 0m ? value : fallback;
        }
    }
}
