namespace AniloxRoll.Monitor.Core.Data
{
    internal static class UiInteractionDefaults
    {
        public const decimal CamOpsStep = 0.1m;
        public const decimal RollSpeedStep = 1m;
        public const decimal CameraStartStep = 1m;
        public const decimal CropStep = 1m;
        public const decimal BackgroundSampleSecondsStep = 1m;
        public const decimal ColumnNormalizationStep = 0.1m;
        public const decimal RowNormalizationStep = 0.1m;
        public const decimal ThinLineRemovalStep = 1m;
        public const decimal ColumnMeanThresholdStep = 0.1m;
        public const decimal ColumnMaxThresholdStep = 0.1m;
        public const decimal RowMeanThresholdStep = 0.1m;
        public const decimal RowMaxThresholdStep = 0.1m;
        public const decimal CaptureDurationSecondsStep = 1m;
        public const decimal WaterfallHeightStep = 1000m;
        public const decimal YearlyYieldStep = 1000m;
        public const decimal MonthlyYieldStep = 100m;
        public const decimal DailyYieldStep = 10m;
        public const decimal LocalFreeSpaceGbStep = 1m;
        public const decimal LogRetentionHoursStep = 1m;
        public const decimal LightChannelStep = 1m;
        public const decimal LightBrightnessStep = 1m;
        public const decimal IoPortStep = 1m;
    }
}
