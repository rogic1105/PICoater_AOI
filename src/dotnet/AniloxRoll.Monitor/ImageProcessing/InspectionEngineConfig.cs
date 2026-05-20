namespace AniloxRoll.Monitor.Core.Services
{
    public static class InspectionEngineConfig
    {
        public const int MaxWidth = 16384;
        public const int MaxHeight = 10000;
        public const int MaxThumbnailSide = 2000;

        public const float DefaultBgSigma = 2.0f;
        public const float DefaultRidgeSigma = 9.0f;
        public const float DefaultHessianMaxFactor = 2.0f;

        public const string DefaultRidgeMode = "vertical";

        // 壓縮存檔預設值（InspectionRecipe.SaveResizeScale / SaveJpgQuality 的唯一參考來源）
        public const int DefaultSaveResizeScale = 5;
        public const int DefaultSaveJpgQuality  = 90;
    }
}
