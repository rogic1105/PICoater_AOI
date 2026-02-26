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
    }
}
