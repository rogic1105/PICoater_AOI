namespace AniloxRoll.Monitor.Core.Services
{
    public static class InspectionEngineConfig
    {
        public const int MaxWidth = 16384;
        public const int MaxHeight = 10000;
        public const int MaxThumbnailSide = 2000;

        /// <summary>背景採集（ComputeColumnMean / StandardBgSub 取背景）的離群門檻。</summary>
        public const float DefaultBgSigma = 2.0f;
        /// <summary>每幀去背（Process 內 Step1 column mean）的離群門檻。
        /// 歷史：舊 native 在每幀路徑硬編 sigma=1、無視傳入參數；4b 參數 honest 化後由 app 明確傳 1.0 → 行為不變。
        /// 與 DefaultBgSigma（背景採集 2.0）本來就是兩個不同 sigma，勿合併。</summary>
        public const float PerFrameBgSigma = 1.0f;
        public const float DefaultRidgeSigma = 9.0f;
        public const float DefaultHessianMaxFactor = 2.0f;

        public const string DefaultRidgeMode = "vertical";

        // 壓縮存檔預設值（InspectionRecipe.SaveResizeScale / SaveJpgQuality 的唯一參考來源）
        public const int DefaultSaveResizeScale = 5;
        public const int DefaultSaveJpgQuality  = 90;
    }
}
