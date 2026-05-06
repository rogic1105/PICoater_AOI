namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>所有 InspectionSettings 子物件的預設值集中定義。
    /// Model 初始值與 ParseJson fallback 均引用此處，確保兩者一致。
    ///
    /// 格式：  預設值常數  ← PropertyGrid 顯示名稱  [Category]
    /// </summary>
    internal static class InspectionDefaults
    {
        // ── 1. 機台佈局 ────────────────────────────────────────────────────
        public const double CamOps      = 33.0;    // OPS (um)       ← Cam 1~7
        public const double CamPos_Cam1 = 0.0;     // Start (mm)     ← Cam 1
        public const double CamPos_Cam2 = 400.0;   // Start (mm)     ← Cam 2
        public const double CamPos_Cam3 = 800.0;   // Start (mm)     ← Cam 3
        public const double CamPos_Cam4 = 1200.0;  // Start (mm)     ← Cam 4
        public const double CamPos_Cam5 = 1600.0;  // Start (mm)     ← Cam 5
        public const double CamPos_Cam6 = 2000.0;  // Start (mm)     ← Cam 6
        public const double CamPos_Cam7 = 2400.0;  // Start (mm)     ← Cam 7
        public const double TrimHeadMm  = 0.0;     // 去頭 (mm)
        public const double TrimTailMm  = 0.0;     // 去尾 (mm)

        // ── 2. 檢測配方 ────────────────────────────────────────────────────
        public static readonly BackgroundAlgorithm Algorithm  = BackgroundAlgorithm.SingleFrameBgSub; // 去背演算法
        public static readonly RidgeDirection       RidgeDir  = RidgeDirection.Vertical;              // 檢出方向
        public const float  HessianMaxFactor        = 1.0f;   // 正規值
        public const float  ErrorValueMean          = 0.3f;   // Mura 圖表 > 平均閾值
        public const float  ErrorValueMax           = 0.5f;   // Mura 圖表 > 最大閾值
        public const int    BackgroundSampleSeconds = 3;      // 取樣秒數
        public const double AniloxRollSpeedMPerMin  = 10.0;   // 輪速 (m/min)

        // ── 3. 圖表設定 ────────────────────────────────────────────────────
        public static readonly ChartScaleMode ScaleMode = ChartScaleMode.Fixed;      // 統計圖表 > 數量範圍
        public const int YearlyYMax  = 60000;      // 統計圖表 > 月產量
        public const int MonthlyYMax = 2000;       // 統計圖表 > 日產量
        public const int DailyYMax   = 300;        // 統計圖表 > 時產量
        public static readonly StitchMode DefaultStitch = StitchMode.Vertical;     // 主畫面 > 合圖方式
        public const bool EnableMuraEnhance   = false;       // 監控強化
        public const bool EnableReviewEnhance = false;       // 回顧強化

        // ── 4. 儲存設定 ────────────────────────────────────────────────────
        public const bool   EnableAutoCapture = true;                              // 存檔
        public const bool   SaveOriginalBmp   = false;                             // 存原圖
        public const string CaptureRootPath   = @"D:\AniloxCaptures";             // 存圖目錄
        public const string BackgroundPath    = @"D:\AniloxCaptures\bg";          // 存背景目錄
        public const int    LocalMinFreeGB    = 100;                               // 預留空間 (GB)
        public const string RemotePath        = @"\\192.168.10.20\AniloxStorage"; // 遠端路徑
        public const string RemoteConfigPath  = @"\\192.168.10.20\AniloxConfig";  // 遠端設定路徑

        // ── 5. IO 設定 ─────────────────────────────────────────────────────
        public const bool   PlcEnabled = true;             // 啟用 IO
        public const string PlcIp      = "192.168.255.1"; // IO IP
        public const int    PlcPort    = 502;              // IO Port

        // ── 6. 相機設定 ────────────────────────────────────────────────────
        public const string DcfPath = @"D:\AniloxCaptures\dcf\Radient_Config.dcf"; // 設定檔

        // ── 7. 光源設定 ────────────────────────────────────────────────────
        public const bool   LightEnabled    = true;    // 啟用光源
        public const string LightComPort    = "COM17"; // COM Port
        public const int    LightChannel    = 1;       // 通道
        public const int    LightBrightness = 128;     // 亮度
        public const int    LightWarmupMs   = 300;     // 暖機延遲 (ms)
    }
}
