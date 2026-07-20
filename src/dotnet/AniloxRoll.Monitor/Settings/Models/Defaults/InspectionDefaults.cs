namespace AniloxRoll.Monitor.Core.Data
{
    using TanukiCv.Controls;

    /// <summary>所有 InspectionSettings 子物件的預設值集中定義。
    /// Model 初始值與 ParseJson fallback 均引用此處，確保兩者一致。
    ///
    /// 格式：  預設值常數  ← PropertyGrid 顯示名稱  [Category]
    /// </summary>
    internal static class InspectionDefaults
    {
        // ── 1. 機台佈局 ────────────────────────────────────────────────────
        public const double CamOps      = 24.4140625; // OPS (um)       ← Cam 1~7（PICoater 機台實測）
        public const double CamPos_Cam1 = 0.0;        // Start (mm)     ← Cam 1
        public const double CamPos_Cam2 = 345.0;      // Start (mm)     ← Cam 2
        public const double CamPos_Cam3 = 690.0;      // Start (mm)     ← Cam 3
        public const double CamPos_Cam4 = 1035.0;     // Start (mm)     ← Cam 4
        public const double CamPos_Cam5 = 1380.0;     // Start (mm)     ← Cam 5
        public const double CamPos_Cam6 = 1725.0;     // Start (mm)     ← Cam 6
        public const double CamPos_Cam7 = 2070.0;     // Start (mm)     ← Cam 7
        public const double TrimHeadMm  = 0.0;        // 去頭 (mm)
        public const double TrimTailMm  = 0.0;        // 去尾 (mm)

        // ── 2. 檢測配方 ────────────────────────────────────────────────────
        public static readonly BackgroundAlgorithm Algorithm  = BackgroundAlgorithm.SingleFrameBgSub; // 去背演算法
        public static readonly RidgeDirection       RidgeDir  = RidgeDirection.Both;                 // 檢出方向
        public const float  HessianMaxFactorV       = 0.3f;   // 欄正規值（同時當作 capture-time 送進 native 的 HM）
        public const float  HessianMaxFactorH       = 0.3f;   // 列正規值（view-time only，僅作 H 曲線顯示縮放）
        public const float  ErrorValueMeanV         = 0.2f;   // Mura 圖表 > 欄平均閾值
        public const float  ErrorValueMaxV          = 0.6f;   // Mura 圖表 > 欄最大閾值（PICoater 機台實測）
        public const float  ErrorValueMeanH         = 0.2f;   // Mura 圖表 > 列平均閾值
        public const float  ErrorValueMaxH          = 0.6f;   // Mura 圖表 > 列最大閾值（PICoater 機台實測）
        public const int    BackgroundSampleSeconds = 3;      // 背景採樣秒數
        public const int    GrabLimitSeconds        = 10;     // 單次抓取上限秒數
        public const double AniloxRollSpeedMPerMin  = 40.0;   // 輪速 (m/min)

        // ── 3. 圖表設定 ────────────────────────────────────────────────────
        public static readonly ChartScaleMode ScaleMode = ChartScaleMode.Auto;       // 統計圖表 > 數量範圍
        public const int YearlyYMax  = 50000;      // 統計圖表 > 月產量
        public const int MonthlyYMax = 2000;       // 統計圖表 > 日產量
        public const int DailyYMax   = 300;        // 統計圖表 > 時產量
        public static readonly StitchMode DefaultStitch = StitchMode.Global;        // 主畫面 > 合圖方式
        public const bool EnableMuraEnhance   = false;       // 監控強化
        public const bool EnableReviewEnhance = false;       // 回顧強化
        public static readonly MainDisplayMode MainDisplay = MainDisplayMode.Waterfall; // 主畫面顯示（即時 / 瀑布）
        public static readonly VerticalDisplayDirection VerticalDirection = VerticalDisplayDirection.BottomToTop; // 主畫面上下方向
        public static readonly LiveLodMode     LiveLod     = LiveLodMode.CPU;             // 動態LOD（Off / GPU / CPU）
        public const int       WaterfallTotalHeight = 30000;                              // 瀑布圖虛擬長圖總高（px）；點兩下 fit 到此
        public static readonly WaterfallFullMode WaterfallFullMode = WaterfallFullMode.Restart; // 瀑布滿了：重來 / 循環

        // ── 4. 儲存設定 ────────────────────────────────────────────────────
        public const bool   EnableAutoCapture = true;                              // 存檔
        public const bool   SaveOriginalBmp   = false;                             // 存原圖
        public const string AniloxRootPath    = @"D:\Anilox";                      // Anilox 根目錄（Captures/Bg/Logs 為子目錄；Dcf 跟 exe 走）
        public const int    LocalMinFreeGB    = 100;                               // 預留空間 (GB)
        public const int    LogRetentionHours = 168;                               // Log 保留 7 天
        public const string RemotePath        = @"\\192.168.10.20\Anilox\Captures"; // 遠端路徑
        public const string RemoteConfigPath  = @"\\192.168.10.20\Anilox\Config";   // 遠端設定路徑

        // ── 5. IO 設定 ─────────────────────────────────────────────────────
        public const string IoModel   = "ET-7044";        // IO 型號（對應 IoModuleFactory）
        public const bool   IoEnabled = true;             // 啟用 IO
        public const string IoIp      = "192.168.255.1"; // IO IP
        public const int    IoPort    = 502;              // IO Port

        // ── 6. 相機設定 ────────────────────────────────────────────────────
        public const string DcfPath = @"Config\Radient_Config.dcf";        // 設定檔（相對 exe，build 自動複製）

        // ── 7. 光源設定 ────────────────────────────────────────────────────
        public const bool   LightEnabled    = true;    // 啟用光源
        public const string LightComPort    = "COM17"; // COM Port
        public const int    LightChannel    = 1;       // 通道
        public const int    LightBrightness = 255;     // 亮度
        public const int    LightWarmupMs   = 300;     // 暖機延遲 (ms)
    }
}
