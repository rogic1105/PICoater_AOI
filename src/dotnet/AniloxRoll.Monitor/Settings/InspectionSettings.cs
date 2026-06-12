using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class InspectionSettings
    {
        // 子物件隱藏於 PropertyGrid，屬性直接平鋪於各 Category（兩層顯示）
        [Browsable(false)] public MachineLayoutConfig MachineLayout { get; set; } = new MachineLayoutConfig();
        [Browsable(false)] public AcquisitionSettings Acquisition   { get; set; } = new AcquisitionSettings();
        [Browsable(false)] public InspectionRecipe    Recipe        { get; set; } = new InspectionRecipe();
        [Browsable(false)] public ChartSettings       Chart         { get; set; } = new ChartSettings();
        [Browsable(false)] public ImageViewSettings   ImageView     { get; set; } = new ImageViewSettings();
        [Browsable(false)] public StorageSettings     Storage       { get; set; } = new StorageSettings();
        [Browsable(false)] public CameraParamSettings CameraParam   { get; set; } = new CameraParamSettings();
        [Browsable(false)] public LightSettings      Light         { get; set; } = new LightSettings();

        public void Validate()
        {
            if (MachineLayout == null) MachineLayout = new MachineLayoutConfig();
            if (Acquisition == null) Acquisition = new AcquisitionSettings();
            if (Recipe == null) Recipe = new InspectionRecipe();
            if (Chart == null) Chart = new ChartSettings();
            if (ImageView == null) ImageView = new ImageViewSettings();
            if (Storage == null) Storage = new StorageSettings();
            if (CameraParam == null) CameraParam = new CameraParamSettings();
            if (Light == null) Light = new LightSettings();

            MachineLayout.Validate();
            Acquisition.Validate();
            Recipe.Validate();
            Chart.Validate();
            ImageView.Validate();
            Storage.Validate();
            CameraParam.Validate();
            Light.Validate();
        }

        public double[] GetCameraOpsUmArray() => MachineLayout.GetCameraOpsUmArray();
        public double[] GetCameraStartPositionMmArray() => MachineLayout.GetCameraStartPositionMmArray();

        // ===== 0. 機台設定（寫入 app-mode.json，不存入 inspection-settings.json）=====
        [Category("0. 機台設定")]
        [DisplayName("機台角色")]
        [Description("檢測模式 = 檢測機；儲存模式 = 儲存機。變更後重開程式生效。")]
        public MachineRole AppRole { get; set; } = MachineRole.Inspection;

        // ===== 1. 機台佈局 =====
        // 屬性名前綴 aa/ab.../ba/bb... 控制 PropertyGrid 字母排序順序
        [Category("1. 機台佈局")][DisplayName("─ OPS (um) ─")][ReadOnly(true)]
        public string aa_OpsHeader => "";
        [Category("1. 機台佈局")][DisplayName("Cam 1")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ab_OpsCam1 { get => MachineLayout.Ops.Cam1; set => MachineLayout.Ops.Cam1 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 2")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ac_OpsCam2 { get => MachineLayout.Ops.Cam2; set => MachineLayout.Ops.Cam2 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 3")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ad_OpsCam3 { get => MachineLayout.Ops.Cam3; set => MachineLayout.Ops.Cam3 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 4")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ae_OpsCam4 { get => MachineLayout.Ops.Cam4; set => MachineLayout.Ops.Cam4 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 5")][TypeConverter(typeof(LeftAlignNumericConverter))] public double af_OpsCam5 { get => MachineLayout.Ops.Cam5; set => MachineLayout.Ops.Cam5 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 6")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ag_OpsCam6 { get => MachineLayout.Ops.Cam6; set => MachineLayout.Ops.Cam6 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 7")][TypeConverter(typeof(LeftAlignNumericConverter))] public double ah_OpsCam7 { get => MachineLayout.Ops.Cam7; set => MachineLayout.Ops.Cam7 = value; }
        [Category("1. 機台佈局")][DisplayName("A輪速度 (m/min)")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public double ai_OpsSpeed { get => Recipe.AniloxRollSpeedMPerMin; set => Recipe.AniloxRollSpeedMPerMin = value; }

        [Category("1. 機台佈局")][DisplayName("─ Start (mm) ─")][ReadOnly(true)]
        public string ba_StartHeader => "";
        [Category("1. 機台佈局")][DisplayName("Cam 1")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bb_StartCam1 { get => MachineLayout.StartPosition.Cam1; set => MachineLayout.StartPosition.Cam1 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 2")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bc_StartCam2 { get => MachineLayout.StartPosition.Cam2; set => MachineLayout.StartPosition.Cam2 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 3")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bd_StartCam3 { get => MachineLayout.StartPosition.Cam3; set => MachineLayout.StartPosition.Cam3 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 4")][TypeConverter(typeof(LeftAlignNumericConverter))] public double be_StartCam4 { get => MachineLayout.StartPosition.Cam4; set => MachineLayout.StartPosition.Cam4 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 5")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bf_StartCam5 { get => MachineLayout.StartPosition.Cam5; set => MachineLayout.StartPosition.Cam5 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 6")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bg_StartCam6 { get => MachineLayout.StartPosition.Cam6; set => MachineLayout.StartPosition.Cam6 = value; }
        [Category("1. 機台佈局")][DisplayName("Cam 7")][TypeConverter(typeof(LeftAlignNumericConverter))] public double bh_StartCam7 { get => MachineLayout.StartPosition.Cam7; set => MachineLayout.StartPosition.Cam7 = value; }

        [Category("1. 機台佈局")][DisplayName("─ Crop (mm) ─")][ReadOnly(true)]
        public string ca_CropHeader => "";
        [Category("1. 機台佈局")][DisplayName("去頭")][TypeConverter(typeof(LeftAlignNumericConverter))] public double cb_CropHead { get => MachineLayout.Crop.TrimHeadMm; set => MachineLayout.Crop.TrimHeadMm = value; }
        [Category("1. 機台佈局")][DisplayName("去尾")][TypeConverter(typeof(LeftAlignNumericConverter))] public double cc_CropTail { get => MachineLayout.Crop.TrimTailMm; set => MachineLayout.Crop.TrimTailMm = value; }

        // 向後相容：CsvConfigSnapshot.FromSettings 直接取值
        [Browsable(false)] public double TrimHeadMm => MachineLayout.TrimHeadMm;
        [Browsable(false)] public double TrimTailMm => MachineLayout.TrimTailMm;

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public CameraOpsConfig Ops => MachineLayout.Ops;
        [Browsable(false)] public CameraStartPositionConfig StartPosition => MachineLayout.StartPosition;
        [Browsable(false)] public CameraCropConfig Crop => MachineLayout.Crop;
        [Browsable(false)] public double Cam1_Ops { get => MachineLayout.Cam1_Ops; set => MachineLayout.Cam1_Ops = value; }
        [Browsable(false)] public double Cam2_Ops { get => MachineLayout.Cam2_Ops; set => MachineLayout.Cam2_Ops = value; }
        [Browsable(false)] public double Cam3_Ops { get => MachineLayout.Cam3_Ops; set => MachineLayout.Cam3_Ops = value; }
        [Browsable(false)] public double Cam4_Ops { get => MachineLayout.Cam4_Ops; set => MachineLayout.Cam4_Ops = value; }
        [Browsable(false)] public double Cam5_Ops { get => MachineLayout.Cam5_Ops; set => MachineLayout.Cam5_Ops = value; }
        [Browsable(false)] public double Cam6_Ops { get => MachineLayout.Cam6_Ops; set => MachineLayout.Cam6_Ops = value; }
        [Browsable(false)] public double Cam7_Ops { get => MachineLayout.Cam7_Ops; set => MachineLayout.Cam7_Ops = value; }

        // ===== 2. 檢測設定 =====
        [Category("2. 檢測設定")][DisplayName("─ 演算法 ─")][ReadOnly(true)]
        public string da_AlgorithmHeader => "";
        [Category("2. 檢測設定")][DisplayName("去背演算法")]
        public BackgroundAlgorithm db_Algorithm { get => Recipe.Algorithm; set => Recipe.Algorithm = value; }
        [Category("2. 檢測設定")][DisplayName("垂直正規值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float dc_HessianMaxFactorV { get => Recipe.HessianMaxFactorV; set => Recipe.HessianMaxFactorV = value; }
        [Category("2. 檢測設定")][DisplayName("水平正規值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float dd_HessianMaxFactorH { get => Recipe.HessianMaxFactorH; set => Recipe.HessianMaxFactorH = value; }

        [Category("2. 檢測設定")][DisplayName("─ 檢出標準 ─")][ReadOnly(true)]
        public string ea_DetectionHeader => "";
        [Category("2. 檢測設定")][DisplayName("檢出方向")]
        public RidgeDirection eb_RidgeDir { get => Recipe.RidgeDir; set => Recipe.RidgeDir = value; }
        [Category("2. 檢測設定")][DisplayName("垂直平均閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ec_ErrorValueMeanV { get => Recipe.ErrorValueMeanV; set => Recipe.ErrorValueMeanV = value; }
        [Category("2. 檢測設定")][DisplayName("垂直最大閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ed_ErrorValueMaxV  { get => Recipe.ErrorValueMaxV;  set => Recipe.ErrorValueMaxV  = value; }
        [Category("2. 檢測設定")][DisplayName("水平平均閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ee_ErrorValueMeanH { get => Recipe.ErrorValueMeanH; set => Recipe.ErrorValueMeanH = value; }
        [Category("2. 檢測設定")][DisplayName("水平最大閾值")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public float ef_ErrorValueMaxH  { get => Recipe.ErrorValueMaxH;  set => Recipe.ErrorValueMaxH  = value; }

        [Category("2. 檢測設定")][DisplayName("─ 背景校正 ─")][ReadOnly(true)]
        public string fa_BgHeader => "";
        [Category("2. 檢測設定")][DisplayName("取時間 (sec)")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public int fb_BackgroundSampleSeconds { get => Recipe.BackgroundSampleSeconds; set => Recipe.BackgroundSampleSeconds = value; }

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public BackgroundAlgorithm Algorithm       { get => Recipe.Algorithm;       set => Recipe.Algorithm       = value; }
        [Browsable(false)] public RidgeDirection      RidgeDir        { get => Recipe.RidgeDir;        set => Recipe.RidgeDir        = value; }
        [Browsable(false)] public float  HessianMaxFactorV      { get => Recipe.HessianMaxFactorV;      set => Recipe.HessianMaxFactorV      = value; }
        [Browsable(false)] public float  HessianMaxFactorH      { get => Recipe.HessianMaxFactorH;      set => Recipe.HessianMaxFactorH      = value; }
        [Browsable(false)] public int    BackgroundSampleSeconds { get => Recipe.BackgroundSampleSeconds; set => Recipe.BackgroundSampleSeconds = value; }
        [Browsable(false)] public double AniloxRollSpeedMPerMin  { get => Recipe.AniloxRollSpeedMPerMin;  set => Recipe.AniloxRollSpeedMPerMin  = value; }

        // ===== 3. 圖表設定 =====
        [Category("3. 圖表設定")][DisplayName("─ 檢測報表 ─")][ReadOnly(true)]
        public string ga_ChartHeader => "";
        [Category("3. 圖表設定")][DisplayName("y座標")]
        public ChartScaleMode gb_ChartScaleMode { get => Chart.ScaleMode; set => Chart.ScaleMode = value; }
        [Category("3. 圖表設定")][DisplayName("月產量")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public int gc_YearlyYMax { get => Chart.YearlyYMax; set => Chart.YearlyYMax = value; }
        [Category("3. 圖表設定")][DisplayName("日產量")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public int gd_MonthlyYMax { get => Chart.MonthlyYMax; set => Chart.MonthlyYMax = value; }
        [Category("3. 圖表設定")][DisplayName("時產量")][TypeConverter(typeof(LeftAlignNumericConverter))]
        public int ge_DailyYMax { get => Chart.DailyYMax; set => Chart.DailyYMax = value; }

        [Category("3. 圖表設定")][DisplayName("─ 主畫面 ─")][ReadOnly(true)]
        public string ha_DisplayHeader => "";
        // 合圖方式選項已退場（2026-06-13 上機決策：app 永遠 Global 合圖；單張模式留 sample）。
        // PG 隱藏 + setter 強制 Global（絞殺式：Vertical 分支變死路，Stage4 刪死碼）。
        [Browsable(false)]
        public StitchMode hb_StitchMode { get => StitchMode.Global; set => ImageView.StitchMode = StitchMode.Global; }
        [Category("3. 圖表設定")][DisplayName("監控強化")][TypeConverter(typeof(BoolYesNoConverter))]
        public bool hc_EnableMuraEnhance { get => ImageView.EnableMuraEnhance; set => ImageView.EnableMuraEnhance = value; }
        [Category("3. 圖表設定")][DisplayName("回顧強化")][TypeConverter(typeof(BoolYesNoConverter))]
        public bool hd_EnableReviewEnhance { get => ImageView.EnableReviewEnhance; set => ImageView.EnableReviewEnhance = value; }
        [Category("3. 圖表設定")][DisplayName("主畫面顯示")][Description("MilDirect=MIL 直繪（現狀）；SmartCanvas=CPU 繪、跟回顧畫布同源。變更後重開抓取生效。")]
        public MainDisplayMode he_MainDisplay { get => ImageView.MainDisplay; set => ImageView.MainDisplay = value; }
        [Category("3. 圖表設定")][DisplayName("動態LOD")][Description("Off=關；GPU=TanukiCv GPU 縮；CPU=純 CPU 縮。放大巨圖看細節用（顯示成本大降）。SmartCanvas 模式即時生效。")]
        public LiveLodMode hf_LiveLod { get => ImageView.LiveLod; set => ImageView.LiveLod = value; }

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public bool EnableMuraEnhance   { get => ImageView.EnableMuraEnhance;   set => ImageView.EnableMuraEnhance   = value; }
        [Browsable(false)] public bool EnableReviewEnhance { get => ImageView.EnableReviewEnhance; set => ImageView.EnableReviewEnhance = value; }
        [Browsable(false)] public ChartScaleMode ChartScaleMode { get => Chart.ScaleMode; set => Chart.ScaleMode = value; }
        [Browsable(false)] public int ChartDataYieldYearlyYMax  { get => Chart.YearlyYMax;  set => Chart.YearlyYMax  = value; }
        [Browsable(false)] public int ChartDataYieldMonthlyYMax { get => Chart.MonthlyYMax; set => Chart.MonthlyYMax = value; }
        [Browsable(false)] public int ChartDataYieldDailyYMax   { get => Chart.DailyYMax;   set => Chart.DailyYMax   = value; }
        [Browsable(false)] public StitchMode StitchMode { get => ImageView.StitchMode; set => ImageView.StitchMode = value; }
        [Browsable(false)] public LiveLodMode LiveLod { get => ImageView.LiveLod; set => ImageView.LiveLod = value; }
        [Browsable(false)] public float ErrorValueMeanV { get => Recipe.ErrorValueMeanV; set => Recipe.ErrorValueMeanV = value; }
        [Browsable(false)] public float ErrorValueMaxV  { get => Recipe.ErrorValueMaxV;  set => Recipe.ErrorValueMaxV  = value; }
        [Browsable(false)] public float ErrorValueMeanH { get => Recipe.ErrorValueMeanH; set => Recipe.ErrorValueMeanH = value; }
        [Browsable(false)] public float ErrorValueMaxH  { get => Recipe.ErrorValueMaxH;  set => Recipe.ErrorValueMaxH  = value; }

        // ===== 4. 儲存設定 =====
        [Category("4. 儲存設定")][DisplayName("Anilox 根目錄")][PropertyOrder(1)]  public string AniloxRootPath       { get => Storage.AniloxRootPath;       set => Storage.AniloxRootPath       = value; }
        // 子目錄路徑：PropertyGrid 不顯示，由 AniloxRootPath 推算
        [Browsable(false)] public string CaptureRootPath => Storage.CaptureRootPath;
        [Category("4. 儲存設定")][DisplayName("預留空間 (GB)")][PropertyOrder(2)][TypeConverter(typeof(LeftAlignNumericConverter))] public int    LocalMinFreeGB       { get => Storage.LocalMinFreeGB;       set => Storage.LocalMinFreeGB       = value; }
        [Category("4. 儲存設定")][DisplayName("遠端路徑")][PropertyOrder(3)]      public string RemotePath           { get => Storage.RemotePath;           set => Storage.RemotePath           = value; }
        [Category("4. 儲存設定")][DisplayName("存檔")][PropertyOrder(4)][TypeConverter(typeof(BoolYesNoConverter))]          public bool   EnableAutoCapture    { get => Storage.EnableAutoCapture;    set => Storage.EnableAutoCapture    = value; }
        [Category("4. 儲存設定")][DisplayName("存原圖")][PropertyOrder(5)][TypeConverter(typeof(BoolYesNoConverter))]        public bool   SaveOriginalBmp      { get => Storage.SaveOriginalBmp;      set => Storage.SaveOriginalBmp      = value; }
        // 開發者設定：PropertyGrid 不顯示，部署時直接改 JSON
        [Browsable(false)] public string RemoteConfigPath { get => Storage.RemoteConfigPath; set => Storage.RemoteConfigPath = value; }

        // DcfPath 固定為 Config\Radient_Config.dcf（跟 exe 走，build 自動複製）；PG 隱藏不讓使用者改
        [Browsable(false)] public string DcfPath { get => CameraParam.DcfPath; set => CameraParam.DcfPath = value; }

        // ===== 6. 光源設定 =====
        [Category("5. 光源設定")][DisplayName("啟用光源")][TypeConverter(typeof(BoolYesNoConverter))]      public bool   LightEnabled    { get => Light.Enabled;    set => Light.Enabled    = value; }
        [Category("5. 光源設定")][DisplayName("COM Port")]      public string LightComPort    { get => Light.ComPort;    set => Light.ComPort    = value; }
        [Category("5. 光源設定")][DisplayName("通道")][TypeConverter(typeof(LeftAlignNumericConverter))]          public int    LightChannel    { get => Light.Channel;    set => Light.Channel    = value; }
        [Category("5. 光源設定")][DisplayName("亮度")][TypeConverter(typeof(LeftAlignNumericConverter))]          public int    LightBrightness { get => Light.Brightness; set => Light.Brightness = value; }
        [Category("5. 光源設定")][DisplayName("暖機延遲 (ms)")][TypeConverter(typeof(LeftAlignNumericConverter))] public int    LightWarmupMs   { get => Light.WarmupMs;   set => Light.WarmupMs   = value; }

        // ===== 7. IO 設定 =====
        [Category("6. IO設定")][DisplayName("IO IP")][PropertyOrder(1)]   public string IoIp      { get; set; } = InspectionDefaults.IoIp;
        [Category("6. IO設定")][DisplayName("IO 型號")][PropertyOrder(2)] public string IoModel { get; set; } = InspectionDefaults.IoModel;
        [Category("6. IO設定")][DisplayName("啟用 IO")][PropertyOrder(3)][TypeConverter(typeof(BoolYesNoConverter))]  public bool   IoEnabled { get; set; } = InspectionDefaults.IoEnabled;
        [Category("6. IO設定")][DisplayName("IO Port")][PropertyOrder(4)][TypeConverter(typeof(LeftAlignNumericConverter))]  public int    IoPort    { get; set; } = InspectionDefaults.IoPort;
        // Mura 檢出（DO1 MURA_DET）：runtime toggle，不持久化 — 每次啟動為 false（避免漏檢）
        [Category("6. IO設定")][DisplayName("Mura檢出")][PropertyOrder(5)][TypeConverter(typeof(BoolYesNoConverter))]  public bool   MuraDetectPaused { get; set; } = false;

        // ===== 8. 開發者（PG 隱藏，編輯 inspection-settings.json 啟用） =====
        // FSM Action Logger 開關。對應 docs/dev/fsm/ + Services/UiActionLogger.cs。
        // 預設 false（production 零 overhead）；改 true 後重啟程式生效，log 寫 D:\Anilox\Logs\ui-actions-YYYYMMDD.jsonl。
        [Browsable(false)] public bool DebugUiActionLog { get; set; } = false;
    }
}
