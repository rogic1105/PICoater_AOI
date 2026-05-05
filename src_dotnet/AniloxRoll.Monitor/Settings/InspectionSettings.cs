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
        [Description("Inspection = 檢測機；Storage = 儲存機。變更後重開程式生效。")]
        public MachineRole AppRole { get; set; } = MachineRole.Inspection;

        // ===== 1. 機台佈局 =====
        [Category("1. 機台佈局")][DisplayName("OPS (um)")]
        public CameraOpsConfig Ops => MachineLayout.Ops;

        [Category("1. 機台佈局")][DisplayName("Start (mm)")]
        public CameraStartPositionConfig StartPosition => MachineLayout.StartPosition;

        [Category("1. 機台佈局")][DisplayName("Crop")]
        public CameraCropConfig Crop => MachineLayout.Crop;

        // 向後相容：CsvConfigSnapshot.FromSettings 直接取值
        [Browsable(false)] public double TrimHeadMm => MachineLayout.TrimHeadMm;
        [Browsable(false)] public double TrimTailMm => MachineLayout.TrimTailMm;

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public double Cam1_Ops { get => MachineLayout.Cam1_Ops; set => MachineLayout.Cam1_Ops = value; }
        [Browsable(false)] public double Cam2_Ops { get => MachineLayout.Cam2_Ops; set => MachineLayout.Cam2_Ops = value; }
        [Browsable(false)] public double Cam3_Ops { get => MachineLayout.Cam3_Ops; set => MachineLayout.Cam3_Ops = value; }
        [Browsable(false)] public double Cam4_Ops { get => MachineLayout.Cam4_Ops; set => MachineLayout.Cam4_Ops = value; }
        [Browsable(false)] public double Cam5_Ops { get => MachineLayout.Cam5_Ops; set => MachineLayout.Cam5_Ops = value; }
        [Browsable(false)] public double Cam6_Ops { get => MachineLayout.Cam6_Ops; set => MachineLayout.Cam6_Ops = value; }
        [Browsable(false)] public double Cam7_Ops { get => MachineLayout.Cam7_Ops; set => MachineLayout.Cam7_Ops = value; }

        // ===== 2. 檢測配方 =====
        [Category("2. 檢測配方")][DisplayName("去背演算法")]  public BackgroundAlgorithm Algorithm       { get => Recipe.Algorithm;       set => Recipe.Algorithm       = value; }
        [Category("2. 檢測配方")][DisplayName("Ridge方向")]   public RidgeDirection      RidgeDir        { get => Recipe.RidgeDir;        set => Recipe.RidgeDir        = value; }
        [Category("2. 檢測配方")][DisplayName("正規值")]      public float HessianMaxFactor { get => Recipe.HessianMaxFactor; set => Recipe.HessianMaxFactor = value; }
        [Category("2. 檢測配方")][DisplayName("取樣秒數")]    public int   BackgroundSampleSeconds { get => Recipe.BackgroundSampleSeconds; set => Recipe.BackgroundSampleSeconds = value; }
        [Category("2. 檢測配方")][DisplayName("輪速 (m/min)")] public double AniloxRollSpeedMPerMin { get => Recipe.AniloxRollSpeedMPerMin; set => Recipe.AniloxRollSpeedMPerMin = value; }
        [Category("2. 檢測配方")][DisplayName("監控強化")]        public bool   EnableMuraEnhance     { get => Recipe.EnableMuraEnhance;     set => Recipe.EnableMuraEnhance     = value; }
        [Category("2. 檢測配方")][DisplayName("回顧強化")]        public bool   EnableReviewEnhance   { get => Recipe.EnableReviewEnhance;   set => Recipe.EnableReviewEnhance   = value; }

        // ===== 3. 報表設定 =====
        [Category("3. 報表設定")][DisplayName("統計圖表")]
        public ChartSettings StatisticsChart => Chart;

        private MuraChartConfig _muraChart;
        [Category("3. 報表設定")][DisplayName("Mura 圖表")]
        public MuraChartConfig MuraChart => _muraChart ?? (_muraChart = new MuraChartConfig(Recipe));

        [Category("3. 報表設定")][DisplayName("主畫面")]
        public ImageViewSettings ImageViewDisplay => ImageView;

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public ChartScaleMode ChartScaleMode { get => Chart.ScaleMode; set => Chart.ScaleMode = value; }
        [Browsable(false)] public int ChartYearlyYMax  { get => Chart.YearlyYMax;  set => Chart.YearlyYMax  = value; }
        [Browsable(false)] public int ChartMonthlyYMax { get => Chart.MonthlyYMax; set => Chart.MonthlyYMax = value; }
        [Browsable(false)] public int ChartDailyYMax   { get => Chart.DailyYMax;   set => Chart.DailyYMax   = value; }
        [Browsable(false)] public StitchMode StitchMode { get => ImageView.StitchMode; set => ImageView.StitchMode = value; }
        [Browsable(false)] public float ErrorValueMean { get => Recipe.ErrorValueMean; set => Recipe.ErrorValueMean = value; }
        [Browsable(false)] public float ErrorValueMax  { get => Recipe.ErrorValueMax;  set => Recipe.ErrorValueMax  = value; }

        // ===== 4. 儲存設定 =====
        [Category("4. 儲存設定")][DisplayName("存檔")]       public bool   EnableAutoCapture    { get => Storage.EnableAutoCapture;    set => Storage.EnableAutoCapture    = value; }
        [Category("4. 儲存設定")][DisplayName("存原圖")]     public bool   SaveOriginalBmp      { get => Storage.SaveOriginalBmp;      set => Storage.SaveOriginalBmp      = value; }
        [Category("4. 儲存設定")][DisplayName("存圖目錄")]   public string CaptureRootPath { get => Storage.CaptureRootPath; set => Storage.CaptureRootPath = value; }
        [Category("4. 儲存設定")][DisplayName("存背景目錄")] public string BackgroundPath  { get => Storage.BackgroundPath;  set => Storage.BackgroundPath  = value; }
        [Category("4. 儲存設定")][DisplayName("預留空間 (GB)")]  public int LocalMinFreeGB { get => Storage.LocalMinFreeGB; set => Storage.LocalMinFreeGB = value; }
        [Category("4. 儲存設定")][DisplayName("遠端路徑")]        public string RemotePath        { get => Storage.RemotePath;        set => Storage.RemotePath        = value; }
        [Category("4. 儲存設定")][DisplayName("遠端設定路徑")]    public string RemoteConfigPath  { get => Storage.RemoteConfigPath;  set => Storage.RemoteConfigPath  = value; }

        // ===== 5. IO設定 =====
        [Category("5. IO設定")][DisplayName("啟用 IO")]  public bool   PlcEnabled { get; set; } = true;
        [Category("5. IO設定")][DisplayName("IO IP")]    public string PlcIp      { get; set; } = "192.168.255.1";
        [Category("5. IO設定")][DisplayName("IO Port")]  public int    PlcPort    { get; set; } = 502;

        // ===== 6. 相機設定 =====
        [Category("6. 相機設定")][DisplayName("設定檔")]
        [Editor("System.Windows.Forms.Design.FileNameEditor, System.Design, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a", typeof(System.Drawing.Design.UITypeEditor))]
        public string DcfPath { get => CameraParam.DcfPath; set => CameraParam.DcfPath = value; }

        // ===== 7. 光源設定 =====
        [Category("7. 光源設定")][DisplayName("啟用光源")]       public bool   LightEnabled    { get => Light.Enabled;    set => Light.Enabled    = value; }
        [Category("7. 光源設定")][DisplayName("COM Port")]     public string LightComPort    { get => Light.ComPort;    set => Light.ComPort    = value; }
        [Category("7. 光源設定")][DisplayName("通道")]         public int    LightChannel    { get => Light.Channel;    set => Light.Channel    = value; }
        [Category("7. 光源設定")][DisplayName("亮度")]         public int    LightBrightness { get => Light.Brightness; set => Light.Brightness = value; }
        [Category("7. 光源設定")][DisplayName("暖機延遲 (ms)")] public int    LightWarmupMs   { get => Light.WarmupMs;   set => Light.WarmupMs   = value; }
    }
}
