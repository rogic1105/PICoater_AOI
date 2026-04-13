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

        public void Validate()
        {
            if (MachineLayout == null) MachineLayout = new MachineLayoutConfig();
            if (Acquisition == null) Acquisition = new AcquisitionSettings();
            if (Recipe == null) Recipe = new InspectionRecipe();
            if (Chart == null) Chart = new ChartSettings();
            if (ImageView == null) ImageView = new ImageViewSettings();
            if (Storage == null) Storage = new StorageSettings();

            MachineLayout.Validate();
            Acquisition.Validate();
            Recipe.Validate();
            Chart.Validate();
            ImageView.Validate();
            Storage.Validate();
        }

        public double[] GetCameraOpsUmArray() => MachineLayout.GetCameraOpsUmArray();
        public double[] GetCameraStartPositionMmArray() => MachineLayout.GetCameraStartPositionMmArray();

        // ===== 1. 機台佈局 =====
        [Category("1. 機台佈局")][DisplayName("OPS (um)")]
        public CameraOpsConfig Ops => MachineLayout.Ops;

        [Category("1. 機台佈局")][DisplayName("Start (mm)")]
        public CameraStartPositionConfig StartPosition => MachineLayout.StartPosition;

        // 向後相容：程式碼中直接存取的快捷屬性
        [Browsable(false)] public double Cam1_Ops { get => MachineLayout.Cam1_Ops; set => MachineLayout.Cam1_Ops = value; }
        [Browsable(false)] public double Cam2_Ops { get => MachineLayout.Cam2_Ops; set => MachineLayout.Cam2_Ops = value; }
        [Browsable(false)] public double Cam3_Ops { get => MachineLayout.Cam3_Ops; set => MachineLayout.Cam3_Ops = value; }
        [Browsable(false)] public double Cam4_Ops { get => MachineLayout.Cam4_Ops; set => MachineLayout.Cam4_Ops = value; }
        [Browsable(false)] public double Cam5_Ops { get => MachineLayout.Cam5_Ops; set => MachineLayout.Cam5_Ops = value; }
        [Browsable(false)] public double Cam6_Ops { get => MachineLayout.Cam6_Ops; set => MachineLayout.Cam6_Ops = value; }
        [Browsable(false)] public double Cam7_Ops { get => MachineLayout.Cam7_Ops; set => MachineLayout.Cam7_Ops = value; }

        // ===== 2. 檢測配方 =====
        [Category("2. 檢測配方")][DisplayName("去背演算法")]   public BackgroundAlgorithm Algorithm       { get => Recipe.Algorithm;       set => Recipe.Algorithm       = value; }
        [Category("2. 檢測配方")][DisplayName("Ridge 方向")]   public RidgeDirection      RidgeDir        { get => Recipe.RidgeDir;        set => Recipe.RidgeDir        = value; }
        [Category("2. 檢測配方")][DisplayName("正規值")]       public float HessianMaxFactor { get => Recipe.HessianMaxFactor; set => Recipe.HessianMaxFactor = value; }
        [Category("2. 檢測配方")][DisplayName("背景取樣秒數")] public int   BackgroundSampleSeconds { get => Recipe.BackgroundSampleSeconds; set => Recipe.BackgroundSampleSeconds = value; }
        [Category("2. 檢測配方")][DisplayName("A輪速度 (m/min)")] public double AniloxRollSpeedMPerMin { get => Recipe.AniloxRollSpeedMPerMin; set => Recipe.AniloxRollSpeedMPerMin = value; }

        // ===== 3. 檢測報表設定 =====
        [Category("3. 檢測報表設定")][DisplayName("統計圖表")]
        public ChartSettings StatisticsChart => Chart;

        private MuraChartConfig _muraChart;
        [Category("3. 檢測報表設定")][DisplayName("Mura 圖表")]
        public MuraChartConfig MuraChart => _muraChart ?? (_muraChart = new MuraChartConfig(Recipe));

        [Category("3. 檢測報表設定")][DisplayName("圖面")]
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
        [Category("4. 儲存設定")][DisplayName("本地上限(GB)")]   public int    LocalMaxGB      { get => Storage.LocalMaxGB;      set => Storage.LocalMaxGB      = value; }
        [Category("4. 儲存設定")][DisplayName("異常保護天數")]    public int    FailProtectDays { get => Storage.FailProtectDays; set => Storage.FailProtectDays = value; }
        [Category("4. 儲存設定")][DisplayName("遠端路徑")]       public string RemotePath      { get => Storage.RemotePath;      set => Storage.RemotePath      = value; }

        // ===== 5. IO 模組設定 =====
        [Category("5. IO 模組設定")][DisplayName("啟用 IO")]  public bool   PlcEnabled { get; set; } = true;
        [Category("5. IO 模組設定")][DisplayName("IO IP")]    public string PlcIp      { get; set; } = "192.168.255.1";
        [Category("5. IO 模組設定")][DisplayName("IO Port")]  public int    PlcPort    { get; set; } = 502;

    }
}
