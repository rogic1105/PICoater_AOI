namespace AniloxRoll.Monitor.Forms
{
    partial class AniloxRollForm
    {
        /// <summary>
        /// 設計工具所需的變數。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清除任何使用中的資源。
        /// </summary>
        /// <param name="disposing">如果應該處置受控資源則為 true，否則為 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form 設計工具產生的程式碼

        /// <summary>
        /// 此為設計工具支援所需的方法 - 請勿使用程式碼編輯器修改
        /// 這個方法的內容。
        /// </summary>
        private void InitializeComponent()
        {
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea1 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend1 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series1 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea3 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend3 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series3 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea4 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend4 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series4 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea5 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend5 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series5 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea6 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend6 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series6 = new System.Windows.Forms.DataVisualization.Charting.Series();
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea7 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend7 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series7 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.tabMain = new System.Windows.Forms.TabControl();
            this.tabPageLiveView = new System.Windows.Forms.TabPage();
            this.btnViewBackground = new System.Windows.Forms.Button();
            this.lblBgBinInfo = new System.Windows.Forms.Label();
            this.btnGetBackground = new System.Windows.Forms.Button();
            this.muraChartHorizontalLive = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.chartLiveOverview = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.muraChartVerticalLive = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.btnCameraGrab = new System.Windows.Forms.Button();
            this.panelMainDisplay = new System.Windows.Forms.Panel();
            this.panelLiveCam7 = new System.Windows.Forms.Panel();
            this.panelLiveCam6 = new System.Windows.Forms.Panel();
            this.panelLiveCam5 = new System.Windows.Forms.Panel();
            this.panelLiveCam4 = new System.Windows.Forms.Panel();
            this.panelLiveCam3 = new System.Windows.Forms.Panel();
            this.panelLiveCam2 = new System.Windows.Forms.Panel();
            this.panelLiveCam1 = new System.Windows.Forms.Panel();
            this.tabPageReview = new System.Windows.Forms.TabPage();
            this.chartMuraHorizontal = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.chartOverview = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.grpReviewTimePeriod = new System.Windows.Forms.GroupBox();
            this.cbDate = new System.Windows.Forms.ComboBox();
            this.cbTime = new System.Windows.Forms.ComboBox();
            this.btnPeriodNext = new System.Windows.Forms.Button();
            this.btnPeriodPrev = new System.Windows.Forms.Button();
            this.chartMuraVertical = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.btnSelectFolder = new System.Windows.Forms.Button();
            this.pbCam1 = new System.Windows.Forms.PictureBox();
            this.pbCam2 = new System.Windows.Forms.PictureBox();
            this.pbCam3 = new System.Windows.Forms.PictureBox();
            this.pbCam4 = new System.Windows.Forms.PictureBox();
            this.pbCam5 = new System.Windows.Forms.PictureBox();
            this.pbCam6 = new System.Windows.Forms.PictureBox();
            this.pbCam7 = new System.Windows.Forms.PictureBox();
            this.grpReviewGrabNav = new System.Windows.Forms.GroupBox();
            this.cbReviewGrabId = new System.Windows.Forms.ComboBox();
            this.btnGrabIdPrev = new System.Windows.Forms.Button();
            this.btnGrabIdNext = new System.Windows.Forms.Button();
            this.canvasMain = new AOI.SDK.UI.SmartCanvas();
            this.tabPageData = new System.Windows.Forms.TabPage();
            this.chartMuraProfile = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.panelStatCam7 = new System.Windows.Forms.Panel();
            this.lblChartMonthlyUnit = new System.Windows.Forms.Label();
            this.lblChartDailyUnit = new System.Windows.Forms.Label();
            this.lblChartYearlyUnit = new System.Windows.Forms.Label();
            this.grpDataSingleSheet = new System.Windows.Forms.GroupBox();
            this.btnGrabIdDataPrev = new System.Windows.Forms.Button();
            this.btnGrabIdDataNext = new System.Windows.Forms.Button();
            this.cbDataGrabId = new System.Windows.Forms.ComboBox();
            this.lblChartNavMonth = new System.Windows.Forms.Label();
            this.cbChartYear = new System.Windows.Forms.ComboBox();
            this.lblChartNavDay = new System.Windows.Forms.Label();
            this.cbChartMonth = new System.Windows.Forms.ComboBox();
            this.lblChartNavYear = new System.Windows.Forms.Label();
            this.cbChartDay = new System.Windows.Forms.ComboBox();
            this.chartYearly = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.chartMonthly = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.chartDaily = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.btnShowFail = new System.Windows.Forms.Button();
            this.listViewGrabDetail = new System.Windows.Forms.ListView();
            this.groupBoxGrabIdRange = new System.Windows.Forms.GroupBox();
            this.cbGrabIdStart = new System.Windows.Forms.ComboBox();
            this.cbGrabIdEnd = new System.Windows.Forms.ComboBox();
            this.lblGrabIdEndLabel = new System.Windows.Forms.Label();
            this.lblGrabIdStartLabel = new System.Windows.Forms.Label();
            this.groupBoxTimeRange = new System.Windows.Forms.GroupBox();
            this.lblStartTimeHeader = new System.Windows.Forms.Label();
            this.cbStartDate = new System.Windows.Forms.ComboBox();
            this.cbStartTime = new System.Windows.Forms.ComboBox();
            this.lblEndTimeHeader = new System.Windows.Forms.Label();
            this.cbEndDate = new System.Windows.Forms.ComboBox();
            this.cbEndTime = new System.Windows.Forms.ComboBox();
            this.btnSelectDataFolder = new System.Windows.Forms.Button();
            this.panelStatCam6 = new System.Windows.Forms.Panel();
            this.panelStatCam5 = new System.Windows.Forms.Panel();
            this.panelStatCam4 = new System.Windows.Forms.Panel();
            this.panelStatCam3 = new System.Windows.Forms.Panel();
            this.panelStatCam2 = new System.Windows.Forms.Panel();
            this.panelStatCam1 = new System.Windows.Forms.Panel();
            this.propertyGridSettings = new System.Windows.Forms.PropertyGrid();
            this.helpRichText = new System.Windows.Forms.RichTextBox();
            this.statusBarMain = new System.Windows.Forms.StatusStrip();
            this.lblPixelInfo = new System.Windows.Forms.ToolStripStatusLabel();
            this.tabControlRight = new System.Windows.Forms.TabControl();
            this.tabPageInspSettings = new System.Windows.Forms.TabPage();
            this.tabPageCamera = new System.Windows.Forms.TabPage();
            this.tabControlCamTabs = new System.Windows.Forms.TabControl();
            this.tabPageExposure = new System.Windows.Forms.TabPage();
            this.panelExpAll = new System.Windows.Forms.Panel();
            this.lblExpAllUnit = new System.Windows.Forms.Label();
            this.trackBarExpAll = new System.Windows.Forms.TrackBar();
            this.numExpAll = new System.Windows.Forms.NumericUpDown();
            this.lblExpAll = new System.Windows.Forms.Label();
            this.panelExpCam7 = new System.Windows.Forms.Panel();
            this.label13 = new System.Windows.Forms.Label();
            this.trackBarExpCam7 = new System.Windows.Forms.TrackBar();
            this.numExpCam7 = new System.Windows.Forms.NumericUpDown();
            this.label14 = new System.Windows.Forms.Label();
            this.panelExpCam6 = new System.Windows.Forms.Panel();
            this.label11 = new System.Windows.Forms.Label();
            this.trackBarExpCam6 = new System.Windows.Forms.TrackBar();
            this.numExpCam6 = new System.Windows.Forms.NumericUpDown();
            this.label12 = new System.Windows.Forms.Label();
            this.panelExpCam5 = new System.Windows.Forms.Panel();
            this.label9 = new System.Windows.Forms.Label();
            this.trackBarExpCam5 = new System.Windows.Forms.TrackBar();
            this.numExpCam5 = new System.Windows.Forms.NumericUpDown();
            this.label10 = new System.Windows.Forms.Label();
            this.panelExpCam4 = new System.Windows.Forms.Panel();
            this.label7 = new System.Windows.Forms.Label();
            this.trackBarExpCam4 = new System.Windows.Forms.TrackBar();
            this.numExpCam4 = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.panelExpCam3 = new System.Windows.Forms.Panel();
            this.label5 = new System.Windows.Forms.Label();
            this.trackBarExpCam3 = new System.Windows.Forms.TrackBar();
            this.numExpCam3 = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.panelExpCam2 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.trackBarExpCam2 = new System.Windows.Forms.TrackBar();
            this.numExpCam2 = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.panelExpCam1 = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.trackBarExpCam1 = new System.Windows.Forms.TrackBar();
            this.numExpCam1 = new System.Windows.Forms.NumericUpDown();
            this.lblExposure = new System.Windows.Forms.Label();
            this.tabPageLineRate = new System.Windows.Forms.TabPage();
            this.panelLrAll = new System.Windows.Forms.Panel();
            this.lblLrAllUnit = new System.Windows.Forms.Label();
            this.trackBarLrAll = new System.Windows.Forms.TrackBar();
            this.numLrAll = new System.Windows.Forms.NumericUpDown();
            this.lblLrAll = new System.Windows.Forms.Label();
            this.panelLrCam7 = new System.Windows.Forms.Panel();
            this.label15 = new System.Windows.Forms.Label();
            this.trackBarLrCam7 = new System.Windows.Forms.TrackBar();
            this.numLrCam7 = new System.Windows.Forms.NumericUpDown();
            this.label16 = new System.Windows.Forms.Label();
            this.panelLrCam6 = new System.Windows.Forms.Panel();
            this.label17 = new System.Windows.Forms.Label();
            this.trackBarLrCam6 = new System.Windows.Forms.TrackBar();
            this.numLrCam6 = new System.Windows.Forms.NumericUpDown();
            this.label18 = new System.Windows.Forms.Label();
            this.panelLrCam5 = new System.Windows.Forms.Panel();
            this.label19 = new System.Windows.Forms.Label();
            this.trackBarLrCam5 = new System.Windows.Forms.TrackBar();
            this.numLrCam5 = new System.Windows.Forms.NumericUpDown();
            this.label20 = new System.Windows.Forms.Label();
            this.panelLrCam4 = new System.Windows.Forms.Panel();
            this.label21 = new System.Windows.Forms.Label();
            this.trackBarLrCam4 = new System.Windows.Forms.TrackBar();
            this.numLrCam4 = new System.Windows.Forms.NumericUpDown();
            this.label22 = new System.Windows.Forms.Label();
            this.panelLrCam3 = new System.Windows.Forms.Panel();
            this.label23 = new System.Windows.Forms.Label();
            this.trackBarLrCam3 = new System.Windows.Forms.TrackBar();
            this.numLrCam3 = new System.Windows.Forms.NumericUpDown();
            this.label24 = new System.Windows.Forms.Label();
            this.panelLrCam2 = new System.Windows.Forms.Panel();
            this.label25 = new System.Windows.Forms.Label();
            this.trackBarLrCam2 = new System.Windows.Forms.TrackBar();
            this.numLrCam2 = new System.Windows.Forms.NumericUpDown();
            this.label26 = new System.Windows.Forms.Label();
            this.panelLrCam1 = new System.Windows.Forms.Panel();
            this.label27 = new System.Windows.Forms.Label();
            this.trackBarLrCam1 = new System.Windows.Forms.TrackBar();
            this.numLrCam1 = new System.Windows.Forms.NumericUpDown();
            this.lblGrabHeight = new System.Windows.Forms.Label();
            this.tabPageGrabHeight = new System.Windows.Forms.TabPage();
            this.panelHtAll = new System.Windows.Forms.Panel();
            this.lblHtAllUnit = new System.Windows.Forms.Label();
            this.trackBarHtAll = new System.Windows.Forms.TrackBar();
            this.numHtAll = new System.Windows.Forms.NumericUpDown();
            this.lblHtAll = new System.Windows.Forms.Label();
            this.panelHtCam7 = new System.Windows.Forms.Panel();
            this.label1 = new System.Windows.Forms.Label();
            this.trackBarHtCam7 = new System.Windows.Forms.TrackBar();
            this.numHtCam7 = new System.Windows.Forms.NumericUpDown();
            this.label28 = new System.Windows.Forms.Label();
            this.panelHtCam6 = new System.Windows.Forms.Panel();
            this.label29 = new System.Windows.Forms.Label();
            this.trackBarHtCam6 = new System.Windows.Forms.TrackBar();
            this.numHtCam6 = new System.Windows.Forms.NumericUpDown();
            this.label30 = new System.Windows.Forms.Label();
            this.panelHtCam5 = new System.Windows.Forms.Panel();
            this.label31 = new System.Windows.Forms.Label();
            this.trackBarHtCam5 = new System.Windows.Forms.TrackBar();
            this.numHtCam5 = new System.Windows.Forms.NumericUpDown();
            this.label32 = new System.Windows.Forms.Label();
            this.panelHtCam4 = new System.Windows.Forms.Panel();
            this.label33 = new System.Windows.Forms.Label();
            this.trackBarHtCam4 = new System.Windows.Forms.TrackBar();
            this.numHtCam4 = new System.Windows.Forms.NumericUpDown();
            this.label34 = new System.Windows.Forms.Label();
            this.panelHtCam3 = new System.Windows.Forms.Panel();
            this.label35 = new System.Windows.Forms.Label();
            this.trackBarHtCam3 = new System.Windows.Forms.TrackBar();
            this.numHtCam3 = new System.Windows.Forms.NumericUpDown();
            this.label36 = new System.Windows.Forms.Label();
            this.panelHtCam2 = new System.Windows.Forms.Panel();
            this.label37 = new System.Windows.Forms.Label();
            this.trackBarHtCam2 = new System.Windows.Forms.TrackBar();
            this.numHtCam2 = new System.Windows.Forms.NumericUpDown();
            this.label38 = new System.Windows.Forms.Label();
            this.panelHtCam1 = new System.Windows.Forms.Panel();
            this.label39 = new System.Windows.Forms.Label();
            this.trackBarHtCam1 = new System.Windows.Forms.TrackBar();
            this.numHtCam1 = new System.Windows.Forms.NumericUpDown();
            this.label40 = new System.Windows.Forms.Label();
            this.tabPageSystem = new System.Windows.Forms.TabPage();
            this.listViewHardware = new System.Windows.Forms.ListView();
            this.label41 = new System.Windows.Forms.Label();
            this.listViewEngine = new System.Windows.Forms.ListView();
            this.lblEngineConst = new System.Windows.Forms.Label();
            this.listViewCameras = new System.Windows.Forms.ListView();
            this.lblCamHardware = new System.Windows.Forms.Label();
            this.panelStatusBar = new System.Windows.Forms.TableLayoutPanel();
            this.lblCamCount = new System.Windows.Forms.Label();
            this.lblIoState = new System.Windows.Forms.Label();
            this.lblIoConn = new System.Windows.Forms.Label();
            this.lblLightConn = new System.Windows.Forms.Label();
            this.lblStorageConn = new System.Windows.Forms.Label();
            this.panelIo = new System.Windows.Forms.TableLayoutPanel();
            this.lblIoDiAlive = new System.Windows.Forms.Label();
            this.lblIoDiStart = new System.Windows.Forms.Label();
            this.lblIoDoPcAlive = new System.Windows.Forms.Label();
            this.lblIoDoMura = new System.Windows.Forms.Label();
            this.lblIoDoPcBusy = new System.Windows.Forms.Label();
            this.backgroundWorker1 = new System.ComponentModel.BackgroundWorker();
            this.tabMain.SuspendLayout();
            this.tabPageLiveView.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.muraChartHorizontalLive)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartLiveOverview)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.muraChartVerticalLive)).BeginInit();
            this.tabPageReview.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraHorizontal)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartOverview)).BeginInit();
            this.grpReviewTimePeriod.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraVertical)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).BeginInit();
            this.grpReviewGrabNav.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).BeginInit();
            this.tabPageData.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraProfile)).BeginInit();
            this.grpDataSingleSheet.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartYearly)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartMonthly)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartDaily)).BeginInit();
            this.groupBoxGrabIdRange.SuspendLayout();
            this.groupBoxTimeRange.SuspendLayout();
            this.statusBarMain.SuspendLayout();
            this.tabControlRight.SuspendLayout();
            this.tabPageInspSettings.SuspendLayout();
            this.tabPageCamera.SuspendLayout();
            this.tabControlCamTabs.SuspendLayout();
            this.tabPageExposure.SuspendLayout();
            this.panelExpAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpAll)).BeginInit();
            this.panelExpCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).BeginInit();
            this.panelExpCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).BeginInit();
            this.panelExpCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).BeginInit();
            this.panelExpCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).BeginInit();
            this.panelExpCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).BeginInit();
            this.panelExpCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).BeginInit();
            this.panelExpCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).BeginInit();
            this.tabPageLineRate.SuspendLayout();
            this.panelLrAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrAll)).BeginInit();
            this.panelLrCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).BeginInit();
            this.panelLrCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).BeginInit();
            this.panelLrCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).BeginInit();
            this.panelLrCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).BeginInit();
            this.panelLrCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).BeginInit();
            this.panelLrCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).BeginInit();
            this.panelLrCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).BeginInit();
            this.tabPageGrabHeight.SuspendLayout();
            this.panelHtAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtAll)).BeginInit();
            this.panelHtCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).BeginInit();
            this.panelHtCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).BeginInit();
            this.panelHtCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).BeginInit();
            this.panelHtCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).BeginInit();
            this.panelHtCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).BeginInit();
            this.panelHtCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).BeginInit();
            this.panelHtCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).BeginInit();
            this.tabPageSystem.SuspendLayout();
            this.panelStatusBar.SuspendLayout();
            this.panelIo.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabMain
            // 
            this.tabMain.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tabMain.Controls.Add(this.tabPageLiveView);
            this.tabMain.Controls.Add(this.tabPageReview);
            this.tabMain.Controls.Add(this.tabPageData);
            this.tabMain.Location = new System.Drawing.Point(12, 35);
            this.tabMain.Multiline = true;
            this.tabMain.Name = "tabMain";
            this.tabMain.SelectedIndex = 0;
            this.tabMain.Size = new System.Drawing.Size(1028, 660);
            this.tabMain.TabIndex = 1;
            // 
            // tabPageLiveView
            // 
            this.tabPageLiveView.Controls.Add(this.btnViewBackground);
            this.tabPageLiveView.Controls.Add(this.lblBgBinInfo);
            this.tabPageLiveView.Controls.Add(this.btnGetBackground);
            this.tabPageLiveView.Controls.Add(this.muraChartHorizontalLive);
            this.tabPageLiveView.Controls.Add(this.chartLiveOverview);
            this.tabPageLiveView.Controls.Add(this.muraChartVerticalLive);
            this.tabPageLiveView.Controls.Add(this.btnCameraGrab);
            this.tabPageLiveView.Controls.Add(this.panelMainDisplay);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam7);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam6);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam5);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam4);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam3);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam2);
            this.tabPageLiveView.Controls.Add(this.panelLiveCam1);
            this.tabPageLiveView.Location = new System.Drawing.Point(4, 25);
            this.tabPageLiveView.Name = "tabPageLiveView";
            this.tabPageLiveView.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageLiveView.Size = new System.Drawing.Size(1020, 631);
            this.tabPageLiveView.TabIndex = 0;
            this.tabPageLiveView.Text = "即時監控";
            this.tabPageLiveView.UseVisualStyleBackColor = true;
            // 
            // btnViewBackground
            // 
            this.btnViewBackground.Location = new System.Drawing.Point(900, 564);
            this.btnViewBackground.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnViewBackground.Name = "btnViewBackground";
            this.btnViewBackground.Size = new System.Drawing.Size(113, 35);
            this.btnViewBackground.TabIndex = 21;
            this.btnViewBackground.Text = "預覽背景";
            this.btnViewBackground.UseVisualStyleBackColor = true;
            // 
            // lblBgBinInfo
            // 
            this.lblBgBinInfo.Font = new System.Drawing.Font("Microsoft Sans Serif", 7.5F);
            this.lblBgBinInfo.ForeColor = System.Drawing.Color.Gray;
            this.lblBgBinInfo.Location = new System.Drawing.Point(900, 609);
            this.lblBgBinInfo.Name = "lblBgBinInfo";
            this.lblBgBinInfo.Size = new System.Drawing.Size(113, 16);
            this.lblBgBinInfo.TabIndex = 22;
            this.lblBgBinInfo.Text = "光源0  曝光 0us";
            this.lblBgBinInfo.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // btnGetBackground
            // 
            this.btnGetBackground.Location = new System.Drawing.Point(900, 527);
            this.btnGetBackground.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnGetBackground.Name = "btnGetBackground";
            this.btnGetBackground.Size = new System.Drawing.Size(113, 36);
            this.btnGetBackground.TabIndex = 20;
            this.btnGetBackground.Text = "取得背景";
            this.btnGetBackground.UseVisualStyleBackColor = true;
            // 
            // muraChartHorizontalLive
            // 
            this.muraChartHorizontalLive.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea1.Name = "ChartArea1";
            this.muraChartHorizontalLive.ChartAreas.Add(chartArea1);
            legend1.Name = "Legend1";
            this.muraChartHorizontalLive.Legends.Add(legend1);
            this.muraChartHorizontalLive.Location = new System.Drawing.Point(897, 190);
            this.muraChartHorizontalLive.Name = "muraChartHorizontalLive";
            series1.ChartArea = "ChartArea1";
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.muraChartHorizontalLive.Series.Add(series1);
            this.muraChartHorizontalLive.Size = new System.Drawing.Size(117, 333);
            this.muraChartHorizontalLive.TabIndex = 19;
            this.muraChartHorizontalLive.Text = "chart1";
            // 
            // chartLiveOverview
            // 
            this.chartLiveOverview.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea2.Name = "ChartArea1";
            this.chartLiveOverview.ChartAreas.Add(chartArea2);
            legend2.Name = "Legend1";
            this.chartLiveOverview.Legends.Add(legend2);
            this.chartLiveOverview.Location = new System.Drawing.Point(6, 88);
            this.chartLiveOverview.Name = "chartLiveOverview";
            series2.ChartArea = "ChartArea1";
            series2.Legend = "Legend1";
            series2.Name = "Series1";
            this.chartLiveOverview.Series.Add(series2);
            this.chartLiveOverview.Size = new System.Drawing.Size(888, 96);
            this.chartLiveOverview.TabIndex = 18;
            this.chartLiveOverview.Text = "chart1";
            // 
            // muraChartVerticalLive
            // 
            this.muraChartVerticalLive.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea3.Name = "ChartArea1";
            this.muraChartVerticalLive.ChartAreas.Add(chartArea3);
            legend3.Name = "Legend1";
            this.muraChartVerticalLive.Legends.Add(legend3);
            this.muraChartVerticalLive.Location = new System.Drawing.Point(6, 529);
            this.muraChartVerticalLive.Name = "muraChartVerticalLive";
            series3.ChartArea = "ChartArea1";
            series3.Legend = "Legend1";
            series3.Name = "Series1";
            this.muraChartVerticalLive.Series.Add(series3);
            this.muraChartVerticalLive.Size = new System.Drawing.Size(888, 96);
            this.muraChartVerticalLive.TabIndex = 17;
            this.muraChartVerticalLive.Text = "chart1";
            // 
            // btnCameraGrab
            // 
            this.btnCameraGrab.Location = new System.Drawing.Point(900, 7);
            this.btnCameraGrab.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnCameraGrab.Name = "btnCameraGrab";
            this.btnCameraGrab.Size = new System.Drawing.Size(114, 79);
            this.btnCameraGrab.TabIndex = 4;
            this.btnCameraGrab.Text = "開始抓取";
            this.btnCameraGrab.UseVisualStyleBackColor = true;
            this.btnCameraGrab.Click += new System.EventHandler(this.btnCameraGrab_Click);
            // 
            // panelMainDisplay
            // 
            this.panelMainDisplay.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelMainDisplay.Location = new System.Drawing.Point(6, 190);
            this.panelMainDisplay.Name = "panelMainDisplay";
            this.panelMainDisplay.Size = new System.Drawing.Size(888, 333);
            this.panelMainDisplay.TabIndex = 1;
            // 
            // panelLiveCam7
            // 
            this.panelLiveCam7.Location = new System.Drawing.Point(774, 6);
            this.panelLiveCam7.Name = "panelLiveCam7";
            this.panelLiveCam7.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam7.TabIndex = 1;
            // 
            // panelLiveCam6
            // 
            this.panelLiveCam6.Location = new System.Drawing.Point(646, 6);
            this.panelLiveCam6.Name = "panelLiveCam6";
            this.panelLiveCam6.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam6.TabIndex = 1;
            // 
            // panelLiveCam5
            // 
            this.panelLiveCam5.Location = new System.Drawing.Point(518, 6);
            this.panelLiveCam5.Name = "panelLiveCam5";
            this.panelLiveCam5.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam5.TabIndex = 1;
            // 
            // panelLiveCam4
            // 
            this.panelLiveCam4.Location = new System.Drawing.Point(390, 6);
            this.panelLiveCam4.Name = "panelLiveCam4";
            this.panelLiveCam4.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam4.TabIndex = 1;
            // 
            // panelLiveCam3
            // 
            this.panelLiveCam3.Location = new System.Drawing.Point(262, 6);
            this.panelLiveCam3.Name = "panelLiveCam3";
            this.panelLiveCam3.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam3.TabIndex = 1;
            // 
            // panelLiveCam2
            // 
            this.panelLiveCam2.Location = new System.Drawing.Point(134, 6);
            this.panelLiveCam2.Name = "panelLiveCam2";
            this.panelLiveCam2.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam2.TabIndex = 1;
            // 
            // panelLiveCam1
            // 
            this.panelLiveCam1.Location = new System.Drawing.Point(6, 6);
            this.panelLiveCam1.Name = "panelLiveCam1";
            this.panelLiveCam1.Size = new System.Drawing.Size(120, 80);
            this.panelLiveCam1.TabIndex = 0;
            // 
            // tabPageReview
            // 
            this.tabPageReview.Controls.Add(this.chartMuraHorizontal);
            this.tabPageReview.Controls.Add(this.chartOverview);
            this.tabPageReview.Controls.Add(this.grpReviewTimePeriod);
            this.tabPageReview.Controls.Add(this.chartMuraVertical);
            this.tabPageReview.Controls.Add(this.btnSelectFolder);
            this.tabPageReview.Controls.Add(this.pbCam1);
            this.tabPageReview.Controls.Add(this.pbCam2);
            this.tabPageReview.Controls.Add(this.pbCam3);
            this.tabPageReview.Controls.Add(this.pbCam4);
            this.tabPageReview.Controls.Add(this.pbCam5);
            this.tabPageReview.Controls.Add(this.pbCam6);
            this.tabPageReview.Controls.Add(this.pbCam7);
            this.tabPageReview.Controls.Add(this.grpReviewGrabNav);
            this.tabPageReview.Controls.Add(this.canvasMain);
            this.tabPageReview.Location = new System.Drawing.Point(4, 25);
            this.tabPageReview.Name = "tabPageReview";
            this.tabPageReview.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageReview.Size = new System.Drawing.Size(1020, 631);
            this.tabPageReview.TabIndex = 1;
            this.tabPageReview.Text = "歷史查詢";
            this.tabPageReview.UseVisualStyleBackColor = true;
            // 
            // chartMuraHorizontal
            // 
            this.chartMuraHorizontal.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea4.Name = "ChartArea1";
            this.chartMuraHorizontal.ChartAreas.Add(chartArea4);
            legend4.Name = "Legend1";
            this.chartMuraHorizontal.Legends.Add(legend4);
            this.chartMuraHorizontal.Location = new System.Drawing.Point(897, 190);
            this.chartMuraHorizontal.Name = "chartMuraHorizontal";
            series4.ChartArea = "ChartArea1";
            series4.Legend = "Legend1";
            series4.Name = "Series1";
            this.chartMuraHorizontal.Series.Add(series4);
            this.chartMuraHorizontal.Size = new System.Drawing.Size(117, 333);
            this.chartMuraHorizontal.TabIndex = 58;
            this.chartMuraHorizontal.Text = "chart1";
            // 
            // chartOverview
            // 
            this.chartOverview.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea5.Name = "ChartArea1";
            this.chartOverview.ChartAreas.Add(chartArea5);
            legend5.Name = "Legend1";
            this.chartOverview.Legends.Add(legend5);
            this.chartOverview.Location = new System.Drawing.Point(6, 88);
            this.chartOverview.Name = "chartOverview";
            series5.ChartArea = "ChartArea1";
            series5.Legend = "Legend1";
            series5.Name = "Series1";
            this.chartOverview.Series.Add(series5);
            this.chartOverview.Size = new System.Drawing.Size(888, 96);
            this.chartOverview.TabIndex = 57;
            this.chartOverview.Text = "chart1";
            // 
            // grpReviewTimePeriod
            // 
            this.grpReviewTimePeriod.Controls.Add(this.cbDate);
            this.grpReviewTimePeriod.Controls.Add(this.cbTime);
            this.grpReviewTimePeriod.Controls.Add(this.btnPeriodNext);
            this.grpReviewTimePeriod.Controls.Add(this.btnPeriodPrev);
            this.grpReviewTimePeriod.Location = new System.Drawing.Point(899, 529);
            this.grpReviewTimePeriod.Name = "grpReviewTimePeriod";
            this.grpReviewTimePeriod.Size = new System.Drawing.Size(119, 102);
            this.grpReviewTimePeriod.TabIndex = 56;
            this.grpReviewTimePeriod.TabStop = false;
            this.grpReviewTimePeriod.Text = "時序選擇";
            // 
            // cbDate
            // 
            this.cbDate.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbDate.FormattingEnabled = true;
            this.cbDate.Location = new System.Drawing.Point(6, 17);
            this.cbDate.Name = "cbDate";
            this.cbDate.Size = new System.Drawing.Size(108, 23);
            this.cbDate.TabIndex = 15;
            // 
            // cbTime
            // 
            this.cbTime.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbTime.FormattingEnabled = true;
            this.cbTime.Location = new System.Drawing.Point(6, 46);
            this.cbTime.Name = "cbTime";
            this.cbTime.Size = new System.Drawing.Size(107, 23);
            this.cbTime.TabIndex = 16;
            // 
            // btnPeriodNext
            // 
            this.btnPeriodNext.Location = new System.Drawing.Point(69, 68);
            this.btnPeriodNext.Name = "btnPeriodNext";
            this.btnPeriodNext.Size = new System.Drawing.Size(44, 28);
            this.btnPeriodNext.TabIndex = 31;
            this.btnPeriodNext.Text = ">";
            this.btnPeriodNext.UseVisualStyleBackColor = true;
            this.btnPeriodNext.Click += new System.EventHandler(this.btnPeriodNext_Click);
            // 
            // btnPeriodPrev
            // 
            this.btnPeriodPrev.Location = new System.Drawing.Point(6, 68);
            this.btnPeriodPrev.Name = "btnPeriodPrev";
            this.btnPeriodPrev.Size = new System.Drawing.Size(44, 28);
            this.btnPeriodPrev.TabIndex = 30;
            this.btnPeriodPrev.Text = "<";
            this.btnPeriodPrev.UseVisualStyleBackColor = true;
            this.btnPeriodPrev.Click += new System.EventHandler(this.btnPeriodPrev_Click);
            // 
            // chartMuraVertical
            // 
            this.chartMuraVertical.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea6.Name = "ChartArea1";
            this.chartMuraVertical.ChartAreas.Add(chartArea6);
            legend6.Name = "Legend1";
            this.chartMuraVertical.Legends.Add(legend6);
            this.chartMuraVertical.Location = new System.Drawing.Point(6, 529);
            this.chartMuraVertical.Name = "chartMuraVertical";
            series6.ChartArea = "ChartArea1";
            series6.Legend = "Legend1";
            series6.Name = "Series1";
            this.chartMuraVertical.Series.Add(series6);
            this.chartMuraVertical.Size = new System.Drawing.Size(888, 96);
            this.chartMuraVertical.TabIndex = 16;
            this.chartMuraVertical.Text = "chart1";
            // 
            // btnSelectFolder
            // 
            this.btnSelectFolder.Font = new System.Drawing.Font("微軟正黑體", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.btnSelectFolder.Location = new System.Drawing.Point(900, 6);
            this.btnSelectFolder.Name = "btnSelectFolder";
            this.btnSelectFolder.Size = new System.Drawing.Size(115, 81);
            this.btnSelectFolder.TabIndex = 23;
            this.btnSelectFolder.Text = "讀取資料";
            this.btnSelectFolder.UseVisualStyleBackColor = true;
            this.btnSelectFolder.Click += new System.EventHandler(this.btnSelectFolder_Click);
            // 
            // pbCam1
            // 
            this.pbCam1.Location = new System.Drawing.Point(6, 6);
            this.pbCam1.Name = "pbCam1";
            this.pbCam1.Size = new System.Drawing.Size(120, 80);
            this.pbCam1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam1.TabIndex = 8;
            this.pbCam1.TabStop = false;
            // 
            // pbCam2
            // 
            this.pbCam2.Location = new System.Drawing.Point(134, 6);
            this.pbCam2.Name = "pbCam2";
            this.pbCam2.Size = new System.Drawing.Size(120, 80);
            this.pbCam2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam2.TabIndex = 9;
            this.pbCam2.TabStop = false;
            // 
            // pbCam3
            // 
            this.pbCam3.Location = new System.Drawing.Point(262, 6);
            this.pbCam3.Name = "pbCam3";
            this.pbCam3.Size = new System.Drawing.Size(120, 80);
            this.pbCam3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam3.TabIndex = 10;
            this.pbCam3.TabStop = false;
            // 
            // pbCam4
            // 
            this.pbCam4.Location = new System.Drawing.Point(390, 6);
            this.pbCam4.Name = "pbCam4";
            this.pbCam4.Size = new System.Drawing.Size(120, 80);
            this.pbCam4.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam4.TabIndex = 11;
            this.pbCam4.TabStop = false;
            // 
            // pbCam5
            // 
            this.pbCam5.Location = new System.Drawing.Point(518, 6);
            this.pbCam5.Name = "pbCam5";
            this.pbCam5.Size = new System.Drawing.Size(120, 80);
            this.pbCam5.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam5.TabIndex = 12;
            this.pbCam5.TabStop = false;
            // 
            // pbCam6
            // 
            this.pbCam6.Location = new System.Drawing.Point(646, 6);
            this.pbCam6.Name = "pbCam6";
            this.pbCam6.Size = new System.Drawing.Size(120, 80);
            this.pbCam6.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam6.TabIndex = 13;
            this.pbCam6.TabStop = false;
            // 
            // pbCam7
            // 
            this.pbCam7.Location = new System.Drawing.Point(774, 6);
            this.pbCam7.Name = "pbCam7";
            this.pbCam7.Size = new System.Drawing.Size(120, 80);
            this.pbCam7.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam7.TabIndex = 14;
            this.pbCam7.TabStop = false;
            // 
            // grpReviewGrabNav
            // 
            this.grpReviewGrabNav.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.grpReviewGrabNav.Controls.Add(this.cbReviewGrabId);
            this.grpReviewGrabNav.Controls.Add(this.btnGrabIdPrev);
            this.grpReviewGrabNav.Controls.Add(this.btnGrabIdNext);
            this.grpReviewGrabNav.Location = new System.Drawing.Point(899, 93);
            this.grpReviewGrabNav.Name = "grpReviewGrabNav";
            this.grpReviewGrabNav.Size = new System.Drawing.Size(118, 96);
            this.grpReviewGrabNav.TabIndex = 55;
            this.grpReviewGrabNav.TabStop = false;
            this.grpReviewGrabNav.Text = "序號選擇";
            // 
            // cbReviewGrabId
            // 
            this.cbReviewGrabId.FormattingEnabled = true;
            this.cbReviewGrabId.Location = new System.Drawing.Point(10, 21);
            this.cbReviewGrabId.Name = "cbReviewGrabId";
            this.cbReviewGrabId.Size = new System.Drawing.Size(104, 23);
            this.cbReviewGrabId.TabIndex = 0;
            // 
            // btnGrabIdPrev
            // 
            this.btnGrabIdPrev.Location = new System.Drawing.Point(10, 50);
            this.btnGrabIdPrev.Name = "btnGrabIdPrev";
            this.btnGrabIdPrev.Size = new System.Drawing.Size(44, 28);
            this.btnGrabIdPrev.TabIndex = 1;
            this.btnGrabIdPrev.Text = "<";
            this.btnGrabIdPrev.UseVisualStyleBackColor = true;
            // 
            // btnGrabIdNext
            // 
            this.btnGrabIdNext.Location = new System.Drawing.Point(70, 50);
            this.btnGrabIdNext.Name = "btnGrabIdNext";
            this.btnGrabIdNext.Size = new System.Drawing.Size(44, 28);
            this.btnGrabIdNext.TabIndex = 2;
            this.btnGrabIdNext.Text = ">";
            this.btnGrabIdNext.UseVisualStyleBackColor = true;
            // 
            // canvasMain
            // 
            this.canvasMain.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.canvasMain.BackColor = System.Drawing.Color.Black;
            this.canvasMain.ClampPan = false;
            this.canvasMain.Location = new System.Drawing.Point(6, 190);
            this.canvasMain.Name = "canvasMain";
            this.canvasMain.Size = new System.Drawing.Size(888, 333);
            this.canvasMain.TabIndex = 7;
            this.canvasMain.TabStop = false;
            // 
            // tabPageData
            // 
            this.tabPageData.Controls.Add(this.chartMuraProfile);
            this.tabPageData.Controls.Add(this.panelStatCam7);
            this.tabPageData.Controls.Add(this.lblChartMonthlyUnit);
            this.tabPageData.Controls.Add(this.lblChartDailyUnit);
            this.tabPageData.Controls.Add(this.lblChartYearlyUnit);
            this.tabPageData.Controls.Add(this.grpDataSingleSheet);
            this.tabPageData.Controls.Add(this.lblChartNavMonth);
            this.tabPageData.Controls.Add(this.cbChartYear);
            this.tabPageData.Controls.Add(this.lblChartNavDay);
            this.tabPageData.Controls.Add(this.cbChartMonth);
            this.tabPageData.Controls.Add(this.lblChartNavYear);
            this.tabPageData.Controls.Add(this.cbChartDay);
            this.tabPageData.Controls.Add(this.chartYearly);
            this.tabPageData.Controls.Add(this.chartMonthly);
            this.tabPageData.Controls.Add(this.chartDaily);
            this.tabPageData.Controls.Add(this.btnShowFail);
            this.tabPageData.Controls.Add(this.listViewGrabDetail);
            this.tabPageData.Controls.Add(this.groupBoxGrabIdRange);
            this.tabPageData.Controls.Add(this.groupBoxTimeRange);
            this.tabPageData.Controls.Add(this.btnSelectDataFolder);
            this.tabPageData.Controls.Add(this.panelStatCam6);
            this.tabPageData.Controls.Add(this.panelStatCam5);
            this.tabPageData.Controls.Add(this.panelStatCam4);
            this.tabPageData.Controls.Add(this.panelStatCam3);
            this.tabPageData.Controls.Add(this.panelStatCam2);
            this.tabPageData.Controls.Add(this.panelStatCam1);
            this.tabPageData.Location = new System.Drawing.Point(4, 25);
            this.tabPageData.Name = "tabPageData";
            this.tabPageData.Size = new System.Drawing.Size(1020, 631);
            this.tabPageData.TabIndex = 2;
            this.tabPageData.Text = "檢測報表";
            this.tabPageData.UseVisualStyleBackColor = true;
            // 
            // chartMuraProfile
            // 
            this.chartMuraProfile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            chartArea7.Name = "ChartArea1";
            this.chartMuraProfile.ChartAreas.Add(chartArea7);
            legend7.Name = "Legend1";
            this.chartMuraProfile.Legends.Add(legend7);
            this.chartMuraProfile.Location = new System.Drawing.Point(6, 88);
            this.chartMuraProfile.Name = "chartMuraProfile";
            series7.ChartArea = "ChartArea1";
            series7.Legend = "Legend1";
            series7.Name = "Series1";
            this.chartMuraProfile.Series.Add(series7);
            this.chartMuraProfile.Size = new System.Drawing.Size(888, 96);
            this.chartMuraProfile.TabIndex = 58;
            this.chartMuraProfile.Text = "chartMuraProfile";
            // 
            // panelStatCam7
            // 
            this.panelStatCam7.Location = new System.Drawing.Point(774, 6);
            this.panelStatCam7.Name = "panelStatCam7";
            this.panelStatCam7.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam7.TabIndex = 3;
            // 
            // lblChartMonthlyUnit
            // 
            this.lblChartMonthlyUnit.AutoSize = true;
            this.lblChartMonthlyUnit.Location = new System.Drawing.Point(978, 482);
            this.lblChartMonthlyUnit.Name = "lblChartMonthlyUnit";
            this.lblChartMonthlyUnit.Size = new System.Drawing.Size(22, 15);
            this.lblChartMonthlyUnit.TabIndex = 53;
            this.lblChartMonthlyUnit.Text = "日";
            // 
            // lblChartDailyUnit
            // 
            this.lblChartDailyUnit.AutoSize = true;
            this.lblChartDailyUnit.Location = new System.Drawing.Point(298, 511);
            this.lblChartDailyUnit.Name = "lblChartDailyUnit";
            this.lblChartDailyUnit.Size = new System.Drawing.Size(22, 15);
            this.lblChartDailyUnit.TabIndex = 52;
            this.lblChartDailyUnit.Text = "時";
            // 
            // lblChartYearlyUnit
            // 
            this.lblChartYearlyUnit.AutoSize = true;
            this.lblChartYearlyUnit.Location = new System.Drawing.Point(298, 285);
            this.lblChartYearlyUnit.Name = "lblChartYearlyUnit";
            this.lblChartYearlyUnit.Size = new System.Drawing.Size(22, 15);
            this.lblChartYearlyUnit.TabIndex = 51;
            this.lblChartYearlyUnit.Text = "月";
            // 
            // grpDataSingleSheet
            // 
            this.grpDataSingleSheet.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.grpDataSingleSheet.Controls.Add(this.btnGrabIdDataPrev);
            this.grpDataSingleSheet.Controls.Add(this.btnGrabIdDataNext);
            this.grpDataSingleSheet.Controls.Add(this.cbDataGrabId);
            this.grpDataSingleSheet.Location = new System.Drawing.Point(899, 93);
            this.grpDataSingleSheet.Name = "grpDataSingleSheet";
            this.grpDataSingleSheet.Size = new System.Drawing.Size(118, 96);
            this.grpDataSingleSheet.TabIndex = 50;
            this.grpDataSingleSheet.TabStop = false;
            this.grpDataSingleSheet.Text = "序號選擇";
            // 
            // btnGrabIdDataPrev
            // 
            this.btnGrabIdDataPrev.Location = new System.Drawing.Point(10, 50);
            this.btnGrabIdDataPrev.Name = "btnGrabIdDataPrev";
            this.btnGrabIdDataPrev.Size = new System.Drawing.Size(44, 28);
            this.btnGrabIdDataPrev.TabIndex = 43;
            this.btnGrabIdDataPrev.Text = "<";
            this.btnGrabIdDataPrev.UseVisualStyleBackColor = true;
            // 
            // btnGrabIdDataNext
            // 
            this.btnGrabIdDataNext.Location = new System.Drawing.Point(70, 50);
            this.btnGrabIdDataNext.Name = "btnGrabIdDataNext";
            this.btnGrabIdDataNext.Size = new System.Drawing.Size(44, 28);
            this.btnGrabIdDataNext.TabIndex = 44;
            this.btnGrabIdDataNext.Text = ">";
            this.btnGrabIdDataNext.UseVisualStyleBackColor = true;
            // 
            // cbDataGrabId
            // 
            this.cbDataGrabId.FormattingEnabled = true;
            this.cbDataGrabId.Location = new System.Drawing.Point(10, 21);
            this.cbDataGrabId.Name = "cbDataGrabId";
            this.cbDataGrabId.Size = new System.Drawing.Size(104, 23);
            this.cbDataGrabId.TabIndex = 42;
            // 
            // lblChartNavMonth
            // 
            this.lblChartNavMonth.AutoSize = true;
            this.lblChartNavMonth.Location = new System.Drawing.Point(978, 369);
            this.lblChartNavMonth.Name = "lblChartNavMonth";
            this.lblChartNavMonth.Size = new System.Drawing.Size(22, 15);
            this.lblChartNavMonth.TabIndex = 42;
            this.lblChartNavMonth.Text = "月";
            // 
            // cbChartYear
            // 
            this.cbChartYear.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cbChartYear.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbChartYear.Location = new System.Drawing.Point(963, 230);
            this.cbChartYear.Name = "cbChartYear";
            this.cbChartYear.Size = new System.Drawing.Size(50, 23);
            this.cbChartYear.TabIndex = 43;
            // 
            // lblChartNavDay
            // 
            this.lblChartNavDay.AutoSize = true;
            this.lblChartNavDay.Location = new System.Drawing.Point(298, 398);
            this.lblChartNavDay.Name = "lblChartNavDay";
            this.lblChartNavDay.Size = new System.Drawing.Size(22, 15);
            this.lblChartNavDay.TabIndex = 41;
            this.lblChartNavDay.Text = "日";
            // 
            // cbChartMonth
            // 
            this.cbChartMonth.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cbChartMonth.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbChartMonth.Location = new System.Drawing.Point(963, 343);
            this.cbChartMonth.Name = "cbChartMonth";
            this.cbChartMonth.Size = new System.Drawing.Size(50, 23);
            this.cbChartMonth.TabIndex = 46;
            // 
            // lblChartNavYear
            // 
            this.lblChartNavYear.AutoSize = true;
            this.lblChartNavYear.Location = new System.Drawing.Point(978, 256);
            this.lblChartNavYear.Name = "lblChartNavYear";
            this.lblChartNavYear.Size = new System.Drawing.Size(22, 15);
            this.lblChartNavYear.TabIndex = 40;
            this.lblChartNavYear.Text = "年";
            // 
            // cbChartDay
            // 
            this.cbChartDay.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cbChartDay.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbChartDay.Location = new System.Drawing.Point(963, 456);
            this.cbChartDay.Name = "cbChartDay";
            this.cbChartDay.Size = new System.Drawing.Size(50, 23);
            this.cbChartDay.TabIndex = 49;
            // 
            // chartYearly
            // 
            this.chartYearly.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.chartYearly.Location = new System.Drawing.Point(296, 195);
            this.chartYearly.Name = "chartYearly";
            this.chartYearly.Size = new System.Drawing.Size(721, 105);
            this.chartYearly.TabIndex = 38;
            this.chartYearly.Text = "chartYearly";
            // 
            // chartMonthly
            // 
            this.chartMonthly.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.chartMonthly.Location = new System.Drawing.Point(296, 308);
            this.chartMonthly.Name = "chartMonthly";
            this.chartMonthly.Size = new System.Drawing.Size(721, 105);
            this.chartMonthly.TabIndex = 39;
            this.chartMonthly.Text = "chartMonthly";
            // 
            // chartDaily
            // 
            this.chartDaily.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.chartDaily.Location = new System.Drawing.Point(296, 421);
            this.chartDaily.Name = "chartDaily";
            this.chartDaily.Size = new System.Drawing.Size(721, 105);
            this.chartDaily.TabIndex = 40;
            this.chartDaily.Text = "chartDaily";
            // 
            // btnShowFail
            // 
            this.btnShowFail.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.btnShowFail.Font = new System.Drawing.Font("新細明體", 8F);
            this.btnShowFail.Location = new System.Drawing.Point(900, 53);
            this.btnShowFail.Name = "btnShowFail";
            this.btnShowFail.Size = new System.Drawing.Size(115, 33);
            this.btnShowFail.TabIndex = 41;
            this.btnShowFail.Text = "△ 顯示異常";
            this.btnShowFail.UseVisualStyleBackColor = true;
            // 
            // listViewGrabDetail
            // 
            this.listViewGrabDetail.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left)));
            this.listViewGrabDetail.HideSelection = false;
            this.listViewGrabDetail.Location = new System.Drawing.Point(6, 195);
            this.listViewGrabDetail.Name = "listViewGrabDetail";
            this.listViewGrabDetail.Size = new System.Drawing.Size(284, 433);
            this.listViewGrabDetail.TabIndex = 37;
            this.listViewGrabDetail.UseCompatibleStateImageBehavior = false;
            // 
            // groupBoxGrabIdRange
            // 
            this.groupBoxGrabIdRange.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBoxGrabIdRange.Controls.Add(this.cbGrabIdStart);
            this.groupBoxGrabIdRange.Controls.Add(this.cbGrabIdEnd);
            this.groupBoxGrabIdRange.Controls.Add(this.lblGrabIdEndLabel);
            this.groupBoxGrabIdRange.Controls.Add(this.lblGrabIdStartLabel);
            this.groupBoxGrabIdRange.Location = new System.Drawing.Point(718, 532);
            this.groupBoxGrabIdRange.Name = "groupBoxGrabIdRange";
            this.groupBoxGrabIdRange.Size = new System.Drawing.Size(176, 92);
            this.groupBoxGrabIdRange.TabIndex = 36;
            this.groupBoxGrabIdRange.TabStop = false;
            this.groupBoxGrabIdRange.Text = "序號範圍";
            // 
            // cbGrabIdStart
            // 
            this.cbGrabIdStart.FormattingEnabled = true;
            this.cbGrabIdStart.Location = new System.Drawing.Point(50, 53);
            this.cbGrabIdStart.Name = "cbGrabIdStart";
            this.cbGrabIdStart.Size = new System.Drawing.Size(104, 23);
            this.cbGrabIdStart.TabIndex = 42;
            // 
            // cbGrabIdEnd
            // 
            this.cbGrabIdEnd.FormattingEnabled = true;
            this.cbGrabIdEnd.Location = new System.Drawing.Point(50, 24);
            this.cbGrabIdEnd.Name = "cbGrabIdEnd";
            this.cbGrabIdEnd.Size = new System.Drawing.Size(104, 23);
            this.cbGrabIdEnd.TabIndex = 43;
            // 
            // lblGrabIdEndLabel
            // 
            this.lblGrabIdEndLabel.AutoSize = true;
            this.lblGrabIdEndLabel.Location = new System.Drawing.Point(7, 27);
            this.lblGrabIdEndLabel.Name = "lblGrabIdEndLabel";
            this.lblGrabIdEndLabel.Size = new System.Drawing.Size(37, 15);
            this.lblGrabIdEndLabel.TabIndex = 41;
            this.lblGrabIdEndLabel.Text = "結束";
            // 
            // lblGrabIdStartLabel
            // 
            this.lblGrabIdStartLabel.AutoSize = true;
            this.lblGrabIdStartLabel.Location = new System.Drawing.Point(7, 58);
            this.lblGrabIdStartLabel.Name = "lblGrabIdStartLabel";
            this.lblGrabIdStartLabel.Size = new System.Drawing.Size(37, 15);
            this.lblGrabIdStartLabel.TabIndex = 40;
            this.lblGrabIdStartLabel.Text = "開始";
            // 
            // groupBoxTimeRange
            // 
            this.groupBoxTimeRange.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.groupBoxTimeRange.Controls.Add(this.lblStartTimeHeader);
            this.groupBoxTimeRange.Controls.Add(this.cbStartDate);
            this.groupBoxTimeRange.Controls.Add(this.cbStartTime);
            this.groupBoxTimeRange.Controls.Add(this.lblEndTimeHeader);
            this.groupBoxTimeRange.Controls.Add(this.cbEndDate);
            this.groupBoxTimeRange.Controls.Add(this.cbEndTime);
            this.groupBoxTimeRange.Location = new System.Drawing.Point(423, 532);
            this.groupBoxTimeRange.Name = "groupBoxTimeRange";
            this.groupBoxTimeRange.Size = new System.Drawing.Size(203, 92);
            this.groupBoxTimeRange.TabIndex = 35;
            this.groupBoxTimeRange.TabStop = false;
            this.groupBoxTimeRange.Text = "時序範圍";
            // 
            // lblStartTimeHeader
            // 
            this.lblStartTimeHeader.AutoSize = true;
            this.lblStartTimeHeader.Location = new System.Drawing.Point(6, 19);
            this.lblStartTimeHeader.Name = "lblStartTimeHeader";
            this.lblStartTimeHeader.Size = new System.Drawing.Size(67, 15);
            this.lblStartTimeHeader.TabIndex = 36;
            this.lblStartTimeHeader.Text = "開始時間";
            // 
            // cbStartDate
            // 
            this.cbStartDate.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbStartDate.FormattingEnabled = true;
            this.cbStartDate.Location = new System.Drawing.Point(6, 37);
            this.cbStartDate.Name = "cbStartDate";
            this.cbStartDate.Size = new System.Drawing.Size(90, 23);
            this.cbStartDate.TabIndex = 21;
            // 
            // cbStartTime
            // 
            this.cbStartTime.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbStartTime.FormattingEnabled = true;
            this.cbStartTime.Location = new System.Drawing.Point(6, 63);
            this.cbStartTime.Name = "cbStartTime";
            this.cbStartTime.Size = new System.Drawing.Size(90, 23);
            this.cbStartTime.TabIndex = 22;
            // 
            // lblEndTimeHeader
            // 
            this.lblEndTimeHeader.AutoSize = true;
            this.lblEndTimeHeader.Location = new System.Drawing.Point(100, 19);
            this.lblEndTimeHeader.Name = "lblEndTimeHeader";
            this.lblEndTimeHeader.Size = new System.Drawing.Size(67, 15);
            this.lblEndTimeHeader.TabIndex = 39;
            this.lblEndTimeHeader.Text = "結束時間";
            // 
            // cbEndDate
            // 
            this.cbEndDate.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbEndDate.FormattingEnabled = true;
            this.cbEndDate.Location = new System.Drawing.Point(100, 37);
            this.cbEndDate.Name = "cbEndDate";
            this.cbEndDate.Size = new System.Drawing.Size(90, 23);
            this.cbEndDate.TabIndex = 27;
            // 
            // cbEndTime
            // 
            this.cbEndTime.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbEndTime.FormattingEnabled = true;
            this.cbEndTime.Location = new System.Drawing.Point(100, 63);
            this.cbEndTime.Name = "cbEndTime";
            this.cbEndTime.Size = new System.Drawing.Size(90, 23);
            this.cbEndTime.TabIndex = 28;
            // 
            // btnSelectDataFolder
            // 
            this.btnSelectDataFolder.Font = new System.Drawing.Font("微軟正黑體", 10.2F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(136)));
            this.btnSelectDataFolder.Location = new System.Drawing.Point(900, 6);
            this.btnSelectDataFolder.Name = "btnSelectDataFolder";
            this.btnSelectDataFolder.Size = new System.Drawing.Size(115, 45);
            this.btnSelectDataFolder.TabIndex = 34;
            this.btnSelectDataFolder.Text = "讀取資料";
            this.btnSelectDataFolder.UseVisualStyleBackColor = true;
            // 
            // panelStatCam6
            // 
            this.panelStatCam6.Location = new System.Drawing.Point(646, 6);
            this.panelStatCam6.Name = "panelStatCam6";
            this.panelStatCam6.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam6.TabIndex = 4;
            // 
            // panelStatCam5
            // 
            this.panelStatCam5.Location = new System.Drawing.Point(518, 6);
            this.panelStatCam5.Name = "panelStatCam5";
            this.panelStatCam5.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam5.TabIndex = 5;
            // 
            // panelStatCam4
            // 
            this.panelStatCam4.Location = new System.Drawing.Point(390, 6);
            this.panelStatCam4.Name = "panelStatCam4";
            this.panelStatCam4.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam4.TabIndex = 6;
            // 
            // panelStatCam3
            // 
            this.panelStatCam3.Location = new System.Drawing.Point(262, 6);
            this.panelStatCam3.Name = "panelStatCam3";
            this.panelStatCam3.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam3.TabIndex = 7;
            // 
            // panelStatCam2
            // 
            this.panelStatCam2.Location = new System.Drawing.Point(134, 6);
            this.panelStatCam2.Name = "panelStatCam2";
            this.panelStatCam2.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam2.TabIndex = 8;
            // 
            // panelStatCam1
            // 
            this.panelStatCam1.Location = new System.Drawing.Point(6, 6);
            this.panelStatCam1.Name = "panelStatCam1";
            this.panelStatCam1.Size = new System.Drawing.Size(120, 80);
            this.panelStatCam1.TabIndex = 2;
            // 
            // propertyGridSettings
            // 
            this.propertyGridSettings.Dock = System.Windows.Forms.DockStyle.Fill;
            this.propertyGridSettings.HelpVisible = false;
            this.propertyGridSettings.Location = new System.Drawing.Point(3, 3);
            this.propertyGridSettings.Name = "propertyGridSettings";
            this.propertyGridSettings.Size = new System.Drawing.Size(207, 565);
            this.propertyGridSettings.TabIndex = 0;
            // 
            // helpRichText
            // 
            this.helpRichText.BackColor = System.Drawing.SystemColors.Control;
            this.helpRichText.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.helpRichText.DetectUrls = false;
            this.helpRichText.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.helpRichText.HideSelection = false;
            this.helpRichText.Location = new System.Drawing.Point(3, 568);
            this.helpRichText.Name = "helpRichText";
            this.helpRichText.ReadOnly = true;
            this.helpRichText.Size = new System.Drawing.Size(207, 60);
            this.helpRichText.TabIndex = 1;
            this.helpRichText.TabStop = false;
            this.helpRichText.Text = "";
            // 
            // statusBarMain
            // 
            this.statusBarMain.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusBarMain.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.lblPixelInfo});
            this.statusBarMain.Location = new System.Drawing.Point(0, 696);
            this.statusBarMain.Name = "statusBarMain";
            this.statusBarMain.Size = new System.Drawing.Size(1262, 25);
            this.statusBarMain.TabIndex = 15;
            this.statusBarMain.Text = "statusBarMain";
            // 
            // lblPixelInfo
            // 
            this.lblPixelInfo.Name = "lblPixelInfo";
            this.lblPixelInfo.Size = new System.Drawing.Size(395, 19);
            this.lblPixelInfo.Text = "位置:0.00mm | 座標:(0, 0) | 亮度: 0  | 倍率:0.0x | 平移:(0, 0)";
            // 
            // tabControlRight
            // 
            this.tabControlRight.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tabControlRight.Controls.Add(this.tabPageInspSettings);
            this.tabControlRight.Controls.Add(this.tabPageCamera);
            this.tabControlRight.Controls.Add(this.tabPageSystem);
            this.tabControlRight.Location = new System.Drawing.Point(1041, 35);
            this.tabControlRight.Multiline = true;
            this.tabControlRight.Name = "tabControlRight";
            this.tabControlRight.SelectedIndex = 0;
            this.tabControlRight.Size = new System.Drawing.Size(221, 660);
            this.tabControlRight.TabIndex = 16;
            // 
            // tabPageInspSettings
            // 
            this.tabPageInspSettings.Controls.Add(this.propertyGridSettings);
            this.tabPageInspSettings.Controls.Add(this.helpRichText);
            this.tabPageInspSettings.Location = new System.Drawing.Point(4, 25);
            this.tabPageInspSettings.Name = "tabPageInspSettings";
            this.tabPageInspSettings.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageInspSettings.Size = new System.Drawing.Size(213, 631);
            this.tabPageInspSettings.TabIndex = 0;
            this.tabPageInspSettings.Text = "檢測設定";
            this.tabPageInspSettings.UseVisualStyleBackColor = true;
            // 
            // tabPageCamera
            // 
            this.tabPageCamera.Controls.Add(this.tabControlCamTabs);
            this.tabPageCamera.Location = new System.Drawing.Point(4, 25);
            this.tabPageCamera.Name = "tabPageCamera";
            this.tabPageCamera.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageCamera.Size = new System.Drawing.Size(213, 631);
            this.tabPageCamera.TabIndex = 1;
            this.tabPageCamera.Text = "相機參數";
            this.tabPageCamera.UseVisualStyleBackColor = true;
            // 
            // tabControlCamTabs
            // 
            this.tabControlCamTabs.Controls.Add(this.tabPageExposure);
            this.tabControlCamTabs.Controls.Add(this.tabPageLineRate);
            this.tabControlCamTabs.Controls.Add(this.tabPageGrabHeight);
            this.tabControlCamTabs.Dock = System.Windows.Forms.DockStyle.Fill;
            this.tabControlCamTabs.Location = new System.Drawing.Point(3, 3);
            this.tabControlCamTabs.Name = "tabControlCamTabs";
            this.tabControlCamTabs.SelectedIndex = 0;
            this.tabControlCamTabs.Size = new System.Drawing.Size(207, 625);
            this.tabControlCamTabs.TabIndex = 2;
            // 
            // tabPageExposure
            // 
            this.tabPageExposure.Controls.Add(this.panelExpAll);
            this.tabPageExposure.Controls.Add(this.panelExpCam7);
            this.tabPageExposure.Controls.Add(this.panelExpCam6);
            this.tabPageExposure.Controls.Add(this.panelExpCam5);
            this.tabPageExposure.Controls.Add(this.panelExpCam4);
            this.tabPageExposure.Controls.Add(this.panelExpCam3);
            this.tabPageExposure.Controls.Add(this.panelExpCam2);
            this.tabPageExposure.Controls.Add(this.panelExpCam1);
            this.tabPageExposure.Location = new System.Drawing.Point(4, 25);
            this.tabPageExposure.Name = "tabPageExposure";
            this.tabPageExposure.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageExposure.Size = new System.Drawing.Size(199, 596);
            this.tabPageExposure.TabIndex = 0;
            this.tabPageExposure.Text = "曝光時間";
            this.tabPageExposure.UseVisualStyleBackColor = true;
            // 
            // panelExpAll
            // 
            this.panelExpAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelExpAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpAll.Controls.Add(this.lblExpAllUnit);
            this.panelExpAll.Controls.Add(this.trackBarExpAll);
            this.panelExpAll.Controls.Add(this.numExpAll);
            this.panelExpAll.Controls.Add(this.lblExpAll);
            this.panelExpAll.Location = new System.Drawing.Point(0, 0);
            this.panelExpAll.Name = "panelExpAll";
            this.panelExpAll.Size = new System.Drawing.Size(199, 69);
            this.panelExpAll.TabIndex = 10;
            // 
            // lblExpAllUnit
            // 
            this.lblExpAllUnit.AutoSize = true;
            this.lblExpAllUnit.Location = new System.Drawing.Point(214, 7);
            this.lblExpAllUnit.Name = "lblExpAllUnit";
            this.lblExpAllUnit.Size = new System.Drawing.Size(27, 15);
            this.lblExpAllUnit.TabIndex = 3;
            this.lblExpAllUnit.Text = "μs";
            // 
            // trackBarExpAll
            // 
            this.trackBarExpAll.AutoSize = false;
            this.trackBarExpAll.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpAll.Maximum = 2000;
            this.trackBarExpAll.Minimum = 1;
            this.trackBarExpAll.Name = "trackBarExpAll";
            this.trackBarExpAll.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpAll.TabIndex = 2;
            this.trackBarExpAll.TickFrequency = 200;
            this.trackBarExpAll.Value = 50;
            // 
            // numExpAll
            // 
            this.numExpAll.Location = new System.Drawing.Point(118, 5);
            this.numExpAll.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpAll.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpAll.Name = "numExpAll";
            this.numExpAll.Size = new System.Drawing.Size(70, 25);
            this.numExpAll.TabIndex = 1;
            this.numExpAll.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // lblExpAll
            // 
            this.lblExpAll.AutoSize = true;
            this.lblExpAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblExpAll.Location = new System.Drawing.Point(5, 5);
            this.lblExpAll.Name = "lblExpAll";
            this.lblExpAll.Size = new System.Drawing.Size(38, 15);
            this.lblExpAll.TabIndex = 0;
            this.lblExpAll.Text = "ALL";
            // 
            // panelExpCam7
            // 
            this.panelExpCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam7.Controls.Add(this.label13);
            this.panelExpCam7.Controls.Add(this.trackBarExpCam7);
            this.panelExpCam7.Controls.Add(this.numExpCam7);
            this.panelExpCam7.Controls.Add(this.label14);
            this.panelExpCam7.Location = new System.Drawing.Point(0, 511);
            this.panelExpCam7.Name = "panelExpCam7";
            this.panelExpCam7.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam7.TabIndex = 4;
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(214, 7);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(27, 15);
            this.label13.TabIndex = 3;
            this.label13.Text = "μs";
            // 
            // trackBarExpCam7
            // 
            this.trackBarExpCam7.AutoSize = false;
            this.trackBarExpCam7.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam7.Maximum = 2000;
            this.trackBarExpCam7.Minimum = 1;
            this.trackBarExpCam7.Name = "trackBarExpCam7";
            this.trackBarExpCam7.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam7.TabIndex = 2;
            this.trackBarExpCam7.TickFrequency = 200;
            this.trackBarExpCam7.Value = 50;
            // 
            // numExpCam7
            // 
            this.numExpCam7.Location = new System.Drawing.Point(118, 5);
            this.numExpCam7.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam7.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam7.Name = "numExpCam7";
            this.numExpCam7.Size = new System.Drawing.Size(70, 25);
            this.numExpCam7.TabIndex = 1;
            this.numExpCam7.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(5, 5);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(46, 15);
            this.label14.TabIndex = 0;
            this.label14.Text = "CAM7";
            // 
            // panelExpCam6
            // 
            this.panelExpCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam6.Controls.Add(this.label11);
            this.panelExpCam6.Controls.Add(this.trackBarExpCam6);
            this.panelExpCam6.Controls.Add(this.numExpCam6);
            this.panelExpCam6.Controls.Add(this.label12);
            this.panelExpCam6.Location = new System.Drawing.Point(0, 438);
            this.panelExpCam6.Name = "panelExpCam6";
            this.panelExpCam6.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam6.TabIndex = 4;
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(214, 7);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(27, 15);
            this.label11.TabIndex = 3;
            this.label11.Text = "μs";
            // 
            // trackBarExpCam6
            // 
            this.trackBarExpCam6.AutoSize = false;
            this.trackBarExpCam6.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam6.Maximum = 2000;
            this.trackBarExpCam6.Minimum = 1;
            this.trackBarExpCam6.Name = "trackBarExpCam6";
            this.trackBarExpCam6.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam6.TabIndex = 2;
            this.trackBarExpCam6.TickFrequency = 200;
            this.trackBarExpCam6.Value = 50;
            // 
            // numExpCam6
            // 
            this.numExpCam6.Location = new System.Drawing.Point(118, 5);
            this.numExpCam6.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam6.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam6.Name = "numExpCam6";
            this.numExpCam6.Size = new System.Drawing.Size(70, 25);
            this.numExpCam6.TabIndex = 1;
            this.numExpCam6.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(5, 5);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(46, 15);
            this.label12.TabIndex = 0;
            this.label12.Text = "CAM6";
            // 
            // panelExpCam5
            // 
            this.panelExpCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam5.Controls.Add(this.label9);
            this.panelExpCam5.Controls.Add(this.trackBarExpCam5);
            this.panelExpCam5.Controls.Add(this.numExpCam5);
            this.panelExpCam5.Controls.Add(this.label10);
            this.panelExpCam5.Location = new System.Drawing.Point(0, 365);
            this.panelExpCam5.Name = "panelExpCam5";
            this.panelExpCam5.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam5.TabIndex = 4;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(214, 7);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(27, 15);
            this.label9.TabIndex = 3;
            this.label9.Text = "μs";
            // 
            // trackBarExpCam5
            // 
            this.trackBarExpCam5.AutoSize = false;
            this.trackBarExpCam5.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam5.Maximum = 2000;
            this.trackBarExpCam5.Minimum = 1;
            this.trackBarExpCam5.Name = "trackBarExpCam5";
            this.trackBarExpCam5.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam5.TabIndex = 2;
            this.trackBarExpCam5.TickFrequency = 200;
            this.trackBarExpCam5.Value = 50;
            // 
            // numExpCam5
            // 
            this.numExpCam5.Location = new System.Drawing.Point(118, 5);
            this.numExpCam5.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam5.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam5.Name = "numExpCam5";
            this.numExpCam5.Size = new System.Drawing.Size(70, 25);
            this.numExpCam5.TabIndex = 1;
            this.numExpCam5.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(5, 5);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(46, 15);
            this.label10.TabIndex = 0;
            this.label10.Text = "CAM5";
            // 
            // panelExpCam4
            // 
            this.panelExpCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam4.Controls.Add(this.label7);
            this.panelExpCam4.Controls.Add(this.trackBarExpCam4);
            this.panelExpCam4.Controls.Add(this.numExpCam4);
            this.panelExpCam4.Controls.Add(this.label8);
            this.panelExpCam4.Location = new System.Drawing.Point(0, 292);
            this.panelExpCam4.Name = "panelExpCam4";
            this.panelExpCam4.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam4.TabIndex = 4;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(214, 7);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(27, 15);
            this.label7.TabIndex = 3;
            this.label7.Text = "μs";
            // 
            // trackBarExpCam4
            // 
            this.trackBarExpCam4.AutoSize = false;
            this.trackBarExpCam4.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam4.Maximum = 2000;
            this.trackBarExpCam4.Minimum = 1;
            this.trackBarExpCam4.Name = "trackBarExpCam4";
            this.trackBarExpCam4.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam4.TabIndex = 2;
            this.trackBarExpCam4.TickFrequency = 200;
            this.trackBarExpCam4.Value = 50;
            // 
            // numExpCam4
            // 
            this.numExpCam4.Location = new System.Drawing.Point(118, 5);
            this.numExpCam4.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam4.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam4.Name = "numExpCam4";
            this.numExpCam4.Size = new System.Drawing.Size(70, 25);
            this.numExpCam4.TabIndex = 1;
            this.numExpCam4.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(5, 5);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(46, 15);
            this.label8.TabIndex = 0;
            this.label8.Text = "CAM4";
            // 
            // panelExpCam3
            // 
            this.panelExpCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam3.Controls.Add(this.label5);
            this.panelExpCam3.Controls.Add(this.trackBarExpCam3);
            this.panelExpCam3.Controls.Add(this.numExpCam3);
            this.panelExpCam3.Controls.Add(this.label6);
            this.panelExpCam3.Location = new System.Drawing.Point(0, 219);
            this.panelExpCam3.Name = "panelExpCam3";
            this.panelExpCam3.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam3.TabIndex = 4;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(214, 7);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(27, 15);
            this.label5.TabIndex = 3;
            this.label5.Text = "μs";
            // 
            // trackBarExpCam3
            // 
            this.trackBarExpCam3.AutoSize = false;
            this.trackBarExpCam3.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam3.Maximum = 2000;
            this.trackBarExpCam3.Minimum = 1;
            this.trackBarExpCam3.Name = "trackBarExpCam3";
            this.trackBarExpCam3.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam3.TabIndex = 2;
            this.trackBarExpCam3.TickFrequency = 200;
            this.trackBarExpCam3.Value = 50;
            // 
            // numExpCam3
            // 
            this.numExpCam3.Location = new System.Drawing.Point(118, 5);
            this.numExpCam3.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam3.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam3.Name = "numExpCam3";
            this.numExpCam3.Size = new System.Drawing.Size(70, 25);
            this.numExpCam3.TabIndex = 1;
            this.numExpCam3.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(5, 5);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(46, 15);
            this.label6.TabIndex = 0;
            this.label6.Text = "CAM3";
            // 
            // panelExpCam2
            // 
            this.panelExpCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam2.Controls.Add(this.label3);
            this.panelExpCam2.Controls.Add(this.trackBarExpCam2);
            this.panelExpCam2.Controls.Add(this.numExpCam2);
            this.panelExpCam2.Controls.Add(this.label4);
            this.panelExpCam2.Location = new System.Drawing.Point(0, 146);
            this.panelExpCam2.Name = "panelExpCam2";
            this.panelExpCam2.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam2.TabIndex = 4;
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(214, 7);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(27, 15);
            this.label3.TabIndex = 3;
            this.label3.Text = "μs";
            // 
            // trackBarExpCam2
            // 
            this.trackBarExpCam2.AutoSize = false;
            this.trackBarExpCam2.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam2.Maximum = 2000;
            this.trackBarExpCam2.Minimum = 1;
            this.trackBarExpCam2.Name = "trackBarExpCam2";
            this.trackBarExpCam2.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam2.TabIndex = 2;
            this.trackBarExpCam2.TickFrequency = 200;
            this.trackBarExpCam2.Value = 50;
            // 
            // numExpCam2
            // 
            this.numExpCam2.Location = new System.Drawing.Point(118, 5);
            this.numExpCam2.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam2.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam2.Name = "numExpCam2";
            this.numExpCam2.Size = new System.Drawing.Size(70, 25);
            this.numExpCam2.TabIndex = 1;
            this.numExpCam2.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(5, 5);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(46, 15);
            this.label4.TabIndex = 0;
            this.label4.Text = "CAM2";
            // 
            // panelExpCam1
            // 
            this.panelExpCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam1.Controls.Add(this.label2);
            this.panelExpCam1.Controls.Add(this.trackBarExpCam1);
            this.panelExpCam1.Controls.Add(this.numExpCam1);
            this.panelExpCam1.Controls.Add(this.lblExposure);
            this.panelExpCam1.Location = new System.Drawing.Point(0, 73);
            this.panelExpCam1.Name = "panelExpCam1";
            this.panelExpCam1.Size = new System.Drawing.Size(199, 69);
            this.panelExpCam1.TabIndex = 0;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(214, 7);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(27, 15);
            this.label2.TabIndex = 3;
            this.label2.Text = "μs";
            // 
            // trackBarExpCam1
            // 
            this.trackBarExpCam1.AutoSize = false;
            this.trackBarExpCam1.Location = new System.Drawing.Point(1, 33);
            this.trackBarExpCam1.Maximum = 2000;
            this.trackBarExpCam1.Minimum = 1;
            this.trackBarExpCam1.Name = "trackBarExpCam1";
            this.trackBarExpCam1.Size = new System.Drawing.Size(190, 30);
            this.trackBarExpCam1.TabIndex = 2;
            this.trackBarExpCam1.TickFrequency = 200;
            this.trackBarExpCam1.Value = 50;
            // 
            // numExpCam1
            // 
            this.numExpCam1.Location = new System.Drawing.Point(118, 5);
            this.numExpCam1.Maximum = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            this.numExpCam1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam1.Name = "numExpCam1";
            this.numExpCam1.Size = new System.Drawing.Size(70, 25);
            this.numExpCam1.TabIndex = 1;
            this.numExpCam1.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // lblExposure
            // 
            this.lblExposure.AutoSize = true;
            this.lblExposure.Location = new System.Drawing.Point(5, 5);
            this.lblExposure.Name = "lblExposure";
            this.lblExposure.Size = new System.Drawing.Size(46, 15);
            this.lblExposure.TabIndex = 0;
            this.lblExposure.Text = "CAM1";
            // 
            // tabPageLineRate
            // 
            this.tabPageLineRate.Controls.Add(this.panelLrAll);
            this.tabPageLineRate.Controls.Add(this.panelLrCam7);
            this.tabPageLineRate.Controls.Add(this.panelLrCam6);
            this.tabPageLineRate.Controls.Add(this.panelLrCam5);
            this.tabPageLineRate.Controls.Add(this.panelLrCam4);
            this.tabPageLineRate.Controls.Add(this.panelLrCam3);
            this.tabPageLineRate.Controls.Add(this.panelLrCam2);
            this.tabPageLineRate.Controls.Add(this.panelLrCam1);
            this.tabPageLineRate.Location = new System.Drawing.Point(4, 25);
            this.tabPageLineRate.Name = "tabPageLineRate";
            this.tabPageLineRate.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageLineRate.Size = new System.Drawing.Size(199, 596);
            this.tabPageLineRate.TabIndex = 1;
            this.tabPageLineRate.Text = "線掃速率";
            this.tabPageLineRate.UseVisualStyleBackColor = true;
            // 
            // panelLrAll
            // 
            this.panelLrAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelLrAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrAll.Controls.Add(this.lblLrAllUnit);
            this.panelLrAll.Controls.Add(this.trackBarLrAll);
            this.panelLrAll.Controls.Add(this.numLrAll);
            this.panelLrAll.Controls.Add(this.lblLrAll);
            this.panelLrAll.Location = new System.Drawing.Point(0, 0);
            this.panelLrAll.Name = "panelLrAll";
            this.panelLrAll.Size = new System.Drawing.Size(199, 69);
            this.panelLrAll.TabIndex = 10;
            // 
            // lblLrAllUnit
            // 
            this.lblLrAllUnit.AutoSize = true;
            this.lblLrAllUnit.Location = new System.Drawing.Point(214, 7);
            this.lblLrAllUnit.Name = "lblLrAllUnit";
            this.lblLrAllUnit.Size = new System.Drawing.Size(23, 15);
            this.lblLrAllUnit.TabIndex = 3;
            this.lblLrAllUnit.Text = "Hz";
            // 
            // trackBarLrAll
            // 
            this.trackBarLrAll.AutoSize = false;
            this.trackBarLrAll.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrAll.Maximum = 10000;
            this.trackBarLrAll.Minimum = 100;
            this.trackBarLrAll.Name = "trackBarLrAll";
            this.trackBarLrAll.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrAll.TabIndex = 2;
            this.trackBarLrAll.TickFrequency = 1000;
            this.trackBarLrAll.Value = 3000;
            // 
            // numLrAll
            // 
            this.numLrAll.Location = new System.Drawing.Point(118, 5);
            this.numLrAll.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrAll.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrAll.Name = "numLrAll";
            this.numLrAll.Size = new System.Drawing.Size(70, 25);
            this.numLrAll.TabIndex = 1;
            this.numLrAll.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // lblLrAll
            // 
            this.lblLrAll.AutoSize = true;
            this.lblLrAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblLrAll.Location = new System.Drawing.Point(5, 5);
            this.lblLrAll.Name = "lblLrAll";
            this.lblLrAll.Size = new System.Drawing.Size(38, 15);
            this.lblLrAll.TabIndex = 0;
            this.lblLrAll.Text = "ALL";
            // 
            // panelLrCam7
            // 
            this.panelLrCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam7.Controls.Add(this.label15);
            this.panelLrCam7.Controls.Add(this.trackBarLrCam7);
            this.panelLrCam7.Controls.Add(this.numLrCam7);
            this.panelLrCam7.Controls.Add(this.label16);
            this.panelLrCam7.Location = new System.Drawing.Point(0, 511);
            this.panelLrCam7.Name = "panelLrCam7";
            this.panelLrCam7.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam7.TabIndex = 6;
            // 
            // label15
            // 
            this.label15.AutoSize = true;
            this.label15.Location = new System.Drawing.Point(214, 7);
            this.label15.Name = "label15";
            this.label15.Size = new System.Drawing.Size(23, 15);
            this.label15.TabIndex = 3;
            this.label15.Text = "Hz";
            // 
            // trackBarLrCam7
            // 
            this.trackBarLrCam7.AutoSize = false;
            this.trackBarLrCam7.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam7.Maximum = 2000;
            this.trackBarLrCam7.Minimum = 1;
            this.trackBarLrCam7.Name = "trackBarLrCam7";
            this.trackBarLrCam7.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam7.TabIndex = 2;
            this.trackBarLrCam7.TickFrequency = 200;
            this.trackBarLrCam7.Value = 50;
            // 
            // numLrCam7
            // 
            this.numLrCam7.Location = new System.Drawing.Point(118, 5);
            this.numLrCam7.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam7.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam7.Name = "numLrCam7";
            this.numLrCam7.Size = new System.Drawing.Size(70, 25);
            this.numLrCam7.TabIndex = 1;
            this.numLrCam7.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label16
            // 
            this.label16.AutoSize = true;
            this.label16.Location = new System.Drawing.Point(5, 5);
            this.label16.Name = "label16";
            this.label16.Size = new System.Drawing.Size(46, 15);
            this.label16.TabIndex = 0;
            this.label16.Text = "CAM7";
            // 
            // panelLrCam6
            // 
            this.panelLrCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam6.Controls.Add(this.label17);
            this.panelLrCam6.Controls.Add(this.trackBarLrCam6);
            this.panelLrCam6.Controls.Add(this.numLrCam6);
            this.panelLrCam6.Controls.Add(this.label18);
            this.panelLrCam6.Location = new System.Drawing.Point(0, 438);
            this.panelLrCam6.Name = "panelLrCam6";
            this.panelLrCam6.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam6.TabIndex = 7;
            // 
            // label17
            // 
            this.label17.AutoSize = true;
            this.label17.Location = new System.Drawing.Point(214, 7);
            this.label17.Name = "label17";
            this.label17.Size = new System.Drawing.Size(23, 15);
            this.label17.TabIndex = 3;
            this.label17.Text = "Hz";
            // 
            // trackBarLrCam6
            // 
            this.trackBarLrCam6.AutoSize = false;
            this.trackBarLrCam6.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam6.Maximum = 2000;
            this.trackBarLrCam6.Minimum = 1;
            this.trackBarLrCam6.Name = "trackBarLrCam6";
            this.trackBarLrCam6.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam6.TabIndex = 2;
            this.trackBarLrCam6.TickFrequency = 200;
            this.trackBarLrCam6.Value = 50;
            // 
            // numLrCam6
            // 
            this.numLrCam6.Location = new System.Drawing.Point(118, 5);
            this.numLrCam6.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam6.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam6.Name = "numLrCam6";
            this.numLrCam6.Size = new System.Drawing.Size(70, 25);
            this.numLrCam6.TabIndex = 1;
            this.numLrCam6.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label18
            // 
            this.label18.AutoSize = true;
            this.label18.Location = new System.Drawing.Point(5, 5);
            this.label18.Name = "label18";
            this.label18.Size = new System.Drawing.Size(46, 15);
            this.label18.TabIndex = 0;
            this.label18.Text = "CAM6";
            // 
            // panelLrCam5
            // 
            this.panelLrCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam5.Controls.Add(this.label19);
            this.panelLrCam5.Controls.Add(this.trackBarLrCam5);
            this.panelLrCam5.Controls.Add(this.numLrCam5);
            this.panelLrCam5.Controls.Add(this.label20);
            this.panelLrCam5.Location = new System.Drawing.Point(0, 365);
            this.panelLrCam5.Name = "panelLrCam5";
            this.panelLrCam5.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam5.TabIndex = 8;
            // 
            // label19
            // 
            this.label19.AutoSize = true;
            this.label19.Location = new System.Drawing.Point(214, 7);
            this.label19.Name = "label19";
            this.label19.Size = new System.Drawing.Size(23, 15);
            this.label19.TabIndex = 3;
            this.label19.Text = "Hz";
            // 
            // trackBarLrCam5
            // 
            this.trackBarLrCam5.AutoSize = false;
            this.trackBarLrCam5.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam5.Maximum = 2000;
            this.trackBarLrCam5.Minimum = 1;
            this.trackBarLrCam5.Name = "trackBarLrCam5";
            this.trackBarLrCam5.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam5.TabIndex = 2;
            this.trackBarLrCam5.TickFrequency = 200;
            this.trackBarLrCam5.Value = 50;
            // 
            // numLrCam5
            // 
            this.numLrCam5.Location = new System.Drawing.Point(118, 5);
            this.numLrCam5.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam5.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam5.Name = "numLrCam5";
            this.numLrCam5.Size = new System.Drawing.Size(70, 25);
            this.numLrCam5.TabIndex = 1;
            this.numLrCam5.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label20
            // 
            this.label20.AutoSize = true;
            this.label20.Location = new System.Drawing.Point(5, 5);
            this.label20.Name = "label20";
            this.label20.Size = new System.Drawing.Size(46, 15);
            this.label20.TabIndex = 0;
            this.label20.Text = "CAM5";
            // 
            // panelLrCam4
            // 
            this.panelLrCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam4.Controls.Add(this.label21);
            this.panelLrCam4.Controls.Add(this.trackBarLrCam4);
            this.panelLrCam4.Controls.Add(this.numLrCam4);
            this.panelLrCam4.Controls.Add(this.label22);
            this.panelLrCam4.Location = new System.Drawing.Point(0, 292);
            this.panelLrCam4.Name = "panelLrCam4";
            this.panelLrCam4.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam4.TabIndex = 9;
            // 
            // label21
            // 
            this.label21.AutoSize = true;
            this.label21.Location = new System.Drawing.Point(214, 7);
            this.label21.Name = "label21";
            this.label21.Size = new System.Drawing.Size(23, 15);
            this.label21.TabIndex = 3;
            this.label21.Text = "Hz";
            // 
            // trackBarLrCam4
            // 
            this.trackBarLrCam4.AutoSize = false;
            this.trackBarLrCam4.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam4.Maximum = 2000;
            this.trackBarLrCam4.Minimum = 1;
            this.trackBarLrCam4.Name = "trackBarLrCam4";
            this.trackBarLrCam4.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam4.TabIndex = 2;
            this.trackBarLrCam4.TickFrequency = 200;
            this.trackBarLrCam4.Value = 50;
            // 
            // numLrCam4
            // 
            this.numLrCam4.Location = new System.Drawing.Point(118, 5);
            this.numLrCam4.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam4.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam4.Name = "numLrCam4";
            this.numLrCam4.Size = new System.Drawing.Size(70, 25);
            this.numLrCam4.TabIndex = 1;
            this.numLrCam4.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label22
            // 
            this.label22.AutoSize = true;
            this.label22.Location = new System.Drawing.Point(5, 5);
            this.label22.Name = "label22";
            this.label22.Size = new System.Drawing.Size(46, 15);
            this.label22.TabIndex = 0;
            this.label22.Text = "CAM4";
            // 
            // panelLrCam3
            // 
            this.panelLrCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam3.Controls.Add(this.label23);
            this.panelLrCam3.Controls.Add(this.trackBarLrCam3);
            this.panelLrCam3.Controls.Add(this.numLrCam3);
            this.panelLrCam3.Controls.Add(this.label24);
            this.panelLrCam3.Location = new System.Drawing.Point(0, 219);
            this.panelLrCam3.Name = "panelLrCam3";
            this.panelLrCam3.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam3.TabIndex = 10;
            // 
            // label23
            // 
            this.label23.AutoSize = true;
            this.label23.Location = new System.Drawing.Point(214, 7);
            this.label23.Name = "label23";
            this.label23.Size = new System.Drawing.Size(23, 15);
            this.label23.TabIndex = 3;
            this.label23.Text = "Hz";
            // 
            // trackBarLrCam3
            // 
            this.trackBarLrCam3.AutoSize = false;
            this.trackBarLrCam3.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam3.Maximum = 2000;
            this.trackBarLrCam3.Minimum = 1;
            this.trackBarLrCam3.Name = "trackBarLrCam3";
            this.trackBarLrCam3.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam3.TabIndex = 2;
            this.trackBarLrCam3.TickFrequency = 200;
            this.trackBarLrCam3.Value = 50;
            // 
            // numLrCam3
            // 
            this.numLrCam3.Location = new System.Drawing.Point(118, 5);
            this.numLrCam3.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam3.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam3.Name = "numLrCam3";
            this.numLrCam3.Size = new System.Drawing.Size(70, 25);
            this.numLrCam3.TabIndex = 1;
            this.numLrCam3.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label24
            // 
            this.label24.AutoSize = true;
            this.label24.Location = new System.Drawing.Point(5, 5);
            this.label24.Name = "label24";
            this.label24.Size = new System.Drawing.Size(46, 15);
            this.label24.TabIndex = 0;
            this.label24.Text = "CAM3";
            // 
            // panelLrCam2
            // 
            this.panelLrCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam2.Controls.Add(this.label25);
            this.panelLrCam2.Controls.Add(this.trackBarLrCam2);
            this.panelLrCam2.Controls.Add(this.numLrCam2);
            this.panelLrCam2.Controls.Add(this.label26);
            this.panelLrCam2.Location = new System.Drawing.Point(0, 146);
            this.panelLrCam2.Name = "panelLrCam2";
            this.panelLrCam2.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam2.TabIndex = 11;
            // 
            // label25
            // 
            this.label25.AutoSize = true;
            this.label25.Location = new System.Drawing.Point(214, 7);
            this.label25.Name = "label25";
            this.label25.Size = new System.Drawing.Size(23, 15);
            this.label25.TabIndex = 3;
            this.label25.Text = "Hz";
            // 
            // trackBarLrCam2
            // 
            this.trackBarLrCam2.AutoSize = false;
            this.trackBarLrCam2.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam2.Maximum = 2000;
            this.trackBarLrCam2.Minimum = 1;
            this.trackBarLrCam2.Name = "trackBarLrCam2";
            this.trackBarLrCam2.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam2.TabIndex = 2;
            this.trackBarLrCam2.TickFrequency = 200;
            this.trackBarLrCam2.Value = 50;
            // 
            // numLrCam2
            // 
            this.numLrCam2.Location = new System.Drawing.Point(118, 5);
            this.numLrCam2.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam2.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam2.Name = "numLrCam2";
            this.numLrCam2.Size = new System.Drawing.Size(70, 25);
            this.numLrCam2.TabIndex = 1;
            this.numLrCam2.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // label26
            // 
            this.label26.AutoSize = true;
            this.label26.Location = new System.Drawing.Point(5, 5);
            this.label26.Name = "label26";
            this.label26.Size = new System.Drawing.Size(46, 15);
            this.label26.TabIndex = 0;
            this.label26.Text = "CAM2";
            // 
            // panelLrCam1
            // 
            this.panelLrCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam1.Controls.Add(this.label27);
            this.panelLrCam1.Controls.Add(this.trackBarLrCam1);
            this.panelLrCam1.Controls.Add(this.numLrCam1);
            this.panelLrCam1.Controls.Add(this.lblGrabHeight);
            this.panelLrCam1.Location = new System.Drawing.Point(0, 73);
            this.panelLrCam1.Name = "panelLrCam1";
            this.panelLrCam1.Size = new System.Drawing.Size(199, 69);
            this.panelLrCam1.TabIndex = 5;
            // 
            // label27
            // 
            this.label27.AutoSize = true;
            this.label27.Location = new System.Drawing.Point(214, 7);
            this.label27.Name = "label27";
            this.label27.Size = new System.Drawing.Size(23, 15);
            this.label27.TabIndex = 3;
            this.label27.Text = "Hz";
            // 
            // trackBarLrCam1
            // 
            this.trackBarLrCam1.AutoSize = false;
            this.trackBarLrCam1.Location = new System.Drawing.Point(1, 33);
            this.trackBarLrCam1.Maximum = 10000;
            this.trackBarLrCam1.Minimum = 100;
            this.trackBarLrCam1.Name = "trackBarLrCam1";
            this.trackBarLrCam1.Size = new System.Drawing.Size(190, 30);
            this.trackBarLrCam1.TabIndex = 2;
            this.trackBarLrCam1.TickFrequency = 200;
            this.trackBarLrCam1.Value = 100;
            // 
            // numLrCam1
            // 
            this.numLrCam1.Location = new System.Drawing.Point(118, 5);
            this.numLrCam1.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numLrCam1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numLrCam1.Name = "numLrCam1";
            this.numLrCam1.Size = new System.Drawing.Size(70, 25);
            this.numLrCam1.TabIndex = 1;
            this.numLrCam1.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // lblGrabHeight
            // 
            this.lblGrabHeight.AutoSize = true;
            this.lblGrabHeight.Location = new System.Drawing.Point(5, 5);
            this.lblGrabHeight.Name = "lblGrabHeight";
            this.lblGrabHeight.Size = new System.Drawing.Size(46, 15);
            this.lblGrabHeight.TabIndex = 0;
            this.lblGrabHeight.Text = "CAM1";
            // 
            // tabPageGrabHeight
            // 
            this.tabPageGrabHeight.Controls.Add(this.panelHtAll);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam7);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam6);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam5);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam4);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam3);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam2);
            this.tabPageGrabHeight.Controls.Add(this.panelHtCam1);
            this.tabPageGrabHeight.Location = new System.Drawing.Point(4, 25);
            this.tabPageGrabHeight.Name = "tabPageGrabHeight";
            this.tabPageGrabHeight.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageGrabHeight.Size = new System.Drawing.Size(199, 596);
            this.tabPageGrabHeight.TabIndex = 2;
            this.tabPageGrabHeight.Text = "擷取高度";
            this.tabPageGrabHeight.UseVisualStyleBackColor = true;
            // 
            // panelHtAll
            // 
            this.panelHtAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelHtAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtAll.Controls.Add(this.lblHtAllUnit);
            this.panelHtAll.Controls.Add(this.trackBarHtAll);
            this.panelHtAll.Controls.Add(this.numHtAll);
            this.panelHtAll.Controls.Add(this.lblHtAll);
            this.panelHtAll.Location = new System.Drawing.Point(0, 0);
            this.panelHtAll.Name = "panelHtAll";
            this.panelHtAll.Size = new System.Drawing.Size(199, 69);
            this.panelHtAll.TabIndex = 10;
            // 
            // lblHtAllUnit
            // 
            this.lblHtAllUnit.AutoSize = true;
            this.lblHtAllUnit.Location = new System.Drawing.Point(214, 7);
            this.lblHtAllUnit.Name = "lblHtAllUnit";
            this.lblHtAllUnit.Size = new System.Drawing.Size(21, 15);
            this.lblHtAllUnit.TabIndex = 3;
            this.lblHtAllUnit.Text = "px";
            // 
            // trackBarHtAll
            // 
            this.trackBarHtAll.AutoSize = false;
            this.trackBarHtAll.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtAll.Maximum = 10000;
            this.trackBarHtAll.Minimum = 100;
            this.trackBarHtAll.Name = "trackBarHtAll";
            this.trackBarHtAll.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtAll.TabIndex = 2;
            this.trackBarHtAll.TickFrequency = 1000;
            this.trackBarHtAll.Value = 3000;
            // 
            // numHtAll
            // 
            this.numHtAll.Location = new System.Drawing.Point(118, 5);
            this.numHtAll.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtAll.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtAll.Name = "numHtAll";
            this.numHtAll.Size = new System.Drawing.Size(70, 25);
            this.numHtAll.TabIndex = 1;
            this.numHtAll.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // lblHtAll
            // 
            this.lblHtAll.AutoSize = true;
            this.lblHtAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblHtAll.Location = new System.Drawing.Point(5, 5);
            this.lblHtAll.Name = "lblHtAll";
            this.lblHtAll.Size = new System.Drawing.Size(38, 15);
            this.lblHtAll.TabIndex = 0;
            this.lblHtAll.Text = "ALL";
            // 
            // panelHtCam7
            // 
            this.panelHtCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam7.Controls.Add(this.label1);
            this.panelHtCam7.Controls.Add(this.trackBarHtCam7);
            this.panelHtCam7.Controls.Add(this.numHtCam7);
            this.panelHtCam7.Controls.Add(this.label28);
            this.panelHtCam7.Location = new System.Drawing.Point(0, 511);
            this.panelHtCam7.Name = "panelHtCam7";
            this.panelHtCam7.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam7.TabIndex = 6;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(214, 7);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(21, 15);
            this.label1.TabIndex = 3;
            this.label1.Text = "px";
            // 
            // trackBarHtCam7
            // 
            this.trackBarHtCam7.AutoSize = false;
            this.trackBarHtCam7.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam7.Maximum = 2000;
            this.trackBarHtCam7.Minimum = 1;
            this.trackBarHtCam7.Name = "trackBarHtCam7";
            this.trackBarHtCam7.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam7.TabIndex = 2;
            this.trackBarHtCam7.TickFrequency = 200;
            this.trackBarHtCam7.Value = 50;
            // 
            // numHtCam7
            // 
            this.numHtCam7.Location = new System.Drawing.Point(118, 5);
            this.numHtCam7.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam7.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam7.Name = "numHtCam7";
            this.numHtCam7.Size = new System.Drawing.Size(70, 25);
            this.numHtCam7.TabIndex = 1;
            this.numHtCam7.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label28
            // 
            this.label28.AutoSize = true;
            this.label28.Location = new System.Drawing.Point(5, 5);
            this.label28.Name = "label28";
            this.label28.Size = new System.Drawing.Size(46, 15);
            this.label28.TabIndex = 0;
            this.label28.Text = "CAM7";
            // 
            // panelHtCam6
            // 
            this.panelHtCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam6.Controls.Add(this.label29);
            this.panelHtCam6.Controls.Add(this.trackBarHtCam6);
            this.panelHtCam6.Controls.Add(this.numHtCam6);
            this.panelHtCam6.Controls.Add(this.label30);
            this.panelHtCam6.Location = new System.Drawing.Point(0, 438);
            this.panelHtCam6.Name = "panelHtCam6";
            this.panelHtCam6.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam6.TabIndex = 7;
            // 
            // label29
            // 
            this.label29.AutoSize = true;
            this.label29.Location = new System.Drawing.Point(214, 7);
            this.label29.Name = "label29";
            this.label29.Size = new System.Drawing.Size(21, 15);
            this.label29.TabIndex = 3;
            this.label29.Text = "px";
            // 
            // trackBarHtCam6
            // 
            this.trackBarHtCam6.AutoSize = false;
            this.trackBarHtCam6.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam6.Maximum = 2000;
            this.trackBarHtCam6.Minimum = 1;
            this.trackBarHtCam6.Name = "trackBarHtCam6";
            this.trackBarHtCam6.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam6.TabIndex = 2;
            this.trackBarHtCam6.TickFrequency = 200;
            this.trackBarHtCam6.Value = 50;
            // 
            // numHtCam6
            // 
            this.numHtCam6.Location = new System.Drawing.Point(118, 5);
            this.numHtCam6.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam6.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam6.Name = "numHtCam6";
            this.numHtCam6.Size = new System.Drawing.Size(70, 25);
            this.numHtCam6.TabIndex = 1;
            this.numHtCam6.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label30
            // 
            this.label30.AutoSize = true;
            this.label30.Location = new System.Drawing.Point(5, 5);
            this.label30.Name = "label30";
            this.label30.Size = new System.Drawing.Size(46, 15);
            this.label30.TabIndex = 0;
            this.label30.Text = "CAM6";
            // 
            // panelHtCam5
            // 
            this.panelHtCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam5.Controls.Add(this.label31);
            this.panelHtCam5.Controls.Add(this.trackBarHtCam5);
            this.panelHtCam5.Controls.Add(this.numHtCam5);
            this.panelHtCam5.Controls.Add(this.label32);
            this.panelHtCam5.Location = new System.Drawing.Point(0, 365);
            this.panelHtCam5.Name = "panelHtCam5";
            this.panelHtCam5.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam5.TabIndex = 8;
            // 
            // label31
            // 
            this.label31.AutoSize = true;
            this.label31.Location = new System.Drawing.Point(214, 7);
            this.label31.Name = "label31";
            this.label31.Size = new System.Drawing.Size(21, 15);
            this.label31.TabIndex = 3;
            this.label31.Text = "px";
            // 
            // trackBarHtCam5
            // 
            this.trackBarHtCam5.AutoSize = false;
            this.trackBarHtCam5.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam5.Maximum = 2000;
            this.trackBarHtCam5.Minimum = 1;
            this.trackBarHtCam5.Name = "trackBarHtCam5";
            this.trackBarHtCam5.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam5.TabIndex = 2;
            this.trackBarHtCam5.TickFrequency = 200;
            this.trackBarHtCam5.Value = 50;
            // 
            // numHtCam5
            // 
            this.numHtCam5.Location = new System.Drawing.Point(118, 5);
            this.numHtCam5.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam5.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam5.Name = "numHtCam5";
            this.numHtCam5.Size = new System.Drawing.Size(70, 25);
            this.numHtCam5.TabIndex = 1;
            this.numHtCam5.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label32
            // 
            this.label32.AutoSize = true;
            this.label32.Location = new System.Drawing.Point(5, 5);
            this.label32.Name = "label32";
            this.label32.Size = new System.Drawing.Size(46, 15);
            this.label32.TabIndex = 0;
            this.label32.Text = "CAM5";
            // 
            // panelHtCam4
            // 
            this.panelHtCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam4.Controls.Add(this.label33);
            this.panelHtCam4.Controls.Add(this.trackBarHtCam4);
            this.panelHtCam4.Controls.Add(this.numHtCam4);
            this.panelHtCam4.Controls.Add(this.label34);
            this.panelHtCam4.Location = new System.Drawing.Point(0, 292);
            this.panelHtCam4.Name = "panelHtCam4";
            this.panelHtCam4.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam4.TabIndex = 9;
            // 
            // label33
            // 
            this.label33.AutoSize = true;
            this.label33.Location = new System.Drawing.Point(214, 7);
            this.label33.Name = "label33";
            this.label33.Size = new System.Drawing.Size(21, 15);
            this.label33.TabIndex = 3;
            this.label33.Text = "px";
            // 
            // trackBarHtCam4
            // 
            this.trackBarHtCam4.AutoSize = false;
            this.trackBarHtCam4.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam4.Maximum = 2000;
            this.trackBarHtCam4.Minimum = 1;
            this.trackBarHtCam4.Name = "trackBarHtCam4";
            this.trackBarHtCam4.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam4.TabIndex = 2;
            this.trackBarHtCam4.TickFrequency = 200;
            this.trackBarHtCam4.Value = 50;
            // 
            // numHtCam4
            // 
            this.numHtCam4.Location = new System.Drawing.Point(118, 5);
            this.numHtCam4.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam4.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam4.Name = "numHtCam4";
            this.numHtCam4.Size = new System.Drawing.Size(70, 25);
            this.numHtCam4.TabIndex = 1;
            this.numHtCam4.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label34
            // 
            this.label34.AutoSize = true;
            this.label34.Location = new System.Drawing.Point(5, 5);
            this.label34.Name = "label34";
            this.label34.Size = new System.Drawing.Size(46, 15);
            this.label34.TabIndex = 0;
            this.label34.Text = "CAM4";
            // 
            // panelHtCam3
            // 
            this.panelHtCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam3.Controls.Add(this.label35);
            this.panelHtCam3.Controls.Add(this.trackBarHtCam3);
            this.panelHtCam3.Controls.Add(this.numHtCam3);
            this.panelHtCam3.Controls.Add(this.label36);
            this.panelHtCam3.Location = new System.Drawing.Point(0, 219);
            this.panelHtCam3.Name = "panelHtCam3";
            this.panelHtCam3.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam3.TabIndex = 10;
            // 
            // label35
            // 
            this.label35.AutoSize = true;
            this.label35.Location = new System.Drawing.Point(214, 7);
            this.label35.Name = "label35";
            this.label35.Size = new System.Drawing.Size(21, 15);
            this.label35.TabIndex = 3;
            this.label35.Text = "px";
            // 
            // trackBarHtCam3
            // 
            this.trackBarHtCam3.AutoSize = false;
            this.trackBarHtCam3.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam3.Maximum = 2000;
            this.trackBarHtCam3.Minimum = 1;
            this.trackBarHtCam3.Name = "trackBarHtCam3";
            this.trackBarHtCam3.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam3.TabIndex = 2;
            this.trackBarHtCam3.TickFrequency = 200;
            this.trackBarHtCam3.Value = 50;
            // 
            // numHtCam3
            // 
            this.numHtCam3.Location = new System.Drawing.Point(118, 5);
            this.numHtCam3.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam3.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam3.Name = "numHtCam3";
            this.numHtCam3.Size = new System.Drawing.Size(70, 25);
            this.numHtCam3.TabIndex = 1;
            this.numHtCam3.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label36
            // 
            this.label36.AutoSize = true;
            this.label36.Location = new System.Drawing.Point(5, 5);
            this.label36.Name = "label36";
            this.label36.Size = new System.Drawing.Size(46, 15);
            this.label36.TabIndex = 0;
            this.label36.Text = "CAM3";
            // 
            // panelHtCam2
            // 
            this.panelHtCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam2.Controls.Add(this.label37);
            this.panelHtCam2.Controls.Add(this.trackBarHtCam2);
            this.panelHtCam2.Controls.Add(this.numHtCam2);
            this.panelHtCam2.Controls.Add(this.label38);
            this.panelHtCam2.Location = new System.Drawing.Point(0, 146);
            this.panelHtCam2.Name = "panelHtCam2";
            this.panelHtCam2.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam2.TabIndex = 11;
            // 
            // label37
            // 
            this.label37.AutoSize = true;
            this.label37.Location = new System.Drawing.Point(214, 7);
            this.label37.Name = "label37";
            this.label37.Size = new System.Drawing.Size(21, 15);
            this.label37.TabIndex = 3;
            this.label37.Text = "px";
            // 
            // trackBarHtCam2
            // 
            this.trackBarHtCam2.AutoSize = false;
            this.trackBarHtCam2.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam2.Maximum = 2000;
            this.trackBarHtCam2.Minimum = 1;
            this.trackBarHtCam2.Name = "trackBarHtCam2";
            this.trackBarHtCam2.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam2.TabIndex = 2;
            this.trackBarHtCam2.TickFrequency = 200;
            this.trackBarHtCam2.Value = 50;
            // 
            // numHtCam2
            // 
            this.numHtCam2.Location = new System.Drawing.Point(118, 5);
            this.numHtCam2.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam2.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam2.Name = "numHtCam2";
            this.numHtCam2.Size = new System.Drawing.Size(70, 25);
            this.numHtCam2.TabIndex = 1;
            this.numHtCam2.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label38
            // 
            this.label38.AutoSize = true;
            this.label38.Location = new System.Drawing.Point(5, 5);
            this.label38.Name = "label38";
            this.label38.Size = new System.Drawing.Size(46, 15);
            this.label38.TabIndex = 0;
            this.label38.Text = "CAM2";
            // 
            // panelHtCam1
            // 
            this.panelHtCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam1.Controls.Add(this.label39);
            this.panelHtCam1.Controls.Add(this.trackBarHtCam1);
            this.panelHtCam1.Controls.Add(this.numHtCam1);
            this.panelHtCam1.Controls.Add(this.label40);
            this.panelHtCam1.Location = new System.Drawing.Point(0, 73);
            this.panelHtCam1.Name = "panelHtCam1";
            this.panelHtCam1.Size = new System.Drawing.Size(199, 69);
            this.panelHtCam1.TabIndex = 5;
            // 
            // label39
            // 
            this.label39.AutoSize = true;
            this.label39.Location = new System.Drawing.Point(214, 7);
            this.label39.Name = "label39";
            this.label39.Size = new System.Drawing.Size(21, 15);
            this.label39.TabIndex = 3;
            this.label39.Text = "px";
            // 
            // trackBarHtCam1
            // 
            this.trackBarHtCam1.AutoSize = false;
            this.trackBarHtCam1.Location = new System.Drawing.Point(1, 33);
            this.trackBarHtCam1.Maximum = 2000;
            this.trackBarHtCam1.Minimum = 1;
            this.trackBarHtCam1.Name = "trackBarHtCam1";
            this.trackBarHtCam1.Size = new System.Drawing.Size(190, 30);
            this.trackBarHtCam1.TabIndex = 2;
            this.trackBarHtCam1.TickFrequency = 200;
            this.trackBarHtCam1.Value = 50;
            // 
            // numHtCam1
            // 
            this.numHtCam1.Location = new System.Drawing.Point(118, 5);
            this.numHtCam1.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam1.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numHtCam1.Name = "numHtCam1";
            this.numHtCam1.Size = new System.Drawing.Size(70, 25);
            this.numHtCam1.TabIndex = 1;
            this.numHtCam1.Value = new decimal(new int[] {
            2000,
            0,
            0,
            0});
            // 
            // label40
            // 
            this.label40.AutoSize = true;
            this.label40.Location = new System.Drawing.Point(5, 5);
            this.label40.Name = "label40";
            this.label40.Size = new System.Drawing.Size(46, 15);
            this.label40.TabIndex = 0;
            this.label40.Text = "CAM1";
            // 
            // tabPageSystem
            // 
            this.tabPageSystem.Controls.Add(this.listViewHardware);
            this.tabPageSystem.Controls.Add(this.label41);
            this.tabPageSystem.Controls.Add(this.listViewEngine);
            this.tabPageSystem.Controls.Add(this.lblEngineConst);
            this.tabPageSystem.Controls.Add(this.listViewCameras);
            this.tabPageSystem.Controls.Add(this.lblCamHardware);
            this.tabPageSystem.Location = new System.Drawing.Point(4, 25);
            this.tabPageSystem.Name = "tabPageSystem";
            this.tabPageSystem.Size = new System.Drawing.Size(213, 631);
            this.tabPageSystem.TabIndex = 2;
            this.tabPageSystem.Text = "系統資訊";
            this.tabPageSystem.UseVisualStyleBackColor = true;
            // 
            // listViewHardware
            // 
            this.listViewHardware.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listViewHardware.FullRowSelect = true;
            this.listViewHardware.GridLines = true;
            this.listViewHardware.HideSelection = false;
            this.listViewHardware.Location = new System.Drawing.Point(3, 495);
            this.listViewHardware.Name = "listViewHardware";
            this.listViewHardware.Size = new System.Drawing.Size(201, 133);
            this.listViewHardware.TabIndex = 7;
            this.listViewHardware.UseCompatibleStateImageBehavior = false;
            this.listViewHardware.View = System.Windows.Forms.View.Details;
            // 
            // label41
            // 
            this.label41.AutoSize = true;
            this.label41.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.label41.Location = new System.Drawing.Point(4, 477);
            this.label41.Name = "label41";
            this.label41.Size = new System.Drawing.Size(103, 15);
            this.label41.TabIndex = 6;
            this.label41.Text = "【硬體參數】";
            // 
            // listViewEngine
            // 
            this.listViewEngine.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listViewEngine.FullRowSelect = true;
            this.listViewEngine.GridLines = true;
            this.listViewEngine.HideSelection = false;
            this.listViewEngine.Location = new System.Drawing.Point(3, 187);
            this.listViewEngine.Name = "listViewEngine";
            this.listViewEngine.Size = new System.Drawing.Size(201, 284);
            this.listViewEngine.TabIndex = 3;
            this.listViewEngine.UseCompatibleStateImageBehavior = false;
            this.listViewEngine.View = System.Windows.Forms.View.Details;
            // 
            // lblEngineConst
            // 
            this.lblEngineConst.AutoSize = true;
            this.lblEngineConst.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblEngineConst.Location = new System.Drawing.Point(4, 169);
            this.lblEngineConst.Name = "lblEngineConst";
            this.lblEngineConst.Size = new System.Drawing.Size(135, 15);
            this.lblEngineConst.TabIndex = 2;
            this.lblEngineConst.Text = "【影像引擎常數】";
            // 
            // listViewCameras
            // 
            this.listViewCameras.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listViewCameras.FullRowSelect = true;
            this.listViewCameras.GridLines = true;
            this.listViewCameras.HideSelection = false;
            this.listViewCameras.Location = new System.Drawing.Point(3, 25);
            this.listViewCameras.Name = "listViewCameras";
            this.listViewCameras.Size = new System.Drawing.Size(201, 138);
            this.listViewCameras.TabIndex = 1;
            this.listViewCameras.UseCompatibleStateImageBehavior = false;
            this.listViewCameras.View = System.Windows.Forms.View.Details;
            // 
            // lblCamHardware
            // 
            this.lblCamHardware.AutoSize = true;
            this.lblCamHardware.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblCamHardware.Location = new System.Drawing.Point(3, 5);
            this.lblCamHardware.Name = "lblCamHardware";
            this.lblCamHardware.Size = new System.Drawing.Size(103, 15);
            this.lblCamHardware.TabIndex = 0;
            this.lblCamHardware.Text = "【相機設定】";
            // 
            // panelStatusBar
            // 
            this.panelStatusBar.ColumnCount = 6;
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 14F));
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 14F));
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 14F));
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 14F));
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 14F));
            this.panelStatusBar.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 30F));
            this.panelStatusBar.Controls.Add(this.lblCamCount, 0, 0);
            this.panelStatusBar.Controls.Add(this.lblIoState, 1, 0);
            this.panelStatusBar.Controls.Add(this.lblIoConn, 2, 0);
            this.panelStatusBar.Controls.Add(this.lblLightConn, 3, 0);
            this.panelStatusBar.Controls.Add(this.lblStorageConn, 4, 0);
            this.panelStatusBar.Controls.Add(this.panelIo, 5, 0);
            this.panelStatusBar.Dock = System.Windows.Forms.DockStyle.Top;
            this.panelStatusBar.Location = new System.Drawing.Point(0, 0);
            this.panelStatusBar.Name = "panelStatusBar";
            this.panelStatusBar.RowCount = 1;
            this.panelStatusBar.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.panelStatusBar.Size = new System.Drawing.Size(1262, 32);
            this.panelStatusBar.TabIndex = 17;
            // 
            // lblCamCount
            // 
            this.lblCamCount.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.lblCamCount.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblCamCount.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblCamCount.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.lblCamCount.ForeColor = System.Drawing.Color.White;
            this.lblCamCount.Location = new System.Drawing.Point(3, 0);
            this.lblCamCount.Name = "lblCamCount";
            this.lblCamCount.Size = new System.Drawing.Size(170, 32);
            this.lblCamCount.TabIndex = 0;
            this.lblCamCount.Text = "相機: --/7";
            this.lblCamCount.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblIoState
            // 
            this.lblIoState.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.lblIoState.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoState.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoState.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.lblIoState.ForeColor = System.Drawing.Color.White;
            this.lblIoState.Location = new System.Drawing.Point(179, 0);
            this.lblIoState.Name = "lblIoState";
            this.lblIoState.Size = new System.Drawing.Size(170, 32);
            this.lblIoState.TabIndex = 2;
            this.lblIoState.Text = "● 狀態: --";
            this.lblIoState.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblIoConn
            // 
            this.lblIoConn.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.lblIoConn.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoConn.Cursor = System.Windows.Forms.Cursors.Hand;
            this.lblIoConn.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoConn.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.lblIoConn.ForeColor = System.Drawing.Color.White;
            this.lblIoConn.Location = new System.Drawing.Point(355, 0);
            this.lblIoConn.Name = "lblIoConn";
            this.lblIoConn.Size = new System.Drawing.Size(170, 32);
            this.lblIoConn.TabIndex = 3;
            this.lblIoConn.Text = "● IO: --";
            this.lblIoConn.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.lblIoConn.Click += new System.EventHandler(this.lblIoConn_Click);
            // 
            // lblLightConn
            // 
            this.lblLightConn.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.lblLightConn.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblLightConn.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblLightConn.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.lblLightConn.ForeColor = System.Drawing.Color.White;
            this.lblLightConn.Location = new System.Drawing.Point(531, 0);
            this.lblLightConn.Name = "lblLightConn";
            this.lblLightConn.Size = new System.Drawing.Size(170, 32);
            this.lblLightConn.TabIndex = 4;
            this.lblLightConn.Text = "● 光源: --";
            this.lblLightConn.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblStorageConn
            // 
            this.lblStorageConn.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(117)))), ((int)(((byte)(117)))), ((int)(((byte)(117)))));
            this.lblStorageConn.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblStorageConn.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblStorageConn.Font = new System.Drawing.Font("Microsoft Sans Serif", 9F);
            this.lblStorageConn.ForeColor = System.Drawing.Color.White;
            this.lblStorageConn.Location = new System.Drawing.Point(707, 0);
            this.lblStorageConn.Name = "lblStorageConn";
            this.lblStorageConn.Size = new System.Drawing.Size(170, 32);
            this.lblStorageConn.TabIndex = 5;
            this.lblStorageConn.Text = "● 儲存電腦: --";
            this.lblStorageConn.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // panelIo
            // 
            this.panelIo.ColumnCount = 5;
            this.panelIo.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.panelIo.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.panelIo.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.panelIo.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.panelIo.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 20F));
            this.panelIo.Controls.Add(this.lblIoDiAlive, 0, 0);
            this.panelIo.Controls.Add(this.lblIoDiStart, 1, 0);
            this.panelIo.Controls.Add(this.lblIoDoPcAlive, 2, 0);
            this.panelIo.Controls.Add(this.lblIoDoMura, 3, 0);
            this.panelIo.Controls.Add(this.lblIoDoPcBusy, 4, 0);
            this.panelIo.Dock = System.Windows.Forms.DockStyle.Fill;
            this.panelIo.Location = new System.Drawing.Point(880, 0);
            this.panelIo.Margin = new System.Windows.Forms.Padding(0);
            this.panelIo.Name = "panelIo";
            this.panelIo.RowCount = 1;
            this.panelIo.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.panelIo.Size = new System.Drawing.Size(382, 32);
            this.panelIo.TabIndex = 1;
            // 
            // lblIoDiAlive
            // 
            this.lblIoDiAlive.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(60)))), ((int)(((byte)(60)))), ((int)(((byte)(60)))));
            this.lblIoDiAlive.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoDiAlive.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoDiAlive.Font = new System.Drawing.Font("Microsoft Sans Serif", 7F);
            this.lblIoDiAlive.ForeColor = System.Drawing.Color.White;
            this.lblIoDiAlive.Location = new System.Drawing.Point(3, 0);
            this.lblIoDiAlive.Name = "lblIoDiAlive";
            this.lblIoDiAlive.Size = new System.Drawing.Size(70, 32);
            this.lblIoDiAlive.TabIndex = 0;
            this.lblIoDiAlive.Text = "DI0\r\nNKN_ALV";
            this.lblIoDiAlive.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblIoDiStart
            // 
            this.lblIoDiStart.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(60)))), ((int)(((byte)(60)))), ((int)(((byte)(60)))));
            this.lblIoDiStart.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoDiStart.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoDiStart.Font = new System.Drawing.Font("Microsoft Sans Serif", 7F);
            this.lblIoDiStart.ForeColor = System.Drawing.Color.White;
            this.lblIoDiStart.Location = new System.Drawing.Point(79, 0);
            this.lblIoDiStart.Name = "lblIoDiStart";
            this.lblIoDiStart.Size = new System.Drawing.Size(70, 32);
            this.lblIoDiStart.TabIndex = 1;
            this.lblIoDiStart.Text = "DI1\r\nINSPECT";
            this.lblIoDiStart.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblIoDoPcAlive
            // 
            this.lblIoDoPcAlive.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(60)))), ((int)(((byte)(60)))), ((int)(((byte)(60)))));
            this.lblIoDoPcAlive.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoDoPcAlive.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoDoPcAlive.Font = new System.Drawing.Font("Microsoft Sans Serif", 7F);
            this.lblIoDoPcAlive.ForeColor = System.Drawing.Color.White;
            this.lblIoDoPcAlive.Location = new System.Drawing.Point(155, 0);
            this.lblIoDoPcAlive.Name = "lblIoDoPcAlive";
            this.lblIoDoPcAlive.Size = new System.Drawing.Size(70, 32);
            this.lblIoDoPcAlive.TabIndex = 2;
            this.lblIoDoPcAlive.Text = "DO0\r\nPC_ALV";
            this.lblIoDoPcAlive.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // lblIoDoMura
            // 
            this.lblIoDoMura.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(60)))), ((int)(((byte)(60)))), ((int)(((byte)(60)))));
            this.lblIoDoMura.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoDoMura.Cursor = System.Windows.Forms.Cursors.Hand;
            this.lblIoDoMura.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoDoMura.Font = new System.Drawing.Font("Microsoft Sans Serif", 7F);
            this.lblIoDoMura.ForeColor = System.Drawing.Color.White;
            this.lblIoDoMura.Location = new System.Drawing.Point(231, 0);
            this.lblIoDoMura.Name = "lblIoDoMura";
            this.lblIoDoMura.Size = new System.Drawing.Size(70, 32);
            this.lblIoDoMura.TabIndex = 3;
            this.lblIoDoMura.Text = "DO1\r\nMURA_DET";
            this.lblIoDoMura.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            this.lblIoDoMura.Click += new System.EventHandler(this.lblIoDoMura_Click);
            // 
            // lblIoDoPcBusy
            // 
            this.lblIoDoPcBusy.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(60)))), ((int)(((byte)(60)))), ((int)(((byte)(60)))));
            this.lblIoDoPcBusy.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.lblIoDoPcBusy.Dock = System.Windows.Forms.DockStyle.Fill;
            this.lblIoDoPcBusy.Font = new System.Drawing.Font("Microsoft Sans Serif", 7F);
            this.lblIoDoPcBusy.ForeColor = System.Drawing.Color.White;
            this.lblIoDoPcBusy.Location = new System.Drawing.Point(307, 0);
            this.lblIoDoPcBusy.Name = "lblIoDoPcBusy";
            this.lblIoDoPcBusy.Size = new System.Drawing.Size(72, 32);
            this.lblIoDoPcBusy.TabIndex = 4;
            this.lblIoDoPcBusy.Text = "DO2\r\nINSPECT";
            this.lblIoDoPcBusy.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
            // 
            // AniloxRollForm
            // 
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;
            this.ClientSize = new System.Drawing.Size(1262, 721);
            this.Controls.Add(this.panelStatusBar);
            this.Controls.Add(this.tabControlRight);
            this.Controls.Add(this.statusBarMain);
            this.Controls.Add(this.tabMain);
            this.Name = "AniloxRollForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "AniloxRoll Monitor";
            this.tabMain.ResumeLayout(false);
            this.tabPageLiveView.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.muraChartHorizontalLive)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartLiveOverview)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.muraChartVerticalLive)).EndInit();
            this.tabPageReview.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraHorizontal)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartOverview)).EndInit();
            this.grpReviewTimePeriod.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraVertical)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).EndInit();
            this.grpReviewGrabNav.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).EndInit();
            this.tabPageData.ResumeLayout(false);
            this.tabPageData.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMuraProfile)).EndInit();
            this.grpDataSingleSheet.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.chartYearly)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartMonthly)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartDaily)).EndInit();
            this.groupBoxGrabIdRange.ResumeLayout(false);
            this.groupBoxGrabIdRange.PerformLayout();
            this.groupBoxTimeRange.ResumeLayout(false);
            this.groupBoxTimeRange.PerformLayout();
            this.statusBarMain.ResumeLayout(false);
            this.statusBarMain.PerformLayout();
            this.tabControlRight.ResumeLayout(false);
            this.tabPageInspSettings.ResumeLayout(false);
            this.tabPageCamera.ResumeLayout(false);
            this.tabControlCamTabs.ResumeLayout(false);
            this.tabPageExposure.ResumeLayout(false);
            this.panelExpAll.ResumeLayout(false);
            this.panelExpAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpAll)).EndInit();
            this.panelExpCam7.ResumeLayout(false);
            this.panelExpCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).EndInit();
            this.panelExpCam6.ResumeLayout(false);
            this.panelExpCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).EndInit();
            this.panelExpCam5.ResumeLayout(false);
            this.panelExpCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).EndInit();
            this.panelExpCam4.ResumeLayout(false);
            this.panelExpCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).EndInit();
            this.panelExpCam3.ResumeLayout(false);
            this.panelExpCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).EndInit();
            this.panelExpCam2.ResumeLayout(false);
            this.panelExpCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).EndInit();
            this.panelExpCam1.ResumeLayout(false);
            this.panelExpCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).EndInit();
            this.tabPageLineRate.ResumeLayout(false);
            this.panelLrAll.ResumeLayout(false);
            this.panelLrAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrAll)).EndInit();
            this.panelLrCam7.ResumeLayout(false);
            this.panelLrCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).EndInit();
            this.panelLrCam6.ResumeLayout(false);
            this.panelLrCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).EndInit();
            this.panelLrCam5.ResumeLayout(false);
            this.panelLrCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).EndInit();
            this.panelLrCam4.ResumeLayout(false);
            this.panelLrCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).EndInit();
            this.panelLrCam3.ResumeLayout(false);
            this.panelLrCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).EndInit();
            this.panelLrCam2.ResumeLayout(false);
            this.panelLrCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).EndInit();
            this.panelLrCam1.ResumeLayout(false);
            this.panelLrCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).EndInit();
            this.tabPageGrabHeight.ResumeLayout(false);
            this.panelHtAll.ResumeLayout(false);
            this.panelHtAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtAll)).EndInit();
            this.panelHtCam7.ResumeLayout(false);
            this.panelHtCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).EndInit();
            this.panelHtCam6.ResumeLayout(false);
            this.panelHtCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).EndInit();
            this.panelHtCam5.ResumeLayout(false);
            this.panelHtCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).EndInit();
            this.panelHtCam4.ResumeLayout(false);
            this.panelHtCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).EndInit();
            this.panelHtCam3.ResumeLayout(false);
            this.panelHtCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).EndInit();
            this.panelHtCam2.ResumeLayout(false);
            this.panelHtCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).EndInit();
            this.panelHtCam1.ResumeLayout(false);
            this.panelHtCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).EndInit();
            this.tabPageSystem.ResumeLayout(false);
            this.tabPageSystem.PerformLayout();
            this.panelStatusBar.ResumeLayout(false);
            this.panelIo.ResumeLayout(false);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private AOI.SDK.UI.SmartCanvas canvasMain;
        private System.Windows.Forms.TabControl tabMain;
        private System.Windows.Forms.TabPage tabPageLiveView;
        private System.Windows.Forms.TabPage tabPageReview;
        private System.Windows.Forms.PictureBox pbCam7;
        private System.Windows.Forms.PictureBox pbCam6;
        private System.Windows.Forms.PictureBox pbCam5;
        private System.Windows.Forms.PictureBox pbCam4;
        private System.Windows.Forms.PictureBox pbCam3;
        private System.Windows.Forms.PictureBox pbCam2;
        private System.Windows.Forms.PictureBox pbCam1;
        private System.Windows.Forms.ComboBox cbDate;
        private System.Windows.Forms.ComboBox cbTime;
        private System.Windows.Forms.Button btnSelectFolder;
        private System.Windows.Forms.StatusStrip statusBarMain;
        private System.Windows.Forms.ToolStripStatusLabel lblPixelInfo;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartMuraVertical;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartYearly;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartMonthly;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartDaily;
        private System.Windows.Forms.Button btnShowFail;
        private System.Windows.Forms.PropertyGrid propertyGridSettings;
        private System.Windows.Forms.RichTextBox helpRichText;
        private System.Windows.Forms.Panel panelLiveCam1;
        private System.Windows.Forms.Panel panelMainDisplay;
        private System.Windows.Forms.Panel panelLiveCam7;
        private System.Windows.Forms.Panel panelLiveCam6;
        private System.Windows.Forms.Panel panelLiveCam5;
        private System.Windows.Forms.Panel panelLiveCam4;
        private System.Windows.Forms.Panel panelLiveCam3;
        private System.Windows.Forms.Panel panelLiveCam2;
        private System.Windows.Forms.Button btnCameraGrab;
        private System.Windows.Forms.Button btnPeriodPrev;
        private System.Windows.Forms.Button btnPeriodNext;
        private System.Windows.Forms.TabControl tabControlRight;
        private System.Windows.Forms.TabPage tabPageInspSettings;
        private System.Windows.Forms.TabPage tabPageCamera;
        private System.Windows.Forms.TabPage tabPageSystem;
        private System.Windows.Forms.Panel panelExpCam1;
        private System.Windows.Forms.Label lblExposure;
        private System.Windows.Forms.NumericUpDown numExpCam1;
        private System.Windows.Forms.TrackBar trackBarExpCam1;
        private System.Windows.Forms.Label lblCamHardware;
        private System.Windows.Forms.ListView listViewCameras;
        private System.Windows.Forms.Label lblEngineConst;
        private System.Windows.Forms.ListView listViewEngine;
        private System.Windows.Forms.TabControl tabControlCamTabs;
        private System.Windows.Forms.TabPage tabPageExposure;
        private System.Windows.Forms.TabPage tabPageLineRate;
        private System.Windows.Forms.TabPage tabPageGrabHeight;
        private System.Windows.Forms.Panel panelExpAll;
        private System.Windows.Forms.Label lblExpAll;
        private System.Windows.Forms.TrackBar trackBarExpAll;
        private System.Windows.Forms.NumericUpDown numExpAll;
        private System.Windows.Forms.Label lblExpAllUnit;
        private System.Windows.Forms.Panel panelLrAll;
        private System.Windows.Forms.Label lblLrAll;
        private System.Windows.Forms.TrackBar trackBarLrAll;
        private System.Windows.Forms.NumericUpDown numLrAll;
        private System.Windows.Forms.Label lblLrAllUnit;
        private System.Windows.Forms.Panel panelHtAll;
        private System.Windows.Forms.Label lblHtAll;
        private System.Windows.Forms.TrackBar trackBarHtAll;
        private System.Windows.Forms.NumericUpDown numHtAll;
        private System.Windows.Forms.Label lblHtAllUnit;
        private System.Windows.Forms.Panel panelExpCam7;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TrackBar trackBarExpCam7;
        private System.Windows.Forms.NumericUpDown numExpCam7;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Panel panelExpCam6;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TrackBar trackBarExpCam6;
        private System.Windows.Forms.NumericUpDown numExpCam6;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Panel panelExpCam5;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TrackBar trackBarExpCam5;
        private System.Windows.Forms.NumericUpDown numExpCam5;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Panel panelExpCam4;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TrackBar trackBarExpCam4;
        private System.Windows.Forms.NumericUpDown numExpCam4;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Panel panelExpCam3;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TrackBar trackBarExpCam3;
        private System.Windows.Forms.NumericUpDown numExpCam3;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Panel panelExpCam2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TrackBar trackBarExpCam2;
        private System.Windows.Forms.NumericUpDown numExpCam2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Panel panelLrCam7;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.TrackBar trackBarLrCam7;
        private System.Windows.Forms.NumericUpDown numLrCam7;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Panel panelLrCam6;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.TrackBar trackBarLrCam6;
        private System.Windows.Forms.NumericUpDown numLrCam6;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.Panel panelLrCam5;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TrackBar trackBarLrCam5;
        private System.Windows.Forms.NumericUpDown numLrCam5;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.Panel panelLrCam4;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.TrackBar trackBarLrCam4;
        private System.Windows.Forms.NumericUpDown numLrCam4;
        private System.Windows.Forms.Label label22;
        private System.Windows.Forms.Panel panelLrCam3;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.TrackBar trackBarLrCam3;
        private System.Windows.Forms.NumericUpDown numLrCam3;
        private System.Windows.Forms.Label label24;
        private System.Windows.Forms.Panel panelLrCam2;
        private System.Windows.Forms.Label label25;
        private System.Windows.Forms.TrackBar trackBarLrCam2;
        private System.Windows.Forms.NumericUpDown numLrCam2;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.Panel panelLrCam1;
        private System.Windows.Forms.Label label27;
        private System.Windows.Forms.TrackBar trackBarLrCam1;
        private System.Windows.Forms.NumericUpDown numLrCam1;
        private System.Windows.Forms.Label lblGrabHeight;
        private System.Windows.Forms.Panel panelHtCam7;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar trackBarHtCam7;
        private System.Windows.Forms.NumericUpDown numHtCam7;
        private System.Windows.Forms.Label label28;
        private System.Windows.Forms.Panel panelHtCam6;
        private System.Windows.Forms.Label label29;
        private System.Windows.Forms.TrackBar trackBarHtCam6;
        private System.Windows.Forms.NumericUpDown numHtCam6;
        private System.Windows.Forms.Label label30;
        private System.Windows.Forms.Panel panelHtCam5;
        private System.Windows.Forms.Label label31;
        private System.Windows.Forms.TrackBar trackBarHtCam5;
        private System.Windows.Forms.NumericUpDown numHtCam5;
        private System.Windows.Forms.Label label32;
        private System.Windows.Forms.Panel panelHtCam4;
        private System.Windows.Forms.Label label33;
        private System.Windows.Forms.TrackBar trackBarHtCam4;
        private System.Windows.Forms.NumericUpDown numHtCam4;
        private System.Windows.Forms.Label label34;
        private System.Windows.Forms.Panel panelHtCam3;
        private System.Windows.Forms.Label label35;
        private System.Windows.Forms.TrackBar trackBarHtCam3;
        private System.Windows.Forms.NumericUpDown numHtCam3;
        private System.Windows.Forms.Label label36;
        private System.Windows.Forms.Panel panelHtCam2;
        private System.Windows.Forms.Label label37;
        private System.Windows.Forms.TrackBar trackBarHtCam2;
        private System.Windows.Forms.NumericUpDown numHtCam2;
        private System.Windows.Forms.Label label38;
        private System.Windows.Forms.Panel panelHtCam1;
        private System.Windows.Forms.Label label39;
        private System.Windows.Forms.TrackBar trackBarHtCam1;
        private System.Windows.Forms.NumericUpDown numHtCam1;
        private System.Windows.Forms.Label label40;
        private System.Windows.Forms.TabPage tabPageData;
        private System.Windows.Forms.Panel panelStatCam7;
        private System.Windows.Forms.Panel panelStatCam6;
        private System.Windows.Forms.Panel panelStatCam5;
        private System.Windows.Forms.Panel panelStatCam4;
        private System.Windows.Forms.Panel panelStatCam3;
        private System.Windows.Forms.Panel panelStatCam2;
        private System.Windows.Forms.Panel panelStatCam1;
        private System.Windows.Forms.ComboBox cbStartDate;
        private System.Windows.Forms.ComboBox cbStartTime;
        private System.Windows.Forms.ComboBox cbEndDate;
        private System.Windows.Forms.ComboBox cbEndTime;
        private System.Windows.Forms.Button btnSelectDataFolder;
        private System.Windows.Forms.GroupBox groupBoxGrabIdRange;
        private System.Windows.Forms.ComboBox cbGrabIdStart;
        private System.Windows.Forms.ComboBox cbGrabIdEnd;
        private System.Windows.Forms.Label lblGrabIdEndLabel;
        private System.Windows.Forms.Label lblGrabIdStartLabel;
        private System.Windows.Forms.GroupBox groupBoxTimeRange;
        private System.Windows.Forms.Label lblStartTimeHeader;
        private System.Windows.Forms.Label lblEndTimeHeader;
        private System.Windows.Forms.ListView listViewGrabDetail;
        private System.Windows.Forms.TableLayoutPanel panelStatusBar;
        private System.Windows.Forms.Label lblIoConn;
        private System.Windows.Forms.Label lblIoState;
        private System.Windows.Forms.Label lblLightConn;
        private System.Windows.Forms.Label lblStorageConn;
        private System.Windows.Forms.TableLayoutPanel panelIo;
        private System.Windows.Forms.Label lblIoDiAlive;
        private System.Windows.Forms.Label lblIoDiStart;
        private System.Windows.Forms.Label lblIoDoPcAlive;
        private System.Windows.Forms.Label lblIoDoMura;
        private System.Windows.Forms.Label lblIoDoPcBusy;
        private System.Windows.Forms.Label lblCamCount;
        private System.Windows.Forms.ComboBox cbChartYear;
        private System.Windows.Forms.ComboBox cbChartMonth;
        private System.Windows.Forms.ComboBox cbChartDay;
        private System.Windows.Forms.Label lblChartNavMonth;
        private System.Windows.Forms.Label lblChartNavDay;
        private System.Windows.Forms.Label lblChartNavYear;
        private System.Windows.Forms.GroupBox grpDataSingleSheet;
        private System.Windows.Forms.Button btnGrabIdDataPrev;
        private System.Windows.Forms.Button btnGrabIdDataNext;
        private System.Windows.Forms.ComboBox cbDataGrabId;
        private System.Windows.Forms.GroupBox grpReviewGrabNav;
        private System.Windows.Forms.ComboBox cbReviewGrabId;
        private System.Windows.Forms.Button btnGrabIdPrev;
        private System.Windows.Forms.Button btnGrabIdNext;
        private System.Windows.Forms.Label lblChartMonthlyUnit;
        private System.Windows.Forms.Label lblChartDailyUnit;
        private System.Windows.Forms.Label lblChartYearlyUnit;
        private System.Windows.Forms.GroupBox grpReviewTimePeriod;
        private System.ComponentModel.BackgroundWorker backgroundWorker1;
        private System.Windows.Forms.DataVisualization.Charting.Chart muraChartVerticalLive;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartOverview;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartLiveOverview;
        private System.Windows.Forms.DataVisualization.Charting.Chart muraChartHorizontalLive;
        private System.Windows.Forms.Button btnGetBackground;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartMuraHorizontal;
        private System.Windows.Forms.ListView listViewHardware;
        private System.Windows.Forms.Label label41;
        private System.Windows.Forms.Button btnViewBackground;
        private System.Windows.Forms.Label lblBgBinInfo;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartMuraProfile;
    }
}