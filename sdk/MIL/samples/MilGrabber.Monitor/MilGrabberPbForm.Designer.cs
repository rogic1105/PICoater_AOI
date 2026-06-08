namespace MilGrabber.Monitor
{
    partial class MilGrabberPbForm
    {
        private System.ComponentModel.IContainer components = null;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        private void InitializeComponent()
        {
            this.btnInit = new System.Windows.Forms.Button();
            this.btnGrab = new System.Windows.Forms.Button();
            this.btnRelease = new System.Windows.Forms.Button();
            this.btnFetchInfo = new System.Windows.Forms.Button();
            this.chkFlipVertical = new System.Windows.Forms.CheckBox();
            this.lblResize = new System.Windows.Forms.Label();
            this.numResize = new System.Windows.Forms.NumericUpDown();
            this.lblFov = new System.Windows.Forms.Label();
            this.numFovMm = new System.Windows.Forms.NumericUpDown();
            this.chkMerge = new System.Windows.Forms.CheckBox();
            this.chkLod = new System.Windows.Forms.CheckBox();
            this._rbModePb = new System.Windows.Forms.RadioButton();
            this._rbModeMil = new System.Windows.Forms.RadioButton();
            this._lblTiming = new System.Windows.Forms.Label();
            this.panelMain = new System.Windows.Forms.Panel();
            this.lvCameras = new System.Windows.Forms.ListView();
            this.colCamCamera = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamFps = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamTargetFps = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamLineRate = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamLineRateMax = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamExpSet = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamExpMeas = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamFrames = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamMissed = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamGrabMiss = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamResolution = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamScanMode = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamFpga = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamTemp = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamMemFree = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamPcieLanes = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colCamPcieSpeed = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.lvEngine = new System.Windows.Forms.ListView();
            this.colEngParam = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.colEngValue = ((System.Windows.Forms.ColumnHeader)(new System.Windows.Forms.ColumnHeader()));
            this.tabParams = new System.Windows.Forms.TabControl();
            this.tabExposure = new System.Windows.Forms.TabPage();
            this.panelExpAll = new System.Windows.Forms.Panel();
            this.lblExpAll = new System.Windows.Forms.Label();
            this.trackBarExpAll = new System.Windows.Forms.TrackBar();
            this.numExpAll = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam1 = new System.Windows.Forms.Panel();
            this.lblExpCam1 = new System.Windows.Forms.Label();
            this.trackBarExpCam1 = new System.Windows.Forms.TrackBar();
            this.numExpCam1 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam2 = new System.Windows.Forms.Panel();
            this.lblExpCam2 = new System.Windows.Forms.Label();
            this.trackBarExpCam2 = new System.Windows.Forms.TrackBar();
            this.numExpCam2 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam3 = new System.Windows.Forms.Panel();
            this.lblExpCam3 = new System.Windows.Forms.Label();
            this.trackBarExpCam3 = new System.Windows.Forms.TrackBar();
            this.numExpCam3 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam4 = new System.Windows.Forms.Panel();
            this.lblExpCam4 = new System.Windows.Forms.Label();
            this.trackBarExpCam4 = new System.Windows.Forms.TrackBar();
            this.numExpCam4 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam5 = new System.Windows.Forms.Panel();
            this.lblExpCam5 = new System.Windows.Forms.Label();
            this.trackBarExpCam5 = new System.Windows.Forms.TrackBar();
            this.numExpCam5 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam6 = new System.Windows.Forms.Panel();
            this.lblExpCam6 = new System.Windows.Forms.Label();
            this.trackBarExpCam6 = new System.Windows.Forms.TrackBar();
            this.numExpCam6 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam7 = new System.Windows.Forms.Panel();
            this.lblExpCam7 = new System.Windows.Forms.Label();
            this.trackBarExpCam7 = new System.Windows.Forms.TrackBar();
            this.numExpCam7 = new System.Windows.Forms.NumericUpDown();
            this.panelExpCam8 = new System.Windows.Forms.Panel();
            this.lblExpCam8 = new System.Windows.Forms.Label();
            this.trackBarExpCam8 = new System.Windows.Forms.TrackBar();
            this.numExpCam8 = new System.Windows.Forms.NumericUpDown();
            this.tabLineRate = new System.Windows.Forms.TabPage();
            this.panelLrAll = new System.Windows.Forms.Panel();
            this.lblLrAll = new System.Windows.Forms.Label();
            this.trackBarLrAll = new System.Windows.Forms.TrackBar();
            this.numLrAll = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam1 = new System.Windows.Forms.Panel();
            this.lblLrCam1 = new System.Windows.Forms.Label();
            this.trackBarLrCam1 = new System.Windows.Forms.TrackBar();
            this.numLrCam1 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam2 = new System.Windows.Forms.Panel();
            this.lblLrCam2 = new System.Windows.Forms.Label();
            this.trackBarLrCam2 = new System.Windows.Forms.TrackBar();
            this.numLrCam2 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam3 = new System.Windows.Forms.Panel();
            this.lblLrCam3 = new System.Windows.Forms.Label();
            this.trackBarLrCam3 = new System.Windows.Forms.TrackBar();
            this.numLrCam3 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam4 = new System.Windows.Forms.Panel();
            this.lblLrCam4 = new System.Windows.Forms.Label();
            this.trackBarLrCam4 = new System.Windows.Forms.TrackBar();
            this.numLrCam4 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam5 = new System.Windows.Forms.Panel();
            this.lblLrCam5 = new System.Windows.Forms.Label();
            this.trackBarLrCam5 = new System.Windows.Forms.TrackBar();
            this.numLrCam5 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam6 = new System.Windows.Forms.Panel();
            this.lblLrCam6 = new System.Windows.Forms.Label();
            this.trackBarLrCam6 = new System.Windows.Forms.TrackBar();
            this.numLrCam6 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam7 = new System.Windows.Forms.Panel();
            this.lblLrCam7 = new System.Windows.Forms.Label();
            this.trackBarLrCam7 = new System.Windows.Forms.TrackBar();
            this.numLrCam7 = new System.Windows.Forms.NumericUpDown();
            this.panelLrCam8 = new System.Windows.Forms.Panel();
            this.lblLrCam8 = new System.Windows.Forms.Label();
            this.trackBarLrCam8 = new System.Windows.Forms.TrackBar();
            this.numLrCam8 = new System.Windows.Forms.NumericUpDown();
            this.tabHeight = new System.Windows.Forms.TabPage();
            this.panelHtAll = new System.Windows.Forms.Panel();
            this.lblHtAll = new System.Windows.Forms.Label();
            this.trackBarHtAll = new System.Windows.Forms.TrackBar();
            this.numHtAll = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam1 = new System.Windows.Forms.Panel();
            this.lblHtCam1 = new System.Windows.Forms.Label();
            this.trackBarHtCam1 = new System.Windows.Forms.TrackBar();
            this.numHtCam1 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam2 = new System.Windows.Forms.Panel();
            this.lblHtCam2 = new System.Windows.Forms.Label();
            this.trackBarHtCam2 = new System.Windows.Forms.TrackBar();
            this.numHtCam2 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam3 = new System.Windows.Forms.Panel();
            this.lblHtCam3 = new System.Windows.Forms.Label();
            this.trackBarHtCam3 = new System.Windows.Forms.TrackBar();
            this.numHtCam3 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam4 = new System.Windows.Forms.Panel();
            this.lblHtCam4 = new System.Windows.Forms.Label();
            this.trackBarHtCam4 = new System.Windows.Forms.TrackBar();
            this.numHtCam4 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam5 = new System.Windows.Forms.Panel();
            this.lblHtCam5 = new System.Windows.Forms.Label();
            this.trackBarHtCam5 = new System.Windows.Forms.TrackBar();
            this.numHtCam5 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam6 = new System.Windows.Forms.Panel();
            this.lblHtCam6 = new System.Windows.Forms.Label();
            this.trackBarHtCam6 = new System.Windows.Forms.TrackBar();
            this.numHtCam6 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam7 = new System.Windows.Forms.Panel();
            this.lblHtCam7 = new System.Windows.Forms.Label();
            this.trackBarHtCam7 = new System.Windows.Forms.TrackBar();
            this.numHtCam7 = new System.Windows.Forms.NumericUpDown();
            this.panelHtCam8 = new System.Windows.Forms.Panel();
            this.lblHtCam8 = new System.Windows.Forms.Label();
            this.trackBarHtCam8 = new System.Windows.Forms.TrackBar();
            this.numHtCam8 = new System.Windows.Forms.NumericUpDown();
            this.panelCam0 = new System.Windows.Forms.Panel();
            this.panelCam1 = new System.Windows.Forms.Panel();
            this.panelCam2 = new System.Windows.Forms.Panel();
            this.panelCam3 = new System.Windows.Forms.Panel();
            this.panelCam4 = new System.Windows.Forms.Panel();
            this.panelCam5 = new System.Windows.Forms.Panel();
            this.panelCam6 = new System.Windows.Forms.Panel();
            this.panelCam7 = new System.Windows.Forms.Panel();
            ((System.ComponentModel.ISupportInitialize)(this.numResize)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFovMm)).BeginInit();
            this.tabParams.SuspendLayout();
            this.tabExposure.SuspendLayout();
            this.panelExpAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpAll)).BeginInit();
            this.panelExpCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).BeginInit();
            this.panelExpCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).BeginInit();
            this.panelExpCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).BeginInit();
            this.panelExpCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).BeginInit();
            this.panelExpCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).BeginInit();
            this.panelExpCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).BeginInit();
            this.panelExpCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).BeginInit();
            this.panelExpCam8.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam8)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam8)).BeginInit();
            this.tabLineRate.SuspendLayout();
            this.panelLrAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrAll)).BeginInit();
            this.panelLrCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).BeginInit();
            this.panelLrCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).BeginInit();
            this.panelLrCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).BeginInit();
            this.panelLrCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).BeginInit();
            this.panelLrCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).BeginInit();
            this.panelLrCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).BeginInit();
            this.panelLrCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).BeginInit();
            this.panelLrCam8.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam8)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam8)).BeginInit();
            this.tabHeight.SuspendLayout();
            this.panelHtAll.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtAll)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtAll)).BeginInit();
            this.panelHtCam1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).BeginInit();
            this.panelHtCam2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).BeginInit();
            this.panelHtCam3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).BeginInit();
            this.panelHtCam4.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).BeginInit();
            this.panelHtCam5.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).BeginInit();
            this.panelHtCam6.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).BeginInit();
            this.panelHtCam7.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).BeginInit();
            this.panelHtCam8.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam8)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam8)).BeginInit();
            this.SuspendLayout();
            // 
            // btnInit
            // 
            this.btnInit.Location = new System.Drawing.Point(963, 17);
            this.btnInit.Name = "btnInit";
            this.btnInit.Size = new System.Drawing.Size(110, 32);
            this.btnInit.TabIndex = 0;
            this.btnInit.Text = "初始化";
            this.btnInit.UseVisualStyleBackColor = true;
            this.btnInit.Click += new System.EventHandler(this.btnInit_Click);
            // 
            // btnGrab
            // 
            this.btnGrab.Location = new System.Drawing.Point(965, 93);
            this.btnGrab.Name = "btnGrab";
            this.btnGrab.Size = new System.Drawing.Size(110, 32);
            this.btnGrab.TabIndex = 1;
            this.btnGrab.Text = "開始抓取";
            this.btnGrab.UseVisualStyleBackColor = true;
            this.btnGrab.Click += new System.EventHandler(this.btnGrab_Click);
            // 
            // btnRelease
            // 
            this.btnRelease.Location = new System.Drawing.Point(965, 131);
            this.btnRelease.Name = "btnRelease";
            this.btnRelease.Size = new System.Drawing.Size(110, 32);
            this.btnRelease.TabIndex = 2;
            this.btnRelease.Text = "釋放";
            this.btnRelease.UseVisualStyleBackColor = true;
            this.btnRelease.Click += new System.EventHandler(this.btnRelease_Click);
            // 
            // btnFetchInfo
            // 
            this.btnFetchInfo.Location = new System.Drawing.Point(963, 55);
            this.btnFetchInfo.Name = "btnFetchInfo";
            this.btnFetchInfo.Size = new System.Drawing.Size(110, 32);
            this.btnFetchInfo.TabIndex = 3;
            this.btnFetchInfo.Text = "抓取相機資訊";
            this.btnFetchInfo.UseVisualStyleBackColor = true;
            this.btnFetchInfo.Click += new System.EventHandler(this.btnFetchInfo_Click);
            // 
            // chkFlipVertical
            // 
            this.chkFlipVertical.AutoSize = true;
            this.chkFlipVertical.Location = new System.Drawing.Point(965, 194);
            this.chkFlipVertical.Name = "chkFlipVertical";
            this.chkFlipVertical.Size = new System.Drawing.Size(89, 19);
            this.chkFlipVertical.TabIndex = 4;
            this.chkFlipVertical.Text = "上下翻轉";
            this.chkFlipVertical.UseVisualStyleBackColor = true;
            this.chkFlipVertical.CheckedChanged += new System.EventHandler(this.chkFlipVertical_CheckedChanged);
            // 
            // lblResize
            // 
            this.lblResize.AutoSize = true;
            this.lblResize.Location = new System.Drawing.Point(164, 272);
            this.lblResize.Name = "lblResize";
            this.lblResize.Size = new System.Drawing.Size(67, 15);
            this.lblResize.TabIndex = 32;
            this.lblResize.Text = "縮圖倍率";
            // 
            // numResize
            // 
            this.numResize.Location = new System.Drawing.Point(234, 270);
            this.numResize.Maximum = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.numResize.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numResize.Name = "numResize";
            this.numResize.Size = new System.Drawing.Size(48, 25);
            this.numResize.TabIndex = 5;
            this.numResize.Value = new decimal(new int[] {
            4,
            0,
            0,
            0});
            this.numResize.ValueChanged += new System.EventHandler(this.numResize_ValueChanged);
            // 
            // lblFov
            // 
            this.lblFov.AutoSize = true;
            this.lblFov.Location = new System.Drawing.Point(14, 272);
            this.lblFov.Name = "lblFov";
            this.lblFov.Size = new System.Drawing.Size(67, 15);
            this.lblFov.TabIndex = 43;
            this.lblFov.Text = "FOV(mm)";
            // 
            // numFovMm
            // 
            this.numFovMm.DecimalPlaces = 1;
            this.numFovMm.Location = new System.Drawing.Point(84, 270);
            this.numFovMm.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numFovMm.Name = "numFovMm";
            this.numFovMm.Size = new System.Drawing.Size(64, 25);
            this.numFovMm.TabIndex = 8;
            this.numFovMm.Value = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numFovMm.ValueChanged += new System.EventHandler(this.numFovMm_ValueChanged);
            // 
            // chkMerge
            // 
            this.chkMerge.AutoSize = true;
            this.chkMerge.Location = new System.Drawing.Point(965, 169);
            this.chkMerge.Name = "chkMerge";
            this.chkMerge.Size = new System.Drawing.Size(59, 19);
            this.chkMerge.TabIndex = 6;
            this.chkMerge.Text = "合圖";
            this.chkMerge.UseVisualStyleBackColor = true;
            this.chkMerge.CheckedChanged += new System.EventHandler(this.chkMerge_CheckedChanged);
            // 
            // chkLod
            // 
            this.chkLod.AutoSize = true;
            this.chkLod.Location = new System.Drawing.Point(965, 219);
            this.chkLod.Name = "chkLod";
            this.chkLod.Size = new System.Drawing.Size(88, 19);
            this.chkLod.TabIndex = 7;
            this.chkLod.Text = "動態LOD";
            this.chkLod.UseVisualStyleBackColor = true;
            this.chkLod.CheckedChanged += new System.EventHandler(this.chkLod_CheckedChanged);
            // 
            // _rbModePb
            // 
            this._rbModePb.AutoSize = true;
            this._rbModePb.Checked = true;
            this._rbModePb.Location = new System.Drawing.Point(820, 20);
            this._rbModePb.Name = "_rbModePb";
            this._rbModePb.Size = new System.Drawing.Size(91, 19);
            this._rbModePb.TabIndex = 40;
            this._rbModePb.TabStop = true;
            this._rbModePb.Text = "PictureBox";
            this._rbModePb.UseVisualStyleBackColor = true;
            // 
            // _rbModeMil
            // 
            this._rbModeMil.AutoSize = true;
            this._rbModeMil.Location = new System.Drawing.Point(820, 44);
            this._rbModeMil.Name = "_rbModeMil";
            this._rbModeMil.Size = new System.Drawing.Size(89, 19);
            this._rbModeMil.TabIndex = 41;
            this._rbModeMil.Text = "MIL 直繪";
            this._rbModeMil.UseVisualStyleBackColor = true;
            // 
            // _lblTiming
            // 
            this._lblTiming.ForeColor = System.Drawing.Color.DimGray;
            this._lblTiming.Location = new System.Drawing.Point(820, 74);
            this._lblTiming.Name = "_lblTiming";
            this._lblTiming.Size = new System.Drawing.Size(135, 160);
            this._lblTiming.TabIndex = 42;
            this._lblTiming.Text = "(計時)";
            // 
            // panelMain
            // 
            this.panelMain.BackColor = System.Drawing.Color.Black;
            this.panelMain.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelMain.Location = new System.Drawing.Point(12, 298);
            this.panelMain.Name = "panelMain";
            this.panelMain.Size = new System.Drawing.Size(1075, 282);
            this.panelMain.TabIndex = 3;
            // 
            // lvCameras
            // 
            this.lvCameras.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.colCamCamera,
            this.colCamFps,
            this.colCamTargetFps,
            this.colCamLineRate,
            this.colCamLineRateMax,
            this.colCamExpSet,
            this.colCamExpMeas,
            this.colCamFrames,
            this.colCamMissed,
            this.colCamGrabMiss,
            this.colCamResolution,
            this.colCamScanMode,
            this.colCamFpga,
            this.colCamTemp,
            this.colCamMemFree,
            this.colCamPcieLanes,
            this.colCamPcieSpeed});
            this.lvCameras.FullRowSelect = true;
            this.lvCameras.GridLines = true;
            this.lvCameras.HideSelection = false;
            this.lvCameras.Location = new System.Drawing.Point(6, 712);
            this.lvCameras.Name = "lvCameras";
            this.lvCameras.Size = new System.Drawing.Size(1393, 160);
            this.lvCameras.TabIndex = 30;
            this.lvCameras.UseCompatibleStateImageBehavior = false;
            this.lvCameras.View = System.Windows.Forms.View.Details;
            // 
            // colCamCamera
            // 
            this.colCamCamera.Text = "Camera";
            this.colCamCamera.Width = 70;
            // 
            // colCamFps
            // 
            this.colCamFps.Text = "FPS";
            this.colCamFps.Width = 70;
            // 
            // colCamTargetFps
            // 
            this.colCamTargetFps.Text = "Target FPS";
            this.colCamTargetFps.Width = 80;
            // 
            // colCamLineRate
            // 
            this.colCamLineRate.Text = "Line Rate(Hz)";
            this.colCamLineRate.Width = 95;
            // 
            // colCamLineRateMax
            // 
            this.colCamLineRateMax.Text = "Max Line Rate(Hz)";
            this.colCamLineRateMax.Width = 120;
            // 
            // colCamExpSet
            // 
            this.colCamExpSet.Text = "Exp Set(μs)";
            this.colCamExpSet.Width = 100;
            // 
            // colCamExpMeas
            // 
            this.colCamExpMeas.Text = "Exp Meas(μs)";
            this.colCamExpMeas.Width = 100;
            // 
            // colCamFrames
            // 
            this.colCamFrames.Text = "Frames";
            this.colCamFrames.Width = 80;
            // 
            // colCamMissed
            // 
            this.colCamMissed.Text = "Missed";
            this.colCamMissed.Width = 70;
            // 
            // colCamGrabMiss
            // 
            this.colCamGrabMiss.Text = "Grab Miss";
            this.colCamGrabMiss.Width = 75;
            // 
            // colCamResolution
            // 
            this.colCamResolution.Text = "Resolution";
            this.colCamResolution.Width = 110;
            // 
            // colCamScanMode
            // 
            this.colCamScanMode.Text = "Scan Mode";
            this.colCamScanMode.Width = 90;
            // 
            // colCamFpga
            // 
            this.colCamFpga.Text = "FPGA(°C)";
            this.colCamFpga.Width = 75;
            // 
            // colCamTemp
            // 
            this.colCamTemp.Text = "Cam Temp(°C)";
            this.colCamTemp.Width = 90;
            // 
            // colCamMemFree
            // 
            this.colCamMemFree.Text = "Mem Free(MB)";
            this.colCamMemFree.Width = 95;
            // 
            // colCamPcieLanes
            // 
            this.colCamPcieLanes.Text = "PCIe Lanes";
            this.colCamPcieLanes.Width = 80;
            // 
            // colCamPcieSpeed
            // 
            this.colCamPcieSpeed.Text = "PCIe Speed";
            this.colCamPcieSpeed.Width = 80;
            // 
            // lvEngine
            // 
            this.lvEngine.Columns.AddRange(new System.Windows.Forms.ColumnHeader[] {
            this.colEngParam,
            this.colEngValue});
            this.lvEngine.FullRowSelect = true;
            this.lvEngine.GridLines = true;
            this.lvEngine.HideSelection = false;
            this.lvEngine.Location = new System.Drawing.Point(12, 586);
            this.lvEngine.Name = "lvEngine";
            this.lvEngine.Size = new System.Drawing.Size(1075, 118);
            this.lvEngine.TabIndex = 31;
            this.lvEngine.UseCompatibleStateImageBehavior = false;
            this.lvEngine.View = System.Windows.Forms.View.Details;
            // 
            // colEngParam
            // 
            this.colEngParam.Text = "參數";
            this.colEngParam.Width = 170;
            // 
            // colEngValue
            // 
            this.colEngValue.Text = "值";
            this.colEngValue.Width = 130;
            // 
            // tabParams
            // 
            this.tabParams.Controls.Add(this.tabExposure);
            this.tabParams.Controls.Add(this.tabLineRate);
            this.tabParams.Controls.Add(this.tabHeight);
            this.tabParams.Location = new System.Drawing.Point(1089, 17);
            this.tabParams.Name = "tabParams";
            this.tabParams.SelectedIndex = 0;
            this.tabParams.Size = new System.Drawing.Size(314, 691);
            this.tabParams.TabIndex = 5;
            // 
            // tabExposure
            // 
            this.tabExposure.AutoScroll = true;
            this.tabExposure.Controls.Add(this.panelExpAll);
            this.tabExposure.Controls.Add(this.panelExpCam1);
            this.tabExposure.Controls.Add(this.panelExpCam2);
            this.tabExposure.Controls.Add(this.panelExpCam3);
            this.tabExposure.Controls.Add(this.panelExpCam4);
            this.tabExposure.Controls.Add(this.panelExpCam5);
            this.tabExposure.Controls.Add(this.panelExpCam6);
            this.tabExposure.Controls.Add(this.panelExpCam7);
            this.tabExposure.Controls.Add(this.panelExpCam8);
            this.tabExposure.Location = new System.Drawing.Point(4, 25);
            this.tabExposure.Name = "tabExposure";
            this.tabExposure.Padding = new System.Windows.Forms.Padding(3);
            this.tabExposure.Size = new System.Drawing.Size(306, 662);
            this.tabExposure.TabIndex = 0;
            this.tabExposure.Text = "曝光(μs)";
            this.tabExposure.UseVisualStyleBackColor = true;
            // 
            // panelExpAll
            // 
            this.panelExpAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelExpAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpAll.Controls.Add(this.lblExpAll);
            this.panelExpAll.Controls.Add(this.trackBarExpAll);
            this.panelExpAll.Controls.Add(this.numExpAll);
            this.panelExpAll.Location = new System.Drawing.Point(3, 3);
            this.panelExpAll.Name = "panelExpAll";
            this.panelExpAll.Size = new System.Drawing.Size(282, 69);
            this.panelExpAll.TabIndex = 0;
            // 
            // lblExpAll
            // 
            this.lblExpAll.AutoSize = true;
            this.lblExpAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblExpAll.Location = new System.Drawing.Point(5, 7);
            this.lblExpAll.Name = "lblExpAll";
            this.lblExpAll.Size = new System.Drawing.Size(71, 15);
            this.lblExpAll.TabIndex = 0;
            this.lblExpAll.Text = "全部相機";
            // 
            // trackBarExpAll
            // 
            this.trackBarExpAll.AutoSize = false;
            this.trackBarExpAll.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpAll.Maximum = 10000;
            this.trackBarExpAll.Minimum = 1;
            this.trackBarExpAll.Name = "trackBarExpAll";
            this.trackBarExpAll.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpAll.TabIndex = 2;
            this.trackBarExpAll.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpAll.Value = 1000;
            // 
            // numExpAll
            // 
            this.numExpAll.Location = new System.Drawing.Point(204, 5);
            this.numExpAll.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpAll.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpAll.Name = "numExpAll";
            this.numExpAll.Size = new System.Drawing.Size(74, 25);
            this.numExpAll.TabIndex = 1;
            this.numExpAll.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam1
            // 
            this.panelExpCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam1.Controls.Add(this.lblExpCam1);
            this.panelExpCam1.Controls.Add(this.trackBarExpCam1);
            this.panelExpCam1.Controls.Add(this.numExpCam1);
            this.panelExpCam1.Location = new System.Drawing.Point(3, 76);
            this.panelExpCam1.Name = "panelExpCam1";
            this.panelExpCam1.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam1.TabIndex = 1;
            // 
            // lblExpCam1
            // 
            this.lblExpCam1.AutoSize = true;
            this.lblExpCam1.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam1.Name = "lblExpCam1";
            this.lblExpCam1.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam1.TabIndex = 0;
            this.lblExpCam1.Text = "CAM1";
            // 
            // trackBarExpCam1
            // 
            this.trackBarExpCam1.AutoSize = false;
            this.trackBarExpCam1.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam1.Maximum = 10000;
            this.trackBarExpCam1.Minimum = 1;
            this.trackBarExpCam1.Name = "trackBarExpCam1";
            this.trackBarExpCam1.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam1.TabIndex = 2;
            this.trackBarExpCam1.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam1.Value = 1000;
            // 
            // numExpCam1
            // 
            this.numExpCam1.Location = new System.Drawing.Point(204, 5);
            this.numExpCam1.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam1.Name = "numExpCam1";
            this.numExpCam1.Size = new System.Drawing.Size(74, 25);
            this.numExpCam1.TabIndex = 1;
            this.numExpCam1.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam2
            // 
            this.panelExpCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam2.Controls.Add(this.lblExpCam2);
            this.panelExpCam2.Controls.Add(this.trackBarExpCam2);
            this.panelExpCam2.Controls.Add(this.numExpCam2);
            this.panelExpCam2.Location = new System.Drawing.Point(3, 149);
            this.panelExpCam2.Name = "panelExpCam2";
            this.panelExpCam2.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam2.TabIndex = 2;
            // 
            // lblExpCam2
            // 
            this.lblExpCam2.AutoSize = true;
            this.lblExpCam2.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam2.Name = "lblExpCam2";
            this.lblExpCam2.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam2.TabIndex = 0;
            this.lblExpCam2.Text = "CAM2";
            // 
            // trackBarExpCam2
            // 
            this.trackBarExpCam2.AutoSize = false;
            this.trackBarExpCam2.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam2.Maximum = 10000;
            this.trackBarExpCam2.Minimum = 1;
            this.trackBarExpCam2.Name = "trackBarExpCam2";
            this.trackBarExpCam2.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam2.TabIndex = 2;
            this.trackBarExpCam2.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam2.Value = 1000;
            // 
            // numExpCam2
            // 
            this.numExpCam2.Location = new System.Drawing.Point(204, 5);
            this.numExpCam2.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam2.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam2.Name = "numExpCam2";
            this.numExpCam2.Size = new System.Drawing.Size(74, 25);
            this.numExpCam2.TabIndex = 1;
            this.numExpCam2.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam3
            // 
            this.panelExpCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam3.Controls.Add(this.lblExpCam3);
            this.panelExpCam3.Controls.Add(this.trackBarExpCam3);
            this.panelExpCam3.Controls.Add(this.numExpCam3);
            this.panelExpCam3.Location = new System.Drawing.Point(3, 222);
            this.panelExpCam3.Name = "panelExpCam3";
            this.panelExpCam3.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam3.TabIndex = 3;
            // 
            // lblExpCam3
            // 
            this.lblExpCam3.AutoSize = true;
            this.lblExpCam3.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam3.Name = "lblExpCam3";
            this.lblExpCam3.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam3.TabIndex = 0;
            this.lblExpCam3.Text = "CAM3";
            // 
            // trackBarExpCam3
            // 
            this.trackBarExpCam3.AutoSize = false;
            this.trackBarExpCam3.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam3.Maximum = 10000;
            this.trackBarExpCam3.Minimum = 1;
            this.trackBarExpCam3.Name = "trackBarExpCam3";
            this.trackBarExpCam3.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam3.TabIndex = 2;
            this.trackBarExpCam3.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam3.Value = 1000;
            // 
            // numExpCam3
            // 
            this.numExpCam3.Location = new System.Drawing.Point(204, 5);
            this.numExpCam3.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam3.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam3.Name = "numExpCam3";
            this.numExpCam3.Size = new System.Drawing.Size(74, 25);
            this.numExpCam3.TabIndex = 1;
            this.numExpCam3.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam4
            // 
            this.panelExpCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam4.Controls.Add(this.lblExpCam4);
            this.panelExpCam4.Controls.Add(this.trackBarExpCam4);
            this.panelExpCam4.Controls.Add(this.numExpCam4);
            this.panelExpCam4.Location = new System.Drawing.Point(3, 295);
            this.panelExpCam4.Name = "panelExpCam4";
            this.panelExpCam4.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam4.TabIndex = 4;
            // 
            // lblExpCam4
            // 
            this.lblExpCam4.AutoSize = true;
            this.lblExpCam4.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam4.Name = "lblExpCam4";
            this.lblExpCam4.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam4.TabIndex = 0;
            this.lblExpCam4.Text = "CAM4";
            // 
            // trackBarExpCam4
            // 
            this.trackBarExpCam4.AutoSize = false;
            this.trackBarExpCam4.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam4.Maximum = 10000;
            this.trackBarExpCam4.Minimum = 1;
            this.trackBarExpCam4.Name = "trackBarExpCam4";
            this.trackBarExpCam4.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam4.TabIndex = 2;
            this.trackBarExpCam4.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam4.Value = 1000;
            // 
            // numExpCam4
            // 
            this.numExpCam4.Location = new System.Drawing.Point(204, 5);
            this.numExpCam4.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam4.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam4.Name = "numExpCam4";
            this.numExpCam4.Size = new System.Drawing.Size(74, 25);
            this.numExpCam4.TabIndex = 1;
            this.numExpCam4.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam5
            // 
            this.panelExpCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam5.Controls.Add(this.lblExpCam5);
            this.panelExpCam5.Controls.Add(this.trackBarExpCam5);
            this.panelExpCam5.Controls.Add(this.numExpCam5);
            this.panelExpCam5.Location = new System.Drawing.Point(3, 368);
            this.panelExpCam5.Name = "panelExpCam5";
            this.panelExpCam5.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam5.TabIndex = 5;
            // 
            // lblExpCam5
            // 
            this.lblExpCam5.AutoSize = true;
            this.lblExpCam5.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam5.Name = "lblExpCam5";
            this.lblExpCam5.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam5.TabIndex = 0;
            this.lblExpCam5.Text = "CAM5";
            // 
            // trackBarExpCam5
            // 
            this.trackBarExpCam5.AutoSize = false;
            this.trackBarExpCam5.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam5.Maximum = 10000;
            this.trackBarExpCam5.Minimum = 1;
            this.trackBarExpCam5.Name = "trackBarExpCam5";
            this.trackBarExpCam5.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam5.TabIndex = 2;
            this.trackBarExpCam5.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam5.Value = 1000;
            // 
            // numExpCam5
            // 
            this.numExpCam5.Location = new System.Drawing.Point(204, 5);
            this.numExpCam5.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam5.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam5.Name = "numExpCam5";
            this.numExpCam5.Size = new System.Drawing.Size(74, 25);
            this.numExpCam5.TabIndex = 1;
            this.numExpCam5.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam6
            // 
            this.panelExpCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam6.Controls.Add(this.lblExpCam6);
            this.panelExpCam6.Controls.Add(this.trackBarExpCam6);
            this.panelExpCam6.Controls.Add(this.numExpCam6);
            this.panelExpCam6.Location = new System.Drawing.Point(3, 441);
            this.panelExpCam6.Name = "panelExpCam6";
            this.panelExpCam6.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam6.TabIndex = 6;
            // 
            // lblExpCam6
            // 
            this.lblExpCam6.AutoSize = true;
            this.lblExpCam6.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam6.Name = "lblExpCam6";
            this.lblExpCam6.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam6.TabIndex = 0;
            this.lblExpCam6.Text = "CAM6";
            // 
            // trackBarExpCam6
            // 
            this.trackBarExpCam6.AutoSize = false;
            this.trackBarExpCam6.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam6.Maximum = 10000;
            this.trackBarExpCam6.Minimum = 1;
            this.trackBarExpCam6.Name = "trackBarExpCam6";
            this.trackBarExpCam6.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam6.TabIndex = 2;
            this.trackBarExpCam6.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam6.Value = 1000;
            // 
            // numExpCam6
            // 
            this.numExpCam6.Location = new System.Drawing.Point(204, 5);
            this.numExpCam6.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam6.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam6.Name = "numExpCam6";
            this.numExpCam6.Size = new System.Drawing.Size(74, 25);
            this.numExpCam6.TabIndex = 1;
            this.numExpCam6.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam7
            // 
            this.panelExpCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam7.Controls.Add(this.lblExpCam7);
            this.panelExpCam7.Controls.Add(this.trackBarExpCam7);
            this.panelExpCam7.Controls.Add(this.numExpCam7);
            this.panelExpCam7.Location = new System.Drawing.Point(3, 514);
            this.panelExpCam7.Name = "panelExpCam7";
            this.panelExpCam7.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam7.TabIndex = 7;
            // 
            // lblExpCam7
            // 
            this.lblExpCam7.AutoSize = true;
            this.lblExpCam7.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam7.Name = "lblExpCam7";
            this.lblExpCam7.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam7.TabIndex = 0;
            this.lblExpCam7.Text = "CAM7";
            // 
            // trackBarExpCam7
            // 
            this.trackBarExpCam7.AutoSize = false;
            this.trackBarExpCam7.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam7.Maximum = 10000;
            this.trackBarExpCam7.Minimum = 1;
            this.trackBarExpCam7.Name = "trackBarExpCam7";
            this.trackBarExpCam7.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam7.TabIndex = 2;
            this.trackBarExpCam7.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam7.Value = 1000;
            // 
            // numExpCam7
            // 
            this.numExpCam7.Location = new System.Drawing.Point(204, 5);
            this.numExpCam7.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam7.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam7.Name = "numExpCam7";
            this.numExpCam7.Size = new System.Drawing.Size(74, 25);
            this.numExpCam7.TabIndex = 1;
            this.numExpCam7.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelExpCam8
            // 
            this.panelExpCam8.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExpCam8.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExpCam8.Controls.Add(this.lblExpCam8);
            this.panelExpCam8.Controls.Add(this.trackBarExpCam8);
            this.panelExpCam8.Controls.Add(this.numExpCam8);
            this.panelExpCam8.Location = new System.Drawing.Point(3, 587);
            this.panelExpCam8.Name = "panelExpCam8";
            this.panelExpCam8.Size = new System.Drawing.Size(282, 69);
            this.panelExpCam8.TabIndex = 8;
            // 
            // lblExpCam8
            // 
            this.lblExpCam8.AutoSize = true;
            this.lblExpCam8.Location = new System.Drawing.Point(5, 7);
            this.lblExpCam8.Name = "lblExpCam8";
            this.lblExpCam8.Size = new System.Drawing.Size(46, 15);
            this.lblExpCam8.TabIndex = 0;
            this.lblExpCam8.Text = "CAM8";
            // 
            // trackBarExpCam8
            // 
            this.trackBarExpCam8.AutoSize = false;
            this.trackBarExpCam8.Location = new System.Drawing.Point(2, 33);
            this.trackBarExpCam8.Maximum = 10000;
            this.trackBarExpCam8.Minimum = 1;
            this.trackBarExpCam8.Name = "trackBarExpCam8";
            this.trackBarExpCam8.Size = new System.Drawing.Size(280, 30);
            this.trackBarExpCam8.TabIndex = 2;
            this.trackBarExpCam8.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarExpCam8.Value = 1000;
            // 
            // numExpCam8
            // 
            this.numExpCam8.Location = new System.Drawing.Point(204, 5);
            this.numExpCam8.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numExpCam8.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numExpCam8.Name = "numExpCam8";
            this.numExpCam8.Size = new System.Drawing.Size(74, 25);
            this.numExpCam8.TabIndex = 1;
            this.numExpCam8.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // tabLineRate
            // 
            this.tabLineRate.AutoScroll = true;
            this.tabLineRate.Controls.Add(this.panelLrAll);
            this.tabLineRate.Controls.Add(this.panelLrCam1);
            this.tabLineRate.Controls.Add(this.panelLrCam2);
            this.tabLineRate.Controls.Add(this.panelLrCam3);
            this.tabLineRate.Controls.Add(this.panelLrCam4);
            this.tabLineRate.Controls.Add(this.panelLrCam5);
            this.tabLineRate.Controls.Add(this.panelLrCam6);
            this.tabLineRate.Controls.Add(this.panelLrCam7);
            this.tabLineRate.Controls.Add(this.panelLrCam8);
            this.tabLineRate.Location = new System.Drawing.Point(4, 25);
            this.tabLineRate.Name = "tabLineRate";
            this.tabLineRate.Padding = new System.Windows.Forms.Padding(3);
            this.tabLineRate.Size = new System.Drawing.Size(306, 662);
            this.tabLineRate.TabIndex = 1;
            this.tabLineRate.Text = "線掃(Hz)";
            this.tabLineRate.UseVisualStyleBackColor = true;
            // 
            // panelLrAll
            // 
            this.panelLrAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelLrAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrAll.Controls.Add(this.lblLrAll);
            this.panelLrAll.Controls.Add(this.trackBarLrAll);
            this.panelLrAll.Controls.Add(this.numLrAll);
            this.panelLrAll.Location = new System.Drawing.Point(3, 3);
            this.panelLrAll.Name = "panelLrAll";
            this.panelLrAll.Size = new System.Drawing.Size(286, 69);
            this.panelLrAll.TabIndex = 0;
            // 
            // lblLrAll
            // 
            this.lblLrAll.AutoSize = true;
            this.lblLrAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblLrAll.Location = new System.Drawing.Point(5, 7);
            this.lblLrAll.Name = "lblLrAll";
            this.lblLrAll.Size = new System.Drawing.Size(71, 15);
            this.lblLrAll.TabIndex = 0;
            this.lblLrAll.Text = "全部相機";
            // 
            // trackBarLrAll
            // 
            this.trackBarLrAll.AutoSize = false;
            this.trackBarLrAll.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrAll.Maximum = 100000;
            this.trackBarLrAll.Minimum = 100;
            this.trackBarLrAll.Name = "trackBarLrAll";
            this.trackBarLrAll.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrAll.TabIndex = 2;
            this.trackBarLrAll.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrAll.Value = 3000;
            // 
            // numLrAll
            // 
            this.numLrAll.Location = new System.Drawing.Point(204, 5);
            this.numLrAll.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrAll.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrAll.Name = "numLrAll";
            this.numLrAll.Size = new System.Drawing.Size(74, 25);
            this.numLrAll.TabIndex = 1;
            this.numLrAll.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam1
            // 
            this.panelLrCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam1.Controls.Add(this.lblLrCam1);
            this.panelLrCam1.Controls.Add(this.trackBarLrCam1);
            this.panelLrCam1.Controls.Add(this.numLrCam1);
            this.panelLrCam1.Location = new System.Drawing.Point(3, 76);
            this.panelLrCam1.Name = "panelLrCam1";
            this.panelLrCam1.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam1.TabIndex = 1;
            // 
            // lblLrCam1
            // 
            this.lblLrCam1.AutoSize = true;
            this.lblLrCam1.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam1.Name = "lblLrCam1";
            this.lblLrCam1.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam1.TabIndex = 0;
            this.lblLrCam1.Text = "CAM1";
            // 
            // trackBarLrCam1
            // 
            this.trackBarLrCam1.AutoSize = false;
            this.trackBarLrCam1.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam1.Maximum = 100000;
            this.trackBarLrCam1.Minimum = 100;
            this.trackBarLrCam1.Name = "trackBarLrCam1";
            this.trackBarLrCam1.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam1.TabIndex = 2;
            this.trackBarLrCam1.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam1.Value = 3000;
            // 
            // numLrCam1
            // 
            this.numLrCam1.Location = new System.Drawing.Point(204, 5);
            this.numLrCam1.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam1.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam1.Name = "numLrCam1";
            this.numLrCam1.Size = new System.Drawing.Size(74, 25);
            this.numLrCam1.TabIndex = 1;
            this.numLrCam1.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam2
            // 
            this.panelLrCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam2.Controls.Add(this.lblLrCam2);
            this.panelLrCam2.Controls.Add(this.trackBarLrCam2);
            this.panelLrCam2.Controls.Add(this.numLrCam2);
            this.panelLrCam2.Location = new System.Drawing.Point(3, 149);
            this.panelLrCam2.Name = "panelLrCam2";
            this.panelLrCam2.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam2.TabIndex = 2;
            // 
            // lblLrCam2
            // 
            this.lblLrCam2.AutoSize = true;
            this.lblLrCam2.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam2.Name = "lblLrCam2";
            this.lblLrCam2.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam2.TabIndex = 0;
            this.lblLrCam2.Text = "CAM2";
            // 
            // trackBarLrCam2
            // 
            this.trackBarLrCam2.AutoSize = false;
            this.trackBarLrCam2.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam2.Maximum = 100000;
            this.trackBarLrCam2.Minimum = 100;
            this.trackBarLrCam2.Name = "trackBarLrCam2";
            this.trackBarLrCam2.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam2.TabIndex = 2;
            this.trackBarLrCam2.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam2.Value = 3000;
            // 
            // numLrCam2
            // 
            this.numLrCam2.Location = new System.Drawing.Point(204, 5);
            this.numLrCam2.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam2.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam2.Name = "numLrCam2";
            this.numLrCam2.Size = new System.Drawing.Size(74, 25);
            this.numLrCam2.TabIndex = 1;
            this.numLrCam2.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam3
            // 
            this.panelLrCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam3.Controls.Add(this.lblLrCam3);
            this.panelLrCam3.Controls.Add(this.trackBarLrCam3);
            this.panelLrCam3.Controls.Add(this.numLrCam3);
            this.panelLrCam3.Location = new System.Drawing.Point(3, 222);
            this.panelLrCam3.Name = "panelLrCam3";
            this.panelLrCam3.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam3.TabIndex = 3;
            // 
            // lblLrCam3
            // 
            this.lblLrCam3.AutoSize = true;
            this.lblLrCam3.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam3.Name = "lblLrCam3";
            this.lblLrCam3.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam3.TabIndex = 0;
            this.lblLrCam3.Text = "CAM3";
            // 
            // trackBarLrCam3
            // 
            this.trackBarLrCam3.AutoSize = false;
            this.trackBarLrCam3.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam3.Maximum = 100000;
            this.trackBarLrCam3.Minimum = 100;
            this.trackBarLrCam3.Name = "trackBarLrCam3";
            this.trackBarLrCam3.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam3.TabIndex = 2;
            this.trackBarLrCam3.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam3.Value = 3000;
            // 
            // numLrCam3
            // 
            this.numLrCam3.Location = new System.Drawing.Point(204, 5);
            this.numLrCam3.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam3.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam3.Name = "numLrCam3";
            this.numLrCam3.Size = new System.Drawing.Size(74, 25);
            this.numLrCam3.TabIndex = 1;
            this.numLrCam3.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam4
            // 
            this.panelLrCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam4.Controls.Add(this.lblLrCam4);
            this.panelLrCam4.Controls.Add(this.trackBarLrCam4);
            this.panelLrCam4.Controls.Add(this.numLrCam4);
            this.panelLrCam4.Location = new System.Drawing.Point(3, 295);
            this.panelLrCam4.Name = "panelLrCam4";
            this.panelLrCam4.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam4.TabIndex = 4;
            // 
            // lblLrCam4
            // 
            this.lblLrCam4.AutoSize = true;
            this.lblLrCam4.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam4.Name = "lblLrCam4";
            this.lblLrCam4.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam4.TabIndex = 0;
            this.lblLrCam4.Text = "CAM4";
            // 
            // trackBarLrCam4
            // 
            this.trackBarLrCam4.AutoSize = false;
            this.trackBarLrCam4.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam4.Maximum = 100000;
            this.trackBarLrCam4.Minimum = 100;
            this.trackBarLrCam4.Name = "trackBarLrCam4";
            this.trackBarLrCam4.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam4.TabIndex = 2;
            this.trackBarLrCam4.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam4.Value = 3000;
            // 
            // numLrCam4
            // 
            this.numLrCam4.Location = new System.Drawing.Point(204, 5);
            this.numLrCam4.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam4.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam4.Name = "numLrCam4";
            this.numLrCam4.Size = new System.Drawing.Size(74, 25);
            this.numLrCam4.TabIndex = 1;
            this.numLrCam4.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam5
            // 
            this.panelLrCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam5.Controls.Add(this.lblLrCam5);
            this.panelLrCam5.Controls.Add(this.trackBarLrCam5);
            this.panelLrCam5.Controls.Add(this.numLrCam5);
            this.panelLrCam5.Location = new System.Drawing.Point(3, 368);
            this.panelLrCam5.Name = "panelLrCam5";
            this.panelLrCam5.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam5.TabIndex = 5;
            // 
            // lblLrCam5
            // 
            this.lblLrCam5.AutoSize = true;
            this.lblLrCam5.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam5.Name = "lblLrCam5";
            this.lblLrCam5.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam5.TabIndex = 0;
            this.lblLrCam5.Text = "CAM5";
            // 
            // trackBarLrCam5
            // 
            this.trackBarLrCam5.AutoSize = false;
            this.trackBarLrCam5.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam5.Maximum = 100000;
            this.trackBarLrCam5.Minimum = 100;
            this.trackBarLrCam5.Name = "trackBarLrCam5";
            this.trackBarLrCam5.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam5.TabIndex = 2;
            this.trackBarLrCam5.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam5.Value = 3000;
            // 
            // numLrCam5
            // 
            this.numLrCam5.Location = new System.Drawing.Point(204, 5);
            this.numLrCam5.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam5.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam5.Name = "numLrCam5";
            this.numLrCam5.Size = new System.Drawing.Size(74, 25);
            this.numLrCam5.TabIndex = 1;
            this.numLrCam5.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam6
            // 
            this.panelLrCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam6.Controls.Add(this.lblLrCam6);
            this.panelLrCam6.Controls.Add(this.trackBarLrCam6);
            this.panelLrCam6.Controls.Add(this.numLrCam6);
            this.panelLrCam6.Location = new System.Drawing.Point(3, 441);
            this.panelLrCam6.Name = "panelLrCam6";
            this.panelLrCam6.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam6.TabIndex = 6;
            // 
            // lblLrCam6
            // 
            this.lblLrCam6.AutoSize = true;
            this.lblLrCam6.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam6.Name = "lblLrCam6";
            this.lblLrCam6.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam6.TabIndex = 0;
            this.lblLrCam6.Text = "CAM6";
            // 
            // trackBarLrCam6
            // 
            this.trackBarLrCam6.AutoSize = false;
            this.trackBarLrCam6.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam6.Maximum = 100000;
            this.trackBarLrCam6.Minimum = 100;
            this.trackBarLrCam6.Name = "trackBarLrCam6";
            this.trackBarLrCam6.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam6.TabIndex = 2;
            this.trackBarLrCam6.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam6.Value = 3000;
            // 
            // numLrCam6
            // 
            this.numLrCam6.Location = new System.Drawing.Point(204, 5);
            this.numLrCam6.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam6.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam6.Name = "numLrCam6";
            this.numLrCam6.Size = new System.Drawing.Size(74, 25);
            this.numLrCam6.TabIndex = 1;
            this.numLrCam6.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam7
            // 
            this.panelLrCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam7.Controls.Add(this.lblLrCam7);
            this.panelLrCam7.Controls.Add(this.trackBarLrCam7);
            this.panelLrCam7.Controls.Add(this.numLrCam7);
            this.panelLrCam7.Location = new System.Drawing.Point(3, 514);
            this.panelLrCam7.Name = "panelLrCam7";
            this.panelLrCam7.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam7.TabIndex = 7;
            // 
            // lblLrCam7
            // 
            this.lblLrCam7.AutoSize = true;
            this.lblLrCam7.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam7.Name = "lblLrCam7";
            this.lblLrCam7.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam7.TabIndex = 0;
            this.lblLrCam7.Text = "CAM7";
            // 
            // trackBarLrCam7
            // 
            this.trackBarLrCam7.AutoSize = false;
            this.trackBarLrCam7.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam7.Maximum = 100000;
            this.trackBarLrCam7.Minimum = 100;
            this.trackBarLrCam7.Name = "trackBarLrCam7";
            this.trackBarLrCam7.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam7.TabIndex = 2;
            this.trackBarLrCam7.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam7.Value = 3000;
            // 
            // numLrCam7
            // 
            this.numLrCam7.Location = new System.Drawing.Point(204, 5);
            this.numLrCam7.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam7.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam7.Name = "numLrCam7";
            this.numLrCam7.Size = new System.Drawing.Size(74, 25);
            this.numLrCam7.TabIndex = 1;
            this.numLrCam7.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // panelLrCam8
            // 
            this.panelLrCam8.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelLrCam8.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelLrCam8.Controls.Add(this.lblLrCam8);
            this.panelLrCam8.Controls.Add(this.trackBarLrCam8);
            this.panelLrCam8.Controls.Add(this.numLrCam8);
            this.panelLrCam8.Location = new System.Drawing.Point(3, 587);
            this.panelLrCam8.Name = "panelLrCam8";
            this.panelLrCam8.Size = new System.Drawing.Size(286, 69);
            this.panelLrCam8.TabIndex = 8;
            // 
            // lblLrCam8
            // 
            this.lblLrCam8.AutoSize = true;
            this.lblLrCam8.Location = new System.Drawing.Point(5, 7);
            this.lblLrCam8.Name = "lblLrCam8";
            this.lblLrCam8.Size = new System.Drawing.Size(46, 15);
            this.lblLrCam8.TabIndex = 0;
            this.lblLrCam8.Text = "CAM8";
            // 
            // trackBarLrCam8
            // 
            this.trackBarLrCam8.AutoSize = false;
            this.trackBarLrCam8.Location = new System.Drawing.Point(2, 33);
            this.trackBarLrCam8.Maximum = 100000;
            this.trackBarLrCam8.Minimum = 100;
            this.trackBarLrCam8.Name = "trackBarLrCam8";
            this.trackBarLrCam8.Size = new System.Drawing.Size(280, 30);
            this.trackBarLrCam8.TabIndex = 2;
            this.trackBarLrCam8.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarLrCam8.Value = 3000;
            // 
            // numLrCam8
            // 
            this.numLrCam8.Location = new System.Drawing.Point(204, 5);
            this.numLrCam8.Maximum = new decimal(new int[] {
            100000,
            0,
            0,
            0});
            this.numLrCam8.Minimum = new decimal(new int[] {
            100,
            0,
            0,
            0});
            this.numLrCam8.Name = "numLrCam8";
            this.numLrCam8.Size = new System.Drawing.Size(74, 25);
            this.numLrCam8.TabIndex = 1;
            this.numLrCam8.Value = new decimal(new int[] {
            3000,
            0,
            0,
            0});
            // 
            // tabHeight
            // 
            this.tabHeight.AutoScroll = true;
            this.tabHeight.Controls.Add(this.panelHtAll);
            this.tabHeight.Controls.Add(this.panelHtCam1);
            this.tabHeight.Controls.Add(this.panelHtCam2);
            this.tabHeight.Controls.Add(this.panelHtCam3);
            this.tabHeight.Controls.Add(this.panelHtCam4);
            this.tabHeight.Controls.Add(this.panelHtCam5);
            this.tabHeight.Controls.Add(this.panelHtCam6);
            this.tabHeight.Controls.Add(this.panelHtCam7);
            this.tabHeight.Controls.Add(this.panelHtCam8);
            this.tabHeight.Location = new System.Drawing.Point(4, 25);
            this.tabHeight.Name = "tabHeight";
            this.tabHeight.Padding = new System.Windows.Forms.Padding(3);
            this.tabHeight.Size = new System.Drawing.Size(306, 662);
            this.tabHeight.TabIndex = 2;
            this.tabHeight.Text = "高度(px)";
            this.tabHeight.UseVisualStyleBackColor = true;
            // 
            // panelHtAll
            // 
            this.panelHtAll.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtAll.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(230)))), ((int)(((byte)(240)))), ((int)(((byte)(255)))));
            this.panelHtAll.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtAll.Controls.Add(this.lblHtAll);
            this.panelHtAll.Controls.Add(this.trackBarHtAll);
            this.panelHtAll.Controls.Add(this.numHtAll);
            this.panelHtAll.Location = new System.Drawing.Point(3, 3);
            this.panelHtAll.Name = "panelHtAll";
            this.panelHtAll.Size = new System.Drawing.Size(286, 69);
            this.panelHtAll.TabIndex = 0;
            // 
            // lblHtAll
            // 
            this.lblHtAll.AutoSize = true;
            this.lblHtAll.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblHtAll.Location = new System.Drawing.Point(5, 7);
            this.lblHtAll.Name = "lblHtAll";
            this.lblHtAll.Size = new System.Drawing.Size(71, 15);
            this.lblHtAll.TabIndex = 0;
            this.lblHtAll.Text = "全部相機";
            // 
            // trackBarHtAll
            // 
            this.trackBarHtAll.AutoSize = false;
            this.trackBarHtAll.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtAll.Maximum = 10000;
            this.trackBarHtAll.Minimum = 1;
            this.trackBarHtAll.Name = "trackBarHtAll";
            this.trackBarHtAll.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtAll.TabIndex = 2;
            this.trackBarHtAll.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtAll.Value = 1000;
            // 
            // numHtAll
            // 
            this.numHtAll.Location = new System.Drawing.Point(204, 5);
            this.numHtAll.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtAll.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtAll.Name = "numHtAll";
            this.numHtAll.Size = new System.Drawing.Size(74, 25);
            this.numHtAll.TabIndex = 1;
            this.numHtAll.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam1
            // 
            this.panelHtCam1.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam1.Controls.Add(this.lblHtCam1);
            this.panelHtCam1.Controls.Add(this.trackBarHtCam1);
            this.panelHtCam1.Controls.Add(this.numHtCam1);
            this.panelHtCam1.Location = new System.Drawing.Point(3, 76);
            this.panelHtCam1.Name = "panelHtCam1";
            this.panelHtCam1.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam1.TabIndex = 1;
            // 
            // lblHtCam1
            // 
            this.lblHtCam1.AutoSize = true;
            this.lblHtCam1.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam1.Name = "lblHtCam1";
            this.lblHtCam1.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam1.TabIndex = 0;
            this.lblHtCam1.Text = "CAM1";
            // 
            // trackBarHtCam1
            // 
            this.trackBarHtCam1.AutoSize = false;
            this.trackBarHtCam1.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam1.Maximum = 10000;
            this.trackBarHtCam1.Minimum = 1;
            this.trackBarHtCam1.Name = "trackBarHtCam1";
            this.trackBarHtCam1.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam1.TabIndex = 2;
            this.trackBarHtCam1.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam1.Value = 1000;
            // 
            // numHtCam1
            // 
            this.numHtCam1.Location = new System.Drawing.Point(204, 5);
            this.numHtCam1.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam1.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam1.Name = "numHtCam1";
            this.numHtCam1.Size = new System.Drawing.Size(74, 25);
            this.numHtCam1.TabIndex = 1;
            this.numHtCam1.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam2
            // 
            this.panelHtCam2.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam2.Controls.Add(this.lblHtCam2);
            this.panelHtCam2.Controls.Add(this.trackBarHtCam2);
            this.panelHtCam2.Controls.Add(this.numHtCam2);
            this.panelHtCam2.Location = new System.Drawing.Point(3, 149);
            this.panelHtCam2.Name = "panelHtCam2";
            this.panelHtCam2.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam2.TabIndex = 2;
            // 
            // lblHtCam2
            // 
            this.lblHtCam2.AutoSize = true;
            this.lblHtCam2.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam2.Name = "lblHtCam2";
            this.lblHtCam2.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam2.TabIndex = 0;
            this.lblHtCam2.Text = "CAM2";
            // 
            // trackBarHtCam2
            // 
            this.trackBarHtCam2.AutoSize = false;
            this.trackBarHtCam2.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam2.Maximum = 10000;
            this.trackBarHtCam2.Minimum = 1;
            this.trackBarHtCam2.Name = "trackBarHtCam2";
            this.trackBarHtCam2.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam2.TabIndex = 2;
            this.trackBarHtCam2.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam2.Value = 1000;
            // 
            // numHtCam2
            // 
            this.numHtCam2.Location = new System.Drawing.Point(204, 5);
            this.numHtCam2.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam2.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam2.Name = "numHtCam2";
            this.numHtCam2.Size = new System.Drawing.Size(74, 25);
            this.numHtCam2.TabIndex = 1;
            this.numHtCam2.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam3
            // 
            this.panelHtCam3.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam3.Controls.Add(this.lblHtCam3);
            this.panelHtCam3.Controls.Add(this.trackBarHtCam3);
            this.panelHtCam3.Controls.Add(this.numHtCam3);
            this.panelHtCam3.Location = new System.Drawing.Point(3, 222);
            this.panelHtCam3.Name = "panelHtCam3";
            this.panelHtCam3.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam3.TabIndex = 3;
            // 
            // lblHtCam3
            // 
            this.lblHtCam3.AutoSize = true;
            this.lblHtCam3.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam3.Name = "lblHtCam3";
            this.lblHtCam3.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam3.TabIndex = 0;
            this.lblHtCam3.Text = "CAM3";
            // 
            // trackBarHtCam3
            // 
            this.trackBarHtCam3.AutoSize = false;
            this.trackBarHtCam3.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam3.Maximum = 10000;
            this.trackBarHtCam3.Minimum = 1;
            this.trackBarHtCam3.Name = "trackBarHtCam3";
            this.trackBarHtCam3.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam3.TabIndex = 2;
            this.trackBarHtCam3.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam3.Value = 1000;
            // 
            // numHtCam3
            // 
            this.numHtCam3.Location = new System.Drawing.Point(204, 5);
            this.numHtCam3.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam3.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam3.Name = "numHtCam3";
            this.numHtCam3.Size = new System.Drawing.Size(74, 25);
            this.numHtCam3.TabIndex = 1;
            this.numHtCam3.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam4
            // 
            this.panelHtCam4.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam4.Controls.Add(this.lblHtCam4);
            this.panelHtCam4.Controls.Add(this.trackBarHtCam4);
            this.panelHtCam4.Controls.Add(this.numHtCam4);
            this.panelHtCam4.Location = new System.Drawing.Point(3, 295);
            this.panelHtCam4.Name = "panelHtCam4";
            this.panelHtCam4.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam4.TabIndex = 4;
            // 
            // lblHtCam4
            // 
            this.lblHtCam4.AutoSize = true;
            this.lblHtCam4.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam4.Name = "lblHtCam4";
            this.lblHtCam4.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam4.TabIndex = 0;
            this.lblHtCam4.Text = "CAM4";
            // 
            // trackBarHtCam4
            // 
            this.trackBarHtCam4.AutoSize = false;
            this.trackBarHtCam4.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam4.Maximum = 10000;
            this.trackBarHtCam4.Minimum = 1;
            this.trackBarHtCam4.Name = "trackBarHtCam4";
            this.trackBarHtCam4.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam4.TabIndex = 2;
            this.trackBarHtCam4.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam4.Value = 1000;
            // 
            // numHtCam4
            // 
            this.numHtCam4.Location = new System.Drawing.Point(204, 5);
            this.numHtCam4.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam4.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam4.Name = "numHtCam4";
            this.numHtCam4.Size = new System.Drawing.Size(74, 25);
            this.numHtCam4.TabIndex = 1;
            this.numHtCam4.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam5
            // 
            this.panelHtCam5.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam5.Controls.Add(this.lblHtCam5);
            this.panelHtCam5.Controls.Add(this.trackBarHtCam5);
            this.panelHtCam5.Controls.Add(this.numHtCam5);
            this.panelHtCam5.Location = new System.Drawing.Point(3, 368);
            this.panelHtCam5.Name = "panelHtCam5";
            this.panelHtCam5.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam5.TabIndex = 5;
            // 
            // lblHtCam5
            // 
            this.lblHtCam5.AutoSize = true;
            this.lblHtCam5.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam5.Name = "lblHtCam5";
            this.lblHtCam5.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam5.TabIndex = 0;
            this.lblHtCam5.Text = "CAM5";
            // 
            // trackBarHtCam5
            // 
            this.trackBarHtCam5.AutoSize = false;
            this.trackBarHtCam5.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam5.Maximum = 10000;
            this.trackBarHtCam5.Minimum = 1;
            this.trackBarHtCam5.Name = "trackBarHtCam5";
            this.trackBarHtCam5.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam5.TabIndex = 2;
            this.trackBarHtCam5.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam5.Value = 1000;
            // 
            // numHtCam5
            // 
            this.numHtCam5.Location = new System.Drawing.Point(204, 5);
            this.numHtCam5.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam5.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam5.Name = "numHtCam5";
            this.numHtCam5.Size = new System.Drawing.Size(74, 25);
            this.numHtCam5.TabIndex = 1;
            this.numHtCam5.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam6
            // 
            this.panelHtCam6.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam6.Controls.Add(this.lblHtCam6);
            this.panelHtCam6.Controls.Add(this.trackBarHtCam6);
            this.panelHtCam6.Controls.Add(this.numHtCam6);
            this.panelHtCam6.Location = new System.Drawing.Point(3, 441);
            this.panelHtCam6.Name = "panelHtCam6";
            this.panelHtCam6.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam6.TabIndex = 6;
            // 
            // lblHtCam6
            // 
            this.lblHtCam6.AutoSize = true;
            this.lblHtCam6.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam6.Name = "lblHtCam6";
            this.lblHtCam6.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam6.TabIndex = 0;
            this.lblHtCam6.Text = "CAM6";
            // 
            // trackBarHtCam6
            // 
            this.trackBarHtCam6.AutoSize = false;
            this.trackBarHtCam6.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam6.Maximum = 10000;
            this.trackBarHtCam6.Minimum = 1;
            this.trackBarHtCam6.Name = "trackBarHtCam6";
            this.trackBarHtCam6.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam6.TabIndex = 2;
            this.trackBarHtCam6.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam6.Value = 1000;
            // 
            // numHtCam6
            // 
            this.numHtCam6.Location = new System.Drawing.Point(204, 5);
            this.numHtCam6.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam6.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam6.Name = "numHtCam6";
            this.numHtCam6.Size = new System.Drawing.Size(74, 25);
            this.numHtCam6.TabIndex = 1;
            this.numHtCam6.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam7
            // 
            this.panelHtCam7.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam7.Controls.Add(this.lblHtCam7);
            this.panelHtCam7.Controls.Add(this.trackBarHtCam7);
            this.panelHtCam7.Controls.Add(this.numHtCam7);
            this.panelHtCam7.Location = new System.Drawing.Point(3, 514);
            this.panelHtCam7.Name = "panelHtCam7";
            this.panelHtCam7.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam7.TabIndex = 7;
            // 
            // lblHtCam7
            // 
            this.lblHtCam7.AutoSize = true;
            this.lblHtCam7.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam7.Name = "lblHtCam7";
            this.lblHtCam7.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam7.TabIndex = 0;
            this.lblHtCam7.Text = "CAM7";
            // 
            // trackBarHtCam7
            // 
            this.trackBarHtCam7.AutoSize = false;
            this.trackBarHtCam7.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam7.Maximum = 10000;
            this.trackBarHtCam7.Minimum = 1;
            this.trackBarHtCam7.Name = "trackBarHtCam7";
            this.trackBarHtCam7.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam7.TabIndex = 2;
            this.trackBarHtCam7.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam7.Value = 1000;
            // 
            // numHtCam7
            // 
            this.numHtCam7.Location = new System.Drawing.Point(204, 5);
            this.numHtCam7.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam7.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam7.Name = "numHtCam7";
            this.numHtCam7.Size = new System.Drawing.Size(74, 25);
            this.numHtCam7.TabIndex = 1;
            this.numHtCam7.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelHtCam8
            // 
            this.panelHtCam8.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelHtCam8.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelHtCam8.Controls.Add(this.lblHtCam8);
            this.panelHtCam8.Controls.Add(this.trackBarHtCam8);
            this.panelHtCam8.Controls.Add(this.numHtCam8);
            this.panelHtCam8.Location = new System.Drawing.Point(3, 587);
            this.panelHtCam8.Name = "panelHtCam8";
            this.panelHtCam8.Size = new System.Drawing.Size(286, 69);
            this.panelHtCam8.TabIndex = 8;
            // 
            // lblHtCam8
            // 
            this.lblHtCam8.AutoSize = true;
            this.lblHtCam8.Location = new System.Drawing.Point(5, 7);
            this.lblHtCam8.Name = "lblHtCam8";
            this.lblHtCam8.Size = new System.Drawing.Size(46, 15);
            this.lblHtCam8.TabIndex = 0;
            this.lblHtCam8.Text = "CAM8";
            // 
            // trackBarHtCam8
            // 
            this.trackBarHtCam8.AutoSize = false;
            this.trackBarHtCam8.Location = new System.Drawing.Point(2, 33);
            this.trackBarHtCam8.Maximum = 10000;
            this.trackBarHtCam8.Minimum = 1;
            this.trackBarHtCam8.Name = "trackBarHtCam8";
            this.trackBarHtCam8.Size = new System.Drawing.Size(280, 30);
            this.trackBarHtCam8.TabIndex = 2;
            this.trackBarHtCam8.TickStyle = System.Windows.Forms.TickStyle.None;
            this.trackBarHtCam8.Value = 1000;
            // 
            // numHtCam8
            // 
            this.numHtCam8.Location = new System.Drawing.Point(204, 5);
            this.numHtCam8.Maximum = new decimal(new int[] {
            10000,
            0,
            0,
            0});
            this.numHtCam8.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.numHtCam8.Name = "numHtCam8";
            this.numHtCam8.Size = new System.Drawing.Size(74, 25);
            this.numHtCam8.TabIndex = 1;
            this.numHtCam8.Value = new decimal(new int[] {
            1000,
            0,
            0,
            0});
            // 
            // panelCam0
            // 
            this.panelCam0.BackColor = System.Drawing.Color.Black;
            this.panelCam0.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam0.Location = new System.Drawing.Point(12, 17);
            this.panelCam0.Name = "panelCam0";
            this.panelCam0.Size = new System.Drawing.Size(194, 118);
            this.panelCam0.TabIndex = 10;
            // 
            // panelCam1
            // 
            this.panelCam1.BackColor = System.Drawing.Color.Black;
            this.panelCam1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam1.Location = new System.Drawing.Point(212, 17);
            this.panelCam1.Name = "panelCam1";
            this.panelCam1.Size = new System.Drawing.Size(194, 118);
            this.panelCam1.TabIndex = 11;
            // 
            // panelCam2
            // 
            this.panelCam2.BackColor = System.Drawing.Color.Black;
            this.panelCam2.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam2.Location = new System.Drawing.Point(412, 17);
            this.panelCam2.Name = "panelCam2";
            this.panelCam2.Size = new System.Drawing.Size(194, 118);
            this.panelCam2.TabIndex = 12;
            // 
            // panelCam3
            // 
            this.panelCam3.BackColor = System.Drawing.Color.Black;
            this.panelCam3.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam3.Location = new System.Drawing.Point(612, 17);
            this.panelCam3.Name = "panelCam3";
            this.panelCam3.Size = new System.Drawing.Size(194, 118);
            this.panelCam3.TabIndex = 13;
            // 
            // panelCam4
            // 
            this.panelCam4.BackColor = System.Drawing.Color.Black;
            this.panelCam4.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam4.Location = new System.Drawing.Point(12, 141);
            this.panelCam4.Name = "panelCam4";
            this.panelCam4.Size = new System.Drawing.Size(194, 118);
            this.panelCam4.TabIndex = 14;
            // 
            // panelCam5
            // 
            this.panelCam5.BackColor = System.Drawing.Color.Black;
            this.panelCam5.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam5.Location = new System.Drawing.Point(212, 141);
            this.panelCam5.Name = "panelCam5";
            this.panelCam5.Size = new System.Drawing.Size(194, 118);
            this.panelCam5.TabIndex = 15;
            // 
            // panelCam6
            // 
            this.panelCam6.BackColor = System.Drawing.Color.Black;
            this.panelCam6.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam6.Location = new System.Drawing.Point(412, 141);
            this.panelCam6.Name = "panelCam6";
            this.panelCam6.Size = new System.Drawing.Size(194, 118);
            this.panelCam6.TabIndex = 16;
            // 
            // panelCam7
            // 
            this.panelCam7.BackColor = System.Drawing.Color.Black;
            this.panelCam7.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelCam7.Location = new System.Drawing.Point(612, 141);
            this.panelCam7.Name = "panelCam7";
            this.panelCam7.Size = new System.Drawing.Size(194, 118);
            this.panelCam7.TabIndex = 17;
            // 
            // MilGrabberPbForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1415, 875);
            this.Controls.Add(this.panelCam0);
            this.Controls.Add(this.panelCam1);
            this.Controls.Add(this.panelCam2);
            this.Controls.Add(this.panelCam3);
            this.Controls.Add(this.panelCam4);
            this.Controls.Add(this.panelCam5);
            this.Controls.Add(this.panelCam6);
            this.Controls.Add(this.panelCam7);
            this.Controls.Add(this.tabParams);
            this.Controls.Add(this.lvEngine);
            this.Controls.Add(this.lvCameras);
            this.Controls.Add(this.panelMain);
            this.Controls.Add(this.chkFlipVertical);
            this.Controls.Add(this.lblResize);
            this.Controls.Add(this.numResize);
            this.Controls.Add(this.lblFov);
            this.Controls.Add(this.numFovMm);
            this.Controls.Add(this.chkMerge);
            this.Controls.Add(this.chkLod);
            this.Controls.Add(this._rbModePb);
            this.Controls.Add(this._rbModeMil);
            this.Controls.Add(this._lblTiming);
            this.Controls.Add(this.btnFetchInfo);
            this.Controls.Add(this.btnRelease);
            this.Controls.Add(this.btnGrab);
            this.Controls.Add(this.btnInit);
            this.Name = "MilGrabberPbForm";
            this.Text = "MilGrabber.Monitor — 多相機即時監控";
            ((System.ComponentModel.ISupportInitialize)(this.numResize)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numFovMm)).EndInit();
            this.tabParams.ResumeLayout(false);
            this.tabExposure.ResumeLayout(false);
            this.panelExpAll.ResumeLayout(false);
            this.panelExpAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpAll)).EndInit();
            this.panelExpCam1.ResumeLayout(false);
            this.panelExpCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).EndInit();
            this.panelExpCam2.ResumeLayout(false);
            this.panelExpCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).EndInit();
            this.panelExpCam3.ResumeLayout(false);
            this.panelExpCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).EndInit();
            this.panelExpCam4.ResumeLayout(false);
            this.panelExpCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).EndInit();
            this.panelExpCam5.ResumeLayout(false);
            this.panelExpCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).EndInit();
            this.panelExpCam6.ResumeLayout(false);
            this.panelExpCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).EndInit();
            this.panelExpCam7.ResumeLayout(false);
            this.panelExpCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).EndInit();
            this.panelExpCam8.ResumeLayout(false);
            this.panelExpCam8.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam8)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam8)).EndInit();
            this.tabLineRate.ResumeLayout(false);
            this.panelLrAll.ResumeLayout(false);
            this.panelLrAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrAll)).EndInit();
            this.panelLrCam1.ResumeLayout(false);
            this.panelLrCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).EndInit();
            this.panelLrCam2.ResumeLayout(false);
            this.panelLrCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).EndInit();
            this.panelLrCam3.ResumeLayout(false);
            this.panelLrCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).EndInit();
            this.panelLrCam4.ResumeLayout(false);
            this.panelLrCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).EndInit();
            this.panelLrCam5.ResumeLayout(false);
            this.panelLrCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).EndInit();
            this.panelLrCam6.ResumeLayout(false);
            this.panelLrCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).EndInit();
            this.panelLrCam7.ResumeLayout(false);
            this.panelLrCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).EndInit();
            this.panelLrCam8.ResumeLayout(false);
            this.panelLrCam8.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam8)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam8)).EndInit();
            this.tabHeight.ResumeLayout(false);
            this.panelHtAll.ResumeLayout(false);
            this.panelHtAll.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtAll)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtAll)).EndInit();
            this.panelHtCam1.ResumeLayout(false);
            this.panelHtCam1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).EndInit();
            this.panelHtCam2.ResumeLayout(false);
            this.panelHtCam2.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).EndInit();
            this.panelHtCam3.ResumeLayout(false);
            this.panelHtCam3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).EndInit();
            this.panelHtCam4.ResumeLayout(false);
            this.panelHtCam4.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).EndInit();
            this.panelHtCam5.ResumeLayout(false);
            this.panelHtCam5.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).EndInit();
            this.panelHtCam6.ResumeLayout(false);
            this.panelHtCam6.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).EndInit();
            this.panelHtCam7.ResumeLayout(false);
            this.panelHtCam7.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).EndInit();
            this.panelHtCam8.ResumeLayout(false);
            this.panelHtCam8.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam8)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam8)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnInit;
        private System.Windows.Forms.Button btnGrab;
        private System.Windows.Forms.Button btnRelease;
        private System.Windows.Forms.Button btnFetchInfo;
        private System.Windows.Forms.CheckBox chkFlipVertical;
        private System.Windows.Forms.Label lblResize;
        private System.Windows.Forms.NumericUpDown numResize;
        private System.Windows.Forms.Label lblFov;
        private System.Windows.Forms.NumericUpDown numFovMm;
        private System.Windows.Forms.CheckBox chkMerge;
        private System.Windows.Forms.CheckBox chkLod;
        private System.Windows.Forms.RadioButton _rbModePb;
        private System.Windows.Forms.RadioButton _rbModeMil;
        private System.Windows.Forms.Label _lblTiming;
        private System.Windows.Forms.Panel panelMain;
        private System.Windows.Forms.ListView lvCameras;
        private System.Windows.Forms.ColumnHeader colCamCamera;
        private System.Windows.Forms.ColumnHeader colCamFps;
        private System.Windows.Forms.ColumnHeader colCamTargetFps;
        private System.Windows.Forms.ColumnHeader colCamLineRate;
        private System.Windows.Forms.ColumnHeader colCamLineRateMax;
        private System.Windows.Forms.ColumnHeader colCamExpSet;
        private System.Windows.Forms.ColumnHeader colCamExpMeas;
        private System.Windows.Forms.ColumnHeader colCamFrames;
        private System.Windows.Forms.ColumnHeader colCamMissed;
        private System.Windows.Forms.ColumnHeader colCamGrabMiss;
        private System.Windows.Forms.ColumnHeader colCamResolution;
        private System.Windows.Forms.ColumnHeader colCamScanMode;
        private System.Windows.Forms.ColumnHeader colCamFpga;
        private System.Windows.Forms.ColumnHeader colCamTemp;
        private System.Windows.Forms.ColumnHeader colCamMemFree;
        private System.Windows.Forms.ColumnHeader colCamPcieLanes;
        private System.Windows.Forms.ColumnHeader colCamPcieSpeed;
        private System.Windows.Forms.ListView lvEngine;
        private System.Windows.Forms.ColumnHeader colEngParam;
        private System.Windows.Forms.ColumnHeader colEngValue;
        private System.Windows.Forms.TabControl tabParams;
        private System.Windows.Forms.TabPage tabExposure;
        private System.Windows.Forms.TabPage tabLineRate;
        private System.Windows.Forms.TabPage tabHeight;
        // ── 曝光 tab 控制項（固定 8 相機 + 全部相機列）──
        private System.Windows.Forms.Panel panelExpAll;
        private System.Windows.Forms.Label lblExpAll;
        private System.Windows.Forms.TrackBar trackBarExpAll;
        private System.Windows.Forms.NumericUpDown numExpAll;
        private System.Windows.Forms.Panel panelExpCam1;
        private System.Windows.Forms.Label lblExpCam1;
        private System.Windows.Forms.TrackBar trackBarExpCam1;
        private System.Windows.Forms.NumericUpDown numExpCam1;
        private System.Windows.Forms.Panel panelExpCam2;
        private System.Windows.Forms.Label lblExpCam2;
        private System.Windows.Forms.TrackBar trackBarExpCam2;
        private System.Windows.Forms.NumericUpDown numExpCam2;
        private System.Windows.Forms.Panel panelExpCam3;
        private System.Windows.Forms.Label lblExpCam3;
        private System.Windows.Forms.TrackBar trackBarExpCam3;
        private System.Windows.Forms.NumericUpDown numExpCam3;
        private System.Windows.Forms.Panel panelExpCam4;
        private System.Windows.Forms.Label lblExpCam4;
        private System.Windows.Forms.TrackBar trackBarExpCam4;
        private System.Windows.Forms.NumericUpDown numExpCam4;
        private System.Windows.Forms.Panel panelExpCam5;
        private System.Windows.Forms.Label lblExpCam5;
        private System.Windows.Forms.TrackBar trackBarExpCam5;
        private System.Windows.Forms.NumericUpDown numExpCam5;
        private System.Windows.Forms.Panel panelExpCam6;
        private System.Windows.Forms.Label lblExpCam6;
        private System.Windows.Forms.TrackBar trackBarExpCam6;
        private System.Windows.Forms.NumericUpDown numExpCam6;
        private System.Windows.Forms.Panel panelExpCam7;
        private System.Windows.Forms.Label lblExpCam7;
        private System.Windows.Forms.TrackBar trackBarExpCam7;
        private System.Windows.Forms.NumericUpDown numExpCam7;
        private System.Windows.Forms.Panel panelExpCam8;
        private System.Windows.Forms.Label lblExpCam8;
        private System.Windows.Forms.TrackBar trackBarExpCam8;
        private System.Windows.Forms.NumericUpDown numExpCam8;
        // ── 線掃 tab 控制項 ──
        private System.Windows.Forms.Panel panelLrAll;
        private System.Windows.Forms.Label lblLrAll;
        private System.Windows.Forms.TrackBar trackBarLrAll;
        private System.Windows.Forms.NumericUpDown numLrAll;
        private System.Windows.Forms.Panel panelLrCam1;
        private System.Windows.Forms.Label lblLrCam1;
        private System.Windows.Forms.TrackBar trackBarLrCam1;
        private System.Windows.Forms.NumericUpDown numLrCam1;
        private System.Windows.Forms.Panel panelLrCam2;
        private System.Windows.Forms.Label lblLrCam2;
        private System.Windows.Forms.TrackBar trackBarLrCam2;
        private System.Windows.Forms.NumericUpDown numLrCam2;
        private System.Windows.Forms.Panel panelLrCam3;
        private System.Windows.Forms.Label lblLrCam3;
        private System.Windows.Forms.TrackBar trackBarLrCam3;
        private System.Windows.Forms.NumericUpDown numLrCam3;
        private System.Windows.Forms.Panel panelLrCam4;
        private System.Windows.Forms.Label lblLrCam4;
        private System.Windows.Forms.TrackBar trackBarLrCam4;
        private System.Windows.Forms.NumericUpDown numLrCam4;
        private System.Windows.Forms.Panel panelLrCam5;
        private System.Windows.Forms.Label lblLrCam5;
        private System.Windows.Forms.TrackBar trackBarLrCam5;
        private System.Windows.Forms.NumericUpDown numLrCam5;
        private System.Windows.Forms.Panel panelLrCam6;
        private System.Windows.Forms.Label lblLrCam6;
        private System.Windows.Forms.TrackBar trackBarLrCam6;
        private System.Windows.Forms.NumericUpDown numLrCam6;
        private System.Windows.Forms.Panel panelLrCam7;
        private System.Windows.Forms.Label lblLrCam7;
        private System.Windows.Forms.TrackBar trackBarLrCam7;
        private System.Windows.Forms.NumericUpDown numLrCam7;
        private System.Windows.Forms.Panel panelLrCam8;
        private System.Windows.Forms.Label lblLrCam8;
        private System.Windows.Forms.TrackBar trackBarLrCam8;
        private System.Windows.Forms.NumericUpDown numLrCam8;
        // ── 高度 tab 控制項 ──
        private System.Windows.Forms.Panel panelHtAll;
        private System.Windows.Forms.Label lblHtAll;
        private System.Windows.Forms.TrackBar trackBarHtAll;
        private System.Windows.Forms.NumericUpDown numHtAll;
        private System.Windows.Forms.Panel panelHtCam1;
        private System.Windows.Forms.Label lblHtCam1;
        private System.Windows.Forms.TrackBar trackBarHtCam1;
        private System.Windows.Forms.NumericUpDown numHtCam1;
        private System.Windows.Forms.Panel panelHtCam2;
        private System.Windows.Forms.Label lblHtCam2;
        private System.Windows.Forms.TrackBar trackBarHtCam2;
        private System.Windows.Forms.NumericUpDown numHtCam2;
        private System.Windows.Forms.Panel panelHtCam3;
        private System.Windows.Forms.Label lblHtCam3;
        private System.Windows.Forms.TrackBar trackBarHtCam3;
        private System.Windows.Forms.NumericUpDown numHtCam3;
        private System.Windows.Forms.Panel panelHtCam4;
        private System.Windows.Forms.Label lblHtCam4;
        private System.Windows.Forms.TrackBar trackBarHtCam4;
        private System.Windows.Forms.NumericUpDown numHtCam4;
        private System.Windows.Forms.Panel panelHtCam5;
        private System.Windows.Forms.Label lblHtCam5;
        private System.Windows.Forms.TrackBar trackBarHtCam5;
        private System.Windows.Forms.NumericUpDown numHtCam5;
        private System.Windows.Forms.Panel panelHtCam6;
        private System.Windows.Forms.Label lblHtCam6;
        private System.Windows.Forms.TrackBar trackBarHtCam6;
        private System.Windows.Forms.NumericUpDown numHtCam6;
        private System.Windows.Forms.Panel panelHtCam7;
        private System.Windows.Forms.Label lblHtCam7;
        private System.Windows.Forms.TrackBar trackBarHtCam7;
        private System.Windows.Forms.NumericUpDown numHtCam7;
        private System.Windows.Forms.Panel panelHtCam8;
        private System.Windows.Forms.Label lblHtCam8;
        private System.Windows.Forms.TrackBar trackBarHtCam8;
        private System.Windows.Forms.NumericUpDown numHtCam8;
        private System.Windows.Forms.Panel panelCam0;
        private System.Windows.Forms.Panel panelCam1;
        private System.Windows.Forms.Panel panelCam2;
        private System.Windows.Forms.Panel panelCam3;
        private System.Windows.Forms.Panel panelCam4;
        private System.Windows.Forms.Panel panelCam5;
        private System.Windows.Forms.Panel panelCam6;
        private System.Windows.Forms.Panel panelCam7;
    }
}
