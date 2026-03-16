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
            this.tabMain = new System.Windows.Forms.TabControl();
            this.tabPageLiveView = new System.Windows.Forms.TabPage();
            this.checkBoxEnableImageProcessing = new System.Windows.Forms.CheckBox();
            this.btnCameraFree = new System.Windows.Forms.Button();
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
            this.label_sec = new System.Windows.Forms.Label();
            this.label_min = new System.Windows.Forms.Label();
            this.label_hr = new System.Windows.Forms.Label();
            this.label_day = new System.Windows.Forms.Label();
            this.label_mon = new System.Windows.Forms.Label();
            this.label_yr = new System.Windows.Forms.Label();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.canvasMain = new AOI.SDK.UI.SmartCanvas();
            this.chartMura = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.btnSelectFolder = new System.Windows.Forms.Button();
            this.btnShowProcessed = new System.Windows.Forms.Button();
            this.pbCam1 = new System.Windows.Forms.PictureBox();
            this.btnShowOriginal = new System.Windows.Forms.Button();
            this.pbCam2 = new System.Windows.Forms.PictureBox();
            this.cbSec = new System.Windows.Forms.ComboBox();
            this.pbCam3 = new System.Windows.Forms.PictureBox();
            this.cbMin = new System.Windows.Forms.ComboBox();
            this.pbCam4 = new System.Windows.Forms.PictureBox();
            this.cbHour = new System.Windows.Forms.ComboBox();
            this.pbCam5 = new System.Windows.Forms.PictureBox();
            this.cbDay = new System.Windows.Forms.ComboBox();
            this.pbCam6 = new System.Windows.Forms.PictureBox();
            this.cbMonth = new System.Windows.Forms.ComboBox();
            this.pbCam7 = new System.Windows.Forms.PictureBox();
            this.cbYear = new System.Windows.Forms.ComboBox();
            this.btnLastPeriod = new System.Windows.Forms.Button();
            this.btnNextPeriod = new System.Windows.Forms.Button();
            this.tabPageData = new System.Windows.Forms.TabPage();
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.lblPixelInfo = new System.Windows.Forms.ToolStripStatusLabel();
            this.tabControlRight = new System.Windows.Forms.TabControl();
            this.tabPageInspSettings = new System.Windows.Forms.TabPage();
            this.tabPageCamera = new System.Windows.Forms.TabPage();
            this.tabControlCamTabs = new System.Windows.Forms.TabControl();
            this.tabPageExposure = new System.Windows.Forms.TabPage();
            this.panel14 = new System.Windows.Forms.Panel();
            this.label13 = new System.Windows.Forms.Label();
            this.trackBarExpCam7 = new System.Windows.Forms.TrackBar();
            this.numExpCam7 = new System.Windows.Forms.NumericUpDown();
            this.label14 = new System.Windows.Forms.Label();
            this.panel13 = new System.Windows.Forms.Panel();
            this.label11 = new System.Windows.Forms.Label();
            this.trackBarExpCam6 = new System.Windows.Forms.TrackBar();
            this.numExpCam6 = new System.Windows.Forms.NumericUpDown();
            this.label12 = new System.Windows.Forms.Label();
            this.panel12 = new System.Windows.Forms.Panel();
            this.label9 = new System.Windows.Forms.Label();
            this.trackBarExpCam5 = new System.Windows.Forms.TrackBar();
            this.numExpCam5 = new System.Windows.Forms.NumericUpDown();
            this.label10 = new System.Windows.Forms.Label();
            this.panel11 = new System.Windows.Forms.Panel();
            this.label7 = new System.Windows.Forms.Label();
            this.trackBarExpCam4 = new System.Windows.Forms.TrackBar();
            this.numExpCam4 = new System.Windows.Forms.NumericUpDown();
            this.label8 = new System.Windows.Forms.Label();
            this.panel10 = new System.Windows.Forms.Panel();
            this.label5 = new System.Windows.Forms.Label();
            this.trackBarExpCam3 = new System.Windows.Forms.TrackBar();
            this.numExpCam3 = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.panel9 = new System.Windows.Forms.Panel();
            this.label3 = new System.Windows.Forms.Label();
            this.trackBarExpCam2 = new System.Windows.Forms.TrackBar();
            this.numExpCam2 = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.panelExposure = new System.Windows.Forms.Panel();
            this.label2 = new System.Windows.Forms.Label();
            this.trackBarExpCam1 = new System.Windows.Forms.TrackBar();
            this.numExpCam1 = new System.Windows.Forms.NumericUpDown();
            this.lblExposure = new System.Windows.Forms.Label();
            this.tabPageLineRate = new System.Windows.Forms.TabPage();
            this.panel15 = new System.Windows.Forms.Panel();
            this.label15 = new System.Windows.Forms.Label();
            this.trackBarLrCam7 = new System.Windows.Forms.TrackBar();
            this.numLrCam7 = new System.Windows.Forms.NumericUpDown();
            this.label16 = new System.Windows.Forms.Label();
            this.panel16 = new System.Windows.Forms.Panel();
            this.label17 = new System.Windows.Forms.Label();
            this.trackBarLrCam6 = new System.Windows.Forms.TrackBar();
            this.numLrCam6 = new System.Windows.Forms.NumericUpDown();
            this.label18 = new System.Windows.Forms.Label();
            this.panel17 = new System.Windows.Forms.Panel();
            this.label19 = new System.Windows.Forms.Label();
            this.trackBarLrCam5 = new System.Windows.Forms.TrackBar();
            this.numLrCam5 = new System.Windows.Forms.NumericUpDown();
            this.label20 = new System.Windows.Forms.Label();
            this.panel18 = new System.Windows.Forms.Panel();
            this.label21 = new System.Windows.Forms.Label();
            this.trackBarLrCam4 = new System.Windows.Forms.TrackBar();
            this.numLrCam4 = new System.Windows.Forms.NumericUpDown();
            this.label22 = new System.Windows.Forms.Label();
            this.panel19 = new System.Windows.Forms.Panel();
            this.label23 = new System.Windows.Forms.Label();
            this.trackBarLrCam3 = new System.Windows.Forms.TrackBar();
            this.numLrCam3 = new System.Windows.Forms.NumericUpDown();
            this.label24 = new System.Windows.Forms.Label();
            this.panel20 = new System.Windows.Forms.Panel();
            this.label25 = new System.Windows.Forms.Label();
            this.trackBarLrCam2 = new System.Windows.Forms.TrackBar();
            this.numLrCam2 = new System.Windows.Forms.NumericUpDown();
            this.label26 = new System.Windows.Forms.Label();
            this.panelGrabHeight = new System.Windows.Forms.Panel();
            this.label27 = new System.Windows.Forms.Label();
            this.trackBarLrCam1 = new System.Windows.Forms.TrackBar();
            this.numLrCam1 = new System.Windows.Forms.NumericUpDown();
            this.lblGrabHeight = new System.Windows.Forms.Label();
            this.tabPageGrabHeight = new System.Windows.Forms.TabPage();
            this.panel21 = new System.Windows.Forms.Panel();
            this.label1 = new System.Windows.Forms.Label();
            this.trackBarHtCam7 = new System.Windows.Forms.TrackBar();
            this.numHtCam7 = new System.Windows.Forms.NumericUpDown();
            this.label28 = new System.Windows.Forms.Label();
            this.panel22 = new System.Windows.Forms.Panel();
            this.label29 = new System.Windows.Forms.Label();
            this.trackBarHtCam6 = new System.Windows.Forms.TrackBar();
            this.numHtCam6 = new System.Windows.Forms.NumericUpDown();
            this.label30 = new System.Windows.Forms.Label();
            this.panel23 = new System.Windows.Forms.Panel();
            this.label31 = new System.Windows.Forms.Label();
            this.trackBarHtCam5 = new System.Windows.Forms.TrackBar();
            this.numHtCam5 = new System.Windows.Forms.NumericUpDown();
            this.label32 = new System.Windows.Forms.Label();
            this.panel24 = new System.Windows.Forms.Panel();
            this.label33 = new System.Windows.Forms.Label();
            this.trackBarHtCam4 = new System.Windows.Forms.TrackBar();
            this.numHtCam4 = new System.Windows.Forms.NumericUpDown();
            this.label34 = new System.Windows.Forms.Label();
            this.panel25 = new System.Windows.Forms.Panel();
            this.label35 = new System.Windows.Forms.Label();
            this.trackBarHtCam3 = new System.Windows.Forms.TrackBar();
            this.numHtCam3 = new System.Windows.Forms.NumericUpDown();
            this.label36 = new System.Windows.Forms.Label();
            this.panel26 = new System.Windows.Forms.Panel();
            this.label37 = new System.Windows.Forms.Label();
            this.trackBarHtCam2 = new System.Windows.Forms.TrackBar();
            this.numHtCam2 = new System.Windows.Forms.NumericUpDown();
            this.label38 = new System.Windows.Forms.Label();
            this.panel27 = new System.Windows.Forms.Panel();
            this.label39 = new System.Windows.Forms.Label();
            this.trackBarHtCam1 = new System.Windows.Forms.TrackBar();
            this.numHtCam1 = new System.Windows.Forms.NumericUpDown();
            this.label40 = new System.Windows.Forms.Label();
            this.tabPageSystem = new System.Windows.Forms.TabPage();
            this.listViewEngine = new System.Windows.Forms.ListView();
            this.lblEngineConst = new System.Windows.Forms.Label();
            this.listViewCameras = new System.Windows.Forms.ListView();
            this.lblCamHardware = new System.Windows.Forms.Label();
            this.tabMain.SuspendLayout();
            this.tabPageLiveView.SuspendLayout();
            this.tabPageReview.SuspendLayout();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartMura)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).BeginInit();
            this.statusStrip1.SuspendLayout();
            this.tabControlRight.SuspendLayout();
            this.tabPageInspSettings.SuspendLayout();
            this.tabPageCamera.SuspendLayout();
            this.tabControlCamTabs.SuspendLayout();
            this.tabPageExposure.SuspendLayout();
            this.panel14.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).BeginInit();
            this.panel13.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).BeginInit();
            this.panel12.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).BeginInit();
            this.panel11.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).BeginInit();
            this.panel10.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).BeginInit();
            this.panel9.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).BeginInit();
            this.panelExposure.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).BeginInit();
            this.tabPageLineRate.SuspendLayout();
            this.panel15.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).BeginInit();
            this.panel16.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).BeginInit();
            this.panel17.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).BeginInit();
            this.panel18.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).BeginInit();
            this.panel19.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).BeginInit();
            this.panel20.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).BeginInit();
            this.panelGrabHeight.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).BeginInit();
            this.tabPageGrabHeight.SuspendLayout();
            this.panel21.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).BeginInit();
            this.panel22.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).BeginInit();
            this.panel23.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).BeginInit();
            this.panel24.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).BeginInit();
            this.panel25.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).BeginInit();
            this.panel26.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).BeginInit();
            this.panel27.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).BeginInit();
            this.tabPageSystem.SuspendLayout();
            this.SuspendLayout();
            // 
            // tabMain
            // 
            this.tabMain.Controls.Add(this.tabPageLiveView);
            this.tabMain.Controls.Add(this.tabPageReview);
            this.tabMain.Controls.Add(this.tabPageData);
            this.tabMain.Location = new System.Drawing.Point(12, 12);
            this.tabMain.Name = "tabMain";
            this.tabMain.SelectedIndex = 0;
            this.tabMain.Size = new System.Drawing.Size(1191, 674);
            this.tabMain.TabIndex = 1;
            // 
            // tabPageLiveView
            // 
            this.tabPageLiveView.Controls.Add(this.checkBoxEnableImageProcessing);
            this.tabPageLiveView.Controls.Add(this.btnCameraFree);
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
            this.tabPageLiveView.Size = new System.Drawing.Size(1183, 645);
            this.tabPageLiveView.TabIndex = 0;
            this.tabPageLiveView.Text = "即時監控";
            this.tabPageLiveView.UseVisualStyleBackColor = true;
            // 
            // checkBoxEnableImageProcessing
            // 
            this.checkBoxEnableImageProcessing.AutoSize = true;
            this.checkBoxEnableImageProcessing.Checked = false;
            this.checkBoxEnableImageProcessing.CheckState = System.Windows.Forms.CheckState.Unchecked;
            this.checkBoxEnableImageProcessing.Location = new System.Drawing.Point(1088, 123);
            this.checkBoxEnableImageProcessing.Name = "checkBoxEnableImageProcessing";
            this.checkBoxEnableImageProcessing.Size = new System.Drawing.Size(89, 19);
            this.checkBoxEnableImageProcessing.TabIndex = 8;
            this.checkBoxEnableImageProcessing.Text = "影像處理";
            this.checkBoxEnableImageProcessing.UseVisualStyleBackColor = true;
            this.checkBoxEnableImageProcessing.CheckedChanged += new System.EventHandler(this.checkBoxEnableImageProcessing_CheckedChanged);
            // 
            // btnCameraFree
            // 
            this.btnCameraFree.Location = new System.Drawing.Point(1084, 68);
            this.btnCameraFree.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnCameraFree.Name = "btnCameraFree";
            this.btnCameraFree.Size = new System.Drawing.Size(93, 49);
            this.btnCameraFree.TabIndex = 5;
            this.btnCameraFree.Text = "Free";
            this.btnCameraFree.UseVisualStyleBackColor = true;
            this.btnCameraFree.Click += new System.EventHandler(this.btnCameraFree_Click);
            // 
            // btnCameraGrab
            // 
            this.btnCameraGrab.Location = new System.Drawing.Point(1084, 7);
            this.btnCameraGrab.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnCameraGrab.Name = "btnCameraGrab";
            this.btnCameraGrab.Size = new System.Drawing.Size(93, 49);
            this.btnCameraGrab.TabIndex = 4;
            this.btnCameraGrab.Text = "Grab";
            this.btnCameraGrab.UseVisualStyleBackColor = true;
            this.btnCameraGrab.Click += new System.EventHandler(this.btnCameraGrab_Click);
            // 
            // panelMainDisplay
            // 
            this.panelMainDisplay.Location = new System.Drawing.Point(6, 123);
            this.panelMainDisplay.Name = "panelMainDisplay";
            this.panelMainDisplay.Size = new System.Drawing.Size(1072, 347);
            this.panelMainDisplay.TabIndex = 1;
            // 
            // panelLiveCam7
            // 
            this.panelLiveCam7.Location = new System.Drawing.Point(930, 6);
            this.panelLiveCam7.Name = "panelLiveCam7";
            this.panelLiveCam7.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam7.TabIndex = 1;
            // 
            // panelLiveCam6
            // 
            this.panelLiveCam6.Location = new System.Drawing.Point(776, 6);
            this.panelLiveCam6.Name = "panelLiveCam6";
            this.panelLiveCam6.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam6.TabIndex = 1;
            // 
            // panelLiveCam5
            // 
            this.panelLiveCam5.Location = new System.Drawing.Point(622, 6);
            this.panelLiveCam5.Name = "panelLiveCam5";
            this.panelLiveCam5.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam5.TabIndex = 1;
            // 
            // panelLiveCam4
            // 
            this.panelLiveCam4.Location = new System.Drawing.Point(468, 6);
            this.panelLiveCam4.Name = "panelLiveCam4";
            this.panelLiveCam4.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam4.TabIndex = 1;
            // 
            // panelLiveCam3
            // 
            this.panelLiveCam3.Location = new System.Drawing.Point(314, 6);
            this.panelLiveCam3.Name = "panelLiveCam3";
            this.panelLiveCam3.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam3.TabIndex = 1;
            // 
            // panelLiveCam2
            // 
            this.panelLiveCam2.Location = new System.Drawing.Point(160, 6);
            this.panelLiveCam2.Name = "panelLiveCam2";
            this.panelLiveCam2.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam2.TabIndex = 1;
            // 
            // panelLiveCam1
            // 
            this.panelLiveCam1.Location = new System.Drawing.Point(6, 6);
            this.panelLiveCam1.Name = "panelLiveCam1";
            this.panelLiveCam1.Size = new System.Drawing.Size(148, 111);
            this.panelLiveCam1.TabIndex = 0;
            // 
            // tabPageReview
            // 
            this.tabPageReview.Controls.Add(this.label_sec);
            this.tabPageReview.Controls.Add(this.label_min);
            this.tabPageReview.Controls.Add(this.label_hr);
            this.tabPageReview.Controls.Add(this.label_day);
            this.tabPageReview.Controls.Add(this.label_mon);
            this.tabPageReview.Controls.Add(this.label_yr);
            this.tabPageReview.Controls.Add(this.tableLayoutPanel1);
            this.tabPageReview.Controls.Add(this.btnSelectFolder);
            this.tabPageReview.Controls.Add(this.btnShowProcessed);
            this.tabPageReview.Controls.Add(this.pbCam1);
            this.tabPageReview.Controls.Add(this.btnShowOriginal);
            this.tabPageReview.Controls.Add(this.pbCam2);
            this.tabPageReview.Controls.Add(this.cbSec);
            this.tabPageReview.Controls.Add(this.pbCam3);
            this.tabPageReview.Controls.Add(this.cbMin);
            this.tabPageReview.Controls.Add(this.pbCam4);
            this.tabPageReview.Controls.Add(this.cbHour);
            this.tabPageReview.Controls.Add(this.pbCam5);
            this.tabPageReview.Controls.Add(this.cbDay);
            this.tabPageReview.Controls.Add(this.pbCam6);
            this.tabPageReview.Controls.Add(this.cbMonth);
            this.tabPageReview.Controls.Add(this.pbCam7);
            this.tabPageReview.Controls.Add(this.cbYear);
            this.tabPageReview.Controls.Add(this.btnLastPeriod);
            this.tabPageReview.Controls.Add(this.btnNextPeriod);
            this.tabPageReview.Location = new System.Drawing.Point(4, 25);
            this.tabPageReview.Name = "tabPageReview";
            this.tabPageReview.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageReview.Size = new System.Drawing.Size(1183, 645);
            this.tabPageReview.TabIndex = 1;
            this.tabPageReview.Text = "影像回顧";
            this.tabPageReview.UseVisualStyleBackColor = true;
            // 
            // label_sec
            // 
            this.label_sec.AutoSize = true;
            this.label_sec.Location = new System.Drawing.Point(1154, 325);
            this.label_sec.Name = "label_sec";
            this.label_sec.Size = new System.Drawing.Size(22, 15);
            this.label_sec.TabIndex = 29;
            this.label_sec.Text = "秒";
            // 
            // label_min
            // 
            this.label_min.AutoSize = true;
            this.label_min.Location = new System.Drawing.Point(1154, 293);
            this.label_min.Name = "label_min";
            this.label_min.Size = new System.Drawing.Size(22, 15);
            this.label_min.TabIndex = 28;
            this.label_min.Text = "分";
            // 
            // label_hr
            // 
            this.label_hr.AutoSize = true;
            this.label_hr.Location = new System.Drawing.Point(1154, 264);
            this.label_hr.Name = "label_hr";
            this.label_hr.Size = new System.Drawing.Size(22, 15);
            this.label_hr.TabIndex = 27;
            this.label_hr.Text = "時";
            // 
            // label_day
            // 
            this.label_day.AutoSize = true;
            this.label_day.Location = new System.Drawing.Point(1154, 220);
            this.label_day.Name = "label_day";
            this.label_day.Size = new System.Drawing.Size(22, 15);
            this.label_day.TabIndex = 26;
            this.label_day.Text = "日";
            // 
            // label_mon
            // 
            this.label_mon.AutoSize = true;
            this.label_mon.Location = new System.Drawing.Point(1154, 188);
            this.label_mon.Name = "label_mon";
            this.label_mon.Size = new System.Drawing.Size(22, 15);
            this.label_mon.TabIndex = 25;
            this.label_mon.Text = "月";
            // 
            // label_yr
            // 
            this.label_yr.AutoSize = true;
            this.label_yr.Location = new System.Drawing.Point(1154, 162);
            this.label_yr.Name = "label_yr";
            this.label_yr.Size = new System.Drawing.Size(22, 15);
            this.label_yr.TabIndex = 24;
            this.label_yr.Text = "年";
            // 
            // tableLayoutPanel1
            // 
            this.tableLayoutPanel1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.tableLayoutPanel1.ColumnCount = 1;
            this.tableLayoutPanel1.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tableLayoutPanel1.Controls.Add(this.canvasMain, 0, 0);
            this.tableLayoutPanel1.Controls.Add(this.chartMura, 0, 1);
            this.tableLayoutPanel1.Location = new System.Drawing.Point(8, 123);
            this.tableLayoutPanel1.Name = "tableLayoutPanel1";
            this.tableLayoutPanel1.RowCount = 2;
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 70F));
            this.tableLayoutPanel1.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 30F));
            this.tableLayoutPanel1.Size = new System.Drawing.Size(1070, 495);
            this.tableLayoutPanel1.TabIndex = 17;
            // 
            // canvasMain
            // 
            this.canvasMain.BackColor = System.Drawing.Color.Black;
            this.canvasMain.Cursor = System.Windows.Forms.Cursors.Cross;
            this.canvasMain.Dock = System.Windows.Forms.DockStyle.Fill;
            this.canvasMain.Location = new System.Drawing.Point(3, 3);
            this.canvasMain.Name = "canvasMain";
            this.canvasMain.Size = new System.Drawing.Size(1064, 340);
            this.canvasMain.TabIndex = 7;
            this.canvasMain.TabStop = false;
            // 
            // chartMura
            // 
            chartArea1.Name = "ChartArea1";
            this.chartMura.ChartAreas.Add(chartArea1);
            this.chartMura.Dock = System.Windows.Forms.DockStyle.Fill;
            legend1.Name = "Legend1";
            this.chartMura.Legends.Add(legend1);
            this.chartMura.Location = new System.Drawing.Point(3, 349);
            this.chartMura.Name = "chartMura";
            series1.ChartArea = "ChartArea1";
            series1.Legend = "Legend1";
            series1.Name = "Series1";
            this.chartMura.Series.Add(series1);
            this.chartMura.Size = new System.Drawing.Size(1064, 143);
            this.chartMura.TabIndex = 16;
            this.chartMura.Text = "chart1";
            // 
            // btnSelectFolder
            // 
            this.btnSelectFolder.Font = new System.Drawing.Font("新細明體", 8F);
            this.btnSelectFolder.Location = new System.Drawing.Point(1084, 6);
            this.btnSelectFolder.Name = "btnSelectFolder";
            this.btnSelectFolder.Size = new System.Drawing.Size(92, 40);
            this.btnSelectFolder.TabIndex = 23;
            this.btnSelectFolder.Text = "讀取資料夾";
            this.btnSelectFolder.UseVisualStyleBackColor = true;
            this.btnSelectFolder.Click += new System.EventHandler(this.btnSelectFolder_Click);
            // 
            // btnShowProcessed
            // 
            this.btnShowProcessed.Font = new System.Drawing.Font("新細明體", 8F);
            this.btnShowProcessed.Location = new System.Drawing.Point(1084, 98);
            this.btnShowProcessed.Name = "btnShowProcessed";
            this.btnShowProcessed.Size = new System.Drawing.Size(92, 40);
            this.btnShowProcessed.TabIndex = 22;
            this.btnShowProcessed.Text = "計算mura";
            this.btnShowProcessed.UseVisualStyleBackColor = true;
            this.btnShowProcessed.Click += new System.EventHandler(this.btnShowProcessed_Click);
            // 
            // pbCam1
            // 
            this.pbCam1.Location = new System.Drawing.Point(6, 6);
            this.pbCam1.Name = "pbCam1";
            this.pbCam1.Size = new System.Drawing.Size(148, 111);
            this.pbCam1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam1.TabIndex = 8;
            this.pbCam1.TabStop = false;
            // 
            // btnShowOriginal
            // 
            this.btnShowOriginal.Font = new System.Drawing.Font("新細明體", 8F);
            this.btnShowOriginal.Location = new System.Drawing.Point(1084, 52);
            this.btnShowOriginal.Name = "btnShowOriginal";
            this.btnShowOriginal.Size = new System.Drawing.Size(92, 40);
            this.btnShowOriginal.TabIndex = 21;
            this.btnShowOriginal.Text = "顯示原圖";
            this.btnShowOriginal.UseVisualStyleBackColor = true;
            this.btnShowOriginal.Click += new System.EventHandler(this.btnShowOriginal_Click);
            // 
            // pbCam2
            // 
            this.pbCam2.Location = new System.Drawing.Point(160, 6);
            this.pbCam2.Name = "pbCam2";
            this.pbCam2.Size = new System.Drawing.Size(148, 111);
            this.pbCam2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam2.TabIndex = 9;
            this.pbCam2.TabStop = false;
            // 
            // cbSec
            // 
            this.cbSec.FormattingEnabled = true;
            this.cbSec.Location = new System.Drawing.Point(1084, 322);
            this.cbSec.Name = "cbSec";
            this.cbSec.Size = new System.Drawing.Size(64, 23);
            this.cbSec.TabIndex = 20;
            // 
            // pbCam3
            // 
            this.pbCam3.Location = new System.Drawing.Point(314, 6);
            this.pbCam3.Name = "pbCam3";
            this.pbCam3.Size = new System.Drawing.Size(148, 111);
            this.pbCam3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam3.TabIndex = 10;
            this.pbCam3.TabStop = false;
            // 
            // cbMin
            // 
            this.cbMin.FormattingEnabled = true;
            this.cbMin.Location = new System.Drawing.Point(1084, 293);
            this.cbMin.Name = "cbMin";
            this.cbMin.Size = new System.Drawing.Size(64, 23);
            this.cbMin.TabIndex = 19;
            // 
            // pbCam4
            // 
            this.pbCam4.Location = new System.Drawing.Point(468, 6);
            this.pbCam4.Name = "pbCam4";
            this.pbCam4.Size = new System.Drawing.Size(148, 111);
            this.pbCam4.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam4.TabIndex = 11;
            this.pbCam4.TabStop = false;
            // 
            // cbHour
            // 
            this.cbHour.FormattingEnabled = true;
            this.cbHour.Location = new System.Drawing.Point(1084, 264);
            this.cbHour.Name = "cbHour";
            this.cbHour.Size = new System.Drawing.Size(64, 23);
            this.cbHour.TabIndex = 18;
            // 
            // pbCam5
            // 
            this.pbCam5.Location = new System.Drawing.Point(622, 6);
            this.pbCam5.Name = "pbCam5";
            this.pbCam5.Size = new System.Drawing.Size(148, 111);
            this.pbCam5.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam5.TabIndex = 12;
            this.pbCam5.TabStop = false;
            // 
            // cbDay
            // 
            this.cbDay.FormattingEnabled = true;
            this.cbDay.Location = new System.Drawing.Point(1084, 217);
            this.cbDay.Name = "cbDay";
            this.cbDay.Size = new System.Drawing.Size(64, 23);
            this.cbDay.TabIndex = 17;
            // 
            // pbCam6
            // 
            this.pbCam6.Location = new System.Drawing.Point(776, 6);
            this.pbCam6.Name = "pbCam6";
            this.pbCam6.Size = new System.Drawing.Size(148, 111);
            this.pbCam6.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam6.TabIndex = 13;
            this.pbCam6.TabStop = false;
            // 
            // cbMonth
            // 
            this.cbMonth.FormattingEnabled = true;
            this.cbMonth.Location = new System.Drawing.Point(1084, 188);
            this.cbMonth.Name = "cbMonth";
            this.cbMonth.Size = new System.Drawing.Size(64, 23);
            this.cbMonth.TabIndex = 16;
            // 
            // pbCam7
            // 
            this.pbCam7.Location = new System.Drawing.Point(930, 6);
            this.pbCam7.Name = "pbCam7";
            this.pbCam7.Size = new System.Drawing.Size(148, 111);
            this.pbCam7.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam7.TabIndex = 14;
            this.pbCam7.TabStop = false;
            // 
            // cbYear
            // 
            this.cbYear.FormattingEnabled = true;
            this.cbYear.Location = new System.Drawing.Point(1084, 159);
            this.cbYear.Name = "cbYear";
            this.cbYear.Size = new System.Drawing.Size(64, 23);
            this.cbYear.TabIndex = 15;
            // 
            // btnLastPeriod
            // 
            this.btnLastPeriod.Location = new System.Drawing.Point(1084, 360);
            this.btnLastPeriod.Name = "btnLastPeriod";
            this.btnLastPeriod.Size = new System.Drawing.Size(44, 28);
            this.btnLastPeriod.TabIndex = 30;
            this.btnLastPeriod.Text = "<";
            this.btnLastPeriod.UseVisualStyleBackColor = true;
            this.btnLastPeriod.Click += new System.EventHandler(this.btnLastPeriod_Click);
            // 
            // btnNextPeriod
            // 
            this.btnNextPeriod.Location = new System.Drawing.Point(1134, 360);
            this.btnNextPeriod.Name = "btnNextPeriod";
            this.btnNextPeriod.Size = new System.Drawing.Size(44, 28);
            this.btnNextPeriod.TabIndex = 31;
            this.btnNextPeriod.Text = ">";
            this.btnNextPeriod.UseVisualStyleBackColor = true;
            this.btnNextPeriod.Click += new System.EventHandler(this.btnNextPeriod_Click);
            // 
            // tabPageData
            // 
            this.tabPageData.Location = new System.Drawing.Point(4, 25);
            this.tabPageData.Name = "tabPageData";
            this.tabPageData.Size = new System.Drawing.Size(1183, 645);
            this.tabPageData.TabIndex = 2;
            this.tabPageData.Text = "檢測數據";
            this.tabPageData.UseVisualStyleBackColor = true;
            // 
            // propertyGrid1
            // 
            this.propertyGrid1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.propertyGrid1.Location = new System.Drawing.Point(3, 3);
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.Size = new System.Drawing.Size(262, 644);
            this.propertyGrid1.TabIndex = 0;
            // 
            // statusStrip1
            // 
            this.statusStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.lblPixelInfo});
            this.statusStrip1.Location = new System.Drawing.Point(0, 689);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(1491, 25);
            this.statusStrip1.TabIndex = 15;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // lblPixelInfo
            // 
            this.lblPixelInfo.Name = "lblPixelInfo";
            this.lblPixelInfo.Size = new System.Drawing.Size(395, 19);
            this.lblPixelInfo.Text = "位置:0.00mm | 座標:(0, 0) | 亮度: 0  | 倍率:0.0x | 平移:(0, 0)";
            // 
            // tabControlRight
            // 
            this.tabControlRight.Controls.Add(this.tabPageInspSettings);
            this.tabControlRight.Controls.Add(this.tabPageCamera);
            this.tabControlRight.Controls.Add(this.tabPageSystem);
            this.tabControlRight.Location = new System.Drawing.Point(1209, 12);
            this.tabControlRight.Multiline = true;
            this.tabControlRight.Name = "tabControlRight";
            this.tabControlRight.SelectedIndex = 0;
            this.tabControlRight.Size = new System.Drawing.Size(276, 679);
            this.tabControlRight.TabIndex = 16;
            // 
            // tabPageInspSettings
            // 
            this.tabPageInspSettings.Controls.Add(this.propertyGrid1);
            this.tabPageInspSettings.Location = new System.Drawing.Point(4, 25);
            this.tabPageInspSettings.Name = "tabPageInspSettings";
            this.tabPageInspSettings.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageInspSettings.Size = new System.Drawing.Size(268, 650);
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
            this.tabPageCamera.Size = new System.Drawing.Size(268, 650);
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
            this.tabControlCamTabs.Size = new System.Drawing.Size(262, 644);
            this.tabControlCamTabs.TabIndex = 2;
            // 
            // tabPageExposure
            // 
            this.tabPageExposure.Controls.Add(this.panel14);
            this.tabPageExposure.Controls.Add(this.panel13);
            this.tabPageExposure.Controls.Add(this.panel12);
            this.tabPageExposure.Controls.Add(this.panel11);
            this.tabPageExposure.Controls.Add(this.panel10);
            this.tabPageExposure.Controls.Add(this.panel9);
            this.tabPageExposure.Controls.Add(this.panelExposure);
            this.tabPageExposure.Location = new System.Drawing.Point(4, 25);
            this.tabPageExposure.Name = "tabPageExposure";
            this.tabPageExposure.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageExposure.Size = new System.Drawing.Size(254, 615);
            this.tabPageExposure.TabIndex = 0;
            this.tabPageExposure.Text = "曝光時間";
            this.tabPageExposure.UseVisualStyleBackColor = true;
            // 
            // panel14
            // 
            this.panel14.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel14.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel14.Controls.Add(this.label13);
            this.panel14.Controls.Add(this.trackBarExpCam7);
            this.panel14.Controls.Add(this.numExpCam7);
            this.panel14.Controls.Add(this.label14);
            this.panel14.Location = new System.Drawing.Point(0, 438);
            this.panel14.Name = "panel14";
            this.panel14.Size = new System.Drawing.Size(254, 69);
            this.panel14.TabIndex = 4;
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
            this.trackBarExpCam7.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam7.Size = new System.Drawing.Size(90, 25);
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
            // panel13
            // 
            this.panel13.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel13.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel13.Controls.Add(this.label11);
            this.panel13.Controls.Add(this.trackBarExpCam6);
            this.panel13.Controls.Add(this.numExpCam6);
            this.panel13.Controls.Add(this.label12);
            this.panel13.Location = new System.Drawing.Point(0, 365);
            this.panel13.Name = "panel13";
            this.panel13.Size = new System.Drawing.Size(254, 69);
            this.panel13.TabIndex = 4;
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
            this.trackBarExpCam6.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam6.Size = new System.Drawing.Size(90, 25);
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
            // panel12
            // 
            this.panel12.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel12.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel12.Controls.Add(this.label9);
            this.panel12.Controls.Add(this.trackBarExpCam5);
            this.panel12.Controls.Add(this.numExpCam5);
            this.panel12.Controls.Add(this.label10);
            this.panel12.Location = new System.Drawing.Point(0, 292);
            this.panel12.Name = "panel12";
            this.panel12.Size = new System.Drawing.Size(254, 69);
            this.panel12.TabIndex = 4;
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
            this.trackBarExpCam5.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam5.Size = new System.Drawing.Size(90, 25);
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
            // panel11
            // 
            this.panel11.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel11.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel11.Controls.Add(this.label7);
            this.panel11.Controls.Add(this.trackBarExpCam4);
            this.panel11.Controls.Add(this.numExpCam4);
            this.panel11.Controls.Add(this.label8);
            this.panel11.Location = new System.Drawing.Point(0, 219);
            this.panel11.Name = "panel11";
            this.panel11.Size = new System.Drawing.Size(254, 69);
            this.panel11.TabIndex = 4;
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
            this.trackBarExpCam4.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam4.Size = new System.Drawing.Size(90, 25);
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
            // panel10
            // 
            this.panel10.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel10.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel10.Controls.Add(this.label5);
            this.panel10.Controls.Add(this.trackBarExpCam3);
            this.panel10.Controls.Add(this.numExpCam3);
            this.panel10.Controls.Add(this.label6);
            this.panel10.Location = new System.Drawing.Point(0, 146);
            this.panel10.Name = "panel10";
            this.panel10.Size = new System.Drawing.Size(254, 69);
            this.panel10.TabIndex = 4;
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
            this.trackBarExpCam3.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam3.Size = new System.Drawing.Size(90, 25);
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
            // panel9
            // 
            this.panel9.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel9.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel9.Controls.Add(this.label3);
            this.panel9.Controls.Add(this.trackBarExpCam2);
            this.panel9.Controls.Add(this.numExpCam2);
            this.panel9.Controls.Add(this.label4);
            this.panel9.Location = new System.Drawing.Point(0, 73);
            this.panel9.Name = "panel9";
            this.panel9.Size = new System.Drawing.Size(254, 69);
            this.panel9.TabIndex = 4;
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
            this.trackBarExpCam2.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam2.Size = new System.Drawing.Size(90, 25);
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
            // panelExposure
            // 
            this.panelExposure.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelExposure.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelExposure.Controls.Add(this.label2);
            this.panelExposure.Controls.Add(this.trackBarExpCam1);
            this.panelExposure.Controls.Add(this.numExpCam1);
            this.panelExposure.Controls.Add(this.lblExposure);
            this.panelExposure.Location = new System.Drawing.Point(0, 0);
            this.panelExposure.Name = "panelExposure";
            this.panelExposure.Size = new System.Drawing.Size(254, 69);
            this.panelExposure.TabIndex = 0;
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
            this.trackBarExpCam1.Size = new System.Drawing.Size(248, 30);
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
            this.numExpCam1.Size = new System.Drawing.Size(90, 25);
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
            this.tabPageLineRate.Controls.Add(this.panel15);
            this.tabPageLineRate.Controls.Add(this.panel16);
            this.tabPageLineRate.Controls.Add(this.panel17);
            this.tabPageLineRate.Controls.Add(this.panel18);
            this.tabPageLineRate.Controls.Add(this.panel19);
            this.tabPageLineRate.Controls.Add(this.panel20);
            this.tabPageLineRate.Controls.Add(this.panelGrabHeight);
            this.tabPageLineRate.Location = new System.Drawing.Point(4, 25);
            this.tabPageLineRate.Name = "tabPageLineRate";
            this.tabPageLineRate.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageLineRate.Size = new System.Drawing.Size(254, 615);
            this.tabPageLineRate.TabIndex = 1;
            this.tabPageLineRate.Text = "線掃速率";
            this.tabPageLineRate.UseVisualStyleBackColor = true;
            // 
            // panel15
            // 
            this.panel15.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel15.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel15.Controls.Add(this.label15);
            this.panel15.Controls.Add(this.trackBarLrCam7);
            this.panel15.Controls.Add(this.numLrCam7);
            this.panel15.Controls.Add(this.label16);
            this.panel15.Location = new System.Drawing.Point(0, 438);
            this.panel15.Name = "panel15";
            this.panel15.Size = new System.Drawing.Size(254, 69);
            this.panel15.TabIndex = 6;
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
            this.trackBarLrCam7.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam7.Size = new System.Drawing.Size(90, 25);
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
            // panel16
            // 
            this.panel16.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel16.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel16.Controls.Add(this.label17);
            this.panel16.Controls.Add(this.trackBarLrCam6);
            this.panel16.Controls.Add(this.numLrCam6);
            this.panel16.Controls.Add(this.label18);
            this.panel16.Location = new System.Drawing.Point(0, 365);
            this.panel16.Name = "panel16";
            this.panel16.Size = new System.Drawing.Size(254, 69);
            this.panel16.TabIndex = 7;
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
            this.trackBarLrCam6.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam6.Size = new System.Drawing.Size(90, 25);
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
            // panel17
            // 
            this.panel17.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel17.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel17.Controls.Add(this.label19);
            this.panel17.Controls.Add(this.trackBarLrCam5);
            this.panel17.Controls.Add(this.numLrCam5);
            this.panel17.Controls.Add(this.label20);
            this.panel17.Location = new System.Drawing.Point(0, 292);
            this.panel17.Name = "panel17";
            this.panel17.Size = new System.Drawing.Size(254, 69);
            this.panel17.TabIndex = 8;
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
            this.trackBarLrCam5.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam5.Size = new System.Drawing.Size(90, 25);
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
            // panel18
            // 
            this.panel18.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel18.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel18.Controls.Add(this.label21);
            this.panel18.Controls.Add(this.trackBarLrCam4);
            this.panel18.Controls.Add(this.numLrCam4);
            this.panel18.Controls.Add(this.label22);
            this.panel18.Location = new System.Drawing.Point(0, 219);
            this.panel18.Name = "panel18";
            this.panel18.Size = new System.Drawing.Size(254, 69);
            this.panel18.TabIndex = 9;
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
            this.trackBarLrCam4.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam4.Size = new System.Drawing.Size(90, 25);
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
            // panel19
            // 
            this.panel19.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel19.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel19.Controls.Add(this.label23);
            this.panel19.Controls.Add(this.trackBarLrCam3);
            this.panel19.Controls.Add(this.numLrCam3);
            this.panel19.Controls.Add(this.label24);
            this.panel19.Location = new System.Drawing.Point(0, 146);
            this.panel19.Name = "panel19";
            this.panel19.Size = new System.Drawing.Size(254, 69);
            this.panel19.TabIndex = 10;
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
            this.trackBarLrCam3.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam3.Size = new System.Drawing.Size(90, 25);
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
            // panel20
            // 
            this.panel20.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel20.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel20.Controls.Add(this.label25);
            this.panel20.Controls.Add(this.trackBarLrCam2);
            this.panel20.Controls.Add(this.numLrCam2);
            this.panel20.Controls.Add(this.label26);
            this.panel20.Location = new System.Drawing.Point(0, 73);
            this.panel20.Name = "panel20";
            this.panel20.Size = new System.Drawing.Size(254, 69);
            this.panel20.TabIndex = 11;
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
            this.trackBarLrCam2.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam2.Size = new System.Drawing.Size(90, 25);
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
            // panelGrabHeight
            // 
            this.panelGrabHeight.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panelGrabHeight.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panelGrabHeight.Controls.Add(this.label27);
            this.panelGrabHeight.Controls.Add(this.trackBarLrCam1);
            this.panelGrabHeight.Controls.Add(this.numLrCam1);
            this.panelGrabHeight.Controls.Add(this.lblGrabHeight);
            this.panelGrabHeight.Location = new System.Drawing.Point(0, 0);
            this.panelGrabHeight.Name = "panelGrabHeight";
            this.panelGrabHeight.Size = new System.Drawing.Size(254, 69);
            this.panelGrabHeight.TabIndex = 5;
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
            this.trackBarLrCam1.Size = new System.Drawing.Size(248, 30);
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
            this.numLrCam1.Size = new System.Drawing.Size(90, 25);
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
            this.tabPageGrabHeight.Controls.Add(this.panel21);
            this.tabPageGrabHeight.Controls.Add(this.panel22);
            this.tabPageGrabHeight.Controls.Add(this.panel23);
            this.tabPageGrabHeight.Controls.Add(this.panel24);
            this.tabPageGrabHeight.Controls.Add(this.panel25);
            this.tabPageGrabHeight.Controls.Add(this.panel26);
            this.tabPageGrabHeight.Controls.Add(this.panel27);
            this.tabPageGrabHeight.Location = new System.Drawing.Point(4, 25);
            this.tabPageGrabHeight.Name = "tabPageGrabHeight";
            this.tabPageGrabHeight.Padding = new System.Windows.Forms.Padding(3);
            this.tabPageGrabHeight.Size = new System.Drawing.Size(254, 615);
            this.tabPageGrabHeight.TabIndex = 2;
            this.tabPageGrabHeight.Text = "擷取高度";
            this.tabPageGrabHeight.UseVisualStyleBackColor = true;
            // 
            // panel21
            // 
            this.panel21.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel21.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel21.Controls.Add(this.label1);
            this.panel21.Controls.Add(this.trackBarHtCam7);
            this.panel21.Controls.Add(this.numHtCam7);
            this.panel21.Controls.Add(this.label28);
            this.panel21.Location = new System.Drawing.Point(0, 438);
            this.panel21.Name = "panel21";
            this.panel21.Size = new System.Drawing.Size(254, 69);
            this.panel21.TabIndex = 6;
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
            this.trackBarHtCam7.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam7.Size = new System.Drawing.Size(90, 25);
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
            // panel22
            // 
            this.panel22.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel22.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel22.Controls.Add(this.label29);
            this.panel22.Controls.Add(this.trackBarHtCam6);
            this.panel22.Controls.Add(this.numHtCam6);
            this.panel22.Controls.Add(this.label30);
            this.panel22.Location = new System.Drawing.Point(0, 365);
            this.panel22.Name = "panel22";
            this.panel22.Size = new System.Drawing.Size(254, 69);
            this.panel22.TabIndex = 7;
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
            this.trackBarHtCam6.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam6.Size = new System.Drawing.Size(90, 25);
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
            // panel23
            // 
            this.panel23.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel23.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel23.Controls.Add(this.label31);
            this.panel23.Controls.Add(this.trackBarHtCam5);
            this.panel23.Controls.Add(this.numHtCam5);
            this.panel23.Controls.Add(this.label32);
            this.panel23.Location = new System.Drawing.Point(0, 292);
            this.panel23.Name = "panel23";
            this.panel23.Size = new System.Drawing.Size(254, 69);
            this.panel23.TabIndex = 8;
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
            this.trackBarHtCam5.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam5.Size = new System.Drawing.Size(90, 25);
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
            // panel24
            // 
            this.panel24.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel24.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel24.Controls.Add(this.label33);
            this.panel24.Controls.Add(this.trackBarHtCam4);
            this.panel24.Controls.Add(this.numHtCam4);
            this.panel24.Controls.Add(this.label34);
            this.panel24.Location = new System.Drawing.Point(0, 219);
            this.panel24.Name = "panel24";
            this.panel24.Size = new System.Drawing.Size(254, 69);
            this.panel24.TabIndex = 9;
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
            this.trackBarHtCam4.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam4.Size = new System.Drawing.Size(90, 25);
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
            // panel25
            // 
            this.panel25.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel25.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel25.Controls.Add(this.label35);
            this.panel25.Controls.Add(this.trackBarHtCam3);
            this.panel25.Controls.Add(this.numHtCam3);
            this.panel25.Controls.Add(this.label36);
            this.panel25.Location = new System.Drawing.Point(0, 146);
            this.panel25.Name = "panel25";
            this.panel25.Size = new System.Drawing.Size(254, 69);
            this.panel25.TabIndex = 10;
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
            this.trackBarHtCam3.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam3.Size = new System.Drawing.Size(90, 25);
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
            // panel26
            // 
            this.panel26.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel26.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel26.Controls.Add(this.label37);
            this.panel26.Controls.Add(this.trackBarHtCam2);
            this.panel26.Controls.Add(this.numHtCam2);
            this.panel26.Controls.Add(this.label38);
            this.panel26.Location = new System.Drawing.Point(0, 73);
            this.panel26.Name = "panel26";
            this.panel26.Size = new System.Drawing.Size(254, 69);
            this.panel26.TabIndex = 11;
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
            this.trackBarHtCam2.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam2.Size = new System.Drawing.Size(90, 25);
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
            // panel27
            // 
            this.panel27.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.panel27.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.panel27.Controls.Add(this.label39);
            this.panel27.Controls.Add(this.trackBarHtCam1);
            this.panel27.Controls.Add(this.numHtCam1);
            this.panel27.Controls.Add(this.label40);
            this.panel27.Location = new System.Drawing.Point(0, 0);
            this.panel27.Name = "panel27";
            this.panel27.Size = new System.Drawing.Size(254, 69);
            this.panel27.TabIndex = 5;
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
            this.trackBarHtCam1.Size = new System.Drawing.Size(248, 30);
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
            this.numHtCam1.Size = new System.Drawing.Size(90, 25);
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
            this.tabPageSystem.Controls.Add(this.listViewEngine);
            this.tabPageSystem.Controls.Add(this.lblEngineConst);
            this.tabPageSystem.Controls.Add(this.listViewCameras);
            this.tabPageSystem.Controls.Add(this.lblCamHardware);
            this.tabPageSystem.Location = new System.Drawing.Point(4, 25);
            this.tabPageSystem.Name = "tabPageSystem";
            this.tabPageSystem.Size = new System.Drawing.Size(268, 650);
            this.tabPageSystem.TabIndex = 2;
            this.tabPageSystem.Text = "系統資訊";
            this.tabPageSystem.UseVisualStyleBackColor = true;
            // 
            // listViewEngine
            // 
            this.listViewEngine.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.listViewEngine.FullRowSelect = true;
            this.listViewEngine.GridLines = true;
            this.listViewEngine.HideSelection = false;
            this.listViewEngine.Location = new System.Drawing.Point(3, 336);
            this.listViewEngine.Name = "listViewEngine";
            this.listViewEngine.Size = new System.Drawing.Size(256, 200);
            this.listViewEngine.TabIndex = 3;
            this.listViewEngine.UseCompatibleStateImageBehavior = false;
            this.listViewEngine.View = System.Windows.Forms.View.Details;
            // 
            // lblEngineConst
            // 
            this.lblEngineConst.AutoSize = true;
            this.lblEngineConst.Font = new System.Drawing.Font("新細明體", 9F, System.Drawing.FontStyle.Bold);
            this.lblEngineConst.Location = new System.Drawing.Point(3, 316);
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
            this.listViewCameras.Size = new System.Drawing.Size(256, 271);
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
            this.lblCamHardware.Size = new System.Drawing.Size(135, 15);
            this.lblCamHardware.TabIndex = 0;
            this.lblCamHardware.Text = "【相機硬體設定】";
            // 
            // AniloxRollForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1491, 714);
            this.Controls.Add(this.tabControlRight);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.tabMain);
            this.Name = "AniloxRollForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "AniloxRoll Monitor";
            this.tabMain.ResumeLayout(false);
            this.tabPageLiveView.ResumeLayout(false);
            this.tabPageLiveView.PerformLayout();
            this.tabPageReview.ResumeLayout(false);
            this.tabPageReview.PerformLayout();
            this.tableLayoutPanel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.chartMura)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).EndInit();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.tabControlRight.ResumeLayout(false);
            this.tabPageInspSettings.ResumeLayout(false);
            this.tabPageCamera.ResumeLayout(false);
            this.tabControlCamTabs.ResumeLayout(false);
            this.tabPageExposure.ResumeLayout(false);
            this.panel14.ResumeLayout(false);
            this.panel14.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam7)).EndInit();
            this.panel13.ResumeLayout(false);
            this.panel13.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam6)).EndInit();
            this.panel12.ResumeLayout(false);
            this.panel12.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam5)).EndInit();
            this.panel11.ResumeLayout(false);
            this.panel11.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam4)).EndInit();
            this.panel10.ResumeLayout(false);
            this.panel10.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam3)).EndInit();
            this.panel9.ResumeLayout(false);
            this.panel9.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam2)).EndInit();
            this.panelExposure.ResumeLayout(false);
            this.panelExposure.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarExpCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numExpCam1)).EndInit();
            this.tabPageLineRate.ResumeLayout(false);
            this.panel15.ResumeLayout(false);
            this.panel15.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam7)).EndInit();
            this.panel16.ResumeLayout(false);
            this.panel16.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam6)).EndInit();
            this.panel17.ResumeLayout(false);
            this.panel17.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam5)).EndInit();
            this.panel18.ResumeLayout(false);
            this.panel18.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam4)).EndInit();
            this.panel19.ResumeLayout(false);
            this.panel19.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam3)).EndInit();
            this.panel20.ResumeLayout(false);
            this.panel20.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam2)).EndInit();
            this.panelGrabHeight.ResumeLayout(false);
            this.panelGrabHeight.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarLrCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLrCam1)).EndInit();
            this.tabPageGrabHeight.ResumeLayout(false);
            this.panel21.ResumeLayout(false);
            this.panel21.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam7)).EndInit();
            this.panel22.ResumeLayout(false);
            this.panel22.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam6)).EndInit();
            this.panel23.ResumeLayout(false);
            this.panel23.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam5)).EndInit();
            this.panel24.ResumeLayout(false);
            this.panel24.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam4)).EndInit();
            this.panel25.ResumeLayout(false);
            this.panel25.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam3)).EndInit();
            this.panel26.ResumeLayout(false);
            this.panel26.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam2)).EndInit();
            this.panel27.ResumeLayout(false);
            this.panel27.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarHtCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numHtCam1)).EndInit();
            this.tabPageSystem.ResumeLayout(false);
            this.tabPageSystem.PerformLayout();
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
        private System.Windows.Forms.ComboBox cbSec;
        private System.Windows.Forms.ComboBox cbMin;
        private System.Windows.Forms.ComboBox cbHour;
        private System.Windows.Forms.ComboBox cbDay;
        private System.Windows.Forms.ComboBox cbMonth;
        private System.Windows.Forms.ComboBox cbYear;
        private System.Windows.Forms.Button btnShowProcessed;
        private System.Windows.Forms.Button btnShowOriginal;
        private System.Windows.Forms.Button btnSelectFolder;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel lblPixelInfo;
        private System.Windows.Forms.DataVisualization.Charting.Chart chartMura;
        private System.Windows.Forms.TableLayoutPanel tableLayoutPanel1;
        private System.Windows.Forms.PropertyGrid propertyGrid1;
        private System.Windows.Forms.Label label_sec;
        private System.Windows.Forms.Label label_min;
        private System.Windows.Forms.Label label_hr;
        private System.Windows.Forms.Label label_day;
        private System.Windows.Forms.Label label_mon;
        private System.Windows.Forms.Label label_yr;
        private System.Windows.Forms.Panel panelLiveCam1;
        private System.Windows.Forms.Panel panelMainDisplay;
        private System.Windows.Forms.Panel panelLiveCam7;
        private System.Windows.Forms.Panel panelLiveCam6;
        private System.Windows.Forms.Panel panelLiveCam5;
        private System.Windows.Forms.Panel panelLiveCam4;
        private System.Windows.Forms.Panel panelLiveCam3;
        private System.Windows.Forms.Panel panelLiveCam2;
        private System.Windows.Forms.Button btnCameraFree;
        private System.Windows.Forms.Button btnCameraGrab;
        private System.Windows.Forms.CheckBox checkBoxEnableImageProcessing;
        private System.Windows.Forms.Button btnLastPeriod;
        private System.Windows.Forms.Button btnNextPeriod;
        private System.Windows.Forms.TabControl tabControlRight;
        private System.Windows.Forms.TabPage tabPageInspSettings;
        private System.Windows.Forms.TabPage tabPageCamera;
        private System.Windows.Forms.TabPage tabPageSystem;
        private System.Windows.Forms.Panel panelExposure;
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
        private System.Windows.Forms.Panel panel14;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.TrackBar trackBarExpCam7;
        private System.Windows.Forms.NumericUpDown numExpCam7;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.Panel panel13;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TrackBar trackBarExpCam6;
        private System.Windows.Forms.NumericUpDown numExpCam6;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.Panel panel12;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TrackBar trackBarExpCam5;
        private System.Windows.Forms.NumericUpDown numExpCam5;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Panel panel11;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TrackBar trackBarExpCam4;
        private System.Windows.Forms.NumericUpDown numExpCam4;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Panel panel10;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TrackBar trackBarExpCam3;
        private System.Windows.Forms.NumericUpDown numExpCam3;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Panel panel9;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TrackBar trackBarExpCam2;
        private System.Windows.Forms.NumericUpDown numExpCam2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Panel panel15;
        private System.Windows.Forms.Label label15;
        private System.Windows.Forms.TrackBar trackBarLrCam7;
        private System.Windows.Forms.NumericUpDown numLrCam7;
        private System.Windows.Forms.Label label16;
        private System.Windows.Forms.Panel panel16;
        private System.Windows.Forms.Label label17;
        private System.Windows.Forms.TrackBar trackBarLrCam6;
        private System.Windows.Forms.NumericUpDown numLrCam6;
        private System.Windows.Forms.Label label18;
        private System.Windows.Forms.Panel panel17;
        private System.Windows.Forms.Label label19;
        private System.Windows.Forms.TrackBar trackBarLrCam5;
        private System.Windows.Forms.NumericUpDown numLrCam5;
        private System.Windows.Forms.Label label20;
        private System.Windows.Forms.Panel panel18;
        private System.Windows.Forms.Label label21;
        private System.Windows.Forms.TrackBar trackBarLrCam4;
        private System.Windows.Forms.NumericUpDown numLrCam4;
        private System.Windows.Forms.Label label22;
        private System.Windows.Forms.Panel panel19;
        private System.Windows.Forms.Label label23;
        private System.Windows.Forms.TrackBar trackBarLrCam3;
        private System.Windows.Forms.NumericUpDown numLrCam3;
        private System.Windows.Forms.Label label24;
        private System.Windows.Forms.Panel panel20;
        private System.Windows.Forms.Label label25;
        private System.Windows.Forms.TrackBar trackBarLrCam2;
        private System.Windows.Forms.NumericUpDown numLrCam2;
        private System.Windows.Forms.Label label26;
        private System.Windows.Forms.Panel panelGrabHeight;
        private System.Windows.Forms.Label label27;
        private System.Windows.Forms.TrackBar trackBarLrCam1;
        private System.Windows.Forms.NumericUpDown numLrCam1;
        private System.Windows.Forms.Label lblGrabHeight;
        private System.Windows.Forms.Panel panel21;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar trackBarHtCam7;
        private System.Windows.Forms.NumericUpDown numHtCam7;
        private System.Windows.Forms.Label label28;
        private System.Windows.Forms.Panel panel22;
        private System.Windows.Forms.Label label29;
        private System.Windows.Forms.TrackBar trackBarHtCam6;
        private System.Windows.Forms.NumericUpDown numHtCam6;
        private System.Windows.Forms.Label label30;
        private System.Windows.Forms.Panel panel23;
        private System.Windows.Forms.Label label31;
        private System.Windows.Forms.TrackBar trackBarHtCam5;
        private System.Windows.Forms.NumericUpDown numHtCam5;
        private System.Windows.Forms.Label label32;
        private System.Windows.Forms.Panel panel24;
        private System.Windows.Forms.Label label33;
        private System.Windows.Forms.TrackBar trackBarHtCam4;
        private System.Windows.Forms.NumericUpDown numHtCam4;
        private System.Windows.Forms.Label label34;
        private System.Windows.Forms.Panel panel25;
        private System.Windows.Forms.Label label35;
        private System.Windows.Forms.TrackBar trackBarHtCam3;
        private System.Windows.Forms.NumericUpDown numHtCam3;
        private System.Windows.Forms.Label label36;
        private System.Windows.Forms.Panel panel26;
        private System.Windows.Forms.Label label37;
        private System.Windows.Forms.TrackBar trackBarHtCam2;
        private System.Windows.Forms.NumericUpDown numHtCam2;
        private System.Windows.Forms.Label label38;
        private System.Windows.Forms.Panel panel27;
        private System.Windows.Forms.Label label39;
        private System.Windows.Forms.TrackBar trackBarHtCam1;
        private System.Windows.Forms.NumericUpDown numHtCam1;
        private System.Windows.Forms.Label label40;
        private System.Windows.Forms.TabPage tabPageData;
    }
}