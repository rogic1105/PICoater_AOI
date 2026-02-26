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
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea2 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend2 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series2 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.tabControl = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.checkBoxEnableImageProcessing = new System.Windows.Forms.CheckBox();
            this.btnCameraFree = new System.Windows.Forms.Button();
            this.btnCameraGrab = new System.Windows.Forms.Button();
            this.panel8 = new System.Windows.Forms.Panel();
            this.panel7 = new System.Windows.Forms.Panel();
            this.panel6 = new System.Windows.Forms.Panel();
            this.panel5 = new System.Windows.Forms.Panel();
            this.panel4 = new System.Windows.Forms.Panel();
            this.panel3 = new System.Windows.Forms.Panel();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel1 = new System.Windows.Forms.Panel();
            this.tabPage2 = new System.Windows.Forms.TabPage();
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
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.lblPixelInfo = new System.Windows.Forms.ToolStripStatusLabel();
            this.tabControl.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
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
            this.SuspendLayout();
            // 
            // tabControl
            // 
            this.tabControl.Controls.Add(this.tabPage1);
            this.tabControl.Controls.Add(this.tabPage2);
            this.tabControl.Location = new System.Drawing.Point(12, 12);
            this.tabControl.Name = "tabControl";
            this.tabControl.SelectedIndex = 0;
            this.tabControl.Size = new System.Drawing.Size(1191, 674);
            this.tabControl.TabIndex = 1;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.checkBoxEnableImageProcessing);
            this.tabPage1.Controls.Add(this.btnCameraFree);
            this.tabPage1.Controls.Add(this.btnCameraGrab);
            this.tabPage1.Controls.Add(this.panel8);
            this.tabPage1.Controls.Add(this.panel7);
            this.tabPage1.Controls.Add(this.panel6);
            this.tabPage1.Controls.Add(this.panel5);
            this.tabPage1.Controls.Add(this.panel4);
            this.tabPage1.Controls.Add(this.panel3);
            this.tabPage1.Controls.Add(this.panel2);
            this.tabPage1.Controls.Add(this.panel1);
            this.tabPage1.Location = new System.Drawing.Point(4, 25);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(1183, 645);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "監控";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // checkBoxEnableImageProcessing
            // 
            this.checkBoxEnableImageProcessing.AutoSize = true;
            this.checkBoxEnableImageProcessing.Checked = true;
            this.checkBoxEnableImageProcessing.CheckState = System.Windows.Forms.CheckState.Checked;
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
            // panel8
            // 
            this.panel8.Location = new System.Drawing.Point(6, 123);
            this.panel8.Name = "panel8";
            this.panel8.Size = new System.Drawing.Size(1072, 347);
            this.panel8.TabIndex = 1;
            // 
            // panel7
            // 
            this.panel7.Location = new System.Drawing.Point(930, 6);
            this.panel7.Name = "panel7";
            this.panel7.Size = new System.Drawing.Size(148, 111);
            this.panel7.TabIndex = 1;
            // 
            // panel6
            // 
            this.panel6.Location = new System.Drawing.Point(776, 6);
            this.panel6.Name = "panel6";
            this.panel6.Size = new System.Drawing.Size(148, 111);
            this.panel6.TabIndex = 1;
            // 
            // panel5
            // 
            this.panel5.Location = new System.Drawing.Point(622, 6);
            this.panel5.Name = "panel5";
            this.panel5.Size = new System.Drawing.Size(148, 111);
            this.panel5.TabIndex = 1;
            // 
            // panel4
            // 
            this.panel4.Location = new System.Drawing.Point(468, 6);
            this.panel4.Name = "panel4";
            this.panel4.Size = new System.Drawing.Size(148, 111);
            this.panel4.TabIndex = 1;
            // 
            // panel3
            // 
            this.panel3.Location = new System.Drawing.Point(314, 6);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(148, 111);
            this.panel3.TabIndex = 1;
            // 
            // panel2
            // 
            this.panel2.Location = new System.Drawing.Point(160, 6);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(148, 111);
            this.panel2.TabIndex = 1;
            // 
            // panel1
            // 
            this.panel1.Location = new System.Drawing.Point(6, 6);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(148, 111);
            this.panel1.TabIndex = 0;
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.label_sec);
            this.tabPage2.Controls.Add(this.label_min);
            this.tabPage2.Controls.Add(this.label_hr);
            this.tabPage2.Controls.Add(this.label_day);
            this.tabPage2.Controls.Add(this.label_mon);
            this.tabPage2.Controls.Add(this.label_yr);
            this.tabPage2.Controls.Add(this.tableLayoutPanel1);
            this.tabPage2.Controls.Add(this.btnSelectFolder);
            this.tabPage2.Controls.Add(this.btnShowProcessed);
            this.tabPage2.Controls.Add(this.pbCam1);
            this.tabPage2.Controls.Add(this.btnShowOriginal);
            this.tabPage2.Controls.Add(this.pbCam2);
            this.tabPage2.Controls.Add(this.cbSec);
            this.tabPage2.Controls.Add(this.pbCam3);
            this.tabPage2.Controls.Add(this.cbMin);
            this.tabPage2.Controls.Add(this.pbCam4);
            this.tabPage2.Controls.Add(this.cbHour);
            this.tabPage2.Controls.Add(this.pbCam5);
            this.tabPage2.Controls.Add(this.cbDay);
            this.tabPage2.Controls.Add(this.pbCam6);
            this.tabPage2.Controls.Add(this.cbMonth);
            this.tabPage2.Controls.Add(this.pbCam7);
            this.tabPage2.Controls.Add(this.cbYear);
            this.tabPage2.Controls.Add(this.btnLastPeriod);
            this.tabPage2.Controls.Add(this.btnNextPeriod);
            this.tabPage2.Location = new System.Drawing.Point(4, 25);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(1183, 645);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "回顧";
            this.tabPage2.UseVisualStyleBackColor = true;
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
            chartArea2.Name = "ChartArea1";
            this.chartMura.ChartAreas.Add(chartArea2);
            this.chartMura.Dock = System.Windows.Forms.DockStyle.Fill;
            legend2.Name = "Legend1";
            this.chartMura.Legends.Add(legend2);
            this.chartMura.Location = new System.Drawing.Point(3, 349);
            this.chartMura.Name = "chartMura";
            series2.ChartArea = "ChartArea1";
            series2.Legend = "Legend1";
            series2.Name = "Series1";
            this.chartMura.Series.Add(series2);
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
            // propertyGrid1
            // 
            this.propertyGrid1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.propertyGrid1.Location = new System.Drawing.Point(1205, 37);
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.Size = new System.Drawing.Size(286, 645);
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
            // AniloxRollForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1491, 714);
            this.Controls.Add(this.propertyGrid1);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.tabControl);
            this.Name = "AniloxRollForm";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "AniloxRoll Monitor";
            this.tabControl.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
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
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion
        private AOI.SDK.UI.SmartCanvas canvasMain;
        private System.Windows.Forms.TabControl tabControl;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
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
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel8;
        private System.Windows.Forms.Panel panel7;
        private System.Windows.Forms.Panel panel6;
        private System.Windows.Forms.Panel panel5;
        private System.Windows.Forms.Panel panel4;
        private System.Windows.Forms.Panel panel3;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Button btnCameraFree;
        private System.Windows.Forms.Button btnCameraGrab;
        private System.Windows.Forms.CheckBox checkBoxEnableImageProcessing;
        private System.Windows.Forms.Button btnLastPeriod;
        private System.Windows.Forms.Button btnNextPeriod;
    }
}