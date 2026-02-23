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
            System.Windows.Forms.DataVisualization.Charting.ChartArea chartArea10 = new System.Windows.Forms.DataVisualization.Charting.ChartArea();
            System.Windows.Forms.DataVisualization.Charting.Legend legend10 = new System.Windows.Forms.DataVisualization.Charting.Legend();
            System.Windows.Forms.DataVisualization.Charting.Series series10 = new System.Windows.Forms.DataVisualization.Charting.Series();
            this.tabControl = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.btnSelectFolder = new System.Windows.Forms.Button();
            this.btnShowProcessed = new System.Windows.Forms.Button();
            this.btnShowOriginal = new System.Windows.Forms.Button();
            this.cbSec = new System.Windows.Forms.ComboBox();
            this.cbMin = new System.Windows.Forms.ComboBox();
            this.cbHour = new System.Windows.Forms.ComboBox();
            this.cbDay = new System.Windows.Forms.ComboBox();
            this.cbMonth = new System.Windows.Forms.ComboBox();
            this.cbYear = new System.Windows.Forms.ComboBox();
            this.propertyGrid1 = new System.Windows.Forms.PropertyGrid();
            this.pbCam7 = new System.Windows.Forms.PictureBox();
            this.pbCam6 = new System.Windows.Forms.PictureBox();
            this.pbCam5 = new System.Windows.Forms.PictureBox();
            this.pbCam4 = new System.Windows.Forms.PictureBox();
            this.pbCam3 = new System.Windows.Forms.PictureBox();
            this.pbCam2 = new System.Windows.Forms.PictureBox();
            this.pbCam1 = new System.Windows.Forms.PictureBox();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.lblPixelInfo = new System.Windows.Forms.ToolStripStatusLabel();
            this.chartMura = new System.Windows.Forms.DataVisualization.Charting.Chart();
            this.tableLayoutPanel1 = new System.Windows.Forms.TableLayoutPanel();
            this.canvasMain = new AOI.SDK.UI.SmartCanvas();
            this.label_yr = new System.Windows.Forms.Label();
            this.label_mon = new System.Windows.Forms.Label();
            this.label_day = new System.Windows.Forms.Label();
            this.label_hr = new System.Windows.Forms.Label();
            this.label_min = new System.Windows.Forms.Label();
            this.label_sec = new System.Windows.Forms.Label();
            this.panel1 = new System.Windows.Forms.Panel();
            this.panel2 = new System.Windows.Forms.Panel();
            this.panel3 = new System.Windows.Forms.Panel();
            this.panel4 = new System.Windows.Forms.Panel();
            this.panel5 = new System.Windows.Forms.Panel();
            this.panel6 = new System.Windows.Forms.Panel();
            this.panel7 = new System.Windows.Forms.Panel();
            this.panel8 = new System.Windows.Forms.Panel();
            this.button1 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.label7 = new System.Windows.Forms.Label();
            this.tabControl.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).BeginInit();
            this.statusStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMura)).BeginInit();
            this.tableLayoutPanel1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).BeginInit();
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
            this.tabPage1.Controls.Add(this.label7);
            this.tabPage1.Controls.Add(this.label6);
            this.tabPage1.Controls.Add(this.label5);
            this.tabPage1.Controls.Add(this.label4);
            this.tabPage1.Controls.Add(this.label3);
            this.tabPage1.Controls.Add(this.label2);
            this.tabPage1.Controls.Add(this.label1);
            this.tabPage1.Controls.Add(this.button3);
            this.tabPage1.Controls.Add(this.button2);
            this.tabPage1.Controls.Add(this.button1);
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
            this.tabPage2.Location = new System.Drawing.Point(4, 25);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(1183, 645);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "回顧";
            this.tabPage2.UseVisualStyleBackColor = true;
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
            // cbSec
            // 
            this.cbSec.FormattingEnabled = true;
            this.cbSec.Location = new System.Drawing.Point(1084, 322);
            this.cbSec.Name = "cbSec";
            this.cbSec.Size = new System.Drawing.Size(64, 23);
            this.cbSec.TabIndex = 20;
            // 
            // cbMin
            // 
            this.cbMin.FormattingEnabled = true;
            this.cbMin.Location = new System.Drawing.Point(1084, 293);
            this.cbMin.Name = "cbMin";
            this.cbMin.Size = new System.Drawing.Size(64, 23);
            this.cbMin.TabIndex = 19;
            // 
            // cbHour
            // 
            this.cbHour.FormattingEnabled = true;
            this.cbHour.Location = new System.Drawing.Point(1084, 264);
            this.cbHour.Name = "cbHour";
            this.cbHour.Size = new System.Drawing.Size(64, 23);
            this.cbHour.TabIndex = 18;
            // 
            // cbDay
            // 
            this.cbDay.FormattingEnabled = true;
            this.cbDay.Location = new System.Drawing.Point(1084, 217);
            this.cbDay.Name = "cbDay";
            this.cbDay.Size = new System.Drawing.Size(64, 23);
            this.cbDay.TabIndex = 17;
            // 
            // cbMonth
            // 
            this.cbMonth.FormattingEnabled = true;
            this.cbMonth.Location = new System.Drawing.Point(1084, 188);
            this.cbMonth.Name = "cbMonth";
            this.cbMonth.Size = new System.Drawing.Size(64, 23);
            this.cbMonth.TabIndex = 16;
            // 
            // cbYear
            // 
            this.cbYear.FormattingEnabled = true;
            this.cbYear.Location = new System.Drawing.Point(1084, 159);
            this.cbYear.Name = "cbYear";
            this.cbYear.Size = new System.Drawing.Size(64, 23);
            this.cbYear.TabIndex = 15;
            // 
            // propertyGrid1
            // 
            this.propertyGrid1.Anchor = System.Windows.Forms.AnchorStyles.None;
            this.propertyGrid1.Location = new System.Drawing.Point(1205, 37);
            this.propertyGrid1.Name = "propertyGrid1";
            this.propertyGrid1.Size = new System.Drawing.Size(286, 645);
            this.propertyGrid1.TabIndex = 0;
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
            // pbCam6
            // 
            this.pbCam6.Location = new System.Drawing.Point(776, 6);
            this.pbCam6.Name = "pbCam6";
            this.pbCam6.Size = new System.Drawing.Size(148, 111);
            this.pbCam6.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam6.TabIndex = 13;
            this.pbCam6.TabStop = false;
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
            // pbCam4
            // 
            this.pbCam4.Location = new System.Drawing.Point(468, 6);
            this.pbCam4.Name = "pbCam4";
            this.pbCam4.Size = new System.Drawing.Size(148, 111);
            this.pbCam4.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam4.TabIndex = 11;
            this.pbCam4.TabStop = false;
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
            // pbCam2
            // 
            this.pbCam2.Location = new System.Drawing.Point(160, 6);
            this.pbCam2.Name = "pbCam2";
            this.pbCam2.Size = new System.Drawing.Size(148, 111);
            this.pbCam2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam2.TabIndex = 9;
            this.pbCam2.TabStop = false;
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
            // chartMura
            // 
            chartArea10.Name = "ChartArea1";
            this.chartMura.ChartAreas.Add(chartArea10);
            this.chartMura.Dock = System.Windows.Forms.DockStyle.Fill;
            legend10.Name = "Legend1";
            this.chartMura.Legends.Add(legend10);
            this.chartMura.Location = new System.Drawing.Point(3, 349);
            this.chartMura.Name = "chartMura";
            series10.ChartArea = "ChartArea1";
            series10.Legend = "Legend1";
            series10.Name = "Series1";
            this.chartMura.Series.Add(series10);
            this.chartMura.Size = new System.Drawing.Size(1064, 143);
            this.chartMura.TabIndex = 16;
            this.chartMura.Text = "chart1";
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
            // label_yr
            // 
            this.label_yr.AutoSize = true;
            this.label_yr.Location = new System.Drawing.Point(1154, 162);
            this.label_yr.Name = "label_yr";
            this.label_yr.Size = new System.Drawing.Size(22, 15);
            this.label_yr.TabIndex = 24;
            this.label_yr.Text = "年";
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
            // label_day
            // 
            this.label_day.AutoSize = true;
            this.label_day.Location = new System.Drawing.Point(1154, 220);
            this.label_day.Name = "label_day";
            this.label_day.Size = new System.Drawing.Size(22, 15);
            this.label_day.TabIndex = 26;
            this.label_day.Text = "日";
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
            // label_min
            // 
            this.label_min.AutoSize = true;
            this.label_min.Location = new System.Drawing.Point(1154, 293);
            this.label_min.Name = "label_min";
            this.label_min.Size = new System.Drawing.Size(22, 15);
            this.label_min.TabIndex = 28;
            this.label_min.Text = "分";
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
            // panel1
            // 
            this.panel1.Location = new System.Drawing.Point(6, 6);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(148, 111);
            this.panel1.TabIndex = 0;
            // 
            // panel2
            // 
            this.panel2.Location = new System.Drawing.Point(160, 6);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(148, 111);
            this.panel2.TabIndex = 1;
            // 
            // panel3
            // 
            this.panel3.Location = new System.Drawing.Point(314, 6);
            this.panel3.Name = "panel3";
            this.panel3.Size = new System.Drawing.Size(148, 111);
            this.panel3.TabIndex = 1;
            // 
            // panel4
            // 
            this.panel4.Location = new System.Drawing.Point(468, 6);
            this.panel4.Name = "panel4";
            this.panel4.Size = new System.Drawing.Size(148, 111);
            this.panel4.TabIndex = 1;
            // 
            // panel5
            // 
            this.panel5.Location = new System.Drawing.Point(622, 6);
            this.panel5.Name = "panel5";
            this.panel5.Size = new System.Drawing.Size(148, 111);
            this.panel5.TabIndex = 1;
            // 
            // panel6
            // 
            this.panel6.Location = new System.Drawing.Point(776, 6);
            this.panel6.Name = "panel6";
            this.panel6.Size = new System.Drawing.Size(148, 111);
            this.panel6.TabIndex = 1;
            // 
            // panel7
            // 
            this.panel7.Location = new System.Drawing.Point(930, 6);
            this.panel7.Name = "panel7";
            this.panel7.Size = new System.Drawing.Size(148, 111);
            this.panel7.TabIndex = 1;
            // 
            // panel8
            // 
            this.panel8.Location = new System.Drawing.Point(6, 141);
            this.panel8.Name = "panel8";
            this.panel8.Size = new System.Drawing.Size(1072, 425);
            this.panel8.TabIndex = 1;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(1084, 7);
            this.button1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(93, 29);
            this.button1.TabIndex = 2;
            this.button1.Text = "Allocation MIL";
            this.button1.UseVisualStyleBackColor = true;
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(1084, 81);
            this.button3.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(93, 29);
            this.button3.TabIndex = 5;
            this.button3.Text = "Free";
            this.button3.UseVisualStyleBackColor = true;
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(1084, 44);
            this.button2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(93, 29);
            this.button2.TabIndex = 4;
            this.button2.Text = "Grab";
            this.button2.UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(55, 120);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 15);
            this.label1.TabIndex = 5;
            this.label1.Text = "label1";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(209, 120);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(41, 15);
            this.label2.TabIndex = 6;
            this.label2.Text = "label2";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(368, 120);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(41, 15);
            this.label3.TabIndex = 7;
            this.label3.Text = "label3";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(519, 120);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(41, 15);
            this.label4.TabIndex = 8;
            this.label4.Text = "label4";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(675, 120);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(41, 15);
            this.label5.TabIndex = 9;
            this.label5.Text = "label5";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(825, 120);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(41, 15);
            this.label6.TabIndex = 10;
            this.label6.Text = "label6";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(989, 120);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(41, 15);
            this.label7.TabIndex = 11;
            this.label7.Text = "label7";
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
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).EndInit();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.chartMura)).EndInit();
            this.tableLayoutPanel1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.canvasMain)).EndInit();
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
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label1;
    }
}