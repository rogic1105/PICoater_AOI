namespace AniloxRoll.Monitor
{
    partial class Form1
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
            this.tabControl1 = new System.Windows.Forms.TabControl();
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
            this.pbCam7 = new System.Windows.Forms.PictureBox();
            this.pbCam6 = new System.Windows.Forms.PictureBox();
            this.pbCam5 = new System.Windows.Forms.PictureBox();
            this.pbCam4 = new System.Windows.Forms.PictureBox();
            this.pbCam3 = new System.Windows.Forms.PictureBox();
            this.pbCam2 = new System.Windows.Forms.PictureBox();
            this.pbCam1 = new System.Windows.Forms.PictureBox();
            this.mainCanvas = new AOI.SDK.UI.SmartCanvas();
            this.tabControl1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.mainCanvas)).BeginInit();
            this.SuspendLayout();
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Location = new System.Drawing.Point(3, 3);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(779, 554);
            this.tabControl1.TabIndex = 1;
            // 
            // tabPage1
            // 
            this.tabPage1.Location = new System.Drawing.Point(4, 25);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(771, 525);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Live Monitor";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.btnSelectFolder);
            this.tabPage2.Controls.Add(this.btnShowProcessed);
            this.tabPage2.Controls.Add(this.btnShowOriginal);
            this.tabPage2.Controls.Add(this.cbSec);
            this.tabPage2.Controls.Add(this.cbMin);
            this.tabPage2.Controls.Add(this.cbHour);
            this.tabPage2.Controls.Add(this.cbDay);
            this.tabPage2.Controls.Add(this.cbMonth);
            this.tabPage2.Controls.Add(this.cbYear);
            this.tabPage2.Controls.Add(this.pbCam7);
            this.tabPage2.Controls.Add(this.pbCam6);
            this.tabPage2.Controls.Add(this.pbCam5);
            this.tabPage2.Controls.Add(this.pbCam4);
            this.tabPage2.Controls.Add(this.pbCam3);
            this.tabPage2.Controls.Add(this.pbCam2);
            this.tabPage2.Controls.Add(this.pbCam1);
            this.tabPage2.Controls.Add(this.mainCanvas);
            this.tabPage2.Location = new System.Drawing.Point(4, 25);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(771, 525);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Review & Test";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // btnSelectFolder
            // 
            this.btnSelectFolder.Location = new System.Drawing.Point(15, 406);
            this.btnSelectFolder.Name = "btnSelectFolder";
            this.btnSelectFolder.Size = new System.Drawing.Size(112, 40);
            this.btnSelectFolder.TabIndex = 23;
            this.btnSelectFolder.Text = "讀取資料夾";
            this.btnSelectFolder.UseVisualStyleBackColor = true;
            this.btnSelectFolder.Click += new System.EventHandler(this.btnSelectFolder_Click);
            // 
            // btnShowProcessed
            // 
            this.btnShowProcessed.Location = new System.Drawing.Point(283, 406);
            this.btnShowProcessed.Name = "btnShowProcessed";
            this.btnShowProcessed.Size = new System.Drawing.Size(112, 40);
            this.btnShowProcessed.TabIndex = 22;
            this.btnShowProcessed.Text = "顯示檢測圖";
            this.btnShowProcessed.UseVisualStyleBackColor = true;
            this.btnShowProcessed.Click += new System.EventHandler(this.btnShowProcessed_Click);
            // 
            // btnShowOriginal
            // 
            this.btnShowOriginal.Location = new System.Drawing.Point(143, 406);
            this.btnShowOriginal.Name = "btnShowOriginal";
            this.btnShowOriginal.Size = new System.Drawing.Size(112, 40);
            this.btnShowOriginal.TabIndex = 21;
            this.btnShowOriginal.Text = "顯示原圖";
            this.btnShowOriginal.UseVisualStyleBackColor = true;
            this.btnShowOriginal.Click += new System.EventHandler(this.btnShowOriginal_Click);
            // 
            // cbSec
            // 
            this.cbSec.FormattingEnabled = true;
            this.cbSec.Location = new System.Drawing.Point(401, 452);
            this.cbSec.Name = "cbSec";
            this.cbSec.Size = new System.Drawing.Size(71, 23);
            this.cbSec.TabIndex = 20;
            this.cbSec.SelectedIndexChanged += new System.EventHandler(this.cbSec_SelectedIndexChanged);
            // 
            // cbMin
            // 
            this.cbMin.FormattingEnabled = true;
            this.cbMin.Location = new System.Drawing.Point(324, 452);
            this.cbMin.Name = "cbMin";
            this.cbMin.Size = new System.Drawing.Size(71, 23);
            this.cbMin.TabIndex = 19;
            this.cbMin.SelectedIndexChanged += new System.EventHandler(this.cbMin_SelectedIndexChanged);
            // 
            // cbHour
            // 
            this.cbHour.FormattingEnabled = true;
            this.cbHour.Location = new System.Drawing.Point(246, 452);
            this.cbHour.Name = "cbHour";
            this.cbHour.Size = new System.Drawing.Size(71, 23);
            this.cbHour.TabIndex = 18;
            this.cbHour.SelectedIndexChanged += new System.EventHandler(this.cbHour_SelectedIndexChanged);
            // 
            // cbDay
            // 
            this.cbDay.FormattingEnabled = true;
            this.cbDay.Location = new System.Drawing.Point(169, 452);
            this.cbDay.Name = "cbDay";
            this.cbDay.Size = new System.Drawing.Size(71, 23);
            this.cbDay.TabIndex = 17;
            this.cbDay.SelectedIndexChanged += new System.EventHandler(this.cbDay_SelectedIndexChanged);
            // 
            // cbMonth
            // 
            this.cbMonth.FormattingEnabled = true;
            this.cbMonth.Location = new System.Drawing.Point(92, 452);
            this.cbMonth.Name = "cbMonth";
            this.cbMonth.Size = new System.Drawing.Size(71, 23);
            this.cbMonth.TabIndex = 16;
            this.cbMonth.SelectedIndexChanged += new System.EventHandler(this.cbMonth_SelectedIndexChanged);
            // 
            // cbYear
            // 
            this.cbYear.FormattingEnabled = true;
            this.cbYear.Location = new System.Drawing.Point(15, 452);
            this.cbYear.Name = "cbYear";
            this.cbYear.Size = new System.Drawing.Size(71, 23);
            this.cbYear.TabIndex = 15;
            this.cbYear.SelectedIndexChanged += new System.EventHandler(this.cbYear_SelectedIndexChanged);
            // 
            // pbCam7
            // 
            this.pbCam7.Location = new System.Drawing.Point(651, 9);
            this.pbCam7.Name = "pbCam7";
            this.pbCam7.Size = new System.Drawing.Size(100, 100);
            this.pbCam7.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam7.TabIndex = 14;
            this.pbCam7.TabStop = false;
            // 
            // pbCam6
            // 
            this.pbCam6.Location = new System.Drawing.Point(545, 9);
            this.pbCam6.Name = "pbCam6";
            this.pbCam6.Size = new System.Drawing.Size(100, 100);
            this.pbCam6.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam6.TabIndex = 13;
            this.pbCam6.TabStop = false;
            // 
            // pbCam5
            // 
            this.pbCam5.Location = new System.Drawing.Point(439, 9);
            this.pbCam5.Name = "pbCam5";
            this.pbCam5.Size = new System.Drawing.Size(100, 100);
            this.pbCam5.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam5.TabIndex = 12;
            this.pbCam5.TabStop = false;
            // 
            // pbCam4
            // 
            this.pbCam4.Location = new System.Drawing.Point(333, 9);
            this.pbCam4.Name = "pbCam4";
            this.pbCam4.Size = new System.Drawing.Size(100, 100);
            this.pbCam4.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam4.TabIndex = 11;
            this.pbCam4.TabStop = false;
            // 
            // pbCam3
            // 
            this.pbCam3.Location = new System.Drawing.Point(227, 9);
            this.pbCam3.Name = "pbCam3";
            this.pbCam3.Size = new System.Drawing.Size(100, 100);
            this.pbCam3.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam3.TabIndex = 10;
            this.pbCam3.TabStop = false;
            // 
            // pbCam2
            // 
            this.pbCam2.Location = new System.Drawing.Point(121, 9);
            this.pbCam2.Name = "pbCam2";
            this.pbCam2.Size = new System.Drawing.Size(100, 100);
            this.pbCam2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam2.TabIndex = 9;
            this.pbCam2.TabStop = false;
            // 
            // pbCam1
            // 
            this.pbCam1.Location = new System.Drawing.Point(15, 9);
            this.pbCam1.Name = "pbCam1";
            this.pbCam1.Size = new System.Drawing.Size(100, 100);
            this.pbCam1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pbCam1.TabIndex = 8;
            this.pbCam1.TabStop = false;
            // 
            // mainCanvas
            // 
            this.mainCanvas.BackColor = System.Drawing.Color.Black;
            this.mainCanvas.Cursor = System.Windows.Forms.Cursors.Cross;
            this.mainCanvas.Location = new System.Drawing.Point(15, 115);
            this.mainCanvas.Name = "mainCanvas";
            this.mainCanvas.Size = new System.Drawing.Size(736, 285);
            this.mainCanvas.TabIndex = 7;
            this.mainCanvas.TabStop = false;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(782, 553);
            this.Controls.Add(this.tabControl1);
            this.Name = "Form1";
            this.Text = "Form1";
            this.tabControl1.ResumeLayout(false);
            this.tabPage2.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pbCam7)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam6)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam5)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pbCam1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.mainCanvas)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private AOI.SDK.UI.SmartCanvas mainCanvas;
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
    }
}

