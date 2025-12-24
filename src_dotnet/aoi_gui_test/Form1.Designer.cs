namespace CsTestApp
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
            this.pictureBoxSrc = new System.Windows.Forms.PictureBox();
            this.pictureBoxDst = new System.Windows.Forms.PictureBox();
            this.numfft_th = new System.Windows.Forms.NumericUpDown();
            this.numbw_th = new System.Windows.Forms.NumericUpDown();
            this.btnOpen = new System.Windows.Forms.Button();
            this.btnRun = new System.Windows.Forms.Button();
            this.btnSave = new System.Windows.Forms.Button();
            this.numborder_t = new System.Windows.Forms.NumericUpDown();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSrc)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxDst)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numfft_th)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numbw_th)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numborder_t)).BeginInit();
            this.SuspendLayout();
            // 
            // pictureBoxSrc
            // 
            this.pictureBoxSrc.Location = new System.Drawing.Point(51, 17);
            this.pictureBoxSrc.Margin = new System.Windows.Forms.Padding(2);
            this.pictureBoxSrc.Name = "pictureBoxSrc";
            this.pictureBoxSrc.Size = new System.Drawing.Size(1167, 302);
            this.pictureBoxSrc.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBoxSrc.TabIndex = 0;
            this.pictureBoxSrc.TabStop = false;
            // 
            // pictureBoxDst
            // 
            this.pictureBoxDst.Location = new System.Drawing.Point(51, 338);
            this.pictureBoxDst.Margin = new System.Windows.Forms.Padding(2);
            this.pictureBoxDst.Name = "pictureBoxDst";
            this.pictureBoxDst.Size = new System.Drawing.Size(1167, 302);
            this.pictureBoxDst.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBoxDst.TabIndex = 1;
            this.pictureBoxDst.TabStop = false;
            // 
            // numfft_th
            // 
            this.numfft_th.Location = new System.Drawing.Point(1268, 17);
            this.numfft_th.Margin = new System.Windows.Forms.Padding(2);
            this.numfft_th.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numfft_th.Minimum = new decimal(new int[] {
            255,
            0,
            0,
            -2147483648});
            this.numfft_th.Name = "numfft_th";
            this.numfft_th.Size = new System.Drawing.Size(90, 22);
            this.numfft_th.TabIndex = 2;
            this.numfft_th.Value = new decimal(new int[] {
            99,
            0,
            0,
            0});
            // 
            // numbw_th
            // 
            this.numbw_th.Location = new System.Drawing.Point(1268, 53);
            this.numbw_th.Margin = new System.Windows.Forms.Padding(2);
            this.numbw_th.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numbw_th.Name = "numbw_th";
            this.numbw_th.Size = new System.Drawing.Size(90, 22);
            this.numbw_th.TabIndex = 3;
            this.numbw_th.Value = new decimal(new int[] {
            50,
            0,
            0,
            0});
            // 
            // btnOpen
            // 
            this.btnOpen.Location = new System.Drawing.Point(1278, 140);
            this.btnOpen.Margin = new System.Windows.Forms.Padding(2);
            this.btnOpen.Name = "btnOpen";
            this.btnOpen.Size = new System.Drawing.Size(56, 18);
            this.btnOpen.TabIndex = 4;
            this.btnOpen.Text = "btnOpen";
            this.btnOpen.UseVisualStyleBackColor = true;
            this.btnOpen.Click += new System.EventHandler(this.btnOpen_Click);
            // 
            // btnRun
            // 
            this.btnRun.Location = new System.Drawing.Point(1278, 177);
            this.btnRun.Margin = new System.Windows.Forms.Padding(2);
            this.btnRun.Name = "btnRun";
            this.btnRun.Size = new System.Drawing.Size(56, 18);
            this.btnRun.TabIndex = 5;
            this.btnRun.Text = "btnRun";
            this.btnRun.UseVisualStyleBackColor = true;
            this.btnRun.Click += new System.EventHandler(this.btnRun_Click);
            // 
            // btnSave
            // 
            this.btnSave.Location = new System.Drawing.Point(1278, 211);
            this.btnSave.Margin = new System.Windows.Forms.Padding(2);
            this.btnSave.Name = "btnSave";
            this.btnSave.Size = new System.Drawing.Size(56, 18);
            this.btnSave.TabIndex = 6;
            this.btnSave.Text = "btnSave";
            this.btnSave.UseVisualStyleBackColor = true;
            this.btnSave.Click += new System.EventHandler(this.btnSave_Click);
            // 
            // numborder_t
            // 
            this.numborder_t.Location = new System.Drawing.Point(1268, 88);
            this.numborder_t.Margin = new System.Windows.Forms.Padding(2);
            this.numborder_t.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numborder_t.Name = "numborder_t";
            this.numborder_t.Size = new System.Drawing.Size(90, 22);
            this.numborder_t.TabIndex = 7;
            this.numborder_t.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1369, 675);
            this.Controls.Add(this.numborder_t);
            this.Controls.Add(this.btnSave);
            this.Controls.Add(this.btnRun);
            this.Controls.Add(this.btnOpen);
            this.Controls.Add(this.numbw_th);
            this.Controls.Add(this.numfft_th);
            this.Controls.Add(this.pictureBoxDst);
            this.Controls.Add(this.pictureBoxSrc);
            this.Margin = new System.Windows.Forms.Padding(2);
            this.Name = "Form1";
            this.Text = "Form1";
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxSrc)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxDst)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numfft_th)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numbw_th)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numborder_t)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBoxSrc;
        private System.Windows.Forms.PictureBox pictureBoxDst;
        private System.Windows.Forms.NumericUpDown numfft_th;
        private System.Windows.Forms.NumericUpDown numbw_th;
        private System.Windows.Forms.Button btnOpen;
        private System.Windows.Forms.Button btnRun;
        private System.Windows.Forms.Button btnSave;
        private System.Windows.Forms.NumericUpDown numborder_t;
    }
}

