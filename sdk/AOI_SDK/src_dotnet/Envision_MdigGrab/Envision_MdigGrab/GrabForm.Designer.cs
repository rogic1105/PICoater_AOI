namespace Envision_MdigGrab
{
    partial class GrabForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.panel1 = new System.Windows.Forms.Panel();
            this.button1 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.button3 = new System.Windows.Forms.Button();
            this.panel2 = new System.Windows.Forms.Panel();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.labelCoord1 = new System.Windows.Forms.Label();
            this.checkBoxEnableImageProcessing = new System.Windows.Forms.CheckBox();
            this.statusStripMain = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabelFps = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabelCoord = new System.Windows.Forms.ToolStripStatusLabel();
            this.statusStripMain.SuspendLayout();
            this.SuspendLayout();
            // 
            // panel1
            // 
            this.panel1.Location = new System.Drawing.Point(14, 33);
            this.panel1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(662, 111);
            this.panel1.TabIndex = 0;
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(702, 33);
            this.button1.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(202, 29);
            this.button1.TabIndex = 1;
            this.button1.Text = "Allocation MIL";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // button2
            // 
            this.button2.Location = new System.Drawing.Point(702, 78);
            this.button2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(202, 29);
            this.button2.TabIndex = 2;
            this.button2.Text = "MdigGrab";
            this.button2.UseVisualStyleBackColor = true;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // button3
            // 
            this.button3.Location = new System.Drawing.Point(702, 115);
            this.button3.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(202, 29);
            this.button3.TabIndex = 3;
            this.button3.Text = "Free MIL";
            this.button3.UseVisualStyleBackColor = true;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // panel2
            // 
            this.panel2.Location = new System.Drawing.Point(14, 188);
            this.panel2.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(662, 111);
            this.panel2.TabIndex = 1;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(13, 11);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(41, 15);
            this.label1.TabIndex = 4;
            this.label1.Text = "label1";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(13, 162);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(41, 15);
            this.label2.TabIndex = 5;
            this.label2.Text = "label2";
            // 
            // labelCoord1
            // 
            this.labelCoord1.AutoSize = true;
            this.labelCoord1.Location = new System.Drawing.Point(11, 282);
            this.labelCoord1.Name = "labelCoord1";
            this.labelCoord1.Size = new System.Drawing.Size(41, 15);
            this.labelCoord1.TabIndex = 6;
            this.labelCoord1.Text = "label3";
            // 
            // checkBoxEnableImageProcessing
            // 
            this.checkBoxEnableImageProcessing.AutoSize = true;
            this.checkBoxEnableImageProcessing.Checked = true;
            this.checkBoxEnableImageProcessing.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxEnableImageProcessing.Location = new System.Drawing.Point(702, 161);
            this.checkBoxEnableImageProcessing.Name = "checkBoxEnableImageProcessing";
            this.checkBoxEnableImageProcessing.Size = new System.Drawing.Size(102, 19);
            this.checkBoxEnableImageProcessing.TabIndex = 7;
            this.checkBoxEnableImageProcessing.Text = "影像處理開關";
            this.checkBoxEnableImageProcessing.UseVisualStyleBackColor = true;
            this.checkBoxEnableImageProcessing.CheckedChanged += new System.EventHandler(this.checkBoxEnableImageProcessing_CheckedChanged);
            // 
            // 
            // statusStripMain
            // 
            this.statusStripMain.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.statusStripMain.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabelFps,
            this.toolStripStatusLabelCoord});
            this.statusStripMain.Location = new System.Drawing.Point(0, 303);
            this.statusStripMain.Name = "statusStripMain";
            this.statusStripMain.Size = new System.Drawing.Size(918, 26);
            this.statusStripMain.TabIndex = 8;
            this.statusStripMain.Text = "statusStrip1";
            // 
            // toolStripStatusLabelFps
            // 
            this.toolStripStatusLabelFps.Name = "toolStripStatusLabelFps";
            this.toolStripStatusLabelFps.Size = new System.Drawing.Size(62, 20);
            this.toolStripStatusLabelFps.Text = "FPS: N/A";
            // 
            // toolStripStatusLabelCoord
            // 
            this.toolStripStatusLabelCoord.Name = "toolStripStatusLabelCoord";
            this.toolStripStatusLabelCoord.Size = new System.Drawing.Size(106, 20);
            this.toolStripStatusLabelCoord.Text = "Coord: Ready";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(918, 329);
            this.Controls.Add(this.statusStripMain);
            this.Controls.Add(this.checkBoxEnableImageProcessing);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.labelCoord1);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.button3);
            this.Controls.Add(this.button2);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.panel1);
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Name = "Form1";
            this.Text = "Form1";
            this.statusStripMain.ResumeLayout(false);
            this.statusStripMain.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label labelCoord1;
        private System.Windows.Forms.CheckBox checkBoxEnableImageProcessing;
        private System.Windows.Forms.StatusStrip statusStripMain;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelFps;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabelCoord;
    }
}
