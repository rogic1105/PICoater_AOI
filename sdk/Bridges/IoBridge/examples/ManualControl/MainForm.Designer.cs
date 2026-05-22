namespace IoBridge.ManualControl
{
    partial class MainForm
    {
        private System.ComponentModel.IContainer components = null;

        private System.Windows.Forms.TextBox txtIP;
        private System.Windows.Forms.Button btnConnect;
        private System.Windows.Forms.Button btnDisconnect;
        private System.Windows.Forms.Label lblIP;
        private System.Windows.Forms.GroupBox grpDO;
        private System.Windows.Forms.GroupBox grpDI;
        private System.Windows.Forms.Panel pnlTop;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null)) components.Dispose();
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            this.pnlTop = new System.Windows.Forms.Panel();
            this.lblIP = new System.Windows.Forms.Label();
            this.txtIP = new System.Windows.Forms.TextBox();
            this.btnConnect = new System.Windows.Forms.Button();
            this.btnDisconnect = new System.Windows.Forms.Button();
            this.grpDO = new System.Windows.Forms.GroupBox();
            this.grpDI = new System.Windows.Forms.GroupBox();
            this.pnlTop.SuspendLayout();
            this.SuspendLayout();

            // pnlTop
            this.pnlTop.BackColor = System.Drawing.Color.WhiteSmoke;
            this.pnlTop.Controls.Add(this.lblIP);
            this.pnlTop.Controls.Add(this.txtIP);
            this.pnlTop.Controls.Add(this.btnConnect);
            this.pnlTop.Controls.Add(this.btnDisconnect);
            this.pnlTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.pnlTop.Location = new System.Drawing.Point(0, 0);
            this.pnlTop.Name = "pnlTop";
            this.pnlTop.Size = new System.Drawing.Size(500, 60);
            this.pnlTop.TabIndex = 0;

            // lblIP
            this.lblIP.AutoSize = true;
            this.lblIP.Font = new System.Drawing.Font("Segoe UI", 10F);
            this.lblIP.Location = new System.Drawing.Point(20, 22);
            this.lblIP.Name = "lblIP";
            this.lblIP.Size = new System.Drawing.Size(76, 19);
            this.lblIP.TabIndex = 0;
            this.lblIP.Text = "IP Address:";

            // txtIP
            this.txtIP.Font = new System.Drawing.Font("Consolas", 10F);
            this.txtIP.Location = new System.Drawing.Point(100, 19);
            this.txtIP.Name = "txtIP";
            this.txtIP.Size = new System.Drawing.Size(140, 23);
            this.txtIP.TabIndex = 1;
            this.txtIP.Text = "192.168.255.1";

            // btnConnect
            this.btnConnect.BackColor = System.Drawing.Color.LightSkyBlue;
            this.btnConnect.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnConnect.Location = new System.Drawing.Point(260, 17);
            this.btnConnect.Name = "btnConnect";
            this.btnConnect.Size = new System.Drawing.Size(90, 30);
            this.btnConnect.TabIndex = 2;
            this.btnConnect.Text = "Connect";
            this.btnConnect.UseVisualStyleBackColor = false;
            this.btnConnect.Click += new System.EventHandler(this.btnConnect_Click);

            // btnDisconnect
            this.btnDisconnect.BackColor = System.Drawing.Color.LightPink;
            this.btnDisconnect.Enabled = false;
            this.btnDisconnect.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnDisconnect.Location = new System.Drawing.Point(360, 17);
            this.btnDisconnect.Name = "btnDisconnect";
            this.btnDisconnect.Size = new System.Drawing.Size(90, 30);
            this.btnDisconnect.TabIndex = 3;
            this.btnDisconnect.Text = "Disconnect";
            this.btnDisconnect.UseVisualStyleBackColor = false;
            this.btnDisconnect.Click += new System.EventHandler(this.btnDisconnect_Click);

            // grpDO
            this.grpDO.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
            this.grpDO.Location = new System.Drawing.Point(20, 80);
            this.grpDO.Name = "grpDO";
            this.grpDO.Size = new System.Drawing.Size(220, 400);
            this.grpDO.TabIndex = 1;
            this.grpDO.TabStop = false;
            this.grpDO.Text = "Digital Output (Control)";

            // grpDI
            this.grpDI.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
            this.grpDI.Location = new System.Drawing.Point(260, 80);
            this.grpDI.Name = "grpDI";
            this.grpDI.Size = new System.Drawing.Size(220, 400);
            this.grpDI.TabIndex = 2;
            this.grpDI.TabStop = false;
            this.grpDI.Text = "Digital Input (Status)";

            // MainForm
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(500, 500);
            this.Controls.Add(this.grpDI);
            this.Controls.Add(this.grpDO);
            this.Controls.Add(this.pnlTop);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "ET-7044 Manual Controller";
            this.pnlTop.ResumeLayout(false);
            this.pnlTop.PerformLayout();
            this.ResumeLayout(false);
        }
    }
}
