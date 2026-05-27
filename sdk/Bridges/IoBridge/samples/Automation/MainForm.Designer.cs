namespace IoBridge.Automation
{
    partial class MainForm
    {
        private System.ComponentModel.IContainer components = null;
        private System.Windows.Forms.TextBox txtIP;
        private System.Windows.Forms.Button btnConnect;
        private System.Windows.Forms.Button btnDisconnect;
        private System.Windows.Forms.Button btnMura;
        private System.Windows.Forms.Label lblIP;
        private System.Windows.Forms.Label lblSystemStatus;
        private System.Windows.Forms.Label lblStatusTitle;
        private System.Windows.Forms.GroupBox grpDO;
        private System.Windows.Forms.GroupBox grpDI;
        private System.Windows.Forms.Panel pnlTop;
        private System.Windows.Forms.Panel pnlStatus;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null)) components.Dispose();
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            this.pnlTop = new System.Windows.Forms.Panel();
            this.pnlStatus = new System.Windows.Forms.Panel();
            this.lblSystemStatus = new System.Windows.Forms.Label();
            this.lblStatusTitle = new System.Windows.Forms.Label();
            this.lblIP = new System.Windows.Forms.Label();
            this.txtIP = new System.Windows.Forms.TextBox();
            this.btnConnect = new System.Windows.Forms.Button();
            this.btnDisconnect = new System.Windows.Forms.Button();
            this.btnMura = new System.Windows.Forms.Button();
            this.grpDO = new System.Windows.Forms.GroupBox();
            this.grpDI = new System.Windows.Forms.GroupBox();
            this.pnlTop.SuspendLayout();
            this.pnlStatus.SuspendLayout();
            this.SuspendLayout();

            // pnlTop
            this.pnlTop.BackColor = System.Drawing.Color.FromArgb(45, 45, 48);
            this.pnlTop.Controls.Add(this.pnlStatus);
            this.pnlTop.Controls.Add(this.lblIP);
            this.pnlTop.Controls.Add(this.txtIP);
            this.pnlTop.Controls.Add(this.btnConnect);
            this.pnlTop.Controls.Add(this.btnDisconnect);
            this.pnlTop.Controls.Add(this.btnMura);
            this.pnlTop.Dock = System.Windows.Forms.DockStyle.Top;
            this.pnlTop.Location = new System.Drawing.Point(0, 0);
            this.pnlTop.Name = "pnlTop";
            this.pnlTop.Size = new System.Drawing.Size(780, 80);
            this.pnlTop.TabIndex = 0;

            // lblIP
            this.lblIP.AutoSize = true;
            this.lblIP.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Bold);
            this.lblIP.ForeColor = System.Drawing.Color.WhiteSmoke;
            this.lblIP.Location = new System.Drawing.Point(25, 30);
            this.lblIP.Name = "lblIP";
            this.lblIP.Size = new System.Drawing.Size(81, 19);
            this.lblIP.TabIndex = 0;
            this.lblIP.Text = "IP Address";

            // txtIP
            this.txtIP.BackColor = System.Drawing.Color.FromArgb(60, 60, 65);
            this.txtIP.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.txtIP.Font = new System.Drawing.Font("Consolas", 11F);
            this.txtIP.ForeColor = System.Drawing.Color.White;
            this.txtIP.Location = new System.Drawing.Point(115, 28);
            this.txtIP.Name = "txtIP";
            this.txtIP.Size = new System.Drawing.Size(140, 25);
            this.txtIP.TabIndex = 1;
            this.txtIP.Text = "192.168.255.1";
            this.txtIP.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;

            // btnConnect
            this.btnConnect.BackColor = System.Drawing.Color.FromArgb(0, 122, 204);
            this.btnConnect.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnConnect.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
            this.btnConnect.ForeColor = System.Drawing.Color.White;
            this.btnConnect.Location = new System.Drawing.Point(270, 25);
            this.btnConnect.Name = "btnConnect";
            this.btnConnect.Size = new System.Drawing.Size(100, 32);
            this.btnConnect.TabIndex = 2;
            this.btnConnect.Text = "CONNECT";
            this.btnConnect.UseVisualStyleBackColor = false;
            this.btnConnect.Click += new System.EventHandler(this.btnConnect_Click);

            // btnDisconnect
            this.btnDisconnect.BackColor = System.Drawing.Color.FromArgb(60, 60, 60);
            this.btnDisconnect.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnDisconnect.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
            this.btnDisconnect.ForeColor = System.Drawing.Color.Gray;
            this.btnDisconnect.Location = new System.Drawing.Point(380, 25);
            this.btnDisconnect.Name = "btnDisconnect";
            this.btnDisconnect.Size = new System.Drawing.Size(100, 32);
            this.btnDisconnect.TabIndex = 3;
            this.btnDisconnect.Text = "DISCONNECT";
            this.btnDisconnect.UseVisualStyleBackColor = false;
            this.btnDisconnect.Click += new System.EventHandler(this.btnDisconnect_Click);

            // btnMura
            this.btnMura.BackColor = System.Drawing.Color.FromArgb(60, 60, 60);
            this.btnMura.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnMura.Font = new System.Drawing.Font("Segoe UI", 9F, System.Drawing.FontStyle.Bold);
            this.btnMura.ForeColor = System.Drawing.Color.Gray;
            this.btnMura.Location = new System.Drawing.Point(490, 25);
            this.btnMura.Name = "btnMura";
            this.btnMura.Size = new System.Drawing.Size(80, 32);
            this.btnMura.TabIndex = 5;
            this.btnMura.Text = "MURA";
            this.btnMura.UseVisualStyleBackColor = false;
            this.btnMura.Click += new System.EventHandler(this.btnMura_Click);

            // pnlStatus
            this.pnlStatus.BackColor = System.Drawing.Color.FromArgb(30, 30, 30);
            this.pnlStatus.Controls.Add(this.lblSystemStatus);
            this.pnlStatus.Controls.Add(this.lblStatusTitle);
            this.pnlStatus.Location = new System.Drawing.Point(590, 15);
            this.pnlStatus.Name = "pnlStatus";
            this.pnlStatus.Size = new System.Drawing.Size(160, 50);
            this.pnlStatus.TabIndex = 6;

            // lblStatusTitle
            this.lblStatusTitle.AutoSize = true;
            this.lblStatusTitle.Font = new System.Drawing.Font("Segoe UI", 7F);
            this.lblStatusTitle.ForeColor = System.Drawing.Color.DarkGray;
            this.lblStatusTitle.Location = new System.Drawing.Point(5, 5);
            this.lblStatusTitle.Name = "lblStatusTitle";
            this.lblStatusTitle.Size = new System.Drawing.Size(77, 12);
            this.lblStatusTitle.TabIndex = 0;
            this.lblStatusTitle.Text = "SYSTEM STATUS";

            // lblSystemStatus
            this.lblSystemStatus.Dock = System.Windows.Forms.DockStyle.Bottom;
            this.lblSystemStatus.Font = new System.Drawing.Font("Segoe UI", 14F, System.Drawing.FontStyle.Bold);
            this.lblSystemStatus.ForeColor = System.Drawing.Color.Gray;
            this.lblSystemStatus.Location = new System.Drawing.Point(0, 18);
            this.lblSystemStatus.Name = "lblSystemStatus";
            this.lblSystemStatus.Size = new System.Drawing.Size(160, 32);
            this.lblSystemStatus.TabIndex = 4;
            this.lblSystemStatus.Text = "CLOSED";
            this.lblSystemStatus.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;

            // grpDO
            this.grpDO.BackColor = System.Drawing.Color.FromArgb(37, 37, 38);
            this.grpDO.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Bold);
            this.grpDO.ForeColor = System.Drawing.Color.FromArgb(200, 200, 200);
            this.grpDO.Location = new System.Drawing.Point(25, 100);
            this.grpDO.Name = "grpDO";
            this.grpDO.Size = new System.Drawing.Size(350, 420);
            this.grpDO.TabIndex = 1;
            this.grpDO.TabStop = false;
            this.grpDO.Text = "  Digital Output (PC -> PLC)  ";

            // grpDI
            this.grpDI.BackColor = System.Drawing.Color.FromArgb(37, 37, 38);
            this.grpDI.Font = new System.Drawing.Font("Segoe UI", 10F, System.Drawing.FontStyle.Bold);
            this.grpDI.ForeColor = System.Drawing.Color.FromArgb(200, 200, 200);
            this.grpDI.Location = new System.Drawing.Point(400, 100);
            this.grpDI.Name = "grpDI";
            this.grpDI.Size = new System.Drawing.Size(350, 420);
            this.grpDI.TabIndex = 2;
            this.grpDI.TabStop = false;
            this.grpDI.Text = "  Digital Input (PLC -> PC)  ";

            // MainForm
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.FromArgb(28, 28, 28);
            this.ClientSize = new System.Drawing.Size(780, 550);
            this.Controls.Add(this.grpDI);
            this.Controls.Add(this.grpDO);
            this.Controls.Add(this.pnlTop);
            this.Font = new System.Drawing.Font("Segoe UI", 9F);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "MainForm";
            this.Text = "PICoater Automation Controller";
            this.pnlTop.ResumeLayout(false);
            this.pnlTop.PerformLayout();
            this.pnlStatus.ResumeLayout(false);
            this.pnlStatus.PerformLayout();
            this.ResumeLayout(false);
        }
    }
}
