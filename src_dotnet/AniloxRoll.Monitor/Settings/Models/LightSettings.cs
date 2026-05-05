using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class LightSettings
    {
        [Browsable(false)]
        public bool Enabled { get; set; } = true;

        /// <summary>設定檔記錄的 COM port。啟動時先試此 port，失敗則掃描所有可用 port。</summary>
        [Browsable(false)]
        public string ComPort { get; set; } = "COM17";

        /// <summary>通道（LTS-3DPA24 單通道機型固定 1）。</summary>
        [Browsable(false)]
        public int Channel { get; set; } = 1;

        /// <summary>亮度（0~255）。</summary>
        [Browsable(false)]
        public int Brightness { get; set; } = 128;

        /// <summary>開燈後等待光源穩定的延遲（ms）。Grab 啟動前插入此延遲。</summary>
        [Browsable(false)]
        public int WarmupMs { get; set; } = 300;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(ComPort)) ComPort = "COM1";
            if (Channel < 1) Channel = 1;
            if (Channel > 4) Channel = 4;
            if (Brightness < 0) Brightness = 0;
            if (Brightness > 255) Brightness = 255;
            if (WarmupMs < 0) WarmupMs = 0;
        }
    }
}
