using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class LightSettings
    {
        [Browsable(false)]
        public bool Enabled { get; set; } = InspectionDefaults.LightEnabled;

        /// <summary>設定檔記錄的 COM port。啟動時先試此 port，失敗則掃描所有可用 port。</summary>
        [Browsable(false)]
        public string ComPort { get; set; } = InspectionDefaults.LightComPort;

        /// <summary>通道（LTS-3DPA24 單通道機型固定 1）。</summary>
        [Browsable(false)]
        public int Channel { get; set; } = InspectionDefaults.LightChannel;

        /// <summary>亮度（0~255）。</summary>
        [Browsable(false)]
        public int Brightness { get; set; } = InspectionDefaults.LightBrightness;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(ComPort)) ComPort = InspectionDefaults.LightComPort;
            if (Channel < 1) Channel = InspectionDefaults.LightChannel;
            if (Channel > 4) Channel = 4;
            if (Brightness < 0) Brightness = 0;
            if (Brightness > 255) Brightness = 255;
        }
    }
}
