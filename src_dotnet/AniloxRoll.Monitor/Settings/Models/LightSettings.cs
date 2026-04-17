using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class LightSettings
    {
        [Browsable(false)]
        public bool Enabled { get; set; } = false;

        [Browsable(false)]
        public string ComPort { get; set; } = "COM1";

        [Browsable(false)]
        public int BrightnessCh1 { get; set; } = 128;

        [Browsable(false)]
        public int BrightnessCh2 { get; set; } = 128;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(ComPort)) ComPort = "COM1";
            if (BrightnessCh1 < 0) BrightnessCh1 = 0;
            if (BrightnessCh1 > 255) BrightnessCh1 = 255;
            if (BrightnessCh2 < 0) BrightnessCh2 = 0;
            if (BrightnessCh2 > 255) BrightnessCh2 = 255;
        }
    }
}
