using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class CameraParamSettings
    {
        [Browsable(false)]
        public string DcfPath { get; set; } = @"D:\AniloxCaptures\dcf\Radient_Config.dcf";

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(DcfPath)) DcfPath = @"D:\AniloxCaptures\dcf\Radient_Config.dcf";
        }
    }
}
