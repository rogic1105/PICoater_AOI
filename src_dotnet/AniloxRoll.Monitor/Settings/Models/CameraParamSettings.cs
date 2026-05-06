using System;
using System.ComponentModel;

namespace AniloxRoll.Monitor.Core.Data
{
    [Serializable]
    public class CameraParamSettings
    {
        [Browsable(false)]
        public string DcfPath { get; set; } = InspectionDefaults.DcfPath;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(DcfPath)) DcfPath = InspectionDefaults.DcfPath;
        }
    }
}
