using System;
using System.Drawing;

namespace AniloxRoll.Monitor.Core.Acquisition.Inspection
{
    public class InspectionData : IDisposable
    {
        public Bitmap Image { get; set; }
        public float[] MuraCurveMean { get; set; }
        public float[] MuraCurveMax { get; set; }

        public void Dispose()
        {
            Image?.Dispose();
        }
    }
}