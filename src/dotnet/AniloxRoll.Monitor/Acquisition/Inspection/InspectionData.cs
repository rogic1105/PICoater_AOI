using System;
using System.Drawing;

namespace AniloxRoll.Monitor.Core.Acquisition.Inspection
{
    public class InspectionData : IDisposable
    {
        public Bitmap Image { get; set; }
        public float[] MuraCurveMean { get; set; }
        public float[] MuraCurveMax { get; set; }
        public float[] MuraRowCurveMean { get; set; }
        public float[] MuraRowCurveMax { get; set; }
        public bool IsCompressedJpeg { get; set; }
        public int ScaleFactor { get; set; } = 1;

        public void Dispose()
        {
            Image?.Dispose();
        }
    }
}