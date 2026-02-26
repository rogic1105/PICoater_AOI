using System.Windows.Forms;
using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Forms.Helpers
{
    internal class CameraConfig
    {
        public int Id { get; set; }
        public string SystemDescriptor { get; set; }
        public int SystemNum { get; set; }
        public MIL_INT DevNum { get; set; }
        public string DcfPath { get; set; }
        public Panel DisplayPanel { get; set; }
        public Label StatusLabel { get; set; }
    }
}
