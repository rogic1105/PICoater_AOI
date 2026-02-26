using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 相機硬體/擷取卡配置（偏靜態，通常於安裝時決定）。
    /// </summary>
    public class CameraHardwareConfig
    {
        public int Id { get; set; }
        public string SystemDescriptor { get; set; }
        public int SystemNum { get; set; }
        public MIL_INT DevNum { get; set; }
        public string DcfPath { get; set; }
    }
}
