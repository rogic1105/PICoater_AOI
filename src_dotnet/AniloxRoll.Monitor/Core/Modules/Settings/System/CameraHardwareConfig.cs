using System.Runtime.Serialization;
using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 相機硬體/擷取卡配置（偏靜態，通常於安裝時決定）。
    /// </summary>
    [DataContract]
    public class CameraHardwareConfig
    {
        [DataMember]
        public int Id { get; set; }
        [DataMember]
        public string SystemDescriptor { get; set; }
        [DataMember]
        public int SystemNum { get; set; }
        [DataMember]
        public MIL_INT DevNum { get; set; }
        [DataMember]
        public string DcfPath { get; set; }
    }
}
