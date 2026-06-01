using System.Collections.Generic;
using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// SystemSettings 預設值集中定義（7 cam 硬體拓樸；dcf 共用 InspectionDefaults.DcfPath）。
    /// 原本在 SystemSettings.CreateDefault 內 inline hardcode 7 行，收斂到唯一來源。
    /// </summary>
    internal static class SystemDefaults
    {
        public static List<CameraHardwareConfig> NewCameraDevices() => new List<CameraHardwareConfig>
        {
            new CameraHardwareConfig { Id = 1, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV0, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 2, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV1, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 3, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV2, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 4, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV3, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 5, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV0, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 6, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV1, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 7, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV2, DcfPath = InspectionDefaults.DcfPath },
        };
    }
}
