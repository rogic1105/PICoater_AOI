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
        // DevNum = 板內 0-based device number（== MIL.M_DEVx，M_DEV0=0）。用純整數常數而非 MIL.M_DEVx，
        // 確保 JavaScriptSerializer 能正確 round-trip（MIL_INT 會序列化成 {}，見 CameraHardwareConfig.DevNum）。
        // 拓樸：board0(SystemNum 0)=dev 0~3 共 4 台、board1(SystemNum 1)=dev 0~2 共 3 台 = 7 台。
        public static List<CameraHardwareConfig> NewCameraDevices() => new List<CameraHardwareConfig>
        {
            new CameraHardwareConfig { Id = 1, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = 0, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 2, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = 1, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 3, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = 2, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 4, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = 3, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 5, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = 0, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 6, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = 1, DcfPath = InspectionDefaults.DcfPath },
            new CameraHardwareConfig { Id = 7, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = 2, DcfPath = InspectionDefaults.DcfPath },
        };
    }
}
