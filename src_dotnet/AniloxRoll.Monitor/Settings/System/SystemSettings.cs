using System.Collections.Generic;
using Matrox.MatroxImagingLibrary;

namespace AniloxRoll.Monitor.Core.Data
{
    /// <summary>
    /// 系統層設定（硬體/設備拓樸）。
    /// </summary>
    public class SystemSettings
    {
        public List<CameraHardwareConfig> CameraDevices { get; set; } = new List<CameraHardwareConfig>();

        public static SystemSettings CreateDefault()
        {
            SystemSettings fallback = new SystemSettings
            {
                CameraDevices = new List<CameraHardwareConfig>
                {
                    new CameraHardwareConfig { Id = 1, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV0, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 2, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV1, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 3, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV2, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 4, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 0, DevNum = MIL.M_DEV3, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 5, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV0, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 6, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV1, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" },
                    new CameraHardwareConfig { Id = 7, SystemDescriptor = MIL.M_SYSTEM_RADIENTEVCL, SystemNum = 1, DevNum = MIL.M_DEV2, DcfPath = @"C:\Users\User\Downloads\dcf\Radient_Config.dcf" }
                }
            };

            var loaded = JsonConfigLoader.LoadOrDefault("Config\\system-settings.json", fallback);
            if (loaded.CameraDevices == null || loaded.CameraDevices.Count == 0) return fallback;
            return loaded;
        }
    }
}
