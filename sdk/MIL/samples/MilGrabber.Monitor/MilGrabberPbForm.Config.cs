using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Web.Script.Serialization; // JavaScriptSerializer 解析 system-settings.json
using Matrox.MatroxImagingLibrary;

namespace MilGrabber.Monitor
{
    // MilGrabberPbForm 的「Config 載入」分區：system-settings.json 反序列化 / fallback 預設 / descriptor 映射。
    // 與 UI 無關；巢狀型別 SystemConfig / CameraDeviceConfig 與 _devices 欄位留主檔（多處共用）。
    public partial class MilGrabberPbForm
    {
        // =========================================================================
        // Config 讀取：exe 同目錄 system-settings.json → JavaScriptSerializer 反序列化。
        // 檔不存在 / 解析失敗 / 空清單 → fallback 內建預設 7 相機（_usedFallbackConfig=true）。
        // =========================================================================
        private List<CameraDeviceConfig> LoadDeviceConfig()
        {
            _usedFallbackConfig = false;
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "system-settings.json");

            try
            {
                if (File.Exists(path))
                {
                    string json = File.ReadAllText(path);
                    var cfg = new JavaScriptSerializer().Deserialize<SystemConfig>(json);
                    if (cfg?.CameraDevices != null && cfg.CameraDevices.Count > 0)
                    {
                        ResolveDcfPaths(cfg.CameraDevices);
                        Trace.WriteLine($"[Config] 讀取 {path}：{cfg.CameraDevices.Count} 台相機。");
                        return cfg.CameraDevices;
                    }
                    Trace.WriteLine($"[Config] {path} 內容為空 / 無 CameraDevices，改用 fallback 預設。");
                }
                else
                {
                    Trace.WriteLine($"[Config] 找不到 {path}，改用 fallback 預設。");
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"[Config] 解析 {path} 失敗（{ex.GetType().Name}: {ex.Message}），改用 fallback 預設。");
            }

            _usedFallbackConfig = true;
            return CreateFallbackDevices();
        }

        /// <summary>內建預設：照 system-settings.json 那 7 筆硬編（SystemNum 0 有 4 台、1 有 3 台）。</summary>
        private static List<CameraDeviceConfig> CreateFallbackDevices()
        {
            const string desc = "M_SYSTEM_RADIENTEVCL";
            const string dcf = @"Config\Radient_Config.dcf";
            var devices = new List<CameraDeviceConfig>
            {
                new CameraDeviceConfig { Id = 1, SystemDescriptor = desc, SystemNum = 0, DevNum = 0, DcfPath = dcf },
                new CameraDeviceConfig { Id = 2, SystemDescriptor = desc, SystemNum = 0, DevNum = 1, DcfPath = dcf },
                new CameraDeviceConfig { Id = 3, SystemDescriptor = desc, SystemNum = 0, DevNum = 2, DcfPath = dcf },
                new CameraDeviceConfig { Id = 4, SystemDescriptor = desc, SystemNum = 0, DevNum = 3, DcfPath = dcf },
                new CameraDeviceConfig { Id = 5, SystemDescriptor = desc, SystemNum = 1, DevNum = 0, DcfPath = dcf },
                new CameraDeviceConfig { Id = 6, SystemDescriptor = desc, SystemNum = 1, DevNum = 1, DcfPath = dcf },
                new CameraDeviceConfig { Id = 7, SystemDescriptor = desc, SystemNum = 1, DevNum = 2, DcfPath = dcf },
            };
            ResolveDcfPaths(devices);
            return devices;
        }

        private static void ResolveDcfPaths(IEnumerable<CameraDeviceConfig> devices)
        {
            string baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
            foreach (CameraDeviceConfig device in devices)
            {
                if (device == null || string.IsNullOrWhiteSpace(device.DcfPath))
                    continue;
                if (!Path.IsPathRooted(device.DcfPath))
                    device.DcfPath = Path.GetFullPath(Path.Combine(baseDirectory, device.DcfPath));
            }
        }

        /// <summary>
        /// config 的 SystemDescriptor 字串 → MIL system descriptor。
        /// 照主程式：MsysAlloc 收字串描述子；"M_SYSTEM_RADIENTEVCL" → MIL.M_SYSTEM_RADIENTEVCL 常數
        /// （該常數本身即字串 "M_SYSTEM_RADIENTEVCL"）。未知值原樣傳回。
        /// </summary>
        private static string MapDescriptor(string descriptor)
        {
            if (string.IsNullOrWhiteSpace(descriptor)) return MIL.M_SYSTEM_RADIENTEVCL;
            if (descriptor == "M_SYSTEM_RADIENTEVCL") return MIL.M_SYSTEM_RADIENTEVCL;
            return descriptor; // 其他擷取卡描述子原樣傳遞給 MsysAlloc
        }
    }
}
