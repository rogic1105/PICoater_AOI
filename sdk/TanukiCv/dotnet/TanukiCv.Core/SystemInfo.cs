using System;
using System.Collections.Generic;
using System.Linq;
using System.Management;
using System.Runtime.InteropServices;

namespace TanukiCv.Core
{
    /// <summary>螢幕實體度量（GDI32 GetDeviceCaps）。MmPerPx 是「實體 1:1 / 倍率」計算的關鍵料。</summary>
    public struct ScreenMetrics
    {
        public int HorzMm, VertMm;       // 螢幕物理尺寸 mm
        public int HorzPx, VertPx;       // 有效解析度 px（含 DPI 縮放）
        public int NativeW, NativeH;     // 原生解析度 px
        public int DpiScalePct;          // DPI 縮放 %
        public double MmPerPx;           // 螢幕每像素 mm = HorzMm / HorzPx
    }

    /// <summary>
    /// 通用系統資訊查詢（CPU / RAM / GPU / 螢幕）——純查詢、無 GUI、無 MIL。
    /// 跨專案唯一來源：主程式 listViewHardware、硬體工具、MIL 範例三擊計算共用。
    /// MIL Grabber / 應用層磁碟/Storage 等專屬資訊不在此（留各自原處）。
    /// </summary>
    public static class SystemInfo
    {
        [DllImport("gdi32.dll")] private static extern int GetDeviceCaps(IntPtr hdc, int index);
        [DllImport("user32.dll")] private static extern IntPtr GetDC(IntPtr hwnd);
        [DllImport("user32.dll")] private static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);

        // GetDeviceCaps index
        private const int HORZSIZE = 4, VERTSIZE = 6, HORZRES = 8, VERTRES = 10, LOGPIXELSX = 88, LOGPIXELSY = 90;

        /// <summary>螢幕實體度量（含 MmPerPx）。失敗回傳全 0。</summary>
        public static ScreenMetrics GetScreenMetrics()
        {
            try
            {
                IntPtr hdc = GetDC(IntPtr.Zero);
                try
                {
                    int horzMm = GetDeviceCaps(hdc, HORZSIZE);
                    int vertMm = GetDeviceCaps(hdc, VERTSIZE);
                    int horzPx = GetDeviceCaps(hdc, HORZRES);
                    int vertPx = GetDeviceCaps(hdc, VERTRES);
                    int dpiX = GetDeviceCaps(hdc, LOGPIXELSX);
                    int dpiY = GetDeviceCaps(hdc, LOGPIXELSY);
                    return new ScreenMetrics
                    {
                        HorzMm = horzMm, VertMm = vertMm, HorzPx = horzPx, VertPx = vertPx,
                        NativeW = (int)Math.Round(horzPx * dpiX / 96.0),
                        NativeH = (int)Math.Round(vertPx * dpiY / 96.0),
                        DpiScalePct = (int)Math.Round(dpiX / 96.0 * 100),
                        MmPerPx = horzPx > 0 ? (double)horzMm / horzPx : 0
                    };
                }
                finally { ReleaseDC(IntPtr.Zero, hdc); }
            }
            catch { return default(ScreenMetrics); }
        }

        /// <summary>螢幕顯示列（ScreenSize / NativeRes / EffectiveRes / DpiScale / mm/px）。
        /// 與原 listViewHardware 一致；mm/px 計算的單一格式來源（消費端不必自己組字串）。</summary>
        public static List<KeyValuePair<string, string>> GetScreenRows()
        {
            var rows = new List<KeyValuePair<string, string>>();
            var s = GetScreenMetrics();
            if (s.HorzPx > 0)
            {
                rows.Add(new KeyValuePair<string, string>("ScreenSize", $"{s.HorzMm / 10.0:F1} × {s.VertMm / 10.0:F1} cm"));
                rows.Add(new KeyValuePair<string, string>("NativeRes", $"{s.NativeW} × {s.NativeH}"));
                rows.Add(new KeyValuePair<string, string>("EffectiveRes", $"{s.HorzPx} × {s.VertPx}"));
                rows.Add(new KeyValuePair<string, string>("DpiScale", $"{s.DpiScalePct}%"));
                rows.Add(new KeyValuePair<string, string>("mm/px", $"{s.MmPerPx:F4}"));
            }
            return rows;
        }

        /// <summary>通用計算硬體列（CPU / CPU_Cores / RAM / GPU / GPU_VRAM）。不含螢幕（見 GetScreenRows）。</summary>
        public static List<KeyValuePair<string, string>> GetGenericHardwareRows()
        {
            var rows = new List<KeyValuePair<string, string>>();
            void Add(string k, string v) => rows.Add(new KeyValuePair<string, string>(k, v));

            // ── CPU / RAM ──
            try
            {
                using (var cpu = new ManagementObjectSearcher("SELECT Name, NumberOfCores, NumberOfLogicalProcessors FROM Win32_Processor"))
                    foreach (var obj in cpu.Get())
                    {
                        Add("CPU", obj["Name"]?.ToString().Trim() ?? "N/A");
                        Add("CPU_Cores", $"{obj["NumberOfCores"]}C / {obj["NumberOfLogicalProcessors"]}T");
                        break; // 只取第一顆
                    }

                using (var mem = new ManagementObjectSearcher("SELECT Capacity, Speed, SMBIOSMemoryType FROM Win32_PhysicalMemory"))
                {
                    var sticks = mem.Get().Cast<ManagementObject>().ToArray();
                    int count = sticks.Length; ulong totalBytes = 0; int speed = 0, memType = 0;
                    foreach (var stick in sticks)
                    {
                        totalBytes += (ulong)stick["Capacity"];
                        if (speed == 0 && stick["Speed"] != null) speed = Convert.ToInt32(stick["Speed"]);
                        if (memType == 0 && stick["SMBIOSMemoryType"] != null) memType = Convert.ToInt32(stick["SMBIOSMemoryType"]);
                    }
                    double perStickGb = count > 0 ? (totalBytes / (double)count) / (1024.0 * 1024 * 1024) : 0;
                    string ddrGen = memType == 34 ? "DDR5" : memType == 26 ? "DDR4" : memType == 24 ? "DDR3" : "DDR";
                    string speedStr = speed > 0 ? $"-{speed}" : "";
                    Add("RAM", $"{totalBytes / (1024.0 * 1024 * 1024):F0} GB ({count}×{perStickGb:F0}GB {ddrGen}{speedStr})");
                }
            }
            catch { /* WMI 非關鍵 */ }

            // ── GPU（Registry 查 64-bit VRAM 避免 WMI uint32 溢位）──
            try
            {
                var regVram = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
                try
                {
                    using (var videoKey = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"))
                        if (videoKey != null)
                            foreach (string sub in videoKey.GetSubKeyNames())
                            {
                                if (!int.TryParse(sub, out _)) continue;
                                using (var sk = videoKey.OpenSubKey(sub))
                                {
                                    if (sk == null) continue;
                                    string desc = sk.GetValue("DriverDesc") as string;
                                    if (string.IsNullOrEmpty(desc)) continue;
                                    object qw = sk.GetValue("HardwareInformation.qwMemorySize");
                                    if (qw is long qwVal && qwVal > 0) regVram[desc] = qwVal;
                                    else if (qw is byte[] qwBytes && qwBytes.Length >= 8) regVram[desc] = BitConverter.ToInt64(qwBytes, 0);
                                }
                            }
                }
                catch { /* registry 非關鍵 */ }

                using (var gpu = new ManagementObjectSearcher("SELECT Name, AdapterRAM FROM Win32_VideoController"))
                    foreach (ManagementObject obj in gpu.Get())
                    {
                        string gpuName = obj["Name"]?.ToString() ?? "N/A";
                        long vramBytes = regVram.TryGetValue(gpuName, out long regBytes) && regBytes > 0
                            ? regBytes : Convert.ToUInt32(obj["AdapterRAM"]);
                        double vramGb = vramBytes / (1024.0 * 1024 * 1024);
                        Add("GPU", gpuName);
                        Add("GPU_VRAM", vramGb >= 1.0 ? $"{vramGb:F1} GB" : $"{vramBytes / (1024.0 * 1024):F0} MB");
                    }
            }
            catch { /* WMI 非關鍵 */ }

            return rows;
        }
    }
}
