// PICoater_AOI\src\dotnet\AniloxRoll.Monitor\Program.cs

// PICoater_AOI\src\dotnet\AniloxRoll.Monitor\Program.cs

using System;
using System.Configuration;
using System.IO;
using System.Windows.Forms;
using AniloxRoll.Monitor.Forms;

namespace AniloxRoll.Monitor
{
    internal static class Program
    {
        [STAThread]
        static void Main()
        {
            // Storage 模式不部署 MIL DLL；若其他路徑意外觸發載入，給出明確錯誤而非 crash
            AppDomain.CurrentDomain.AssemblyResolve += (_, args) =>
            {
                if (args.Name != null && args.Name.StartsWith("Matrox.MatroxImagingLibrary",
                        StringComparison.OrdinalIgnoreCase))
                {
                    System.Diagnostics.Trace.TraceWarning(
                        "[Program] Matrox.MatroxImagingLibrary 不存在（Storage 模式）。" +
                        "請確認 InitCameraLayer 在 Storage 模式已跳過。");
                }
                return null;
            };

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            TryDeleteCorruptedUserConfig();
            Application.Run(new AniloxRollForm());
        }

        /// <summary>
        /// 若 user.config 損毀（含 null byte / XML 格式錯誤）會在啟動時拋出
        /// ConfigurationErrorsException，導致程式無法開啟。
        /// 偵測到損毀時自動刪除，讓程式以預設值啟動。
        /// </summary>
        private static void TryDeleteCorruptedUserConfig()
        {
            try
            {
                // 強制讀取一次，讓 .NET 在這裡拋例外而非在 Form 建構時
                var _ = Properties.Settings.Default;
            }
            catch (ConfigurationErrorsException ex)
            {
                string path = (ex.Filename ?? string.Empty);
                if (string.IsNullOrEmpty(path))
                {
                    // 從 inner exception 取得路徑
                    var inner = ex.InnerException as ConfigurationErrorsException;
                    if (inner != null) path = inner.Filename ?? string.Empty;
                }

                if (!string.IsNullOrEmpty(path) && File.Exists(path))
                {
                    try
                    {
                        File.Delete(path);
                        System.Diagnostics.Trace.WriteLine(
                            $"[Program] 已刪除損毀的 user.config：{path}");
                    }
                    catch { /* 刪除失敗就繼續，讓程式嘗試啟動 */ }
                }
            }
        }
    }
}