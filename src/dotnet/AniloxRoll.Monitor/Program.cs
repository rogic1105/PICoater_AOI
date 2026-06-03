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

            // 全域例外攔截：背景執行緒未處理例外（如 startup 期間 CLProtocol/光源/儲存背景回 UI 的
            // BeginInvoke/Invoke 在 Handle 未建立時拋 InvalidOperationException）會直接終止 process（exit 0xffffffff）。
            // 這裡統一攔截並寫完整 stack 到 crash log，UI 執行緒例外改為記錄後續跑（不硬崩）。
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
            Application.ThreadException += (_, e) => LogFatal("ThreadException(UI)", e.Exception);
            AppDomain.CurrentDomain.UnhandledException += (_, e) => LogFatal("UnhandledException(BG)", e.ExceptionObject as Exception);
            System.Threading.Tasks.TaskScheduler.UnobservedTaskException += (_, e) =>
            {
                LogFatal("UnobservedTaskException", e.Exception);
                e.SetObserved();
            };

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            TryDeleteCorruptedUserConfig();
            Application.Run(new AniloxRollForm());
        }

        /// <summary>把致命例外的完整 stack 寫到 %TEMP% 與 exe 目錄的 crash log，並輸出到 Trace（VS 輸出視窗可見）。</summary>
        private static void LogFatal(string source, Exception ex)
        {
            string msg = $"[FATAL][{source}] {DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}\r\n" +
                         (ex?.ToString() ?? "(null exception object)") + "\r\n" +
                         new string('-', 80) + "\r\n";
            System.Diagnostics.Trace.WriteLine(msg);
            foreach (var dir in new[] { Path.GetTempPath(), AppDomain.CurrentDomain.BaseDirectory })
            {
                try { File.AppendAllText(Path.Combine(dir, "AniloxRoll-crash.log"), msg); }
                catch { /* log 失敗不可再丟例外 */ }
            }
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