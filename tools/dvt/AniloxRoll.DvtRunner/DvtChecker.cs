using System;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.DvtRunner
{
    internal sealed class CheckerResult
    {
        public int ExitCode { get; set; }
        public string Output { get; set; }
    }

    internal static class DvtChecker
    {
        public static async Task<CheckerResult> RunAsync(
            string repositoryRoot,
            string logDirectory,
            CancellationToken cancellationToken)
        {
            string script = System.IO.Path.Combine(
                repositoryRoot, "tools", "python", "check_all_flows.py");
            var startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments =
                    "\"" + script + "\" --latest --log-dir \"" + logDirectory + "\"",
                WorkingDirectory = repositoryRoot,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8
            };
            startInfo.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";

            using (var process = Process.Start(startInfo))
            {
                Task<string> stdout = process.StandardOutput.ReadToEndAsync();
                Task<string> stderr = process.StandardError.ReadToEndAsync();
                await Task.Run(() =>
                {
                    while (!process.WaitForExit(200))
                        cancellationToken.ThrowIfCancellationRequested();
                }, cancellationToken);
                string output = await stdout;
                string error = await stderr;
                return new CheckerResult
                {
                    ExitCode = process.ExitCode,
                    Output = output +
                        (string.IsNullOrWhiteSpace(error)
                            ? ""
                            : Environment.NewLine + "[stderr]" + Environment.NewLine + error)
                };
            }
        }
    }
}
