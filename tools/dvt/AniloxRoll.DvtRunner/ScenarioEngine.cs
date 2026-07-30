using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace AniloxRoll.DvtRunner
{
    internal sealed class ScenarioEngine
    {
        private readonly RunnerOptions _options;
        private readonly UiAutomationDriver _ui = new UiAutomationDriver();
        private readonly FlowLogMonitor _log;
        private readonly Dictionary<string, string> _originalProperties =
            new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<string, Process> _helperProcesses =
            new Dictionary<string, Process>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _disabledNetworkAdapters =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private readonly ManualResetEventSlim _pauseGate = new ManualResetEventSlim(true);
        private bool _captureMayBeActive;
        private string _sessionStatePath;
        private byte[] _originalSessionState;
        private bool _sessionStateExisted;

        public ScenarioEngine(RunnerOptions options)
        {
            _options = options;
            _log = new FlowLogMonitor(options.LogDirectory);
            _log.LineObserved += line => Output?.Invoke(line);
        }

        public event Action<StepUpdate> StepChanged;
        public event Action<string> Output;

        public bool IsPaused => !_pauseGate.IsSet;

        public void TogglePause()
        {
            if (_pauseGate.IsSet) _pauseGate.Reset();
            else _pauseGate.Set();
        }

        public async Task RunAsync(
            DvtScenario scenario,
            CancellationToken cancellationToken)
        {
            _log.BeginSession();
            _originalProperties.Clear();
            _helperProcesses.Clear();
            _disabledNetworkAdapters.Clear();
            _captureMayBeActive = false;
            CaptureSessionState();
            try
            {
                for (int i = 0; i < scenario.Steps.Count; i++)
                {
                    WaitWhilePaused(cancellationToken);
                    DvtStep step = scenario.Steps[i];
                    Raise(i, step, StepStatus.Running, "執行中");
                    Output?.Invoke(
                        $"[{step.Contract}] 開始：{step.Title} ({step.Action})");
                    try
                    {
                        string detail = await ExecuteStepAsync(step, cancellationToken);
                        Output?.Invoke(
                            $"[{step.Contract}] 完成：{step.Title} - {detail}");
                        Raise(i, step, StepStatus.Passed, detail);
                    }
                    catch (Exception ex) when (
                        step.Optional && !(ex is OperationCanceledException))
                    {
                        Raise(i, step, StepStatus.Skipped, ex.Message);
                    }
                    catch (Exception ex)
                    {
                        Output?.Invoke(
                            $"[{step.Contract}] 失敗：{step.Title} - {ex.Message}");
                        Raise(i, step, StepStatus.Failed, ex.Message);
                        throw;
                    }
                }
            }
            finally
            {
                _pauseGate.Set();
                try
                {
                    await SafeCleanupAsync();
                }
                finally
                {
                    RestoreSessionState();
                }
            }
        }

        private async Task<string> ExecuteStepAsync(
            DvtStep step,
            CancellationToken cancellationToken)
        {
            string action = step.Action.ToLowerInvariant();
            switch (action)
            {
                case "set-session-value":
                    SetSessionValue(step.Target, step.Value);
                    return step.Target + "=" + step.Value;

                case "launch":
                    await _ui.AttachOrLaunchAsync(
                        _options.AppExePath, step.TimeoutSeconds, cancellationToken);
                    if (!string.IsNullOrWhiteSpace(_options.ProcessIdPath))
                    {
                        string directory =
                            Path.GetDirectoryName(_options.ProcessIdPath);
                        if (!string.IsNullOrEmpty(directory))
                            Directory.CreateDirectory(directory);
                        File.WriteAllText(
                            _options.ProcessIdPath,
                            _ui.ProcessId.ToString(
                                System.Globalization.CultureInfo.InvariantCulture));
                    }
                    return "已連接 AniloxRoll.Monitor PID=" + _ui.ProcessId;

                case "launch-helper":
                    return LaunchHelper(step);

                case "wait-helper-exit":
                    return await WaitForHelperExitAsync(
                        step.Target,
                        step.TimeoutSeconds,
                        cancellationToken);

                case "stop-helper":
                    StopHelper(step.Target);
                    return "helper stopped: " + step.Target;

                case "disable-target-network":
                    return DisableTargetNetwork(step);

                case "enable-target-network":
                    return EnableTargetNetwork(step.Target, false);

                case "wait-element":
                    return await _ui.WaitForElementAsync(
                        step.Target, step.Value, step.TimeoutSeconds, cancellationToken);

                case "set-property":
                    if (!_originalProperties.ContainsKey(step.Target))
                        _originalProperties[step.Target] = _ui.GetPropertyValue(step.Target);
                    await _ui.SetPropertyValueAsync(
                        step.Target, step.Value, step.TimeoutSeconds, cancellationToken);
                    return step.Target + "=" + step.Value;

                case "click":
                    await _ui.ClickAsync(
                        step.Target, step.TimeoutSeconds, cancellationToken);
                    if (string.Equals(
                        step.Target, "開始抓取", StringComparison.Ordinal) ||
                        string.Equals(
                            step.Target, "取得背景", StringComparison.Ordinal))
                        _captureMayBeActive = true;
                    else if (string.Equals(
                        step.Target, "停止抓取", StringComparison.Ordinal))
                        _captureMayBeActive = false;
                    return "已觸發 " + step.Target;

                case "wheel":
                    return await _ui.WheelAsync(
                        step.Target,
                        step.Value,
                        step.TimeoutSeconds,
                        cancellationToken);

                case "drag":
                    return await _ui.DragAsync(
                        step.Target,
                        step.Value,
                        step.TimeoutSeconds,
                        cancellationToken);

                case "select-tab":
                    return await _ui.SelectTabAsync(
                        step.Target, step.TimeoutSeconds, cancellationToken);

                case "confirm-folder":
                    return await _ui.ConfirmFolderAsync(
                        step.Target, step.Value, step.TimeoutSeconds, cancellationToken);

                case "select-combo":
                    return await _ui.SelectComboAsync(
                        step.Target, step.Value, step.TimeoutSeconds, cancellationToken);

                case "wait-log":
                    string evidence = await _log.WaitForAsync(
                        step.Pattern, step.TimeoutSeconds, cancellationToken);
                    if (step.Pattern.Contains("capture gate open"))
                        _captureMayBeActive = true;
                    if (step.Pattern.Contains("background capture end") ||
                        step.Pattern.Contains("capture gate closed"))
                        _captureMayBeActive = false;
                    return evidence;

                case "reset-evidence":
                    _log.ResetEvidence();
                    return "後續只接受本階段新產生的 Flow 證據";

                case "delay":
                    await Task.Delay(
                        TimeSpan.FromSeconds(step.TimeoutSeconds), cancellationToken);
                    return "等待完成";

                case "soak":
                    await _ui.ObserveElementsAsync(
                        step.Target,
                        step.TimeoutSeconds,
                        message => Output?.Invoke(message),
                        cancellationToken);
                    return $"耐久觀察完成 {step.TimeoutSeconds}s";

                case "restore-properties":
                    await RestoreOriginalPropertiesAsync(cancellationToken);
                    return "Runner 修改的設定已還原";

                case "close-app":
                    await _ui.CloseAppAsync(
                        step.TimeoutSeconds, cancellationToken);
                    return "AniloxRoll.Monitor 已正常關閉";

                case "run-checker":
                    CheckerResult result = await DvtChecker.RunAsync(
                        _options.RepositoryRoot,
                        _options.LogDirectory,
                        cancellationToken);
                    Output?.Invoke(result.Output);
                    if (result.ExitCode != 0)
                        throw new InvalidOperationException(
                            "check_all_flows.py reported FAIL; see output below.");
                    return "check_all_flows.py exit=0";

                default:
                    throw new InvalidOperationException("Unsupported action: " + step.Action);
            }
        }

        private async Task SafeCleanupAsync()
        {
            if (_captureMayBeActive)
            {
                Output?.Invoke("[cleanup] 中止仍在進行的 Grab");
                using (var stopTimeout =
                    new CancellationTokenSource(TimeSpan.FromSeconds(15)))
                {
                    try
                    {
                        await _ui.TryStopCaptureAsync(stopTimeout.Token);
                        Output?.Invoke("[cleanup] Grab 停止完成");
                    }
                    catch (Exception ex)
                    {
                        Output?.Invoke(
                            "清理警告：停止抓取失敗：" + ex.Message);
                    }
                }
            }

            StopAllHelpers();
            EnableAllDisabledNetworkAdapters();

            using (var restoreTimeout =
                new CancellationTokenSource(TimeSpan.FromSeconds(30)))
            {
                await RestoreOriginalPropertiesAsync(restoreTimeout.Token);
            }

            if (_options.CloseAppOnCleanup && _ui.IsAttached)
            {
                Output?.Invoke("[cleanup] closing AniloxRoll.Monitor");
                using (var closeTimeout =
                    new CancellationTokenSource(TimeSpan.FromSeconds(65)))
                {
                    try
                    {
                        await _ui.CloseAppAsync(60, closeTimeout.Token);
                        Output?.Invoke("[cleanup] AniloxRoll.Monitor closed");
                        return;
                    }
                    catch (Exception ex)
                    {
                        Output?.Invoke(
                            "[cleanup warning] app close failed: " + ex.Message);
                    }
                }

                try
                {
                    bool exited = _ui.ForceTerminateApp(5);
                    Output?.Invoke(
                        exited
                            ? "[cleanup] AniloxRoll.Monitor force-terminated"
                            : "[cleanup warning] force termination timed out");
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        "[cleanup warning] force termination failed: " +
                        ex.Message);
                }
            }
        }

        private string LaunchHelper(DvtStep step)
        {
            if (string.IsNullOrWhiteSpace(step.Target))
                throw new InvalidDataException("launch-helper requires Target.");
            if (_helperProcesses.ContainsKey(step.Id))
                throw new InvalidOperationException(
                    "Helper step is already running: " + step.Id);

            string path = step.Target;
            if (!Path.IsPathRooted(path))
                path = Path.Combine(_options.RepositoryRoot, path);
            path = Path.GetFullPath(path);
            if (!File.Exists(path))
                throw new FileNotFoundException("Helper executable not found.", path);

            var start = new ProcessStartInfo
            {
                FileName = path,
                Arguments = step.Value ?? string.Empty,
                WorkingDirectory = Path.GetDirectoryName(path),
                UseShellExecute = false,
                CreateNoWindow = true
            };
            Process process = Process.Start(start);
            if (process == null)
                throw new InvalidOperationException("Failed to start helper: " + path);
            _helperProcesses.Add(step.Id, process);
            return "helper " + step.Id + " PID=" + process.Id;
        }

        private async Task<string> WaitForHelperExitAsync(
            string helperId,
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            if (string.IsNullOrWhiteSpace(helperId) ||
                !_helperProcesses.TryGetValue(helperId, out Process process))
                throw new InvalidOperationException(
                    "Unknown helper process: " + helperId);

            DateTime deadline = DateTime.UtcNow.AddSeconds(timeoutSeconds);
            while (!process.HasExited && DateTime.UtcNow < deadline)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Task.Delay(100, cancellationToken);
                process.Refresh();
            }
            if (!process.HasExited)
                throw new TimeoutException("Helper did not exit: " + helperId);

            int exitCode = process.ExitCode;
            process.Dispose();
            _helperProcesses.Remove(helperId);
            if (exitCode != 0)
                throw new InvalidOperationException(
                    "Helper failed: " + helperId + " exit=" + exitCode);
            return "helper " + helperId + " exit=0";
        }

        private void StopHelper(string helperId)
        {
            if (string.IsNullOrWhiteSpace(helperId) ||
                !_helperProcesses.TryGetValue(helperId, out Process process))
                return;
            StopProcess(process);
            _helperProcesses.Remove(helperId);
        }

        private void StopAllHelpers()
        {
            foreach (Process process in _helperProcesses.Values)
                StopProcess(process);
            _helperProcesses.Clear();
        }

        private static void StopProcess(Process process)
        {
            try
            {
                if (!process.HasExited)
                {
                    try { process.CloseMainWindow(); } catch { }
                    if (!process.WaitForExit(1000))
                    {
                        try { process.Kill(); } catch { }
                        process.WaitForExit(3000);
                    }
                }
            }
            finally
            {
                process.Dispose();
            }
        }

        private string DisableTargetNetwork(DvtStep step)
        {
            if (string.IsNullOrWhiteSpace(step.Target))
                throw new InvalidDataException(
                    "disable-target-network requires a UNC Target.");
            if (_disabledNetworkAdapters.ContainsKey(step.Id))
                throw new InvalidOperationException(
                    "Network disable step is already active: " + step.Id);

            string server = ParseUncServer(step.Target);
            System.Net.IPAddress address;
            if (!System.Net.IPAddress.TryParse(server, out address))
                throw new InvalidDataException(
                    "disable-target-network requires an IP-based UNC Target: " +
                    step.Target);

            string command =
                "$route=@(Find-NetRoute -RemoteIPAddress '" + server +
                "' | Where-Object { $_.PSObject.Properties['DestinationPrefix'] })" +
                " | Select-Object -First 1;" +
                "if(-not $route){throw 'No route to target'};" +
                "$adapter=Get-NetAdapter -InterfaceIndex $route.InterfaceIndex " +
                "-ErrorAction Stop;" +
                "if($adapter.Status -ne 'Up'){throw 'Target adapter is not Up'};" +
                "Write-Output ('INDEX=' + $adapter.ifIndex);" +
                "$adapter | Disable-NetAdapter -Confirm:$false";
            string output = RunProcessChecked(
                "powershell.exe",
                "-NoProfile -ExecutionPolicy Bypass -Command \"" +
                command.Replace("\"", "\\\"") + "\"",
                30);

            const string marker = "INDEX=";
            int markerIndex = output.IndexOf(
                marker, StringComparison.OrdinalIgnoreCase);
            if (markerIndex < 0)
                throw new InvalidOperationException(
                    "Unable to identify the disabled network adapter.");
            int endIndex = output.IndexOfAny(
                new[] { '\r', '\n' }, markerIndex);
            string interfaceIndex = output.Substring(
                markerIndex + marker.Length,
                (endIndex < 0 ? output.Length : endIndex) -
                markerIndex - marker.Length).Trim();
            int parsedInterfaceIndex;
            if (!int.TryParse(interfaceIndex, out parsedInterfaceIndex))
                throw new InvalidOperationException(
                    "Invalid disabled network adapter index: " + interfaceIndex);

            _disabledNetworkAdapters.Add(step.Id, interfaceIndex);
            return "network disabled target=" + step.Target +
                " ifIndex=" + interfaceIndex;
        }

        private string EnableTargetNetwork(string disableStepId, bool cleanup)
        {
            string interfaceIndex;
            if (string.IsNullOrWhiteSpace(disableStepId) ||
                !_disabledNetworkAdapters.TryGetValue(
                    disableStepId, out interfaceIndex))
            {
                if (cleanup) return "target network already enabled";
                throw new InvalidOperationException(
                    "Unknown network disable step: " + disableStepId);
            }

            string command =
                "$adapter=Get-NetAdapter -InterfaceIndex " + interfaceIndex +
                " -ErrorAction Stop;" +
                "if($adapter.Status -ne 'Up'){" +
                "$adapter | Enable-NetAdapter -Confirm:$false}";
            RunProcessChecked(
                "powershell.exe",
                "-NoProfile -ExecutionPolicy Bypass -Command \"" +
                command.Replace("\"", "\\\"") + "\"",
                30);
            _disabledNetworkAdapters.Remove(disableStepId);
            return "network enabled ifIndex=" + interfaceIndex;
        }

        private void EnableAllDisabledNetworkAdapters()
        {
            foreach (string stepId in new List<string>(
                _disabledNetworkAdapters.Keys))
            {
                try
                {
                    Output?.Invoke(
                        "[cleanup] " + EnableTargetNetwork(stepId, true));
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        "[cleanup warning] network adapter recovery failed: " +
                        ex.Message);
                }
            }
            _disabledNetworkAdapters.Clear();
        }

        private static string ParseUncServer(string uncPath)
        {
            if (!uncPath.StartsWith(@"\\", StringComparison.Ordinal))
                throw new InvalidDataException(
                    "Network Target must be a UNC path: " + uncPath);
            string[] parts = uncPath.TrimStart('\\').Split('\\');
            if (parts.Length < 2 || string.IsNullOrWhiteSpace(parts[0]))
                throw new InvalidDataException(
                    "Network Target must include server and share: " + uncPath);
            return parts[0];
        }

        private static string RunProcessChecked(
            string fileName,
            string arguments,
            int timeoutSeconds)
        {
            int exitCode;
            string output = RunProcess(
                fileName, arguments, timeoutSeconds, out exitCode);
            if (exitCode != 0)
            {
                throw new InvalidOperationException(
                    Path.GetFileName(fileName) + " failed exit=" + exitCode +
                    ". Run DVT Runner as administrator. " + output.Trim());
            }
            return output;
        }

        private static string RunProcess(
            string fileName,
            string arguments,
            int timeoutSeconds,
            out int exitCode)
        {
            var start = new ProcessStartInfo
            {
                FileName = fileName,
                Arguments = arguments,
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };
            using (Process process = Process.Start(start))
            {
                if (process == null)
                    throw new InvalidOperationException(
                        "Unable to start " + fileName);
                string stdout = process.StandardOutput.ReadToEnd();
                string stderr = process.StandardError.ReadToEnd();
                if (!process.WaitForExit(timeoutSeconds * 1000))
                {
                    try { process.Kill(); } catch { }
                    throw new TimeoutException(
                        Path.GetFileName(fileName) + " timed out.");
                }
                exitCode = process.ExitCode;
                return stdout + Environment.NewLine + stderr;
            }
        }

        private void SetSessionValue(string name, string value)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new InvalidDataException(
                    "set-session-value requires Target.");

            string directory = Path.GetDirectoryName(_sessionStatePath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);

            var serializer = new JavaScriptSerializer();
            Dictionary<string, object> state;
            if (File.Exists(_sessionStatePath))
            {
                string json = File.ReadAllText(
                    _sessionStatePath, Encoding.UTF8);
                state = serializer.Deserialize<Dictionary<string, object>>(json)
                    ?? new Dictionary<string, object>(
                        StringComparer.OrdinalIgnoreCase);
            }
            else
            {
                state = new Dictionary<string, object>(
                    StringComparer.OrdinalIgnoreCase);
            }

            state[name] = value;
            File.WriteAllText(
                _sessionStatePath,
                serializer.Serialize(state),
                new UTF8Encoding(false));
        }

        private async Task RestoreOriginalPropertiesAsync(
            CancellationToken cancellationToken)
        {
            foreach (KeyValuePair<string, string> pair in _originalProperties)
            {
                try
                {
                    Output?.Invoke(
                        $"[cleanup] 還原 {pair.Key}={pair.Value}");
                    await _ui.SetPropertyValueAsync(
                        pair.Key, pair.Value, 5, cancellationToken);
                    Output?.Invoke($"已還原 {pair.Key}={pair.Value}");
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        $"清理警告：{pair.Key} 還原失敗：{ex.Message}");
                }
            }
            _originalProperties.Clear();
        }

        private void CaptureSessionState()
        {
            string exeDirectory = Path.GetDirectoryName(_options.AppExePath);
            _sessionStatePath = Path.Combine(
                exeDirectory ?? string.Empty,
                "Config",
                "session-state.json");
            _sessionStateExisted = File.Exists(_sessionStatePath);
            _originalSessionState = _sessionStateExisted
                ? File.ReadAllBytes(_sessionStatePath)
                : null;
        }

        private void RestoreSessionState()
        {
            if (string.IsNullOrWhiteSpace(_sessionStatePath))
                return;
            try
            {
                if (_sessionStateExisted)
                {
                    Directory.CreateDirectory(
                        Path.GetDirectoryName(_sessionStatePath));
                    File.WriteAllBytes(
                        _sessionStatePath, _originalSessionState);
                }
                else if (File.Exists(_sessionStatePath))
                {
                    File.Delete(_sessionStatePath);
                }
                Output?.Invoke(
                    "[cleanup] session-state.json restored");
            }
            catch (Exception ex)
            {
                Output?.Invoke(
                    "[cleanup warning] session-state restore failed: " +
                    ex.Message);
            }
        }

        private void WaitWhilePaused(CancellationToken cancellationToken)
        {
            while (!_pauseGate.Wait(100))
                cancellationToken.ThrowIfCancellationRequested();
        }

        private void Raise(
            int index,
            DvtStep step,
            StepStatus status,
            string detail)
        {
            StepChanged?.Invoke(new StepUpdate
            {
                Index = index,
                Step = step,
                Status = status,
                Detail = detail
            });
        }
    }
}
