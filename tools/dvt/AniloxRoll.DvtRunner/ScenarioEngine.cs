using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Web.Script.Serialization;

namespace AniloxRoll.DvtRunner
{
    internal sealed class ScenarioEngine
    {
        private const string ScheduledTaskHandlePrefix = "scheduled-task:";
        private const string IoBlockTaskName = "PICoater-DVT-Block-IO502";
        private const string IoUnblockTaskName = "PICoater-DVT-Unblock-IO502";
        private const string StorageBlockTaskName =
            "PICoater-DVT-Block-Storage";
        private const string StorageUnblockTaskName =
            "PICoater-DVT-Unblock-Storage";
        private const string LightDisableTaskName = "PICoater-DVT-Disable-COM17";
        private const string LightEnableTaskName = "PICoater-DVT-Enable-COM17";
        private const string FixedIoBlockedRoutePrefix =
            "192.168.255.1/32";
        private const int FixedIoBlackholeInterfaceIndex = 1;
        private const string FixedIoBlackholeNextHop = "0.0.0.0";
        private const string FixedStorageAddress = "192.168.10.20";
        private const string FixedStorageBlockedRoutePrefix =
            FixedStorageAddress + "/32";
        private const int FixedStorageBlackholeInterfaceIndex = 1;
        private const string FixedStorageBlackholeNextHop = "0.0.0.0";
        private const string RetentionFixtureDirectoryName =
            "PICoater-DVT-Retention";
        private const string RetentionFixtureMarkerName =
            ".dvt-retention-fixture";
        private const string RetentionFixtureMarkerValue =
            "PICoater DVT retention fixture v1";
        private const string RetentionRootPropertyName = "Anilox 根目錄";
        private const string RetentionThresholdPropertyName =
            "預留空間 (GB)";
        private readonly RunnerOptions _options;
        private readonly UiAutomationDriver _ui = new UiAutomationDriver();
        private readonly FlowLogMonitor _log;
        private readonly Dictionary<string, string> _originalProperties =
            new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly Dictionary<string, Process> _helperProcesses =
            new Dictionary<string, Process>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _blockedNetworkTargets =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _blockedTargetPorts =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, string> _disabledSerialDevices =
            new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        private readonly ManualResetEventSlim _pauseGate = new ManualResetEventSlim(true);
        private bool _captureMayBeActive;
        private string _retentionFixtureRoot;
        private string _retentionOldDayDirectory;
        private string _retentionOldDailyCsv;
        private string _retentionNewDayDirectory;
        private string _retentionNewDailyCsv;
        private int _retentionThresholdGb;
        private long _retentionFixtureBytes;
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
            _blockedNetworkTargets.Clear();
            _retentionFixtureRoot = null;
            _retentionOldDayDirectory = null;
            _retentionOldDailyCsv = null;
            _retentionNewDayDirectory = null;
            _retentionNewDailyCsv = null;
            _retentionThresholdGb = 0;
            _retentionFixtureBytes = 0;
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

                case "block-target-port":
                    return BlockTargetPort(step);

                case "unblock-target-port":
                    return UnblockTargetPort(step.Target, false);

                case "disable-serial-device":
                    return DisableSerialDevice(step);

                case "enable-serial-device":
                    return EnableSerialDevice(step.Target, false);

                case "prepare-retention-fixture":
                    return await PrepareRetentionFixtureAsync(
                        step.TimeoutSeconds, cancellationToken);

                case "verify-retention-fixture":
                    return VerifyRetentionFixture();

                case "cleanup-retention-fixture":
                    return CleanupRetentionFixture(false);

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

                case "wait-log-current-combo":
                    string comboValue = await _ui.ReadComboValueAsync(
                        step.Target, step.TimeoutSeconds, cancellationToken);
                    string currentPattern = step.Pattern.Replace(
                        "{value}", Regex.Escape(comboValue));
                    return await _log.WaitForAsync(
                        currentPattern, step.TimeoutSeconds, cancellationToken);

                case "verify-log-min-count":
                    int minimumCount = int.Parse(
                        step.Value,
                        NumberStyles.Integer,
                        CultureInfo.InvariantCulture);
                    int observedCount = await _log.WaitForMinimumCountAsync(
                        step.Pattern,
                        minimumCount,
                        step.TimeoutSeconds,
                        cancellationToken);
                    return $"count={observedCount} minimum={minimumCount}";

                case "verify-range-scroll":
                    return RangeScrollEvidenceVerifier.Verify(
                        _log.GetEvidenceLines());

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
            EnableAllDisabledSerialDevices();
            UnblockAllTargetPorts();
            UnblockAllNetworkTargets();

            using (var restoreTimeout =
                new CancellationTokenSource(TimeSpan.FromSeconds(30)))
            {
                await RestoreOriginalPropertiesAsync(restoreTimeout.Token);
            }
            CleanupRetentionFixture(true);

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

        private async Task<string> PrepareRetentionFixtureAsync(
            int timeoutSeconds,
            CancellationToken cancellationToken)
        {
            string root = Path.GetFullPath(Path.Combine(
                Path.GetTempPath(), RetentionFixtureDirectoryName));
            PrepareEmptyRetentionRoot(root);
            _retentionFixtureRoot = root;

            string markerPath = Path.Combine(
                root, RetentionFixtureMarkerName);
            File.WriteAllText(
                markerPath,
                RetentionFixtureMarkerValue,
                new UTF8Encoding(false));

            var drive = new DriveInfo(Path.GetPathRoot(root));
            long freeBefore = drive.AvailableFreeSpace;
            int thresholdGb = (int)(freeBefore / (1024L * 1024L * 1024L));
            if (thresholdGb < 1)
                throw new InvalidOperationException(
                    "Retention DVT requires at least 1 GiB of free space.");
            if ((long)thresholdGb * 1024L * 1024L * 1024L >=
                drive.TotalSize)
                throw new InvalidOperationException(
                    "Computed retention threshold is not below volume total.");

            DateTime oldDate = DateTime.Today.AddDays(-2);
            DateTime newDate = DateTime.Today.AddDays(-1);
            _retentionOldDayDirectory = GetRetentionDayDirectory(
                root, oldDate);
            _retentionNewDayDirectory = GetRetentionDayDirectory(
                root, newDate);
            Directory.CreateDirectory(_retentionOldDayDirectory);
            Directory.CreateDirectory(_retentionNewDayDirectory);

            _retentionOldDailyCsv = Path.Combine(
                Path.GetDirectoryName(_retentionOldDayDirectory),
                oldDate.ToString("yyyyMMdd", CultureInfo.InvariantCulture) +
                ".csv");
            _retentionNewDailyCsv = Path.Combine(
                Path.GetDirectoryName(_retentionNewDayDirectory),
                newDate.ToString("yyyyMMdd", CultureInfo.InvariantCulture) +
                ".csv");

            long thresholdBytes =
                (long)thresholdGb * 1024L * 1024L * 1024L;
            long bytesToAllocate =
                freeBefore - thresholdBytes + 128L * 1024L * 1024L;
            if (bytesToAllocate < 128L * 1024L * 1024L ||
                bytesToAllocate > 1200L * 1024L * 1024L)
                throw new InvalidOperationException(
                    "Unsafe retention fixture size: " +
                    bytesToAllocate.ToString(
                        CultureInfo.InvariantCulture) + " bytes.");

            string oldArchive = Path.Combine(
                _retentionOldDayDirectory, "retention-old.acap");
            WriteAllocatedFile(
                oldArchive, bytesToAllocate, cancellationToken);
            File.WriteAllText(
                _retentionOldDailyCsv, "old-day", new UTF8Encoding(false));
            File.WriteAllText(
                Path.Combine(
                    _retentionNewDayDirectory, "retention-new.acap"),
                "new-day",
                new UTF8Encoding(false));
            File.WriteAllText(
                _retentionNewDailyCsv, "new-day", new UTF8Encoding(false));

            drive = new DriveInfo(Path.GetPathRoot(root));
            if (drive.AvailableFreeSpace >= thresholdBytes)
                throw new InvalidOperationException(
                    "Fixture did not lower free space below the threshold.");

            if (!_originalProperties.ContainsKey(
                RetentionRootPropertyName))
                _originalProperties[RetentionRootPropertyName] =
                    _ui.GetPropertyValue(RetentionRootPropertyName);
            if (!_originalProperties.ContainsKey(
                RetentionThresholdPropertyName))
                _originalProperties[RetentionThresholdPropertyName] =
                    _ui.GetPropertyValue(RetentionThresholdPropertyName);

            await _ui.SetPropertyValueAsync(
                RetentionRootPropertyName,
                root,
                timeoutSeconds,
                cancellationToken);
            await _ui.SetPropertyValueAsync(
                RetentionThresholdPropertyName,
                thresholdGb.ToString(CultureInfo.InvariantCulture),
                timeoutSeconds,
                cancellationToken);

            _retentionThresholdGb = thresholdGb;
            _retentionFixtureBytes = bytesToAllocate;
            return string.Format(
                CultureInfo.InvariantCulture,
                "root={0} threshold={1}GiB fixture={2} bytes freeBefore={3}",
                root,
                thresholdGb,
                bytesToAllocate,
                freeBefore);
        }

        private string VerifyRetentionFixture()
        {
            if (string.IsNullOrWhiteSpace(_retentionFixtureRoot))
                throw new InvalidOperationException(
                    "Retention fixture was not prepared.");
            if (Directory.Exists(_retentionOldDayDirectory) ||
                File.Exists(_retentionOldDailyCsv))
                throw new InvalidOperationException(
                    "The oldest complete day was not fully deleted.");
            if (!Directory.Exists(_retentionNewDayDirectory) ||
                !File.Exists(_retentionNewDailyCsv) ||
                !File.Exists(Path.Combine(
                    _retentionNewDayDirectory, "retention-new.acap")))
                throw new InvalidOperationException(
                    "The newer complete day was deleted or damaged.");

            var drive = new DriveInfo(
                Path.GetPathRoot(_retentionFixtureRoot));
            long thresholdBytes =
                (long)_retentionThresholdGb * 1024L * 1024L * 1024L;
            if (drive.AvailableFreeSpace < thresholdBytes)
                throw new InvalidOperationException(
                    "Free space remains below the configured threshold.");

            return string.Format(
                CultureInfo.InvariantCulture,
                "oldest=deleted newer=preserved threshold={0}GiB " +
                "free={1} fixture={2}",
                _retentionThresholdGb,
                drive.AvailableFreeSpace,
                _retentionFixtureBytes);
        }

        private string CleanupRetentionFixture(bool cleanup)
        {
            if (string.IsNullOrWhiteSpace(_retentionFixtureRoot))
                return "retention fixture already clean";

            string root = _retentionFixtureRoot;
            _retentionFixtureRoot = null;
            string markerPath = Path.Combine(
                root, RetentionFixtureMarkerName);
            if (!Directory.Exists(root))
                return "retention fixture already absent";
            if (!File.Exists(markerPath) ||
                !string.Equals(
                    File.ReadAllText(markerPath, Encoding.UTF8),
                    RetentionFixtureMarkerValue,
                    StringComparison.Ordinal))
            {
                string message =
                    "Refused to delete retention fixture without marker: " +
                    root;
                if (cleanup)
                {
                    Output?.Invoke("[cleanup warning] " + message);
                    return message;
                }
                throw new InvalidOperationException(message);
            }

            DirectoryInfo info = new DirectoryInfo(root);
            if ((info.Attributes & FileAttributes.ReparsePoint) != 0 ||
                !string.Equals(
                    info.Name,
                    RetentionFixtureDirectoryName,
                    StringComparison.Ordinal))
                throw new InvalidOperationException(
                    "Refused to delete an unsafe retention fixture path: " +
                    root);

            Directory.Delete(root, true);
            return "retention fixture removed: " + root;
        }

        private static void PrepareEmptyRetentionRoot(string root)
        {
            if (Directory.Exists(root))
            {
                string markerPath = Path.Combine(
                    root, RetentionFixtureMarkerName);
                if (!File.Exists(markerPath) ||
                    !string.Equals(
                        File.ReadAllText(markerPath, Encoding.UTF8),
                        RetentionFixtureMarkerValue,
                        StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        "Existing retention test root has no valid marker: " +
                        root);
                DirectoryInfo info = new DirectoryInfo(root);
                if ((info.Attributes & FileAttributes.ReparsePoint) != 0)
                    throw new InvalidOperationException(
                        "Retention test root cannot be a reparse point.");
                Directory.Delete(root, true);
            }
            Directory.CreateDirectory(root);
        }

        private static string GetRetentionDayDirectory(
            string root,
            DateTime date)
        {
            return Path.Combine(
                root,
                "Captures",
                date.ToString("yyyy", CultureInfo.InvariantCulture),
                date.ToString("yyyyMM", CultureInfo.InvariantCulture),
                date.ToString("yyyyMMdd", CultureInfo.InvariantCulture));
        }

        private static void WriteAllocatedFile(
            string path,
            long bytes,
            CancellationToken cancellationToken)
        {
            byte[] buffer = new byte[4 * 1024 * 1024];
            using (var stream = new FileStream(
                path,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                buffer.Length,
                FileOptions.SequentialScan))
            {
                long remaining = bytes;
                while (remaining > 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int count = (int)Math.Min(buffer.Length, remaining);
                    stream.Write(buffer, 0, count);
                    remaining -= count;
                }
                stream.Flush(true);
            }
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
            if (_blockedNetworkTargets.ContainsKey(step.Id))
                throw new InvalidOperationException(
                    "Network disable step is already active: " + step.Id);

            string server = ParseUncServer(step.Target);
            System.Net.IPAddress address;
            if (!System.Net.IPAddress.TryParse(server, out address))
                throw new InvalidDataException(
                    "disable-target-network requires an IP-based UNC Target: " +
                    step.Target);

            if (!string.Equals(
                    server,
                    FixedStorageAddress,
                    StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidOperationException(
                    "Network isolation is restricted to the fixed storage " +
                    "endpoint " + FixedStorageAddress + ".");
            }
            if (!TryStartScheduledTask(StorageBlockTaskName))
            {
                throw new InvalidOperationException(
                    "Pre-authorized storage isolation action is missing. " +
                    "Run tests\\InstallDvtAdminActions.bat once.");
            }

            _blockedNetworkTargets.Add(
                step.Id,
                ScheduledTaskHandlePrefix + StorageUnblockTaskName);
            try
            {
                WaitForPowerShellBoolean(
                    "[bool](Get-NetRoute -DestinationPrefix '" +
                    FixedStorageBlockedRoutePrefix +
                    "' -PolicyStore ActiveStore " +
                    "-ErrorAction SilentlyContinue | Where-Object {" +
                    "$_.InterfaceIndex -eq " +
                    FixedStorageBlackholeInterfaceIndex.ToString() +
                    " -and $_.NextHop -eq '" +
                    FixedStorageBlackholeNextHop +
                    "'})",
                    true,
                    20,
                    "Pre-authorized storage blackhole route did not appear.");
                WaitForTcpReachability(
                    FixedStorageAddress,
                    445,
                    false,
                    20,
                    "Storage SMB remained reachable after installing the " +
                    "blackhole route.");
            }
            catch
            {
                try { EnableTargetNetwork(step.Id, true); }
                catch { }
                throw;
            }
            return "network isolated target=" + step.Target +
                " route=" + FixedStorageBlockedRoutePrefix;
        }

        private string EnableTargetNetwork(string disableStepId, bool cleanup)
        {
            string handle;
            if (string.IsNullOrWhiteSpace(disableStepId) ||
                !_blockedNetworkTargets.TryGetValue(
                    disableStepId, out handle))
            {
                if (cleanup) return "target network already enabled";
                throw new InvalidOperationException(
                    "Unknown network disable step: " + disableStepId);
            }

            if (!handle.StartsWith(
                    ScheduledTaskHandlePrefix,
                    StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "Unsupported network isolation handle: " + handle);
            }

            string taskName =
                handle.Substring(ScheduledTaskHandlePrefix.Length);
            if (!TryStartScheduledTask(taskName))
                throw new InvalidOperationException(
                    "Pre-authorized action is missing: " + taskName);
            WaitForPowerShellBoolean(
                "[bool](Get-NetRoute -DestinationPrefix '" +
                FixedStorageBlockedRoutePrefix +
                "' -PolicyStore ActiveStore " +
                "-ErrorAction SilentlyContinue | Where-Object {" +
                "$_.InterfaceIndex -eq " +
                FixedStorageBlackholeInterfaceIndex.ToString() +
                " -and $_.NextHop -eq '" +
                FixedStorageBlackholeNextHop +
                "'})",
                false,
                20,
                "Pre-authorized storage blackhole route was not removed.");
            WaitForTcpReachability(
                FixedStorageAddress,
                445,
                true,
                20,
                "Storage SMB did not recover after removing the " +
                "blackhole route.");
            _blockedNetworkTargets.Remove(disableStepId);
            return "network restored target=" + FixedStorageAddress;
        }

        private void UnblockAllNetworkTargets()
        {
            foreach (string stepId in new List<string>(
                _blockedNetworkTargets.Keys))
            {
                try
                {
                    Output?.Invoke(
                        "[cleanup] " + EnableTargetNetwork(stepId, true));
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        "[cleanup warning] network target recovery failed: " +
                        ex.Message);
                }
            }
            _blockedNetworkTargets.Clear();
        }

        private string BlockTargetPort(DvtStep step)
        {
            if (string.IsNullOrWhiteSpace(step.Target))
                throw new InvalidDataException(
                    "block-target-port requires an IPv4:port Target.");
            if (_blockedTargetPorts.ContainsKey(step.Id))
                throw new InvalidOperationException(
                    "Target port block step is already active: " + step.Id);

            int separator = step.Target.LastIndexOf(':');
            string addressText = separator > 0
                ? step.Target.Substring(0, separator)
                : string.Empty;
            string portText = separator > 0
                ? step.Target.Substring(separator + 1)
                : string.Empty;
            System.Net.IPAddress address;
            int port;
            if (!System.Net.IPAddress.TryParse(addressText, out address) ||
                address.AddressFamily !=
                    System.Net.Sockets.AddressFamily.InterNetwork ||
                !int.TryParse(portText, out port) ||
                port < 1 ||
                port > 65535)
            {
                throw new InvalidDataException(
                    "block-target-port requires an IPv4:port Target: " +
                    step.Target);
            }

            string ruleName;
            if (string.Equals(
                    addressText,
                    "192.168.255.1",
                    StringComparison.OrdinalIgnoreCase) &&
                port == 502 &&
                TryStartScheduledTask(IoBlockTaskName))
            {
                ruleName =
                    ScheduledTaskHandlePrefix + IoUnblockTaskName;
                _blockedTargetPorts.Add(step.Id, ruleName);
                try
                {
                    WaitForPowerShellBoolean(
                        "[bool](Get-NetRoute -DestinationPrefix '" +
                        FixedIoBlockedRoutePrefix +
                        "' -PolicyStore ActiveStore " +
                        "-ErrorAction SilentlyContinue | Where-Object {" +
                        "$_.InterfaceIndex -eq " +
                        FixedIoBlackholeInterfaceIndex.ToString() +
                        " -and $_.NextHop -eq '" +
                        FixedIoBlackholeNextHop +
                        "'})",
                        true,
                        20,
                        "Pre-authorized IO blackhole route did not appear.");
                    WaitForTcpReachability(
                        addressText,
                        port,
                        false,
                        20,
                        "IO TCP remained reachable after installing the " +
                        "blackhole route.");
                }
                catch
                {
                    try { UnblockTargetPort(step.Id, true); }
                    catch { }
                    throw;
                }
                return "target port blocked by pre-authorized action " +
                    addressText + ":" + portText;
            }

            ruleName =
                "PICoater-DVT-" +
                Process.GetCurrentProcess().Id.ToString() +
                "-" +
                Guid.NewGuid().ToString("N");
            try
            {
                string command =
                    "New-NetFirewallRule -Name '" +
                    PowerShellLiteral(ruleName) +
                    "' -DisplayName '" +
                    PowerShellLiteral(ruleName) +
                    "' -Direction Outbound -Action Block -Protocol TCP " +
                    "-RemoteAddress '" +
                    PowerShellLiteral(addressText) +
                    "' -RemotePort " +
                    port.ToString() +
                    " -Profile Any -ErrorAction Stop | Out-Null";
                RunPowerShellChecked(command, 30);
            }
            catch
            {
                try { RemoveFirewallRule(ruleName); } catch { }
                throw;
            }

            _blockedTargetPorts.Add(step.Id, ruleName);
            return "target port blocked " + addressText + ":" + portText;
        }

        private string UnblockTargetPort(string blockStepId, bool cleanup)
        {
            string ruleName;
            if (string.IsNullOrWhiteSpace(blockStepId) ||
                !_blockedTargetPorts.TryGetValue(blockStepId, out ruleName))
            {
                if (cleanup) return "target port already unblocked";
                throw new InvalidOperationException(
                    "Unknown target port block step: " + blockStepId);
            }

            if (ruleName.StartsWith(
                ScheduledTaskHandlePrefix,
                StringComparison.Ordinal))
            {
                string taskName =
                    ruleName.Substring(ScheduledTaskHandlePrefix.Length);
                if (!TryStartScheduledTask(taskName))
                    throw new InvalidOperationException(
                        "Pre-authorized action is missing: " + taskName);
                WaitForPowerShellBoolean(
                    "[bool](Get-NetRoute -DestinationPrefix '" +
                    FixedIoBlockedRoutePrefix +
                    "' -PolicyStore ActiveStore " +
                    "-ErrorAction SilentlyContinue | Where-Object {" +
                    "$_.InterfaceIndex -eq " +
                    FixedIoBlackholeInterfaceIndex.ToString() +
                    " -and $_.NextHop -eq '" +
                    FixedIoBlackholeNextHop +
                    "'})",
                    false,
                    20,
                    "Pre-authorized IO blackhole route was not removed.");
                WaitForTcpReachability(
                    FixedIoBlockedRoutePrefix.Substring(
                        0,
                        FixedIoBlockedRoutePrefix.IndexOf('/')),
                    502,
                    true,
                    20,
                    "IO TCP did not recover after removing the " +
                    "blackhole route.");
            }
            else
            {
                RemoveFirewallRule(ruleName);
            }
            _blockedTargetPorts.Remove(blockStepId);
            return "target port unblocked rule=" + ruleName;
        }

        private void UnblockAllTargetPorts()
        {
            foreach (string stepId in new List<string>(
                _blockedTargetPorts.Keys))
            {
                try
                {
                    Output?.Invoke(
                        "[cleanup] " + UnblockTargetPort(stepId, true));
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        "[cleanup warning] target port recovery failed: " +
                        ex.Message);
                }
            }
            _blockedTargetPorts.Clear();
        }

        private static void RemoveFirewallRule(string ruleName)
        {
            string command =
                "$rule=Get-NetFirewallRule -Name '" +
                PowerShellLiteral(ruleName) +
                "' -ErrorAction SilentlyContinue;" +
                "if($rule){$rule | Remove-NetFirewallRule -ErrorAction Stop}";
            RunPowerShellChecked(command, 30);
        }

        private string DisableSerialDevice(DvtStep step)
        {
            if (string.IsNullOrWhiteSpace(step.Target))
                throw new InvalidDataException(
                    "disable-serial-device requires a COM port Target.");
            if (_disabledSerialDevices.ContainsKey(step.Id))
                throw new InvalidOperationException(
                    "Serial disable step is already active: " + step.Id);

            string portName = step.Target.Trim();
            string instanceId = ResolveSerialInstanceId(portName);

            if (string.Equals(
                    portName,
                    "COM17",
                    StringComparison.OrdinalIgnoreCase) &&
                TryStartScheduledTask(LightDisableTaskName))
            {
                WaitForPowerShellBoolean(
                    GetPnpDeviceOkExpression(instanceId),
                    false,
                    20,
                    "Pre-authorized COM17 disable did not take effect.");
                _disabledSerialDevices.Add(
                    step.Id,
                    ScheduledTaskHandlePrefix + LightEnableTaskName);
                return "serial device disabled by pre-authorized action port=" +
                    portName;
            }

            try
            {
                string command =
                    "$device=Get-PnpDevice -InstanceId '" +
                    PowerShellLiteral(instanceId) +
                    "' -ErrorAction Stop;" +
                    "if($device.Status -ne 'OK'){" +
                    "throw ('Serial device is not ready: ' + $device.Status)};" +
                    "$device | Disable-PnpDevice -Confirm:$false " +
                    "-ErrorAction Stop";
                RunPowerShellChecked(command, 30);
            }
            catch
            {
                try { EnablePnpDevice(instanceId); } catch { }
                throw;
            }

            _disabledSerialDevices.Add(step.Id, instanceId);
            return "serial device disabled port=" + portName;
        }

        private string EnableSerialDevice(string disableStepId, bool cleanup)
        {
            string instanceId;
            if (string.IsNullOrWhiteSpace(disableStepId) ||
                !_disabledSerialDevices.TryGetValue(
                    disableStepId, out instanceId))
            {
                if (cleanup) return "serial device already enabled";
                throw new InvalidOperationException(
                    "Unknown serial disable step: " + disableStepId);
            }

            if (instanceId.StartsWith(
                ScheduledTaskHandlePrefix,
                StringComparison.Ordinal))
            {
                string taskName =
                    instanceId.Substring(ScheduledTaskHandlePrefix.Length);
                if (!TryStartScheduledTask(taskName))
                    throw new InvalidOperationException(
                        "Pre-authorized action is missing: " + taskName);
                string currentInstanceId = ResolveSerialInstanceId("COM17");
                WaitForPowerShellBoolean(
                    GetPnpDeviceOkExpression(currentInstanceId),
                    true,
                    20,
                    "Pre-authorized COM17 enable did not take effect.");
            }
            else
            {
                EnablePnpDevice(instanceId);
            }
            _disabledSerialDevices.Remove(disableStepId);
            return "serial device enabled";
        }

        private void EnableAllDisabledSerialDevices()
        {
            foreach (string stepId in new List<string>(
                _disabledSerialDevices.Keys))
            {
                try
                {
                    Output?.Invoke(
                        "[cleanup] " + EnableSerialDevice(stepId, true));
                }
                catch (Exception ex)
                {
                    Output?.Invoke(
                        "[cleanup warning] serial device recovery failed: " +
                        ex.Message);
                }
            }
            _disabledSerialDevices.Clear();
        }

        private static void EnablePnpDevice(string instanceId)
        {
            string command =
                "$device=Get-PnpDevice -InstanceId '" +
                PowerShellLiteral(instanceId) +
                "' -ErrorAction Stop;" +
                "if($device.Status -ne 'OK'){" +
                "$device | Enable-PnpDevice -Confirm:$false " +
                "-ErrorAction Stop};" +
                "$deadline=(Get-Date).AddSeconds(15);" +
                "do{Start-Sleep -Milliseconds 250;" +
                "$device=Get-PnpDevice -InstanceId '" +
                PowerShellLiteral(instanceId) +
                "' -ErrorAction Stop}" +
                "while($device.Status -ne 'OK' -and " +
                "(Get-Date) -lt $deadline);" +
                "if($device.Status -ne 'OK'){" +
                "throw ('Serial device did not recover: ' + $device.Status)}";
            RunPowerShellChecked(command, 30);
        }

        private static string ResolveSerialInstanceId(string portName)
        {
            string resolveCommand =
                "$port=Get-PnpDevice -Class Ports -ErrorAction Stop | " +
                "Where-Object {$_.FriendlyName -match '\\(" +
                PowerShellLiteral(portName) +
                "\\)$'} | Select-Object -First 1;" +
                "if(-not $port){throw 'Serial port not found'};" +
                "[Convert]::ToBase64String(" +
                "[Text.Encoding]::UTF8.GetBytes([string]$port.InstanceId))";
            string encoded = LastNonEmptyLine(
                RunPowerShellChecked(resolveCommand, 30));
            try
            {
                string instanceId = Encoding.UTF8.GetString(
                    Convert.FromBase64String(encoded));
                if (!string.IsNullOrWhiteSpace(instanceId))
                    return instanceId;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    "Unable to identify serial device for " + portName,
                    ex);
            }
            throw new InvalidOperationException(
                "Serial device ID is empty for " + portName);
        }

        private static string GetPnpDeviceOkExpression(string instanceId)
        {
            return
                "$device=Get-PnpDevice -InstanceId '" +
                PowerShellLiteral(instanceId) +
                "' -ErrorAction SilentlyContinue;" +
                "[bool]($device -and $device.Status -eq 'OK')";
        }

        private static bool TryStartScheduledTask(string taskName)
        {
            int exitCode;
            RunProcess(
                "schtasks.exe",
                "/Query /TN \"" + taskName + "\"",
                15,
                out exitCode);
            if (exitCode != 0) return false;

            RunProcessChecked(
                "schtasks.exe",
                "/Run /TN \"" + taskName + "\"",
                15);
            return true;
        }

        private static void WaitForPowerShellBoolean(
            string expression,
            bool expected,
            int timeoutSeconds,
            string failureMessage)
        {
            Stopwatch timer = Stopwatch.StartNew();
            while (timer.Elapsed.TotalSeconds < timeoutSeconds)
            {
                string result = LastNonEmptyLine(
                    RunPowerShellChecked(expression, 10));
                bool actual;
                if (bool.TryParse(result, out actual) && actual == expected)
                    return;
                Thread.Sleep(250);
            }
            throw new TimeoutException(failureMessage);
        }

        private static void WaitForTcpReachability(
            string address,
            int port,
            bool expected,
            int timeoutSeconds,
            string failureMessage)
        {
            Stopwatch timer = Stopwatch.StartNew();
            while (timer.Elapsed.TotalSeconds < timeoutSeconds)
            {
                bool reachable = false;
                using (var client = new TcpClient())
                {
                    try
                    {
                        IAsyncResult pending = client.BeginConnect(
                            address,
                            port,
                            null,
                            null);
                        using (WaitHandle waitHandle = pending.AsyncWaitHandle)
                        {
                            if (waitHandle.WaitOne(500))
                            {
                                client.EndConnect(pending);
                                reachable = client.Connected;
                            }
                        }
                    }
                    catch (SocketException)
                    {
                        reachable = false;
                    }
                    catch (ObjectDisposedException)
                    {
                        reachable = false;
                    }
                }

                if (reachable == expected)
                    return;
                Thread.Sleep(250);
            }
            throw new TimeoutException(failureMessage);
        }

        private static string RunPowerShellChecked(
            string command,
            int timeoutSeconds)
        {
            return RunProcessChecked(
                "powershell.exe",
                "-NoProfile -ExecutionPolicy Bypass -Command \"" +
                command.Replace("\"", "\\\"") + "\"",
                timeoutSeconds);
        }

        private static string PowerShellLiteral(string value)
        {
            return (value ?? string.Empty).Replace("'", "''");
        }

        private static string LastNonEmptyLine(string text)
        {
            string[] lines = (text ?? string.Empty).Split(
                new[] { '\r', '\n' },
                StringSplitOptions.RemoveEmptyEntries);
            if (lines.Length == 0)
                throw new InvalidOperationException(
                    "PowerShell command returned no output.");
            return lines[lines.Length - 1].Trim();
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
