using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AniloxRoll.DvtRunner
{
    internal sealed class ScenarioEngine
    {
        private readonly RunnerOptions _options;
        private readonly UiAutomationDriver _ui = new UiAutomationDriver();
        private readonly FlowLogMonitor _log;
        private readonly Dictionary<string, string> _originalProperties =
            new Dictionary<string, string>(StringComparer.Ordinal);
        private readonly ManualResetEventSlim _pauseGate = new ManualResetEventSlim(true);
        private bool _captureMayBeActive;

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
            _captureMayBeActive = false;
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
                await SafeCleanupAsync();
            }
        }

        private async Task<string> ExecuteStepAsync(
            DvtStep step,
            CancellationToken cancellationToken)
        {
            string action = step.Action.ToLowerInvariant();
            switch (action)
            {
                case "launch":
                    await _ui.AttachOrLaunchAsync(
                        _options.AppExePath, step.TimeoutSeconds, cancellationToken);
                    return "已連接 AniloxRoll.Monitor";

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

                case "wait-log":
                    string evidence = await _log.WaitForAsync(
                        step.Pattern, step.TimeoutSeconds, cancellationToken);
                    if (step.Pattern.Contains("background capture end") ||
                        step.Pattern.Contains("capture gate closed"))
                        _captureMayBeActive = false;
                    return evidence;

                case "delay":
                    await Task.Delay(
                        TimeSpan.FromSeconds(step.TimeoutSeconds), cancellationToken);
                    return "等待完成";

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
            using (var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(15)))
            {
                if (_captureMayBeActive)
                {
                    Output?.Invoke("[cleanup] 中止仍在進行的 Grab");
                    try
                    {
                        await _ui.TryStopCaptureAsync(timeout.Token);
                        Output?.Invoke("[cleanup] Grab 停止完成");
                    }
                    catch (Exception ex)
                    {
                        Output?.Invoke(
                            "清理警告：停止抓取失敗：" + ex.Message);
                    }
                }

                await RestoreOriginalPropertiesAsync(timeout.Token);
            }
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
