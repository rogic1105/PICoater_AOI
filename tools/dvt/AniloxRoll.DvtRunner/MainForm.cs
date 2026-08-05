using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace AniloxRoll.DvtRunner
{
    internal sealed class MainForm : Form
    {
        private readonly TextBox _appPath = new TextBox();
        private readonly TextBox _logDirectory = new TextBox();
        private readonly ComboBox _scenario = new ComboBox();
        private readonly Button _start = new Button();
        private readonly Button _pause = new Button();
        private readonly Button _abort = new Button();
        private readonly ListView _steps = new ListView();
        private readonly RichTextBox _output = new RichTextBox();
        private readonly Label _status = new Label();
        private readonly string _autoScenarioId;
        private readonly string _resultPath;
        private readonly string _processIdPath;
        private readonly int? _durationSeconds;
        private readonly string _repositoryRootOverride;
        private readonly string _appPathOverride;
        private readonly string _logDirectoryOverride;
        private IReadOnlyList<DvtScenario> _scenarios;
        private ScenarioEngine _engine;
        private CancellationTokenSource _runCancellation;
        private string _repositoryRoot;

        public MainForm(
            string autoScenarioId = null,
            string resultPath = null,
            string processIdPath = null,
            int? durationSeconds = null,
            string repositoryRoot = null,
            string appPath = null,
            string logDirectory = null)
        {
            _autoScenarioId = autoScenarioId;
            _resultPath = resultPath;
            _processIdPath = processIdPath;
            _durationSeconds = durationSeconds;
            _repositoryRootOverride = repositoryRoot;
            _appPathOverride = appPath;
            _logDirectoryOverride = logDirectory;
            Text = "PICoater DVT Runner";
            StartPosition = FormStartPosition.CenterScreen;
            MinimumSize = new Size(960, 640);
            Size = new Size(1180, 760);
            Font = new Font("Microsoft JhengHei UI", 9F);
            BuildUi();
            Load += OnLoaded;
            FormClosing += OnFormClosing;
        }

        private void BuildUi()
        {
            var root = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3,
                Padding = new Padding(10)
            };
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            root.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            root.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var inputs = new TableLayoutPanel
            {
                Dock = DockStyle.Top,
                AutoSize = true,
                ColumnCount = 4,
                RowCount = 4
            };
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

            AddInput(inputs, 0, "監控程式", _appPath, BrowseExe);
            AddInput(inputs, 1, "Trace 目錄", _logDirectory, BrowseLogs);

            inputs.Controls.Add(new Label
            {
                Text = "測試情境",
                Anchor = AnchorStyles.Left,
                AutoSize = true
            }, 0, 2);
            _scenario.DropDownStyle = ComboBoxStyle.DropDownList;
            _scenario.Dock = DockStyle.Fill;
            _scenario.SelectedIndexChanged += (s, e) => PopulateSteps();
            inputs.Controls.Add(_scenario, 1, 2);
            inputs.SetColumnSpan(_scenario, 3);

            var commands = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                FlowDirection = FlowDirection.LeftToRight
            };
            _start.Text = "開始";
            _start.AutoSize = true;
            _start.Click += async (s, e) => await StartScenarioAsync();
            _pause.Text = "暫停";
            _pause.AutoSize = true;
            _pause.Enabled = false;
            _pause.Click += (s, e) => TogglePause();
            _abort.Text = "中止";
            _abort.AutoSize = true;
            _abort.Enabled = false;
            _abort.Click += (s, e) => _runCancellation?.Cancel();
            commands.Controls.AddRange(new Control[] { _start, _pause, _abort });
            inputs.Controls.Add(commands, 1, 3);
            inputs.SetColumnSpan(commands, 3);

            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal,
                SplitterDistance = 330
            };
            _steps.Dock = DockStyle.Fill;
            _steps.View = View.Details;
            _steps.FullRowSelect = true;
            _steps.GridLines = true;
            _steps.HideSelection = false;
            _steps.Columns.Add("狀態", 85);
            _steps.Columns.Add("契約", 85);
            _steps.Columns.Add("步驟", 310);
            _steps.Columns.Add("等待證據／結果", 620);
            split.Panel1.Controls.Add(_steps);

            _output.Dock = DockStyle.Fill;
            _output.ReadOnly = true;
            _output.BackColor = SystemColors.Window;
            _output.Font = new Font("Consolas", 9F);
            split.Panel2.Controls.Add(_output);

            _status.Dock = DockStyle.Fill;
            _status.AutoSize = true;
            _status.Padding = new Padding(0, 6, 0, 0);
            _status.Text = "尚未開始";

            root.Controls.Add(inputs, 0, 0);
            root.Controls.Add(split, 0, 1);
            root.Controls.Add(_status, 0, 2);
            Controls.Add(root);
        }

        private static void AddInput(
            TableLayoutPanel panel,
            int row,
            string label,
            TextBox textBox,
            EventHandler browse)
        {
            panel.Controls.Add(new Label
            {
                Text = label,
                Anchor = AnchorStyles.Left,
                AutoSize = true
            }, 0, row);
            textBox.Dock = DockStyle.Fill;
            panel.Controls.Add(textBox, 1, row);
            panel.SetColumnSpan(textBox, 2);
            var button = new Button { Text = "瀏覽...", AutoSize = true };
            button.Click += browse;
            panel.Controls.Add(button, 3, row);
        }

        private void OnLoaded(object sender, EventArgs e)
        {
            try
            {
                _repositoryRoot = string.IsNullOrWhiteSpace(_repositoryRootOverride)
                    ? RepositoryLocator.FindRoot()
                    : Path.GetFullPath(_repositoryRootOverride);
                _appPath.Text = string.IsNullOrWhiteSpace(_appPathOverride)
                    ? Path.Combine(
                        _repositoryRoot,
                        "bin",
                        "x64",
                        "Release",
                        "AniloxRoll.Monitor.exe")
                    : Path.GetFullPath(_appPathOverride);
                _logDirectory.Text = string.IsNullOrWhiteSpace(_logDirectoryOverride)
                    ? @"D:\Anilox\Logs"
                    : Path.GetFullPath(_logDirectoryOverride);
                string scenarioDirectory = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory, "Scenarios");
                _scenarios = ScenarioLoader.LoadDirectory(scenarioDirectory);
                foreach (DvtScenario item in _scenarios) _scenario.Items.Add(item);
                if (_scenario.Items.Count == 0)
                {
                    _status.Text = "沒有找到 DVT 情境。";
                    return;
                }

                if (string.IsNullOrWhiteSpace(_autoScenarioId))
                {
                    _scenario.SelectedIndex = 0;
                }
                else
                {
                    DvtScenario selected = _scenarios.FirstOrDefault(
                        item => string.Equals(
                            item.Id,
                            _autoScenarioId,
                            StringComparison.OrdinalIgnoreCase));
                    if (selected == null)
                        throw new InvalidDataException(
                            "找不到指定情境：" + _autoScenarioId);
                    ApplyDurationOverride(selected);
                    _scenario.SelectedItem = selected;
                    BeginInvoke(new Action(async () =>
                    {
                        Hide();
                        bool passed = await StartScenarioAsync();
                        WriteAutomationResult(passed);
                        Environment.ExitCode = passed ? 0 : 1;
                        Close();
                    }));
                }
            }
            catch (Exception ex)
            {
                _status.Text = "初始化失敗：" + ex.Message;
                _start.Enabled = false;
            }
        }

        private void ApplyDurationOverride(DvtScenario scenario)
        {
            DvtStep soak = scenario.Steps.SingleOrDefault(
                step => string.Equals(
                    step.Action, "soak", StringComparison.OrdinalIgnoreCase));
            if (_durationSeconds.HasValue && soak == null)
                throw new InvalidDataException(
                    "指定 --duration-seconds 的情境必須恰有一個 soak 步驟。");
            if (_durationSeconds.HasValue)
                soak.TimeoutSeconds = _durationSeconds.Value;

            int durationSeconds = soak?.TimeoutSeconds ?? 0;
            if (durationSeconds <= 0) return;

            var cycleToken = new System.Text.RegularExpressions.Regex(
                @"\{cycles:(\d+)\}",
                System.Text.RegularExpressions.RegexOptions.CultureInvariant);
            foreach (DvtStep step in scenario.Steps)
            {
                if (string.IsNullOrWhiteSpace(step.Value)) continue;
                step.Value = cycleToken.Replace(
                    step.Value,
                    match =>
                    {
                        int cycleMilliseconds = int.Parse(
                            match.Groups[1].Value,
                            System.Globalization.CultureInfo.InvariantCulture);
                        int cycles = Math.Max(
                            1,
                            (durationSeconds * 1000) /
                            cycleMilliseconds);
                        return cycles.ToString(
                            System.Globalization.CultureInfo.InvariantCulture);
                    });
            }
        }

        private void PopulateSteps()
        {
            _steps.Items.Clear();
            var selected = _scenario.SelectedItem as DvtScenario;
            if (selected == null) return;
            foreach (DvtStep step in selected.Steps)
            {
                var item = new ListViewItem("待執行");
                item.SubItems.Add(step.Contract ?? "DVT");
                item.SubItems.Add(step.Title);
                item.SubItems.Add(step.Pattern ?? step.Target ?? step.Action);
                item.Tag = step;
                _steps.Items.Add(item);
            }
            _status.Text =
                $"已選：{selected.Name}，共 {selected.Steps.Count} 步。{selected.Description}";
        }

        private async Task<bool> StartScenarioAsync()
        {
            var selected = _scenario.SelectedItem as DvtScenario;
            if (selected == null) return false;
            if (!File.Exists(_appPath.Text))
            {
                MessageBox.Show("找不到監控程式：" + _appPath.Text);
                return false;
            }
            if (!Directory.Exists(_logDirectory.Text))
                Directory.CreateDirectory(_logDirectory.Text);

            PopulateSteps();
            _output.Clear();
            SetRunning(true);
            _runCancellation = new CancellationTokenSource();
            _engine = new ScenarioEngine(new RunnerOptions
            {
                RepositoryRoot = _repositoryRoot,
                AppExePath = _appPath.Text,
                LogDirectory = _logDirectory.Text,
                ProcessIdPath = _processIdPath,
                CloseAppOnCleanup = !string.IsNullOrWhiteSpace(_autoScenarioId)
            });
            _engine.StepChanged += update => Ui(() => ApplyStepUpdate(update));
            _engine.Output += text => Ui(() => AppendOutput(text));
            try
            {
                _status.Text = "執行中：" + selected.Name;
                await Task.Run(
                    () => _engine.RunAsync(selected, _runCancellation.Token));
                _status.Text = "PASS：" + selected.Name;
                _status.ForeColor = Color.DarkGreen;
                return true;
            }
            catch (OperationCanceledException)
            {
                _status.Text = "已中止；已嘗試停止 Grab 並還原設定。";
                _status.ForeColor = Color.DarkOrange;
                return false;
            }
            catch (Exception ex)
            {
                _status.Text = "FAIL：" + ex.Message;
                _status.ForeColor = Color.DarkRed;
                AppendOutput(ex.ToString());
                return false;
            }
            finally
            {
                SetRunning(false);
                _runCancellation.Dispose();
                _runCancellation = null;
                _engine = null;
            }
        }

        private void WriteAutomationResult(bool passed)
        {
            if (string.IsNullOrWhiteSpace(_resultPath)) return;
            string directory = Path.GetDirectoryName(_resultPath);
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(
                _resultPath,
                "Result: " + (passed ? "PASS" : "FAIL") +
                Environment.NewLine +
                "Status: " + _status.Text +
                Environment.NewLine + Environment.NewLine +
                _output.Text,
                new UTF8Encoding(false));
        }

        private void ApplyStepUpdate(StepUpdate update)
        {
            if (update.Index < 0 || update.Index >= _steps.Items.Count) return;
            ListViewItem item = _steps.Items[update.Index];
            item.Text = StatusText(update.Status);
            item.SubItems[3].Text = update.Detail ?? "";
            item.BackColor =
                update.Status == StepStatus.Passed ? Color.Honeydew :
                update.Status == StepStatus.Failed ? Color.MistyRose :
                update.Status == StepStatus.Running ? Color.LightYellow :
                update.Status == StepStatus.Skipped ? Color.WhiteSmoke :
                SystemColors.Window;
            if (update.Status == StepStatus.Running)
                _status.Text = "執行中：" + update.Step.Title;
            item.EnsureVisible();
            WriteAutomationProgress(update);
        }

        private void WriteAutomationProgress(StepUpdate update)
        {
            if (string.IsNullOrWhiteSpace(_resultPath)) return;
            try
            {
                string directory = Path.GetDirectoryName(_resultPath);
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                File.WriteAllText(
                    _resultPath,
                    "Result: RUNNING" + Environment.NewLine +
                    "Step: " + (update.Step?.Id ?? "unknown") +
                    Environment.NewLine +
                    "StepStatus: " + update.Status + Environment.NewLine +
                    "Detail: " + (update.Detail ?? string.Empty) +
                    Environment.NewLine + Environment.NewLine +
                    _output.Text,
                    new UTF8Encoding(false));
            }
            catch (IOException)
            {
                // Progress is diagnostic only; the final result remains required.
            }
        }

        private static string StatusText(StepStatus status)
        {
            switch (status)
            {
                case StepStatus.Running: return "執行中";
                case StepStatus.Passed: return "PASS";
                case StepStatus.Failed: return "FAIL";
                case StepStatus.Skipped: return "略過";
                default: return "待執行";
            }
        }

        private void AppendOutput(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            _output.AppendText(text.TrimEnd() + Environment.NewLine);
            _output.SelectionStart = _output.TextLength;
            _output.ScrollToCaret();
        }

        private void TogglePause()
        {
            if (_engine == null) return;
            _engine.TogglePause();
            _pause.Text = _engine.IsPaused ? "繼續" : "暫停";
            _status.Text = _engine.IsPaused ? "已暫停，可在旁觀察畫面。" : "繼續執行。";
        }

        private void SetRunning(bool running)
        {
            _start.Enabled = !running;
            _scenario.Enabled = !running;
            _appPath.Enabled = !running;
            _logDirectory.Enabled = !running;
            _pause.Enabled = running;
            _abort.Enabled = running;
            _pause.Text = "暫停";
            if (running) _status.ForeColor = SystemColors.ControlText;
        }

        private void BrowseExe(object sender, EventArgs e)
        {
            using (var dialog = new OpenFileDialog
            {
                Filter = "AniloxRoll.Monitor.exe|AniloxRoll.Monitor.exe|Executable (*.exe)|*.exe",
                FileName = _appPath.Text
            })
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                    _appPath.Text = dialog.FileName;
            }
        }

        private void BrowseLogs(object sender, EventArgs e)
        {
            using (var dialog = new FolderBrowserDialog
            {
                SelectedPath = _logDirectory.Text
            })
            {
                if (dialog.ShowDialog(this) == DialogResult.OK)
                    _logDirectory.Text = dialog.SelectedPath;
            }
        }

        private void Ui(Action action)
        {
            if (IsDisposed) return;
            if (InvokeRequired) BeginInvoke(action);
            else action();
        }

        private void OnFormClosing(object sender, FormClosingEventArgs e)
        {
            if (_runCancellation == null) return;
            e.Cancel = true;
            _runCancellation.Cancel();
            _status.Text = "正在中止並還原設定，完成後再關閉。";
        }
    }
}
