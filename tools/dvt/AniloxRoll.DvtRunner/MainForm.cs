using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Automation;
using System.Windows.Forms;

namespace AniloxRoll.DvtRunner
{
    internal sealed class MainForm : Form
    {
        private readonly TextBox _appPath = new TextBox();
        private readonly TextBox _logDirectory = new TextBox();
        private readonly TreeView _scenarioTree = new TreeView();
        private readonly Button _start = new Button();
        private readonly Button _pause = new Button();
        private readonly Button _abort = new Button();
        private readonly ListView _steps = new ListView();
        private readonly RichTextBox _output = new RichTextBox();
        private readonly Label _status = new Label();
        private readonly Button _attachMonitor = new Button();
        private readonly Button _pickMonitorElement = new Button();
        private readonly CheckBox _followMonitorFocus = new CheckBox();
        private readonly Button _clearControlFilter = new Button();
        private readonly Label _inspectorStatus = new Label();
        private readonly System.Windows.Forms.Timer _focusInspectorTimer =
            new System.Windows.Forms.Timer();
        private readonly object _runOutputGate = new object();
        private readonly StringBuilder _runOutput = new StringBuilder();
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
        private UiAutomationDriver _inspectorDriver;
        private MonitorElementPicker _elementPicker;
        private string _uiReferenceFilter;
        private string[] _inspectorControlIds = new string[0];
        private string[] _inspectorPropertyNames = new string[0];
        private string _lastFocusedReferenceKey;

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
            MinimumSize = new Size(1100, 680);
            Size = new Size(1500, 860);
            WindowState = FormWindowState.Maximized;
            Font = new Font("Microsoft JhengHei UI", 9F);
            BuildUi();
            _focusInspectorTimer.Interval = 250;
            _focusInspectorTimer.Tick += OnFocusInspectorTick;
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
                RowCount = 3
            };
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            inputs.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));

            AddInput(inputs, 0, "監控程式", _appPath, BrowseExe);
            AddInput(inputs, 1, "Trace 目錄", _logDirectory, BrowseLogs);

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
            inputs.Controls.Add(commands, 1, 2);
            inputs.SetColumnSpan(commands, 3);

            var runnerArea = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal,
                SplitterDistance = 210
            };
            var scenarioArea = new TableLayoutPanel
            {
                Dock = DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 3
            };
            scenarioArea.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            scenarioArea.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            scenarioArea.RowStyles.Add(new RowStyle(SizeType.AutoSize));

            var inspectorCommands = new FlowLayoutPanel
            {
                Dock = DockStyle.Fill,
                AutoSize = true,
                Padding = new Padding(0, 0, 0, 4)
            };
            _attachMonitor.Text = "連接 Monitor";
            _attachMonitor.AutoSize = true;
            _attachMonitor.Click += async (s, e) =>
                await AttachInspectorAsync(bringMonitorForward: false);
            _pickMonitorElement.Text = "選取真實元件";
            _pickMonitorElement.AutoSize = true;
            _pickMonitorElement.Click += async (s, e) =>
                await ToggleElementPickerAsync();
            _followMonitorFocus.Text = "跟隨 Monitor 焦點";
            _followMonitorFocus.AutoSize = true;
            _followMonitorFocus.CheckedChanged += (s, e) =>
            {
                _lastFocusedReferenceKey = null;
                if (_followMonitorFocus.Checked)
                    _focusInspectorTimer.Start();
                else
                    _focusInspectorTimer.Stop();
            };
            _clearControlFilter.Text = "顯示全部 DVT";
            _clearControlFilter.AutoSize = true;
            _clearControlFilter.Enabled = false;
            _clearControlFilter.Click += (s, e) => ApplyUiReferenceFilter(null);
            inspectorCommands.Controls.AddRange(new Control[]
            {
                _attachMonitor,
                _pickMonitorElement,
                _followMonitorFocus,
                _clearControlFilter
            });
            scenarioArea.Controls.Add(inspectorCommands, 0, 0);

            _scenarioTree.Dock = DockStyle.Fill;
            _scenarioTree.HideSelection = false;
            _scenarioTree.FullRowSelect = true;
            _scenarioTree.ShowNodeToolTips = true;
            _scenarioTree.AfterSelect += (s, e) => PopulateSteps();
            scenarioArea.Controls.Add(_scenarioTree, 0, 1);
            _inspectorStatus.Dock = DockStyle.Fill;
            _inspectorStatus.AutoSize = true;
            _inspectorStatus.Padding = new Padding(0, 4, 0, 2);
            _inspectorStatus.Text =
                "直接操作真實 Monitor；需要查關聯測試時按「選取真實元件」。";
            scenarioArea.Controls.Add(_inspectorStatus, 0, 2);
            runnerArea.Panel1.Controls.Add(scenarioArea);

            var split = new SplitContainer
            {
                Dock = DockStyle.Fill,
                Orientation = Orientation.Horizontal,
                SplitterDistance = 300
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
            runnerArea.Panel2.Controls.Add(split);

            _status.Dock = DockStyle.Fill;
            _status.AutoSize = true;
            _status.Padding = new Padding(0, 6, 0, 0);
            _status.Text = "尚未開始";

            root.Controls.Add(inputs, 0, 0);
            root.Controls.Add(runnerArea, 0, 1);
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
                PopulateScenarioTree();
                if (_scenarioTree.Nodes.Count == 0)
                {
                    _status.Text = "沒有找到 DVT 情境。";
                    return;
                }

                if (string.IsNullOrWhiteSpace(_autoScenarioId))
                {
                    SelectFirstScenario();
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
                    SelectScenario(selected);
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
            DvtScenario selected = SelectedScenario;
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
            DvtScenario selected = SelectedScenario;
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
            lock (_runOutputGate)
                _runOutput.Clear();
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
            _engine.Output += HandleOutput;
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
                HandleOutput(ex.ToString());
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
                GetRunOutput(),
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
                    GetRunOutput(),
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

        private void HandleOutput(string text)
        {
            if (string.IsNullOrEmpty(text)) return;
            lock (_runOutputGate)
                _runOutput.AppendLine(text.TrimEnd());

            // Automated runs persist the complete trace in the result file.
            // Rendering every high-density Flow line in a RichTextBox floods
            // the UI queue during large-data scenarios and stalls the runner.
            if (string.IsNullOrWhiteSpace(_autoScenarioId))
                Ui(() => AppendOutput(text));
        }

        private string GetRunOutput()
        {
            lock (_runOutputGate)
                return _runOutput.ToString();
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
            _scenarioTree.Enabled = !running;
            _appPath.Enabled = !running;
            _logDirectory.Enabled = !running;
            _attachMonitor.Enabled = !running;
            _pickMonitorElement.Enabled = !running;
            _followMonitorFocus.Enabled = !running;
            if (running)
            {
                _elementPicker?.Cancel();
                _focusInspectorTimer.Stop();
            }
            else if (_followMonitorFocus.Checked)
            {
                _focusInspectorTimer.Start();
            }
            _clearControlFilter.Enabled = !running &&
                !string.IsNullOrEmpty(_uiReferenceFilter);
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

        private DvtScenario SelectedScenario =>
            _scenarioTree.SelectedNode?.Tag as DvtScenario;

        private void PopulateScenarioTree()
        {
            DvtScenario selected = SelectedScenario;
            _scenarioTree.BeginUpdate();
            try
            {
                _scenarioTree.Nodes.Clear();
                foreach (string category in DvtCategories.Ordered)
                {
                    DvtScenario[] items = _scenarios
                        .Where(item => string.Equals(
                            item.Category,
                            category,
                            StringComparison.OrdinalIgnoreCase))
                        .Where(item => ScenarioReferenceMatcher.Matches(
                            item,
                            _uiReferenceFilter))
                        .OrderBy(item => item.Name)
                        .ToArray();
                    if (items.Length == 0) continue;
                    var categoryNode = new TreeNode(
                        DvtCategories.DisplayName(category) +
                        "（" + items.Length + "）");
                    foreach (DvtScenario item in items)
                    {
                        categoryNode.Nodes.Add(new TreeNode(item.Name)
                        {
                            Tag = item,
                            ToolTipText = item.Description
                        });
                    }
                    categoryNode.Expand();
                    _scenarioTree.Nodes.Add(categoryNode);
                }
            }
            finally
            {
                _scenarioTree.EndUpdate();
            }

            if (selected != null && SelectScenario(selected)) return;
            SelectFirstScenario();
        }

        private void SelectFirstScenario()
        {
            foreach (TreeNode category in _scenarioTree.Nodes)
            {
                if (category.Nodes.Count == 0) continue;
                _scenarioTree.SelectedNode = category.Nodes[0];
                return;
            }
        }

        private bool SelectScenario(DvtScenario scenario)
        {
            if (scenario == null) return false;
            foreach (TreeNode category in _scenarioTree.Nodes)
            {
                foreach (TreeNode node in category.Nodes)
                {
                    if (!ReferenceEquals(node.Tag, scenario)) continue;
                    _scenarioTree.SelectedNode = node;
                    node.EnsureVisible();
                    return true;
                }
            }
            return false;
        }

        private void ApplyUiReferenceFilter(string referenceKey)
        {
            _uiReferenceFilter = string.IsNullOrWhiteSpace(referenceKey)
                ? null
                : referenceKey;
            _clearControlFilter.Enabled =
                _runCancellation == null &&
                !string.IsNullOrEmpty(_uiReferenceFilter);
            PopulateScenarioTree();
            if (string.IsNullOrEmpty(_uiReferenceFilter))
            {
                _inspectorStatus.ForeColor = SystemColors.ControlText;
                _inspectorStatus.Text =
                    "顯示全部 DVT；可直接操作 Monitor，或選取真實元件篩選。";
                return;
            }

            string label = _uiReferenceFilter;
            string value;
            if (MonitorUiReference.TryGetControl(_uiReferenceFilter, out value))
                label = "控制項 " + value;
            else if (MonitorUiReference.TryGetProperty(
                _uiReferenceFilter,
                out value))
                label = "PropertyGrid 參數「" + value + "」";
            _inspectorStatus.ForeColor = SystemColors.ControlText;
            _inspectorStatus.Text = label + "：共 " +
                _scenarios.Count(item => ScenarioReferenceMatcher.Matches(
                    item,
                    _uiReferenceFilter)) +
                " 個關聯 DVT。";
        }

        private async Task<bool> AttachInspectorAsync(bool bringMonitorForward)
        {
            if (!File.Exists(_appPath.Text))
            {
                MessageBox.Show("找不到監控程式：" + _appPath.Text);
                return false;
            }

            _attachMonitor.Enabled = false;
            _inspectorStatus.ForeColor = SystemColors.ControlText;
            _inspectorStatus.Text = "正在連接真實 Monitor...";
            try
            {
                if (_inspectorDriver == null)
                    _inspectorDriver = new UiAutomationDriver();
                using (var cancellation = new CancellationTokenSource(
                    TimeSpan.FromSeconds(45)))
                {
                    await _inspectorDriver.AttachOrLaunchAsync(
                        _appPath.Text,
                        40,
                        cancellation.Token);
                }
                _inspectorControlIds = _scenarios
                    .SelectMany(item => item.ControlRefs)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                _inspectorPropertyNames = _scenarios
                    .SelectMany(item => item.PropertyRefs)
                    .Distinct(StringComparer.Ordinal)
                    .ToArray();
                _inspectorStatus.Text =
                    "已連接 Monitor。直接操作真實頁籤與 PropertyGrid；選取器只攔截一次點擊。";
                if (bringMonitorForward)
                    _inspectorDriver.BringMonitorToForeground();
                return true;
            }
            catch (Exception ex)
            {
                _inspectorStatus.Text = "連接失敗：" + ex.Message;
                _inspectorStatus.ForeColor = Color.DarkRed;
                return false;
            }
            finally
            {
                _attachMonitor.Enabled = _runCancellation == null;
            }
        }

        private async Task ToggleElementPickerAsync()
        {
            if (_runCancellation != null) return;
            if (_elementPicker != null && _elementPicker.IsActive)
            {
                _elementPicker.Cancel();
                return;
            }

            if (_inspectorDriver == null || !_inspectorDriver.IsAttached)
            {
                if (!await AttachInspectorAsync(bringMonitorForward: false))
                    return;
            }

            if (_elementPicker == null)
            {
                _elementPicker = new MonitorElementPicker(
                    this,
                    point => _inspectorDriver.InspectAtScreenPoint(
                        point,
                        _inspectorControlIds,
                        _inspectorPropertyNames));
                _elementPicker.SelectionCompleted += OnElementSelected;
                _elementPicker.Canceled += OnElementPickerCanceled;
            }
            _elementPicker.Start();
            _pickMonitorElement.Text = "取消選取";
            _inspectorStatus.ForeColor = Color.DarkOrange;
            _inspectorStatus.Text =
                "選取模式：移到真實 Monitor；橘框出現後點一下。這一下不會觸發產品功能。";
            _inspectorDriver.BringMonitorToForeground();
        }

        private void OnElementSelected(MonitorLiveSelection selection)
        {
            _pickMonitorElement.Text = "選取真實元件";
            if (selection == null) return;
            ApplyUiReferenceFilter(selection.ReferenceKey);
            _inspectorStatus.Text = selection.DisplayName + "：共 " +
                _scenarios.Count(item => ScenarioReferenceMatcher.Matches(
                    item,
                    selection.ReferenceKey)) +
                " 個關聯 DVT。";
            Activate();
        }

        private void OnElementPickerCanceled()
        {
            _pickMonitorElement.Text = "選取真實元件";
            _inspectorStatus.ForeColor = SystemColors.ControlText;
            _inspectorStatus.Text =
                "已取消選取；Monitor 維持正常操作。";
        }

        private void OnFocusInspectorTick(object sender, EventArgs e)
        {
            if (_runCancellation != null ||
                !_followMonitorFocus.Checked ||
                _elementPicker != null && _elementPicker.IsActive)
                return;
            try
            {
                if (_inspectorDriver == null || !_inspectorDriver.IsAttached)
                    return;
                MonitorLiveSelection selection =
                    _inspectorDriver.InspectFocusedElement(
                        _inspectorControlIds,
                        _inspectorPropertyNames);
                if (selection == null ||
                    string.Equals(
                        selection.ReferenceKey,
                        _lastFocusedReferenceKey,
                        StringComparison.OrdinalIgnoreCase))
                    return;
                _lastFocusedReferenceKey = selection.ReferenceKey;
                ApplyUiReferenceFilter(selection.ReferenceKey);
                _inspectorStatus.Text =
                    "跟隨焦點：" + selection.DisplayName + "。";
            }
            catch (ElementNotAvailableException)
            {
                _lastFocusedReferenceKey = null;
            }
            catch (InvalidOperationException)
            {
                _lastFocusedReferenceKey = null;
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
            if (_runCancellation == null)
            {
                _focusInspectorTimer.Stop();
                _elementPicker?.Dispose();
                return;
            }
            e.Cancel = true;
            _runCancellation.Cancel();
            _status.Text = "正在中止並還原設定，完成後再關閉。";
        }
    }
}
