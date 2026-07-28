using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
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
        private IReadOnlyList<DvtScenario> _scenarios;
        private ScenarioEngine _engine;
        private CancellationTokenSource _runCancellation;
        private string _repositoryRoot;

        public MainForm()
        {
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
                _repositoryRoot = RepositoryLocator.FindRoot();
                _appPath.Text = Path.Combine(
                    _repositoryRoot, "bin", "x64", "Release", "AniloxRoll.Monitor.exe");
                _logDirectory.Text = @"D:\Anilox\Logs";
                string scenarioDirectory = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory, "Scenarios");
                _scenarios = ScenarioLoader.LoadDirectory(scenarioDirectory);
                foreach (DvtScenario item in _scenarios) _scenario.Items.Add(item);
                if (_scenario.Items.Count > 0) _scenario.SelectedIndex = 0;
                else _status.Text = "沒有找到 DVT 情境。";
            }
            catch (Exception ex)
            {
                _status.Text = "初始化失敗：" + ex.Message;
                _start.Enabled = false;
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

        private async Task StartScenarioAsync()
        {
            var selected = _scenario.SelectedItem as DvtScenario;
            if (selected == null) return;
            if (!File.Exists(_appPath.Text))
            {
                MessageBox.Show("找不到監控程式：" + _appPath.Text);
                return;
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
                LogDirectory = _logDirectory.Text
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
            }
            catch (OperationCanceledException)
            {
                _status.Text = "已中止；已嘗試停止 Grab 並還原設定。";
                _status.ForeColor = Color.DarkOrange;
            }
            catch (Exception ex)
            {
                _status.Text = "FAIL：" + ex.Message;
                _status.ForeColor = Color.DarkRed;
                AppendOutput(ex.ToString());
            }
            finally
            {
                SetRunning(false);
                _runCancellation.Dispose();
                _runCancellation = null;
                _engine = null;
            }
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
