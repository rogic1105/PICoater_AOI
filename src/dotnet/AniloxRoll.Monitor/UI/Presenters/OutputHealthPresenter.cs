using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using AniloxRoll.Monitor.Core.Services;

namespace AniloxRoll.Monitor.UI.Presenters
{
    /// <summary>Renders and acknowledges independent output-health incidents in the status strip.</summary>
    internal sealed class OutputHealthPresenter : IDisposable
    {
        private readonly OutputHealthService _service;
        private readonly StatusStrip _statusStrip;
        private readonly ToolStripStatusLabel _informationLabel;
        private readonly Action<Action> _dispatch;
        private readonly Action _refreshInformation;
        private readonly Action<string> _flowLog;
        private readonly Color _noticeColor;
        private readonly Color _outputFaultColor;
        private readonly Color _criticalColor;
        private readonly Dictionary<string, ToolStripStatusLabel> _labels =
            new Dictionary<string, ToolStripStatusLabel>(StringComparer.OrdinalIgnoreCase);
        private bool _disposed;

        public OutputHealthPresenter(
            OutputHealthService service,
            StatusStrip statusStrip,
            ToolStripStatusLabel informationLabel,
            Action<Action> dispatch,
            Action refreshInformation,
            Action<string> flowLog,
            Color noticeColor,
            Color outputFaultColor,
            Color criticalColor)
        {
            _service = service ?? throw new ArgumentNullException(nameof(service));
            _statusStrip = statusStrip ?? throw new ArgumentNullException(nameof(statusStrip));
            _informationLabel = informationLabel ?? throw new ArgumentNullException(nameof(informationLabel));
            _dispatch = dispatch ?? throw new ArgumentNullException(nameof(dispatch));
            _refreshInformation = refreshInformation;
            _flowLog = flowLog;
            _noticeColor = noticeColor;
            _outputFaultColor = outputFaultColor;
            _criticalColor = criticalColor;

            _statusStrip.SizingGrip = false;
            _informationLabel.Spring = true;
            _informationLabel.TextAlign = ContentAlignment.MiddleLeft;
            _service.Changed += Service_Changed;
            ApplySnapshot();
        }

        private void Service_Changed(OutputHealthSnapshot snapshot)
        {
            _dispatch(ApplySnapshot);
        }

        private void ApplySnapshot()
        {
            if (_disposed) return;
            RefreshLabels();
            _refreshInformation?.Invoke();
        }

        private void RefreshLabels()
        {
            OutputHealthSnapshot[] incidents = _service.Incidents;
            var currentCodes = new HashSet<string>(
                incidents.Select(x => x.Code),
                StringComparer.OrdinalIgnoreCase);

            foreach (string obsoleteCode in _labels.Keys
                .Where(code => !currentCodes.Contains(code))
                .ToArray())
            {
                ToolStripStatusLabel obsolete = _labels[obsoleteCode];
                _statusStrip.Items.Remove(obsolete);
                obsolete.Dispose();
                _labels.Remove(obsoleteCode);
            }

            foreach (ToolStripStatusLabel label in _labels.Values)
                _statusStrip.Items.Remove(label);

            int insertIndex = 0;
            foreach (OutputHealthSnapshot incident in incidents)
            {
                ToolStripStatusLabel label;
                if (!_labels.TryGetValue(incident.Code, out label))
                {
                    label = new ToolStripStatusLabel
                    {
                        AutoSize = true,
                        Margin = new Padding(0, 0, 2, 0),
                        Padding = new Padding(6, 0, 6, 0)
                    };
                    label.Click += Label_Click;
                    _labels.Add(incident.Code, label);
                }

                ApplyLabel(label, incident);
                _statusStrip.Items.Insert(insertIndex++, label);
            }
        }

        private void Label_Click(object sender, EventArgs e)
        {
            var label = sender as ToolStripStatusLabel;
            string code = label?.Tag as string;
            if (string.IsNullOrWhiteSpace(code)) return;

            _flowLog?.Invoke($"ui:【產出狀態】確認 code={code}");
            _service.AcknowledgeResolved(code);
        }

        private void ApplyLabel(ToolStripStatusLabel label, OutputHealthSnapshot incident)
        {
            label.Tag = incident.Code;
            label.Text = incident.Message +
                (incident.IsActive ? string.Empty : "（已恢復，點擊關閉）");
            label.ToolTipText = incident.IsActive
                ? "問題尚未排除；恢復後可點擊關閉這一項"
                : "點擊只關閉這一項已恢復的問題";

            if (!incident.IsActive)
            {
                label.BackColor = _noticeColor;
                label.ForeColor = Color.Black;
                return;
            }

            switch (incident.Severity)
            {
                case OutputHealthSeverity.Notice:
                    label.BackColor = _noticeColor;
                    label.ForeColor = Color.Black;
                    break;
                case OutputHealthSeverity.OutputFault:
                    label.BackColor = _outputFaultColor;
                    label.ForeColor = Color.White;
                    break;
                case OutputHealthSeverity.Critical:
                    label.BackColor = _criticalColor;
                    label.ForeColor = Color.White;
                    break;
                default:
                    label.BackColor = SystemColors.Control;
                    label.ForeColor = SystemColors.ControlText;
                    break;
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _service.Changed -= Service_Changed;
            foreach (ToolStripStatusLabel label in _labels.Values)
            {
                label.Click -= Label_Click;
                _statusStrip.Items.Remove(label);
                label.Dispose();
            }
            _labels.Clear();
        }
    }
}
