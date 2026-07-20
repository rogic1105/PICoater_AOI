using System;
using System.Collections.Generic;
using System.Linq;

namespace AniloxRoll.Monitor.Core.Services
{
    internal enum OutputHealthSeverity
    {
        Normal = 0,
        Notice = 1,
        OutputFault = 2,
        Critical = 3
    }

    internal sealed class OutputHealthSnapshot
    {
        public static readonly OutputHealthSnapshot Normal =
            new OutputHealthSnapshot(OutputHealthSeverity.Normal, "none", string.Empty, false);

        public OutputHealthSnapshot(
            OutputHealthSeverity severity, string code, string message, bool isActive)
        {
            Severity = severity;
            Code = code ?? "none";
            Message = message ?? string.Empty;
            IsActive = isActive;
        }

        public OutputHealthSeverity Severity { get; }
        public string Code { get; }
        public string Message { get; }
        public bool IsActive { get; }
    }

    /// <summary>
    /// Owns the operator-visible health state. Producers only report and resolve named incidents;
    /// UI color selection and acknowledgement semantics stay here.
    /// </summary>
    internal sealed class OutputHealthService
    {
        private sealed class Incident
        {
            public string Code;
            public OutputHealthSeverity Severity;
            public string Message;
            public bool IsActive;
            public long Sequence;
        }

        private readonly object _sync = new object();
        private readonly Dictionary<string, Incident> _incidents =
            new Dictionary<string, Incident>(StringComparer.OrdinalIgnoreCase);
        private long _sequence;
        private OutputHealthSnapshot _snapshot = OutputHealthSnapshot.Normal;

        public event Action<OutputHealthSnapshot> Changed;

        public OutputHealthSnapshot Snapshot
        {
            get
            {
                lock (_sync) return _snapshot;
            }
        }

        public OutputHealthSnapshot[] Incidents
        {
            get
            {
                lock (_sync)
                {
                    return GetOrderedIncidentsLocked()
                        .Select(ToSnapshot)
                        .ToArray();
                }
            }
        }

        public void Report(string code, OutputHealthSeverity severity, string message)
        {
            if (string.IsNullOrWhiteSpace(code))
                throw new ArgumentException("An incident code is required.", nameof(code));
            if (severity == OutputHealthSeverity.Normal)
                throw new ArgumentOutOfRangeException(nameof(severity));

            OutputHealthSnapshot notification;
            lock (_sync)
            {
                Incident existing;
                if (_incidents.TryGetValue(code, out existing) &&
                    existing.IsActive &&
                    existing.Severity == severity &&
                    string.Equals(existing.Message, message, StringComparison.Ordinal))
                {
                    return;
                }

                var incident = existing ?? new Incident { Code = code };
                incident.Severity = severity;
                incident.Message = message ?? string.Empty;
                incident.IsActive = true;
                incident.Sequence = ++_sequence;
                _incidents[code] = incident;

                FlowTrace.Log(
                    $"[OutputHealth] raise code={code} severity={severity} message={incident.Message}");
                notification = RecomputeSnapshotLocked() ?? _snapshot;
            }
            RaiseChanged(notification);
        }

        public void Resolve(string code, string message = null)
        {
            if (string.IsNullOrWhiteSpace(code)) return;

            OutputHealthSnapshot notification;
            lock (_sync)
            {
                Incident incident;
                if (!_incidents.TryGetValue(code, out incident) || !incident.IsActive) return;

                incident.IsActive = false;
                if (!string.IsNullOrWhiteSpace(message)) incident.Message = message;
                incident.Sequence = ++_sequence;
                FlowTrace.Log(
                    $"[OutputHealth] resolve code={code} message={incident.Message}");
                notification = RecomputeSnapshotLocked() ?? _snapshot;
            }
            RaiseChanged(notification);
        }

        /// <summary>
        /// Clears one resolved incident. Active problems remain visible even when the operator
        /// clicks their label.
        /// </summary>
        public void AcknowledgeResolved(string code)
        {
            if (string.IsNullOrWhiteSpace(code)) return;

            OutputHealthSnapshot notification;
            lock (_sync)
            {
                Incident incident;
                if (!_incidents.TryGetValue(code, out incident) || incident.IsActive) return;

                _incidents.Remove(code);
                FlowTrace.Log("[OutputHealth] ack codes=" + incident.Code);
                notification = RecomputeSnapshotLocked() ?? _snapshot;
            }
            RaiseChanged(notification);
        }

        private OutputHealthSnapshot RecomputeSnapshotLocked()
        {
            Incident selected = GetOrderedIncidentsLocked().FirstOrDefault();

            var next = selected == null
                ? OutputHealthSnapshot.Normal
                : ToSnapshot(selected);

            if (SameSnapshot(_snapshot, next)) return null;

            OutputHealthSnapshot previous = _snapshot;
            _snapshot = next;
            FlowTrace.Log(
                $"[OutputHealth] state {previous.Severity} -> {next.Severity} " +
                $"code={next.Code} active={next.IsActive}");
            return next;
        }

        private IOrderedEnumerable<Incident> GetOrderedIncidentsLocked()
        {
            return _incidents.Values
                .OrderByDescending(x => x.Severity)
                .ThenByDescending(x => x.IsActive)
                .ThenByDescending(x => x.Sequence);
        }

        private static OutputHealthSnapshot ToSnapshot(Incident incident)
        {
            return new OutputHealthSnapshot(
                incident.Severity, incident.Code, incident.Message, incident.IsActive);
        }

        private static bool SameSnapshot(
            OutputHealthSnapshot left, OutputHealthSnapshot right)
        {
            return left.Severity == right.Severity &&
                   left.IsActive == right.IsActive &&
                   string.Equals(left.Code, right.Code, StringComparison.Ordinal) &&
                   string.Equals(left.Message, right.Message, StringComparison.Ordinal);
        }

        private void RaiseChanged(OutputHealthSnapshot snapshot)
        {
            Changed?.Invoke(snapshot);
        }
    }
}
