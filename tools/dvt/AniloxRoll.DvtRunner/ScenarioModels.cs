using System.Collections.Generic;

namespace AniloxRoll.DvtRunner
{
    internal sealed class DvtScenario
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public List<DvtStep> Steps { get; set; } = new List<DvtStep>();

        public override string ToString() => Name ?? Id ?? "(unnamed)";
    }

    internal sealed class DvtStep
    {
        public string Id { get; set; }
        public string Contract { get; set; }
        public string Title { get; set; }
        public string Action { get; set; }
        public string Target { get; set; }
        public string Value { get; set; }
        public string Pattern { get; set; }
        public int TimeoutSeconds { get; set; } = 30;
        public bool Optional { get; set; }
    }

    internal enum StepStatus
    {
        Pending,
        Running,
        Passed,
        Failed,
        Skipped
    }

    internal sealed class StepUpdate
    {
        public int Index { get; set; }
        public DvtStep Step { get; set; }
        public StepStatus Status { get; set; }
        public string Detail { get; set; }
    }

    internal sealed class RunnerOptions
    {
        public string RepositoryRoot { get; set; }
        public string AppExePath { get; set; }
        public string LogDirectory { get; set; }
        public string ProcessIdPath { get; set; }
        public bool CloseAppOnCleanup { get; set; }
    }
}
