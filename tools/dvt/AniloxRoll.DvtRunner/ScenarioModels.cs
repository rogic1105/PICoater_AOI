using System.Collections.Generic;

namespace AniloxRoll.DvtRunner
{
    internal sealed class DvtScenario
    {
        public string Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public string Category { get; set; }
        public List<string> ControlRefs { get; set; } = new List<string>();
        public List<string> PropertyRefs { get; set; } = new List<string>();
        public List<DvtStep> Steps { get; set; } = new List<DvtStep>();

        public override string ToString() => Name ?? Id ?? "(unnamed)";
    }

    internal static class DvtCategories
    {
        public const string Monitor = "monitor";
        public const string Review = "review";
        public const string Report = "report";
        public const string Bridge = "bridge";

        public static readonly string[] Ordered =
        {
            Monitor,
            Review,
            Report,
            Bridge
        };

        public static bool IsKnown(string category)
        {
            foreach (string known in Ordered)
            {
                if (string.Equals(
                    known,
                    category,
                    System.StringComparison.OrdinalIgnoreCase))
                    return true;
            }
            return false;
        }

        public static string DisplayName(string category)
        {
            if (string.Equals(category, Monitor, System.StringComparison.OrdinalIgnoreCase))
                return "監控";
            if (string.Equals(category, Review, System.StringComparison.OrdinalIgnoreCase))
                return "回顧";
            if (string.Equals(category, Report, System.StringComparison.OrdinalIgnoreCase))
                return "報表";
            if (string.Equals(category, Bridge, System.StringComparison.OrdinalIgnoreCase))
                return "Bridge";
            return category ?? "未分類";
        }
    }

    internal static class ScenarioReferenceMatcher
    {
        public static bool Matches(
            DvtScenario scenario,
            string referenceKey)
        {
            if (scenario == null) return false;
            if (string.IsNullOrWhiteSpace(referenceKey)) return true;

            string value;
            if (MonitorUiReference.TryGetControl(referenceKey, out value))
            {
                foreach (string controlId in scenario.ControlRefs)
                {
                    if (string.Equals(
                        controlId,
                        value,
                        System.StringComparison.OrdinalIgnoreCase))
                        return true;
                }
                return false;
            }

            if (MonitorUiReference.TryGetProperty(referenceKey, out value))
            {
                foreach (string propertyName in scenario.PropertyRefs)
                {
                    if (string.Equals(
                        propertyName,
                        value,
                        System.StringComparison.Ordinal))
                        return true;
                }
            }
            return false;
        }

        public static bool MatchesSearch(
            DvtScenario scenario,
            string searchText)
        {
            if (scenario == null) return false;
            if (string.IsNullOrWhiteSpace(searchText)) return true;

            string[] terms = searchText.Split(
                new[] { ' ', '\t' },
                System.StringSplitOptions.RemoveEmptyEntries);
            foreach (string term in terms)
            {
                if (!ContainsTerm(scenario, term))
                    return false;
            }
            return true;
        }

        private static bool ContainsTerm(
            DvtScenario scenario,
            string term)
        {
            if (Contains(scenario.Id, term) ||
                Contains(scenario.Name, term) ||
                Contains(scenario.Description, term) ||
                Contains(scenario.Category, term) ||
                Contains(DvtCategories.DisplayName(scenario.Category), term))
                return true;

            foreach (string value in scenario.ControlRefs)
            {
                if (Contains(value, term)) return true;
            }
            foreach (string value in scenario.PropertyRefs)
            {
                if (Contains(value, term)) return true;
            }
            foreach (DvtStep step in scenario.Steps)
            {
                if (Contains(step.Id, term) ||
                    Contains(step.Contract, term) ||
                    Contains(step.Title, term) ||
                    Contains(step.Action, term) ||
                    Contains(step.Target, term) ||
                    Contains(step.Value, term))
                    return true;
            }
            return false;
        }

        private static bool Contains(string value, string term)
        {
            return !string.IsNullOrEmpty(value) &&
                value.IndexOf(
                    term,
                    System.StringComparison.OrdinalIgnoreCase) >= 0;
        }
    }

    internal sealed class DvtStep
    {
        public string Id { get; set; }
        public string Contract { get; set; }
        public string Title { get; set; }
        public string Action { get; set; }
        public string Target { get; set; }
        public int TargetOccurrence { get; set; }
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

    internal sealed class OriginalPropertyValue
    {
        public string DisplayName { get; set; }
        public int Occurrence { get; set; }
        public string Value { get; set; }
    }
}
