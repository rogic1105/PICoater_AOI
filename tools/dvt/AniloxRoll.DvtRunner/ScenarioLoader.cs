using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Web.Script.Serialization;

namespace AniloxRoll.DvtRunner
{
    internal static class ScenarioLoader
    {
        private static readonly HashSet<string> Actions =
            new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "set-session-value",
                "launch",
                "launch-helper",
                "wait-helper-exit",
                "stop-helper",
                "disable-target-network",
                "enable-target-network",
                "block-target-port",
                "unblock-target-port",
                "disable-serial-device",
                "enable-serial-device",
                "prepare-retention-fixture",
                "verify-retention-fixture",
                "cleanup-retention-fixture",
                "audit-property-catalog",
                "exercise-property-group",
                "wait-element",
                "set-property",
                "click",
                "click-button",
                "click-control",
                "wheel",
                "drag",
                "select-tab",
                "confirm-folder",
                "select-combo",
                "wait-log",
                "wait-log-current-combo",
                "wait-log-current-period",
                "verify-log-min-count",
                "verify-log-absent",
                "verify-range-scroll",
                "reset-verdict-cache",
                "verify-verdict-cache-performance",
                "verify-verdict-cache-warm-first",
                "verify-monitor-functional-cycle",
                "reset-evidence",
                "delay",
                "soak",
                "restore-properties",
                "close-app",
                "run-checker"
            };

        public static IReadOnlyList<DvtScenario> LoadDirectory(string directory)
        {
            if (!Directory.Exists(directory))
                throw new DirectoryNotFoundException("Scenario directory not found: " + directory);

            var serializer = new JavaScriptSerializer();
            var scenarios = new List<DvtScenario>();
            foreach (string path in Directory.GetFiles(directory, "*.json").OrderBy(p => p))
            {
                string json = File.ReadAllText(path, Encoding.UTF8);
                DvtScenario scenario = serializer.Deserialize<DvtScenario>(json);
                Validate(scenario, path);
                scenarios.Add(scenario);
            }

            if (scenarios.Count == 0)
                throw new InvalidOperationException("No DVT scenarios were found in " + directory);
            PropertyGridCatalog catalog = PropertyGridCatalog.Load();
            ValidatePropertyGridGroupCoverage(scenarios, catalog);
            PopulatePropertyReferences(scenarios, catalog);
            return scenarios;
        }

        private static void ValidatePropertyGridGroupCoverage(
            IReadOnlyList<DvtScenario> scenarios,
            PropertyGridCatalog catalog)
        {
            var exercised = new HashSet<string>(
                scenarios.SelectMany(scenario => scenario.Steps)
                    .Where(step => string.Equals(
                        step.Action,
                        "exercise-property-group",
                        StringComparison.OrdinalIgnoreCase))
                    .Select(step => step.Target),
                StringComparer.OrdinalIgnoreCase);
            string[] missing = catalog.Properties
                .Select(item => item.Group)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Where(group => !exercised.Contains(group))
                .OrderBy(group => group)
                .ToArray();
            if (missing.Length > 0)
            {
                throw new InvalidDataException(
                    "PropertyGrid DVT groups are not exercised by any scenario: " +
                    string.Join(",", missing));
            }
        }

        private static void PopulatePropertyReferences(
            IReadOnlyList<DvtScenario> scenarios,
            PropertyGridCatalog catalog)
        {
            foreach (DvtScenario scenario in scenarios)
            {
                var references = new HashSet<string>(StringComparer.Ordinal);
                foreach (DvtStep step in scenario.Steps)
                {
                    if (string.Equals(
                        step.Action,
                        "set-property",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        if (!string.IsNullOrWhiteSpace(step.Target))
                            references.Add(step.Target);
                        continue;
                    }

                    if (string.Equals(
                        step.Action,
                        "exercise-property-group",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        foreach (PropertyGridCatalogEntry item in
                            catalog.GetGroup(step.Target))
                            references.Add(item.DisplayName);
                        continue;
                    }

                    if (string.Equals(
                        step.Action,
                        "audit-property-catalog",
                        StringComparison.OrdinalIgnoreCase))
                    {
                        foreach (PropertyGridCatalogEntry item in catalog.Properties)
                            references.Add(item.DisplayName);
                    }
                }
                scenario.PropertyRefs = references.OrderBy(value => value).ToList();
            }
        }

        private static void Validate(DvtScenario scenario, string path)
        {
            if (scenario == null || string.IsNullOrWhiteSpace(scenario.Id) ||
                string.IsNullOrWhiteSpace(scenario.Name))
                throw new InvalidDataException(path + ": scenario id/name is required.");
            if (!DvtCategories.IsKnown(scenario.Category))
                throw new InvalidDataException(
                    path + ": category must be monitor, review, report, or bridge.");
            if (scenario.ControlRefs == null || scenario.ControlRefs.Count == 0)
                throw new InvalidDataException(
                    path + ": at least one Monitor controlRefs entry is required.");
            var controlIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (string controlId in scenario.ControlRefs)
            {
                if (string.IsNullOrWhiteSpace(controlId) ||
                    !Regex.IsMatch(
                        controlId,
                        @"^[A-Za-z_][A-Za-z0-9_]*$",
                        RegexOptions.CultureInvariant))
                    throw new InvalidDataException(
                        path + ": invalid Monitor control reference " + controlId);
                if (!controlIds.Add(controlId))
                    throw new InvalidDataException(
                        path + ": duplicate Monitor control reference " + controlId);
            }
            if (scenario.Steps == null || scenario.Steps.Count == 0)
                throw new InvalidDataException(path + ": at least one step is required.");

            var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < scenario.Steps.Count; i++)
            {
                DvtStep step = scenario.Steps[i];
                string prefix = path + ": step " + (i + 1);
                if (step == null || string.IsNullOrWhiteSpace(step.Id) ||
                    string.IsNullOrWhiteSpace(step.Title) ||
                    string.IsNullOrWhiteSpace(step.Action))
                    throw new InvalidDataException(prefix + " requires id/title/action.");
                if (!ids.Add(step.Id))
                    throw new InvalidDataException(prefix + " has duplicate id " + step.Id);
                if (!Actions.Contains(step.Action))
                    throw new InvalidDataException(prefix + " has unsupported action " + step.Action);
                if (step.Action != "run-checker" && string.IsNullOrWhiteSpace(step.Contract))
                    throw new InvalidDataException(prefix + " must reference a verify-flow contract.");
                if (string.Equals(
                        step.Action,
                        "exercise-property-group",
                        StringComparison.OrdinalIgnoreCase) &&
                    string.IsNullOrWhiteSpace(step.Target))
                    throw new InvalidDataException(
                        prefix + " requires a PropertyGrid catalog group in Target.");
                if (step.Action == "wait-log" ||
                    step.Action == "wait-log-current-combo" ||
                    step.Action == "wait-log-current-period" ||
                    step.Action == "verify-log-absent")
                {
                    if (string.IsNullOrWhiteSpace(step.Pattern))
                        throw new InvalidDataException(prefix + " requires a regex pattern.");
                    string pattern = step.Pattern;
                    if (step.Action == "wait-log-current-combo")
                        pattern = pattern.Replace("{value}", "DVT_VALUE");
                    if (step.Action == "wait-log-current-period")
                    {
                        pattern = pattern
                            .Replace("{date}", "DVT_DATE")
                            .Replace("{time}", "DVT_TIME");
                    }
                    _ = new Regex(pattern, RegexOptions.CultureInvariant);
                    if (step.Action == "wait-log-current-combo" &&
                        (string.IsNullOrWhiteSpace(step.Target) ||
                         !step.Pattern.Contains("{value}")))
                        throw new InvalidDataException(
                            prefix + " requires Target and a {value} token.");
                    if (step.Action == "wait-log-current-period" &&
                        (string.IsNullOrWhiteSpace(step.Target) ||
                         string.IsNullOrWhiteSpace(step.Value) ||
                         !step.Pattern.Contains("{date}") ||
                         !step.Pattern.Contains("{time}")))
                        throw new InvalidDataException(
                            prefix + " requires date Target, time Value, and {date}/{time} tokens.");
                }
                if (step.Action == "verify-log-min-count")
                {
                    if (string.IsNullOrWhiteSpace(step.Pattern))
                        throw new InvalidDataException(prefix + " requires a regex pattern.");
                    _ = new Regex(step.Pattern, RegexOptions.CultureInvariant);
                    bool isDurationCycleToken = Regex.IsMatch(
                        step.Value ?? string.Empty,
                        @"^\{cycles:[1-9]\d*\}$",
                        RegexOptions.CultureInvariant);
                    if ((!int.TryParse(step.Value, out int minimumCount) ||
                         minimumCount < 1) &&
                        !isDurationCycleToken)
                        throw new InvalidDataException(
                            prefix + " requires a positive count in Value.");
                }
                if (step.TimeoutSeconds < 0)
                    throw new InvalidDataException(
                        prefix + " timeout cannot be negative.");
                if (step.TimeoutSeconds == 0 &&
                    !string.Equals(
                        step.Action, "wait-element",
                        StringComparison.OrdinalIgnoreCase))
                    step.TimeoutSeconds = 30;
            }
        }
    }
}
