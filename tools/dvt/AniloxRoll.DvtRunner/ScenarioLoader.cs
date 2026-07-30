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
                "wait-element",
                "set-property",
                "click",
                "wheel",
                "drag",
                "select-tab",
                "confirm-folder",
                "select-combo",
                "wait-log",
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
            return scenarios;
        }

        private static void Validate(DvtScenario scenario, string path)
        {
            if (scenario == null || string.IsNullOrWhiteSpace(scenario.Id) ||
                string.IsNullOrWhiteSpace(scenario.Name))
                throw new InvalidDataException(path + ": scenario id/name is required.");
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
                if (step.Action == "wait-log")
                {
                    if (string.IsNullOrWhiteSpace(step.Pattern))
                        throw new InvalidDataException(prefix + " requires a regex pattern.");
                    _ = new Regex(step.Pattern, RegexOptions.CultureInvariant);
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
