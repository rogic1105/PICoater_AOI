import json
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO_DIR = REPO_ROOT / "tools" / "dvt" / "AniloxRoll.DvtRunner" / "Scenarios"
DESIGNER_PATH = (
    REPO_ROOT
    / "src"
    / "dotnet"
    / "AniloxRoll.Monitor"
    / "UI"
    / "Form"
    / "AniloxRollForm.Designer.cs"
)
PROPERTY_CATALOG_PATH = (
    REPO_ROOT
    / "tools"
    / "dvt"
    / "AniloxRoll.DvtRunner"
    / "PropertyGridCoverage.json"
)
LIVE_INSPECTOR_PATH = (
    REPO_ROOT
    / "tools"
    / "dvt"
    / "AniloxRoll.DvtRunner"
    / "MonitorLiveInspector.cs"
)
UI_DRIVER_PATH = LIVE_INSPECTOR_PATH.with_name("UiAutomationDriver.cs")
SCENARIO_MODELS_PATH = LIVE_INSPECTOR_PATH.with_name("ScenarioModels.cs")
MAIN_FORM_PATH = LIVE_INSPECTOR_PATH.with_name("MainForm.cs")
KNOWN_CATEGORIES = {"monitor", "review", "report", "bridge"}
CONTROL_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DvtScenarioCatalogTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = []
        for path in sorted(SCENARIO_DIR.glob("*.json")):
            with path.open("r", encoding="utf-8-sig") as stream:
                self.scenarios.append((path, json.load(stream)))

    def test_every_scenario_has_one_primary_category(self):
        self.assertTrue(self.scenarios)
        categories = set()
        for path, scenario in self.scenarios:
            category = scenario.get("Category")
            self.assertIn(category, KNOWN_CATEGORIES, path.name)
            categories.add(category)
        self.assertEqual(KNOWN_CATEGORIES, categories)

    def test_control_references_are_unique_and_exist_in_monitor_designer(self):
        designer = DESIGNER_PATH.read_text(encoding="utf-8-sig")
        for path, scenario in self.scenarios:
            control_refs = scenario.get("ControlRefs")
            self.assertIsInstance(control_refs, list, path.name)
            self.assertTrue(control_refs, path.name)
            self.assertEqual(
                len(control_refs),
                len(set(control_refs)),
                path.name + " contains duplicate controlRefs",
            )
            for control_id in control_refs:
                self.assertRegex(control_id, CONTROL_ID, path.name)
                self.assertIn(
                    f'this.{control_id}.Name = "{control_id}";',
                    designer,
                    path.name + " references a stale Monitor control",
                )

    def test_property_references_are_derived_from_existing_scenario_steps(self):
        with PROPERTY_CATALOG_PATH.open("r", encoding="utf-8-sig") as stream:
            catalog = json.load(stream)["Properties"]
        display_names = {item["DisplayName"] for item in catalog}
        groups = {item["Group"] for item in catalog}

        for path, scenario in self.scenarios:
            self.assertNotIn(
                "PropertyRefs",
                scenario,
                path.name + " must derive property mapping instead of duplicating it",
            )
            for step in scenario.get("Steps", []):
                action = step.get("Action")
                if action == "set-property":
                    self.assertIn(step.get("Target"), display_names, path.name)
                elif action == "exercise-property-group":
                    self.assertIn(step.get("Target"), groups, path.name)

    def test_live_inspector_uses_real_uia_and_consumes_picker_click(self):
        inspector = LIVE_INSPECTOR_PATH.read_text(encoding="utf-8-sig")
        driver = UI_DRIVER_PATH.read_text(encoding="utf-8-sig")
        self.assertIn("MonitorElementPicker", inspector)
        self.assertIn("HighlightOverlay", inspector)
        self.assertIn("SetWindowsHookEx", inspector)
        self.assertIn("kind == WmLButtonDown", inspector)
        self.assertIn("kind == WmLButtonUp", inspector)
        self.assertGreaterEqual(inspector.count("return new IntPtr(1)"), 3)
        self.assertIn("AutomationElement.FromPoint", driver)
        self.assertIn("AutomationElement.FocusedElement", driver)
        self.assertIn("InspectAtScreenPoint", driver)
        self.assertIn("InspectFocusedElement", driver)
        self.assertNotIn("CopyFromScreen", inspector)
        self.assertNotIn("CopyFromScreen", driver)

    def test_live_inspector_filter_has_one_matcher_owner(self):
        models = SCENARIO_MODELS_PATH.read_text(encoding="utf-8-sig")
        main_form = MAIN_FORM_PATH.read_text(encoding="utf-8-sig")
        self.assertIn("internal static class ScenarioReferenceMatcher", models)
        self.assertIn("MonitorUiReference.TryGetControl", models)
        self.assertIn("MonitorUiReference.TryGetProperty", models)
        self.assertIn("ScenarioReferenceMatcher.Matches", main_form)
        self.assertNotIn("ScenarioMatchesUiReference", main_form)


if __name__ == "__main__":
    unittest.main()
