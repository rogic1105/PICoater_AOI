"""Tests for the focused StopGrab DVT checker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from check_stopgrab_flow import check


class StopGrabFlowTests(unittest.TestCase):
    def test_capture_save_may_finish_during_drain_then_releases(self):
        rows = [
            (1, "StartGrab（cams=7）"),
            (2, "capture plan grab=260722-120000 root=x"),
            (3, "StopGrab"),
            (4, "capture gate closed standby=on"),
            (5, "capture save drain begin grab=260722-120000"),
            (6, "capture archive append grab=260722-120000 cam=1 assets=7 bytes=10"),
            (7, "capture csv firstRecord grab=260722-120000 path=x"),
            (8, "capture save drain done grab=260722-120000"),
            (9, "capture remote release grab=260722-120000 files=2 bytes=20"),
        ]

        stop_windows, violations = check(rows)

        self.assertEqual(1, stop_windows)
        self.assertEqual([], violations)

    def test_persistence_after_drain_is_rejected(self):
        rows = [
            (1, "capture plan grab=260722-120000 root=x"),
            (2, "StopGrab"),
            (3, "capture gate closed standby=on"),
            (4, "capture save drain begin grab=260722-120000"),
            (5, "capture save drain done grab=260722-120000"),
            (6, "capture archive append grab=260722-120000 cam=1 assets=7 bytes=10"),
            (7, "capture remote release grab=260722-120000 files=2 bytes=20"),
        ]

        _, violations = check(rows)

        self.assertTrue(any("capture archive append" in message for _, message in violations))

    def test_capture_stop_requires_remote_release(self):
        rows = [
            (1, "capture plan grab=260722-120000 root=x"),
            (2, "StopGrab"),
            (3, "capture gate closed standby=on"),
            (4, "capture save drain begin grab=260722-120000"),
            (5, "capture save drain done grab=260722-120000"),
        ]

        _, violations = check(rows)

        self.assertTrue(any("not released" in message for _, message in violations))

    def test_non_capture_stop_only_requires_gate(self):
        rows = [
            (1, "StopGrab"),
            (2, "capture gate closed standby=on"),
            (3, "drop drainedFrame after StopGrab cam1"),
        ]

        _, violations = check(rows)

        self.assertEqual([], violations)

    def test_row_curve_presentation_after_stop_is_rejected(self):
        rows = [
            (1, "StopGrab"),
            (2, "display capture quiesce mode=WF"),
            (3, "capture gate closed standby=on"),
            (4, "rowCurve present after=mainImage cams=2 mode=WF"),
        ]

        _, violations = check(rows)

        self.assertTrue(any("rowCurve present" in message for _, message in violations))


if __name__ == "__main__":
    unittest.main()
