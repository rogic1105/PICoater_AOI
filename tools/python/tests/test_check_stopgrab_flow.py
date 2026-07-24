"""Synthetic tests for the standalone StopGrab DVT checker."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from check_stopgrab_flow import check


class StopGrabFlowTests(unittest.TestCase):
    def test_completed_io_tail_allows_one_final_row_presentation(self):
        stop_windows, violations = check(
            [
                (1, "capture tail complete pending="),
                (2, "StopGrab"),
                (3, "capture gate closed standby=on"),
                (4, "LC row rowChart dir=BottomToTop"),
                (5, "rowCurve present after=mainImage cams=2 mode=WF"),
                (6, "drop drainedFrame after StopGrab cam1"),
                (7, "StartGrab cams=2"),
            ]
        )
        self.assertEqual(1, stop_windows)
        self.assertEqual([], violations)

    def test_row_presentation_without_completed_tail_fails(self):
        _, violations = check(
            [
                (1, "StopGrab"),
                (2, "capture gate closed standby=on"),
                (3, "rowCurve present after=mainImage cams=2 mode=WF"),
            ]
        )
        self.assertEqual(1, len(violations))

    def test_more_than_one_final_row_presentation_fails(self):
        _, violations = check(
            [
                (1, "capture tail complete pending="),
                (2, "StopGrab"),
                (3, "capture gate closed standby=on"),
                (4, "rowCurve present after=mainImage cams=2 mode=WF"),
                (5, "rowCurve present after=mainImage cams=2 mode=WF"),
            ]
        )
        self.assertEqual(1, len(violations))


if __name__ == "__main__":
    unittest.main()
