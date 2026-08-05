"""Synthetic-log tests for immediate live OPS/Start layout updates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import CheckStatus, FlowLine, FlowSession
from flow_checks.settings import SettingsFlowValidator


def session(*messages: str) -> FlowSession:
    lines = [
        FlowLine(
            elapsed=float(index),
            timestamp=f"00:00:{index:02d}.000",
            thread=1,
            message=message,
        )
        for index, message in enumerate(messages)
    ]
    return FlowSession(Path("synthetic.log"), lines)


def result(report, rule: str):
    return next(item for item in report.results if item.rule == rule)


class LiveLayoutFlowValidatorTests(unittest.TestCase):
    def test_cam4_ops_is_applied_immediately_during_grab(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[ae_OpsCam4]=11",
                "setting route ae_OpsCam4 owner=LiveLayout effects=None",
                "capture layout pending grab=260804-090000 "
                "setting=ae_OpsCam4 apply=display-now+stop-final",
                "displayLayout applied setting=ae_OpsCam4 refGrid=cam1 "
                "ops=24.41406250|24.41406250|24.41406250|11.00000000|"
                "24.41406250|24.41406250|24.41406250 "
                "start=0.0000|345.0000|690.0000|1035.0000|1380.0000|1725.0000|2070.0000 "
                "speed=40.0000 head=0.00 tail=0.00 "
                "scope=main+column-chart source=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "S7.live-layout").status
        )

    def test_start_rejects_stale_snapshot(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[be_StartCam4]=1200",
                "setting route be_StartCam4 owner=LiveLayout effects=None",
                "displayLayout applied setting=be_StartCam4 refGrid=cam1 "
                "ops=24.41406250|24.41406250|24.41406250|24.41406250|"
                "24.41406250|24.41406250|24.41406250 "
                "start=0.0000|345.0000|690.0000|1035.0000|1380.0000|1725.0000|2070.0000 "
                "speed=40.0000 head=0.00 tail=0.00 "
                "scope=main+column-chart source=unchanged",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S7.live-layout").status
        )

    def test_ops_requires_applied_state(self):
        report = SettingsFlowValidator().validate(
            session(
                "set:[ab_OpsCam1]=11",
                "setting route ab_OpsCam1 owner=LiveLayout effects=None",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "S7.live-layout").status
        )


if __name__ == "__main__":
    unittest.main()
