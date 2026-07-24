"""Synthetic-log tests for capture/output validators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.capture import CaptureFlowValidator
from flow_checks.core import CheckStatus, FlowLine, FlowSession


def validate_config(message: str):
    session = FlowSession(
        Path("synthetic.log"),
        [
            FlowLine(0, "00:00:00.000", 1,
                     "capture plan grab=260721-120000 root=D:\\Anilox imageDir=x csv=y "
                     "archive=260721-120000.acap "
                     "assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r "
                     "preview=1920x1080x3 scale=5"),
            FlowLine(1, "00:00:01.000", 1, message),
        ],
    )
    report = CaptureFlowValidator().validate(session)
    return next(item for item in report.results if item.rule == "C2.cfg-scale")


class CaptureFlowValidatorTests(unittest.TestCase):
    def test_cfg_with_physical_scale_passes(self):
        result = validate_config(
            "capture csv cfg path=x speed=40.0000 lr=3000.00 HM=1/1 ridge=9 thrV=1/1 thrH=1/1"
        )
        self.assertEqual(CheckStatus.PASS, result.status)

    def test_cfg_without_physical_scale_fails(self):
        result = validate_config(
            "capture csv cfg path=x HM=1/1 ridge=9 thrV=1/1 thrH=1/1"
        )
        self.assertEqual(CheckStatus.FAIL, result.status)

    def test_packed_plan_and_finalize_pass(self):
        session = FlowSession(
            Path("synthetic.log"),
            [
                FlowLine(
                    0,
                    "00:00:00.000",
                    1,
                    "capture plan grab=260721-120000 root=D:\\Anilox\\Captures_pack "
                    "imageDir=x csv=y archive=260721-120000.acap "
                    "assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r "
                    "preview=1920x1080x3 scale=5",
                ),
                FlowLine(
                    1,
                    "00:00:01.000",
                    1,
                    "capture finalize grab=260721-120000 "
                    "archive=D:\\Anilox\\Captures_pack\\2026\\202607\\20260721\\260721-120000.acap "
                    "atlas=3 atlasBytes=1234 remoteFiles=2",
                ),
            ],
        )

        report = CaptureFlowValidator().validate(session)
        plan = next(item for item in report.results if item.rule == "C1.plan")
        finalize = next(item for item in report.results if item.rule == "C3.finalize")

        self.assertEqual(CheckStatus.PASS, plan.status)
        self.assertEqual(CheckStatus.PASS, finalize.status)

    def test_finalize_failure_is_reported(self):
        session = FlowSession(
            Path("synthetic.log"),
            [
                FlowLine(
                    0,
                    "00:00:00.000",
                    1,
                    "capture plan grab=260721-120000 root=D:\\Anilox\\Captures_pack "
                    "imageDir=x csv=y archive=260721-120000.acap "
                    "assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r "
                    "preview=1920x1080x3 scale=5",
                ),
                FlowLine(
                    1,
                    "00:00:01.000",
                    1,
                    "capture finalize failed grab=260721-120000 error=IOException",
                ),
            ],
        )

        report = CaptureFlowValidator().validate(session)
        finalize = next(item for item in report.results if item.rule == "C3.finalize")

        self.assertEqual(CheckStatus.FAIL, finalize.status)


if __name__ == "__main__":
    unittest.main()
