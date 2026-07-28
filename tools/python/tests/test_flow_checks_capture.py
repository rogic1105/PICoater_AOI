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
                    "capture plan grab=260721-120000 root=D:\\Anilox\\Captures "
                    "imageDir=x csv=y archive=260721-120000.acap "
                    "assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r "
                    "preview=1920x1080x3 scale=5",
                ),
                FlowLine(
                    1,
                    "00:00:01.000",
                    1,
                    "capture finalize grab=260721-120000 "
                    "archive=D:\\Anilox\\Captures\\2026\\202607\\20260721\\260721-120000.acap "
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
                    "capture plan grab=260721-120000 root=D:\\Anilox\\Captures "
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

    def test_final_layout_with_pending_change_passes(self):
        session = FlowSession(
            Path("synthetic.log"),
            [
                FlowLine(
                    0, "00:00:00.000", 1,
                    "capture plan grab=260728-120000 root=D:\\Anilox "
                    "imageDir=x csv=y archive=260728-120000.acap "
                    "assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r "
                    "preview=1920x1080x3 scale=5",
                ),
                FlowLine(
                    1, "00:00:01.000", 1,
                    "capture layout pending grab=260728-120000 "
                    "setting=cb_CropHead apply=display-now+stop-final",
                ),
                FlowLine(
                    2, "00:00:02.000", 1,
                    "capture layout final grab=260728-120000 "
                    "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                    "speed=40 head=50 tail=10 path=x",
                ),
                FlowLine(
                    3, "00:00:03.000", 1,
                    "capture layout applied grab=260728-120000 timing=stop "
                    "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                    "speed=40 head=50 tail=10 "
                    "render=already-applied source=unchanged",
                ),
                FlowLine(
                    4, "00:00:04.000", 1,
                    "capture finalize grab=260728-120000 "
                    "archive=D:\\Anilox\\260728-120000.acap "
                    "atlas=3 atlasBytes=1234 remoteFiles=2",
                ),
            ],
        )

        report = CaptureFlowValidator().validate(session)
        result = next(
            item for item in report.results if item.rule == "C2.final-layout"
        )
        self.assertEqual(CheckStatus.PASS, result.status)

    def test_pending_layout_without_apply_fails(self):
        session = FlowSession(
            Path("synthetic.log"),
            [
                FlowLine(
                    0, "00:00:00.000", 1,
                    "capture layout pending grab=260728-120000 "
                    "setting=cb_CropHead apply=display-now+stop-final",
                ),
                FlowLine(
                    1, "00:00:01.000", 1,
                    "capture layout final grab=260728-120000 "
                    "ops=1|1|1|1|1|1|1 start=0|1|2|3|4|5|6 "
                    "speed=40 head=50 tail=10 path=x",
                ),
            ],
        )

        report = CaptureFlowValidator().validate(session)
        result = next(
            item for item in report.results if item.rule == "C2.final-layout"
        )
        self.assertEqual(CheckStatus.FAIL, result.status)


if __name__ == "__main__":
    unittest.main()
