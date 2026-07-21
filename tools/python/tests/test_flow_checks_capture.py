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
                     "_raw.jpg _proc_c.jpg _proc_r.jpg _mean_c.bin _max_c.bin _mean_r.bin _max_r.bin"),
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


if __name__ == "__main__":
    unittest.main()
