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
                     "archive=260721-120000.acap scale=4"),
            FlowLine(1, "00:00:01.000", 1, message),
        ],
    )
    report = CaptureFlowValidator().validate(session)
    return next(item for item in report.results if item.rule == "C2.cfg-scale")


class CaptureFlowValidatorTests(unittest.TestCase):
    def test_acap_plan_passes(self):
        session = FlowSession(
            Path("synthetic.log"),
            [FlowLine(
                0, "00:00:00.000", 1,
                "capture plan grab=260721-120000 root=D:\\Anilox "
                "imageDir=x csv=y archive=260721-120000.acap scale=4")],
        )
        report = CaptureFlowValidator().validate(session)
        result = next(item for item in report.results if item.rule == "C1.plan")
        self.assertEqual(CheckStatus.PASS, result.status)

    def test_first_record_requires_archive_append(self):
        lines = [
            FlowLine(0, "00:00:00.000", 1,
                     "capture plan grab=260721-120000 root=D:\\Anilox "
                     "imageDir=x csv=y archive=260721-120000.acap scale=4"),
            FlowLine(1, "00:00:01.000", 2,
                     "capture csv firstRecord grab=260721-120000 path=x file=f "
                     "verdict=PASS peak=0 rowPeak=0 maxCMean=0"),
        ]
        report = CaptureFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines))
        result = next(item for item in report.results if item.rule == "C2.first-record")
        self.assertEqual(CheckStatus.FAIL, result.status)

        lines.insert(
            1,
            FlowLine(0.5, "00:00:00.500", 3,
                     "capture archive append grab=260721-120000 cam=1 "
                     "frame=f assets=7 bytes=123"),
        )
        report = CaptureFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines))
        result = next(item for item in report.results if item.rule == "C2.first-record")
        self.assertEqual(CheckStatus.PASS, result.status)

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

    def test_capture_write_failure_fails_integrity_even_when_health_sequence_is_valid(self):
        lines = [
            FlowLine(0, "00:00:00.000", 1,
                     "capture plan grab=260721-120000 root=D:\\Anilox "
                     "imageDir=x csv=y archive=260721-120000.acap scale=4"),
            FlowLine(1, "00:00:01.000", 2,
                     "capture archive append grab=260721-120000 cam=1 "
                     "frame=f assets=7 bytes=123"),
            FlowLine(2, "00:00:02.000", 3,
                     "[OutputHealth] raise code=CaptureWriteFailure.CAM1 "
                     "severity=OutputFault message=CAM1 存檔失敗"),
            FlowLine(3, "00:00:02.001", 3,
                     "[OutputHealth] state Normal -> OutputFault "
                     "code=CaptureWriteFailure.CAM1 active=True"),
        ]
        report = CaptureFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines))
        result = next(
            item for item in report.results if item.rule == "C3.write-integrity")
        self.assertEqual(CheckStatus.FAIL, result.status)

    def test_capture_stop_drains_before_remote_release(self):
        lines = [
            FlowLine(0, "00:00:00.000", 1,
                     "capture plan grab=260721-120000 root=D:\\Anilox "
                     "imageDir=x csv=y archive=260721-120000.acap scale=4"),
            FlowLine(1, "00:00:01.000", 1, "StopGrab"),
            FlowLine(2, "00:00:02.000", 1,
                     "capture save drain begin grab=260721-120000"),
            FlowLine(3, "00:00:03.000", 2,
                     "capture archive append grab=260721-120000 cam=1 "
                     "frame=f assets=7 bytes=123"),
            FlowLine(4, "00:00:04.000", 1,
                     "capture save drain done grab=260721-120000"),
            FlowLine(5, "00:00:05.000", 1,
                     "capture remote release grab=260721-120000 files=2 bytes=456"),
        ]

        report = CaptureFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines))
        result = next(
            item for item in report.results if item.rule == "C3.delivery-release")

        self.assertEqual(CheckStatus.PASS, result.status)

    def test_archive_append_after_drain_done_fails_delivery_release(self):
        lines = [
            FlowLine(0, "00:00:00.000", 1,
                     "capture plan grab=260721-120000 root=D:\\Anilox "
                     "imageDir=x csv=y archive=260721-120000.acap scale=4"),
            FlowLine(1, "00:00:01.000", 1, "StopGrab"),
            FlowLine(2, "00:00:02.000", 1,
                     "capture save drain begin grab=260721-120000"),
            FlowLine(3, "00:00:03.000", 1,
                     "capture save drain done grab=260721-120000"),
            FlowLine(4, "00:00:04.000", 2,
                     "capture archive append grab=260721-120000 cam=1 "
                     "frame=f assets=7 bytes=123"),
            FlowLine(5, "00:00:05.000", 1,
                     "capture remote release grab=260721-120000 files=2 bytes=456"),
        ]

        report = CaptureFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines))
        result = next(
            item for item in report.results if item.rule == "C3.delivery-release")

        self.assertEqual(CheckStatus.FAIL, result.status)


if __name__ == "__main__":
    unittest.main()
