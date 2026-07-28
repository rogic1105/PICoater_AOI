"""Synthetic-log tests for hardware lifecycle and IO Grab validators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import CheckStatus, FlowLine, FlowSession
from flow_checks.hardware import HardwareFlowValidator


def session(*messages: str) -> FlowSession:
    return FlowSession(
        Path("synthetic.log"),
        [
            FlowLine(float(index), f"00:00:{index:02d}.000", 1, message)
            for index, message in enumerate(messages)
        ],
    )


def result(report, rule: str):
    return next(item for item in report.results if item.rule == rule)


class HardwareFlowValidatorTests(unittest.TestCase):
    def test_serial_controller_restart_and_grab_outcome_pass(self):
        report = HardwareFlowValidator().validate(
            session(
                "IO controller start generation=1 endpoint=127.0.0.1:502",
                "io:DI START 上升緣 → 抓取請求",
                "io:DI START 上升緣 → 開始抓取",
                "IO grab accepted busy=on state=started",
                "IO controller stop generation=1 reason=settings",
                "IO controller start generation=2 endpoint=192.168.255.1:502",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "H1.io-lifecycle").status)
        self.assertEqual(CheckStatus.PASS, result(report, "H3.io-grab").status)

    def test_overlapping_controller_and_missing_grab_outcome_fail(self):
        report = HardwareFlowValidator().validate(
            session(
                "IO controller start generation=1 endpoint=127.0.0.1:502",
                "IO controller start generation=2 endpoint=192.168.255.1:502",
                "io:DI START 上升緣 → 抓取請求",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "H1.io-lifecycle").status)
        self.assertEqual(CheckStatus.FAIL, result(report, "H3.io-grab").status)

    def test_busy_io_request_with_one_rejection_passes(self):
        report = HardwareFlowValidator().validate(
            session(
                "io:DI START 上升緣 → 抓取請求",
                "IO grab rejected busy=off reason=capture-not-ready:manager-busy",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "H3.io-grab").status)

    def test_tail_drain_cannot_be_accepted_as_already_grabbing(self):
        report = HardwareFlowValidator().validate(
            session(
                "capture tail begin cams=1,2 timeoutMs=3500",
                "io:DI START 上升緣 → 抓取請求",
                "IO grab accepted busy=on state=already-grabbing",
                "capture tail complete pending=",
                "StopGrab",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "H3.io-grab").status)

    def test_tail_drain_rejection_keeps_request_valid(self):
        report = HardwareFlowValidator().validate(
            session(
                "capture tail begin cams=1,2 timeoutMs=3500",
                "io:DI START 上升緣 → 抓取請求",
                "IO grab rejected busy=off reason=capture-not-ready:tail-drain",
                "capture tail complete pending=",
                "StopGrab",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "H3.io-grab").status)

    def test_disconnect_during_start_with_rejection_passes(self):
        report = HardwareFlowValidator().validate(
            session(
                "io:DI START 上升緣 → 抓取請求",
                "io:DI START 上升緣 → 開始抓取",
                "StartGrab（cams=4）",
                "⚠ IO 斷線",
                "capture start cancelled before gate reason=io-request-invalid",
                "StopGrab",
                "IO grab rejected busy=off reason=io-disconnected",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "H3.io-grab").status)

    def test_io_stop_policy_accepts_io_and_ignores_fixed_targets(self):
        report = HardwareFlowValidator().validate(
            session(
                "IO grab stop accepted reason=StartLow stopCondition=IoSignal drainTail=True",
                "IO grab stop accepted reason=CommunicationLost stopCondition=IoSignal drainTail=False",
                "IO grab stop ignored reason=CommunicationLost stopCondition=Time captureContinues=True",
                "IO grab stop ignored reason=PlcAliveLost stopCondition=Height captureContinues=True",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "H4.io-stop-policy").status)

    def test_io_stop_policy_rejects_fixed_target_stop(self):
        report = HardwareFlowValidator().validate(
            session(
                "IO grab stop accepted reason=CommunicationLost stopCondition=Time drainTail=False",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "H4.io-stop-policy").status)


if __name__ == "__main__":
    unittest.main()
