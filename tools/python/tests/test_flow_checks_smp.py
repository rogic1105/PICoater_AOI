"""Synthetic-log tests for settings, Mura, and camera-parameter validators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import CheckStatus, FlowLine, FlowSession
from flow_checks.live import LiveFlowValidator
from flow_checks.mura import MuraFlowValidator
from flow_checks.parameter import ParameterFlowValidator
from flow_checks.registry import PENDING_DOMAINS
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


class SettingsFlowValidatorTests(unittest.TestCase):
    def test_direction_and_review_enhance_are_followed_by_required_updates(self):
        report = SettingsFlowValidator().validate(
            session(
                "RV loadGrab done 260720-120000（20ms）",
                "ui:設定[hd_EnableReviewEnhance]=True",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab done 260720-120000（21ms）",
                "ui:設定[hee_VerticalDirection]=TopToBottom",
                "LC row rowView dir=TopToBottom n=-1 total=100mm view 0~100",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S0.format").status)
        self.assertEqual(
            CheckStatus.PASS, result(report, "S2.review-enhance").status
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S3.direction").status)

    def test_direction_without_row_refresh_fails(self):
        report = SettingsFlowValidator().validate(
            session("ui:設定[hee_VerticalDirection]=BottomToTop")
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S3.direction").status)


class MuraFlowValidatorTests(unittest.TestCase):
    def test_edges_health_and_pause_sequences_pass(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（v）mean=0.30/max=0.70（thr 0.20/0.60，IO已連線）",
                "[OutputHealth] raise code=MuraExceed.v severity=Critical message=檢測異常（欄）",
                "MURA 恢復（v）",
                "[OutputHealth] resolve code=MuraExceed.v message=檢測異常（欄）",
                "ui:【暫停Mura檢測】鈕",
                "set:[MuraDetectPaused]=True",
                "MURA 暫停 → 清除 DO1",
                "ui:【暫停Mura檢測】鈕",
                "set:[MuraDetectPaused]=False",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "M1.edges").status)
        self.assertEqual(CheckStatus.PASS, result(report, "M1.health").status)
        self.assertEqual(CheckStatus.PASS, result(report, "M1.pause").status)

    def test_duplicate_edge_fails_and_legacy_health_is_not_covered(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（h）mean=0.30/max=0.70（thr 0.20/0.60，IO未連線→僅畫面警告）",
                "⚠ MURA 超標（h）mean=0.31/max=0.71（thr 0.20/0.60，IO未連線→僅畫面警告）",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "M1.edges").status)
        self.assertEqual(
            CheckStatus.NOT_COVERED, result(report, "M1.health").status
        )

    def test_missing_health_resolve_fails(self):
        report = MuraFlowValidator().validate(
            session(
                "StartGrab（cams=4）",
                "⚠ MURA 超標（v）mean=0.30/max=0.70（thr 0.20/0.60，IO已連線）",
                "[OutputHealth] raise code=MuraExceed.v severity=Critical message=檢測異常（欄）",
                "MURA 恢復（v）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "M1.edges").status)
        self.assertEqual(CheckStatus.FAIL, result(report, "M1.health").status)


class ParameterFlowValidatorTests(unittest.TestCase):
    def test_user_adjustment_after_startup_passes(self):
        report = ParameterFlowValidator().validate(
            session(
                "AllocateCameras begin（expect 7 cams）",
                "AllocateCameras done（配置 4、在線 4/7）",
                "idle",
                "ui:【相機參數】cam2 Height=4000",
                "[UiStall] 200ms（GC0+0 GC1+0 GC2+0）",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "P1.startup").status)
        self.assertEqual(CheckStatus.PASS, result(report, "P1.intent").status)
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.responsiveness").status
        )

    def test_debounced_initialization_intent_fails_startup(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "AllocateCameras begin（expect 7 cams）"),
            FlowLine(0.5, "00:00:00.500", 1, "AllocateCameras done（配置 4、在線 4/7）"),
            FlowLine(0.7, "00:00:00.700", 1, "ui:【相機參數】All HeightAll=3001"),
        ]
        report = ParameterFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines)
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "P1.startup").status)

    def test_registry_has_no_pending_domains(self):
        self.assertEqual((), PENDING_DOMAINS)


class LiveStandbyFlowValidatorTests(unittest.TestCase):
    def test_warm_ready_gate_start_and_stop_pass(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition parameters ready cam2 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=102",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> ImageDisplayView",
                "firstFrame cam2 100x100 -> ImageDisplayView",
                "StopGrab",
                "capture gate closed standby=on",
                "drop drainedFrame after StopGrab cam1",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

    def test_gate_before_all_cameras_are_ready_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition parameters ready cam2 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=False",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_gate_before_capture_plan_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture gate open cams=1 warm=True",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_standby_before_parameter_work_completes_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "StartGrab cams=4",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)
