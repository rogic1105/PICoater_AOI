"""Synthetic-log tests for settings, Mura, and camera-parameter validators."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import CheckStatus, FlowLine, FlowSession
from flow_checks.data import DataFlowValidator
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
                "setting route hd_EnableReviewEnhance owner=Enhance effects=None",
                "RV loadGrab begin 260720-120000（proc=True）",
                "RV loadGrab done 260720-120000（21ms）",
                "ui:設定[hee_VerticalDirection]=TopToBottom",
                "setting route hee_VerticalDirection owner=LiveLayout effects=None",
                "LC row rowView dir=TopToBottom n=-1 total=100mm view 0~100",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S0.format").status)
        self.assertEqual(CheckStatus.PASS, result(report, "S0.route").status)
        self.assertEqual(
            CheckStatus.PASS, result(report, "S2.review-enhance").status
        )
        self.assertEqual(CheckStatus.PASS, result(report, "S3.direction").status)

    def test_direction_without_row_refresh_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[hee_VerticalDirection]=BottomToTop",
                "setting route hee_VerticalDirection owner=LiveLayout effects=None",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S3.direction").status)

    def test_setting_without_route_fails(self):
        report = SettingsFlowValidator().validate(
            session("ui:設定[IoIp]=127.0.0.1")
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S0.route").status)

    def test_unrelated_setting_with_capture_policy_fails(self):
        report = SettingsFlowValidator().validate(
            session(
                "ui:設定[IoIp]=127.0.0.1",
                "setting route IoIp owner=Io effects=CapturePolicy",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "S0.route").status)


class DataFlowValidatorTests(unittest.TestCase):
    def test_timer_starvation_without_ping_or_stack_is_not_a_hard_ui_block(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "ui:【報表序號】→ 260721-080000"),
            FlowLine(1.5, "00:00:01.500", 1, "[UiStall] 1500ms（GC0+2 GC1+1 GC2+0）"),
            FlowLine(1.51, "00:00:01.510", 1, "DT selected 260721-080000 stats=cache list=keep ms=1"),
        ]
        report = DataFlowValidator().validate(FlowSession(Path("synthetic.log"), lines))
        responsiveness = result(report, "U.stall")
        self.assertEqual(CheckStatus.PASS, responsiveness.status)
        self.assertIn("計時器飢餓=1", responsiveness.detail)

    def test_correlated_ping_and_stack_are_a_hard_ui_block(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "ui:【報表序號】→ 260721-080000"),
            FlowLine(0.8, "00:00:00.800", 3, "[UiStack] BlockingCall.Wait ←"),
            FlowLine(1.5, "00:00:01.500", 1, "[UiPing] 1400ms"),
            FlowLine(1.51, "00:00:01.510", 1, "[UiStall] 1500ms（GC0+0 GC1+0 GC2+0）"),
            FlowLine(1.52, "00:00:01.520", 1, "DT selected 260721-080000 stats=cache list=keep ms=1"),
        ]
        report = DataFlowValidator().validate(FlowSession(Path("synthetic.log"), lines))
        responsiveness = result(report, "U.stall")
        self.assertEqual(CheckStatus.FAIL, responsiveness.status)
        self.assertIn("真阻塞=1", responsiveness.detail)

    def test_single_curve_latest_only_allows_stale_intermediate_and_requires_final(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only",
                "ui:【報表序號】→ 260721-080000",
                "DT selected 260721-080000 stats=cache list=keep ms=1",
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve stale-drop 260721-080000",
                "DT row curve load 260721-080001 source=shared storage=summary points=100 pitch=0.010000mm",
                "DT curve load 260721-080001 captures=7 source=shared storage=summary configMs=1 waitMs=2 pathMs=0 mergeMs=0 summaryMs=1 points=100 drawMs=3 totalMs=5",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "D3.curve-policy").status)
        self.assertEqual(CheckStatus.PASS, result(report, "D3.curve").status)
        self.assertEqual(CheckStatus.PASS, result(report, "D3.row-curve").status)

    def test_single_curve_latest_only_fails_when_final_selection_never_applies(self):
        report = DataFlowValidator().validate(
            session(
                "DT curve load policy latest-only shared-loader entries=512 maxMB=256 scale=merged-only",
                "ui:【報表序號】→ 260721-080001",
                "DT selected 260721-080001 stats=cache list=keep ms=1",
                "DT curve stale-drop 260721-080001",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.curve").status)
        self.assertEqual(CheckStatus.FAIL, result(report, "D3.row-curve").status)

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

    def test_live_exposure_uses_fast_path_without_reconfiguration(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】cam2 Exp=4000",
                "exposure live apply begin scope=cam2 gate=open",
                "exposure live apply complete scope=cam2 gate=open elapsedMs=320",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.synchronization").status
        )

    def test_live_exposure_reconfiguration_path_fails(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=6000",
                "parameter reconfigure begin scope=All gate=closed targets=2",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.synchronization").status
        )

    def test_live_exposure_fast_path_can_finish_after_stop(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=3000",
                "exposure live apply begin scope=All gate=open",
                "StopGrab",
                "capture gate closed standby=on",
                "exposure live apply complete scope=All gate=closed elapsedMs=450",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.synchronization").status
        )

    def test_live_exposure_over_five_seconds_fails(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】All ExpAll=3000",
                "exposure live apply begin scope=All gate=open",
                "exposure live apply complete scope=All gate=open elapsedMs=5001",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.synchronization").status
        )

    def test_live_line_rate_or_height_intent_fails_policy(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "ui:【相機參數】cam2 LineRate=6000",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "P1.live-policy").status
        )
        self.assertEqual(
            CheckStatus.NOT_COVERED,
            result(report, "P1.synchronization").status,
        )

    def test_live_backend_block_is_valid_policy_evidence(self):
        report = ParameterFlowValidator().validate(
            session(
                "capture gate open cams=2 warm=True",
                "parameter change blocked scope=cam2 param=Height reason=GrabActive",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "P1.live-policy").status
        )

    def test_registry_has_no_pending_domains(self):
        self.assertEqual((), PENDING_DOMAINS)


class LiveStandbyFlowValidatorTests(unittest.TestCase):
    def test_row_chart_waits_for_main_image_presentation(self):
        report = LiveFlowValidator().validate(
            session(
                "rowCurve present after=mainImage cams=2 mode=WF",
                "LC row rowChart dir=BottomToTop n=100 total=10mm view 0~10 "
                "dataPhys 0~5mm dataChart 0~5",
            )
        )
        self.assertEqual(
            CheckStatus.PASS, result(report, "F2.row-presentation").status
        )

    def test_row_chart_without_main_image_presentation_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "LC row rowChart dir=BottomToTop n=100 total=10mm view 0~10 "
                "dataPhys 0~5mm dataChart 0~5",
            )
        )
        self.assertEqual(
            CheckStatus.FAIL, result(report, "F2.row-presentation").status
        )

    def test_initialization_ignores_later_height_reallocation_metrics(self):
        lines = [
            FlowLine(0.0, "00:00:00.000", 1, "AllocateCameras begin (expect 2 cams)"),
            FlowLine(1.0, "00:00:01.000", 1, "camera init cam=1 phase=acquisition ms=10 size=10x10 thread=1"),
            FlowLine(2.0, "00:00:02.000", 1, "camera init cam=2 phase=acquisition ms=10 size=10x10 thread=1"),
            FlowLine(3.0, "00:00:03.000", 1, "camera init phase=acquisition done cams=2 ms=20"),
            FlowLine(4.0, "00:00:04.000", 1, "camera init phase=processing begin cams=2"),
            FlowLine(5.0, "00:00:05.000", 15, "camera init cam=1 phase=processing ms=10 pinnedMB=1 allocCalls=2 thread=15"),
            FlowLine(6.0, "00:00:06.000", 15, "camera init cam=2 phase=processing ms=10 pinnedMB=1 allocCalls=2 thread=15"),
            FlowLine(7.0, "00:00:07.000", 1, "camera init phase=processing done cams=2 ms=20"),
            FlowLine(8.0, "00:00:08.000", 1, "camera init summary acquisition=20ms processing=20ms total=40ms"),
            FlowLine(9.0, "00:00:09.000", 29, "camera init cam=1 phase=processing ms=8 pinnedMB=1 allocCalls=2 thread=29"),
        ]
        report = LiveFlowValidator().validate(
            FlowSession(Path("synthetic.log"), lines)
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F1.init").status)

    def test_warm_ready_gate_start_and_stop_pass(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition parameters ready cam2 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=102",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=2",
                "acquisition sync paused reason=start attempt=1 cams=2",
                "acquisition sync resumed reason=start attempt=1 cams=2",
                "acquisition sync ready reason=start attempt=1 cam1 system=0 tick=100 freq=1000",
                "acquisition sync ready reason=start attempt=1 cam2 system=0 tick=102 freq=1000",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1,2 spreadTicks=2 spreadMs=2.000 limitMs=5.000 measurable=True aligned=True",
                "acquisition sync complete reason=start attempts=1 cams=2 phase=True",
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

    def test_warm_standby_without_physical_start_sync_fails(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition parameters ready cam2 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=200",
                "StartGrab cams=2",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> Waterfall",
                "firstFrame cam2 100x100 -> Waterfall",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.FAIL, result(report, "F2.standby").status)

    def test_start_phase_retry_can_succeed(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3000",
                "acquisition parameters ready cam2 cl=True lineRate=3000",
                "acquisition standby start cam1",
                "acquisition standby start cam2",
                "acquisition standby ready cam1 tick=100",
                "acquisition standby ready cam2 tick=200",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=2",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1,2 spreadTicks=20 spreadMs=20.000 limitMs=5.000 measurable=True aligned=False",
                "acquisition sync retry reason=start attempt=1 error=PhaseOutOfRange",
                "acquisition sync begin reason=start attempt=2 gate=closed cams=2",
                "acquisition sync ready reason=start attempt=2 cam1 system=0 tick=200 freq=1000",
                "acquisition sync ready reason=start attempt=2 cam2 system=0 tick=202 freq=1000",
                "acquisition sync phase reason=start attempt=2 system=0 cams=1,2 spreadTicks=2 spreadMs=2.000 limitMs=5.000 measurable=True aligned=True",
                "acquisition sync complete reason=start attempts=2 cams=2 phase=True",
                "StartGrab cams=2",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=2 warm=True",
                "firstFrame cam1 100x100 -> Waterfall",
                "firstFrame cam2 100x100 -> Waterfall",
                "StopGrab",
                "capture gate closed standby=on",
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

    def test_exposure_fast_path_keeps_capture_gate_open(self):
        report = LiveFlowValidator().validate(
            session(
                "acquisition parameters ready cam1 cl=True lineRate=3001",
                "acquisition standby start cam1",
                "acquisition standby ready cam1 tick=100",
                "acquisition sync begin reason=start attempt=1 gate=closed cams=1",
                "acquisition sync paused reason=start attempt=1 cams=1",
                "acquisition sync resumed reason=start attempt=1 cams=1",
                "acquisition sync ready reason=start attempt=1 cam1 system=0 tick=100 freq=1000",
                "acquisition sync phase reason=start attempt=1 system=0 cams=1 spreadTicks=0 spreadMs=0.000 limitMs=5.000 measurable=True aligned=True",
                "acquisition sync complete reason=start attempts=1 cams=1 phase=True",
                "StartGrab cams=1",
                "capture plan grab=260720-120000 root=D:\\Anilox",
                "capture gate open cams=1 warm=True",
                "exposure live apply begin scope=cam1 gate=open",
                "exposure live apply complete scope=cam1 gate=open elapsedMs=300",
                "StopGrab",
                "capture gate closed standby=on",
            )
        )
        self.assertEqual(CheckStatus.PASS, result(report, "F2.standby").status)

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
