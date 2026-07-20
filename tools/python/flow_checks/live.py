"""Live-display (F-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class LiveFlowValidator:
    domain = "LIVE"

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        self._check_camera_initialization(session, report)
        self._check_capture_standby(session, report)
        if not any(
            line.message.startswith(("LC ", "IC ", "WF ", "ui:【開始抓取】"))
            for line in session.lines
        ):
            report.add(self.domain, "F0", CheckStatus.NOT_COVERED, "本 session 無監控操作")
            return report

        self._check_drag_first_publish(session, report)
        return report

    def _check_camera_initialization(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        begins = [
            index
            for index, line in enumerate(session.lines)
            if line.message.startswith("AllocateCameras begin")
        ]
        if not begins:
            report.add(self.domain, "F1.init", CheckStatus.NOT_COVERED, "本 session 無相機配置")
            return

        failures = []
        covered = 0
        worst_allocation_stall = 0
        worst_stall = 0
        for sequence, begin_index in enumerate(begins, start=1):
            end_index = begins[sequence] if sequence < len(begins) else len(session.lines)
            lines = session.lines[begin_index:end_index]
            if not any("camera init phase=" in line.message for line in lines):
                continue

            covered += 1
            acquisition = [
                line for line in lines
                if re.match(r"camera init cam=\d+ phase=acquisition ", line.message)
            ]
            processing = [
                line for line in lines
                if re.match(r"camera init cam=\d+ phase=processing ", line.message)
            ]
            acquisition_done = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=acquisition done ")
                ),
                None,
            )
            processing_begin = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=processing begin ")
                ),
                None,
            )
            processing_done = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=processing done ")
                ),
                None,
            )
            summary = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init summary ")
                ),
                None,
            )

            required = (acquisition_done, processing_begin, processing_done, summary)
            if any(line is None for line in required):
                failures.append(f"配置#{sequence} 階段行不完整")
                continue

            expected_match = re.search(r"cams=(\d+)", acquisition_done.message)
            expected = int(expected_match.group(1)) if expected_match else -1
            if len(acquisition) != expected or len(processing) != expected:
                failures.append(
                    f"配置#{sequence} cams={expected} acquisition={len(acquisition)} "
                    f"processing={len(processing)}"
                )

            allocation_calls = []
            for line in processing:
                match = re.search(r"\ballocCalls=(\d+)\b", line.message)
                allocation_calls.append(int(match.group(1)) if match else -1)
            if any(count != 2 for count in allocation_calls):
                failures.append(
                    f"配置#{sequence} processing allocCalls={allocation_calls}"
                )

            if expected > 0:
                acquisition_threads = {line.thread for line in acquisition}
                processing_threads = {line.thread for line in processing}
                if (
                    len(acquisition_threads) != 1
                    or len(processing_threads) != 1
                    or acquisition_threads == processing_threads
                ):
                    failures.append(
                        f"配置#{sequence} acquisitionThreads={sorted(acquisition_threads)} "
                        f"processingThreads={sorted(processing_threads)}"
                    )

            processing_window = [
                line
                for line in lines
                if processing_begin.elapsed <= line.elapsed <= processing_done.elapsed
            ]
            stalls = []
            for line in processing_window:
                match = re.match(r"\[UiStall\]\s+(\d+)ms", line.message)
                if match:
                    stalls.append(int(match.group(1)))
            if stalls:
                worst_stall = max(worst_stall, max(stalls))
                failures.append(
                    f"配置#{sequence} processing 期間 UiStall 最大 {max(stalls)}ms"
                )

            allocation_stalls = []
            for line in lines:
                # WM_TIMER is low priority, so the stall line can be emitted just after
                # the allocation summary even though the measured gap occurred inside it.
                if summary.elapsed + 1 < line.elapsed:
                    continue
                match = re.match(r"\[UiStall\]\s+(\d+)ms", line.message)
                if match:
                    allocation_stalls.append(int(match.group(1)))
            if allocation_stalls:
                worst_allocation_stall = max(
                    worst_allocation_stall, max(allocation_stalls)
                )
                if max(allocation_stalls) > 1000:
                    failures.append(
                        f"配置#{sequence} UI stall 最大 {max(allocation_stalls)}ms"
                    )

        if covered == 0:
            report.add(
                self.domain,
                "F1.init",
                CheckStatus.NOT_COVERED,
                f"配置={len(begins)}；舊版 log 無 phase 儀器",
            )
            return

        report.add(
            self.domain,
            "F1.init",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"配置={covered}；failures={len(failures)}；"
            f"allocationWorstStall={worst_allocation_stall}ms；"
            f"processingWorstStall={worst_stall}ms"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_capture_standby(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        if not any(
            line.message.startswith(
                (
                    "acquisition parameters ready ",
                    "acquisition standby start ",
                    "acquisition standby ready ",
                    "capture gate open ",
                    "capture gate closed ",
                )
            )
            for line in session.lines
        ):
            report.add(
                self.domain,
                "F2.standby",
                CheckStatus.NOT_COVERED,
                "session predates acquisition standby instrumentation",
            )
            return

        parameter_ready_cameras = set()
        ready_cameras = set()
        start_pending = False
        capture_open = False
        stop_pending = False
        plan_ready = False
        background_capture = False
        starts = 0
        stops = 0
        failures = []

        for line in session.lines:
            message = line.message
            parameter_match = re.match(
                r"acquisition parameters ready cam(\d+) "
                r"cl=(True|False) lineRate=([0-9]+(?:\.[0-9]+)?)",
                message,
            )
            if parameter_match:
                parameter_ready_cameras.add(int(parameter_match.group(1)))
                continue

            standby_start_match = re.match(
                r"acquisition standby start cam(\d+)", message
            )
            if standby_start_match:
                camera_id = int(standby_start_match.group(1))
                if camera_id not in parameter_ready_cameras:
                    failures.append(
                        f"standby started before parameters ready for cam{camera_id} "
                        f"at {line.timestamp}"
                    )
                continue

            ready_match = re.match(
                r"acquisition standby ready cam(\d+) tick=(-?\d+)", message
            )
            if ready_match:
                camera_id = int(ready_match.group(1))
                if camera_id not in parameter_ready_cameras:
                    failures.append(
                        f"standby ready before parameters ready for cam{camera_id} "
                        f"at {line.timestamp}"
                    )
                ready_cameras.add(camera_id)
                continue

            if message == "ui:【取得背景】鈕":
                background_capture = True
                continue

            if message.startswith("StartGrab"):
                starts += 1
                start_pending = True
                stop_pending = False
                plan_ready = background_capture
                continue

            if message.startswith("capture plan "):
                plan_ready = True
                continue

            gate_match = re.match(
                r"capture gate open cams=(\d+) warm=(True|False)", message
            )
            if gate_match:
                expected = int(gate_match.group(1))
                warm = gate_match.group(2) == "True"
                if not start_pending:
                    failures.append(f"gate open without StartGrab at {line.timestamp}")
                if not plan_ready:
                    failures.append(
                        f"gate opened before capture plan at {line.timestamp}"
                    )
                if not warm or len(ready_cameras) < expected:
                    failures.append(
                        f"gate opened before warm ready at {line.timestamp}: "
                        f"ready={len(ready_cameras)} expected={expected} warm={warm}"
                    )
                if len(parameter_ready_cameras) < expected:
                    failures.append(
                        f"gate opened before parameters ready at {line.timestamp}: "
                        f"ready={len(parameter_ready_cameras)} expected={expected}"
                    )
                start_pending = False
                plan_ready = False
                capture_open = True
                continue

            if message == "StopGrab":
                stops += 1
                capture_open = False
                stop_pending = True
                background_capture = False
                continue

            if message == "capture gate closed standby=on":
                if not stop_pending:
                    failures.append(f"gate closed without StopGrab at {line.timestamp}")
                stop_pending = False
                continue

            if "firstFrame " in message and not capture_open:
                failures.append(f"firstFrame while capture gate closed at {line.timestamp}")

        if start_pending:
            failures.append("last StartGrab has no capture gate open")
        if stop_pending:
            failures.append("last StopGrab has no capture gate closed")

        report.add(
            self.domain,
            "F2.standby",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"parameterReadyCams={len(parameter_ready_cameras)} "
            f"readyCams={len(ready_cameras)} starts={starts} stops={stops} "
            f"failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_drag_first_publish(self, session: FlowSession, report: CheckReport) -> None:
        starts = []
        failures = []
        active = {"IC": None, "WF": None}

        for line in session.lines:
            message = line.message
            for prefix in active:
                if message == f"{prefix} drag(start)":
                    starts.append((prefix, line.elapsed))
                    active[prefix] = False
                elif message == f"{prefix} drag(view-published)" and active[prefix] is False:
                    active[prefix] = True
                elif message.startswith(f"{prefix} viewEdges") and active[prefix] is not None:
                    if active[prefix] is not True:
                        failures.append(f"{prefix}@{line.elapsed:.3f}s")
                    active[prefix] = None

        for prefix, state in active.items():
            if state is False:
                failures.append(f"{prefix}@未結束")

        if not starts:
            report.add(self.domain, "F6.first-view", CheckStatus.NOT_COVERED, "無主畫面拖曳")
            return
        report.add(
            self.domain,
            "F6.first-view",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"拖曳={len(starts)}；首位移未發布={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
