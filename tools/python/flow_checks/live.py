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
        self._check_capture_head_guard(session, report)
        self._check_waterfall_bootstrap(session, report)
        self._check_capture_view_refire(session, report)
        self._check_row_presentation(session, report)
        self._check_wheel_zoom_floor(session, report)
        if not any(
            line.message.startswith(("LC ", "IC ", "WF ", "ui:【開始抓取】"))
            for line in session.lines
        ):
            report.add(self.domain, "F0", CheckStatus.NOT_COVERED, "本 session 無監控操作")
            return report

        self._check_drag_first_publish(session, report)
        return report

    def _check_capture_head_guard(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        contract_enabled = any(
            line.message == "experiment build=mil-edge-coverage-v8"
            for line in session.lines
        )
        has_head_evidence = any(
            line.message.startswith("capture head frame dropped ")
            for line in session.lines
        )
        if not contract_enabled and not has_head_evidence:
            report.add(
                self.domain,
                "F2.head-guard",
                CheckStatus.NOT_COVERED,
                "session predates cross-boundary head-frame guard",
            )
            return

        expected = None
        dropped = set()
        opens = 0
        completed = 0
        failures = []

        for line in session.lines:
            gate_match = re.match(r"capture gate open cams=(\d+)\b", line.message)
            if gate_match:
                expected = int(gate_match.group(1))
                dropped = set()
                opens += 1
                continue

            drop_match = re.match(
                r"capture head frame dropped cam(\d+) tick=-?\d+ "
                r"reason=cross-boundary$",
                line.message,
            )
            if drop_match:
                camera_id = int(drop_match.group(1))
                if expected is None:
                    failures.append(
                        f"head-frame drop without open gate at {line.timestamp}"
                    )
                elif camera_id in dropped:
                    failures.append(
                        f"duplicate head-frame drop for cam{camera_id} "
                        f"at {line.timestamp}"
                    )
                else:
                    dropped.add(camera_id)
                continue

            first_set_match = re.match(
                r"capture first-set ready path=\S+ cams=([\d,]+) "
                r"aligned=(True|False)$",
                line.message,
            )
            if first_set_match:
                first_set_cameras = {
                    int(value) for value in first_set_match.group(1).split(",")
                }
                if expected is None:
                    failures.append(
                        f"first accepted set without open gate at {line.timestamp}"
                    )
                elif len(dropped) != expected:
                    failures.append(
                        f"first accepted set before all head frames were dropped "
                        f"at {line.timestamp}: dropped={len(dropped)}/{expected}"
                    )
                elif dropped != first_set_cameras:
                    failures.append(
                        f"head-drop cameras {sorted(dropped)} differ from first set "
                        f"{sorted(first_set_cameras)} at {line.timestamp}"
                    )
                else:
                    completed += 1
                expected = None
                dropped = set()
                continue

            if line.message == "StopGrab":
                expected = None
                dropped = set()

        if completed == 0 and not failures:
            report.add(
                self.domain,
                "F2.head-guard",
                CheckStatus.NOT_COVERED,
                f"gateOpens={opens}; no complete first accepted set",
            )
            return

        report.add(
            self.domain,
            "F2.head-guard",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"gateOpens={opens} completed={completed} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_wheel_zoom_floor(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        pattern = re.compile(
            r"^(?:IC|WF|RV) wheelZoom (?:in|out) → "
            r"zoom=(?P<zoom>\d+(?:\.\d+)?)（fit=(?P<fit>\d+(?:\.\d+)?) "
            r"min=(?P<minimum>\d+(?:\.\d+)?) "
            r"content=(?P<width>\d+)x(?P<height>\d+)）$"
        )
        samples = []
        failures = []

        for line in session.lines:
            if " wheelZoom " not in line.message:
                continue
            match = pattern.match(line.message)
            if match is None:
                failures.append(f"{line.timestamp} 舊版或無法解析的 wheelZoom：{line.message}")
                continue

            zoom = float(match.group("zoom"))
            minimum_text = match.group("minimum")
            minimum = float(minimum_text)
            width = int(match.group("width"))
            height = int(match.group("height"))
            expected = max(0.000001, 1.0 / width, 1.0 / height)
            samples.append((zoom, minimum, expected))
            decimal_places = (
                len(minimum_text.split(".", 1)[1])
                if "." in minimum_text
                else 0
            )
            print_rounding = 0.5 * (10 ** -decimal_places)
            if abs(minimum - expected) > max(
                print_rounding + 1e-12, expected * 0.02
            ):
                failures.append(
                    f"{line.timestamp} min={minimum:g} 應為內容下限 {expected:g}"
                )
            if zoom + 0.000001 < minimum:
                failures.append(
                    f"{line.timestamp} zoom={zoom:g} 低於 min={minimum:g}"
                )

        if not samples and not failures:
            report.add(
                self.domain,
                "F6.zoom-floor",
                CheckStatus.NOT_COVERED,
                "本 session 無主畫面滾輪縮放",
            )
            return
        report.add(
            self.domain,
            "F6.zoom-floor",
            CheckStatus.PASS if samples and not failures else CheckStatus.FAIL,
            f"samples={len(samples)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_waterfall_bootstrap(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        bootstrap_lines = [
            line
            for line in session.lines
            if line.message.startswith("WF bootstrap period ")
        ]
        if not bootstrap_lines:
            report.add(
                self.domain,
                "F2.waterfall-bootstrap",
                CheckStatus.NOT_COVERED,
                "本 session 無瀑布預載週期儀器",
            )
            return

        waterfall_mode = False
        prepared = False
        starts = 0
        failures = []
        for line in session.lines:
            message = line.message
            if message == "ApplyMainDisplayMode → Waterfall":
                waterfall_mode = True
                continue
            if message == "ApplyMainDisplayMode → ImageCanvas":
                waterfall_mode = False
                continue
            if message.startswith("WF bootstrap period "):
                prepared = "source=applied-hardware" in message
                if not prepared:
                    failures.append(
                        f"{line.timestamp} 瀑布週期退回第二幀學習：{message}"
                    )
                continue
            if message.startswith("StartGrab") and waterfall_mode:
                starts += 1
                if not prepared:
                    failures.append(
                        f"{line.timestamp} 瀑布 StartGrab 前未預載硬體週期"
                    )
                prepared = False

        report.add(
            self.domain,
            "F2.waterfall-bootstrap",
            CheckStatus.PASS if starts > 0 and not failures else CheckStatus.FAIL,
            f"starts={starts} bootstrap={len(bootstrap_lines)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_capture_view_refire(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        starts = 0
        refires = 0
        awaiting_refire = False
        failures = []

        for line in session.lines:
            message = line.message
            if message.startswith("StartGrab"):
                if awaiting_refire:
                    failures.append(
                        f"{line.timestamp} previous StartGrab has no view-range refire"
                    )
                starts += 1
                awaiting_refire = True
                continue
            if message.startswith("viewRange refire reason=capture-start mode="):
                if awaiting_refire:
                    refires += 1
                    awaiting_refire = False
                continue
            if message.startswith("capture gate open ") and awaiting_refire:
                failures.append(
                    f"{line.timestamp} capture gate opened before view-range refire"
                )
                awaiting_refire = False

        if awaiting_refire:
            failures.append("last StartGrab has no view-range refire")
        if starts == 0:
            report.add(
                self.domain,
                "F2.view-refire",
                CheckStatus.NOT_COVERED,
                "session has no StartGrab",
            )
            return
        report.add(
            self.domain,
            "F2.view-refire",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"starts={starts} refires={refires} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_row_presentation(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        latest_presentation = None
        presentations = 0
        rows = 0
        failures = []

        for line in session.lines:
            if re.match(
                r"^rowCurve present after=mainImage cams=\d+ mode=(IC|WF)$",
                line.message,
            ):
                latest_presentation = line
                presentations += 1
                continue
            if not line.message.startswith("LC row rowChart "):
                continue

            rows += 1
            if latest_presentation is None:
                failures.append(f"{line.timestamp} rowChart 無 mainImage 呈現證據")
                continue
            delay = line.elapsed - latest_presentation.elapsed
            if delay < 0 or delay > 1.0:
                failures.append(
                    f"{line.timestamp} rowChart 距 mainImage 呈現 {delay:.3f}s"
                )

        if rows == 0:
            detail = (
                "已有 mainImage 後 Curve 接受證據，但日常記錄未開啟 rowChart DVT 快照"
                if presentations > 0
                else "本 session 無監控列曲線更新"
            )
            report.add(
                self.domain,
                "F2.row-presentation",
                CheckStatus.NOT_COVERED,
                detail,
            )
            return
        report.add(
            self.domain,
            "F2.row-presentation",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"presentations={presentations} rowUpdates={rows} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

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
            summary_index = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.message.startswith("camera init summary ")
                ),
                None,
            )
            # Height reallocation intentionally reuses the per-camera processing metric later in
            # the session. F1 only owns the initial allocation window ending at its summary.
            initialization_lines = (
                lines if summary_index is None else lines[: summary_index + 1]
            )
            acquisition = [
                line for line in initialization_lines
                if re.match(r"camera init cam=\d+ phase=acquisition ", line.message)
            ]
            processing = [
                line for line in initialization_lines
                if re.match(r"camera init cam=\d+ phase=processing ", line.message)
            ]
            acquisition_done = next(
                (
                    line for line in initialization_lines
                    if line.message.startswith("camera init phase=acquisition done ")
                ),
                None,
            )
            processing_begin = next(
                (
                    line for line in initialization_lines
                    if line.message.startswith("camera init phase=processing begin ")
                ),
                None,
            )
            processing_done = next(
                (
                    line for line in initialization_lines
                    if line.message.startswith("camera init phase=processing done ")
                ),
                None,
            )
            summary = next(
                (
                    line for line in initialization_lines
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
                    "acquisition sync begin ",
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
        start_syncs = 0
        start_sync_ready = False
        standby_phase_verified = False
        standby_start_ready = False
        parameter_sync_ready = False
        tail_active = False
        tail_completed = False
        sync_expected = {}
        sync_ready_cameras = {}
        sync_phase_results = {}
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

            sync_begin_match = re.match(
                r"acquisition sync begin reason=(\S+) attempt=(\d+) "
                r"gate=closed cams=(\d+)",
                message,
            )
            if sync_begin_match:
                reason = sync_begin_match.group(1)
                attempt = int(sync_begin_match.group(2))
                key = (reason, attempt)
                sync_expected[key] = int(sync_begin_match.group(3))
                sync_ready_cameras[key] = set()
                sync_phase_results[key] = []
                if reason == "start":
                    if capture_open:
                        failures.append(
                            f"start synchronization began while capture gate open "
                            f"at {line.timestamp}"
                        )
                    if int(sync_begin_match.group(2)) == 1:
                        start_syncs += 1
                        start_sync_ready = False
                elif reason.startswith("parameter:"):
                    parameter_sync_ready = False
                continue

            sync_ready_match = re.match(
                r"acquisition sync ready reason=(\S+) attempt=(\d+) "
                r"cam(\d+) system=(-?\d+) tick=(-?\d+) freq=(-?\d+)",
                message,
            )
            if sync_ready_match:
                key = (
                    sync_ready_match.group(1),
                    int(sync_ready_match.group(2)),
                )
                sync_ready_cameras.setdefault(key, set()).add(
                    int(sync_ready_match.group(3))
                )
                continue

            sync_phase_match = re.match(
                r"acquisition sync phase reason=(\S+) attempt=(\d+) "
                r"system=(?P<system>-?\d+) cams=(?P<cams>[\d,]+) "
                r"(?:periodMs=[0-9]+(?:\.[0-9]+)? "
                r"periodMismatchMs=[0-9]+(?:\.[0-9]+)? )?"
                r"spreadTicks=(?P<ticks>-?\d+) "
                r"spreadMs=(?P<spread>[0-9]+(?:\.[0-9]+)?) "
                r"limitMs=(?P<limit>[0-9]+(?:\.[0-9]+)?) "
                r"measurable=(?P<measurable>True|False) "
                r"aligned=(?P<aligned>True|False)"
                r"(?: sampleSource=(?P<source>\S+))?",
                message,
            )
            if sync_phase_match:
                measurable = sync_phase_match.group("measurable") == "True"
                aligned = sync_phase_match.group("aligned") == "True"
                spread_ms = float(sync_phase_match.group("spread"))
                limit_ms = float(sync_phase_match.group("limit"))
                if aligned and (not measurable or spread_ms > limit_ms):
                    failures.append(
                        f"invalid aligned phase evidence at {line.timestamp}: "
                        f"measurable={measurable} spread={spread_ms} limit={limit_ms}"
                    )
                if sync_phase_match.group("source") != "warm-snapshot":
                    failures.append(
                        f"sync phase did not use immutable warm snapshot "
                        f"at {line.timestamp}"
                    )
                key = (
                    sync_phase_match.group(1),
                    int(sync_phase_match.group(2)),
                )
                sync_phase_results.setdefault(key, []).append(
                    measurable and aligned and spread_ms <= limit_ms
                )
                continue

            sync_complete_match = re.match(
                r"acquisition sync complete reason=(\S+) attempts=(\d+) "
                r"cams=(\d+) phase=(True|False)",
                message,
            )
            if sync_complete_match:
                reason = sync_complete_match.group(1)
                attempt = int(sync_complete_match.group(2))
                key = (reason, attempt)
                phase_ok = sync_complete_match.group(4) == "True"
                expected = sync_expected.get(key)
                ready_count = len(sync_ready_cameras.get(key, set()))
                phase_results = sync_phase_results.get(key, [])
                evidence_ok = (
                    expected is not None
                    and ready_count == expected
                    and bool(phase_results)
                    and all(phase_results)
                )
                if not phase_ok:
                    failures.append(
                        f"synchronization completed without phase proof "
                        f"at {line.timestamp}"
                    )
                if not evidence_ok:
                    failures.append(
                        f"synchronization evidence incomplete at {line.timestamp}: "
                        f"ready={ready_count}/{expected} phases={phase_results}"
                    )
                phase_ok = phase_ok and evidence_ok
                if reason == "start":
                    start_sync_ready = phase_ok
                elif reason.startswith("parameter:"):
                    parameter_sync_ready = phase_ok
                continue

            if message.startswith(
                ("acquisition sync failed ", "capture synchronize failed ")
            ):
                if "reason=idle" not in message:
                    failures.append(
                        f"acquisition synchronization failed at {line.timestamp}: {message}"
                    )
                start_sync_ready = False
                parameter_sync_ready = False
                continue

            if message.startswith("acquisition phase verified "):
                standby_phase_verified = True
                continue

            if message.startswith("acquisition phase invalidated "):
                standby_phase_verified = False
                standby_start_ready = False
                continue

            start_path_match = re.match(
                r"acquisition start path=(verified-standby|full-sync) cams=(\d+)",
                message,
            )
            if start_path_match:
                path = start_path_match.group(1)
                standby_start_ready = (
                    standby_phase_verified
                    if path == "verified-standby"
                    else start_sync_ready
                )
                if not standby_start_ready:
                    failures.append(
                        f"{path} selected without phase proof at {line.timestamp}"
                    )
                continue

            if message == "ui:【取得背景】鈕":
                background_capture = True
                continue

            if message.startswith("StartGrab"):
                starts += 1
                if not (start_sync_ready or standby_start_ready):
                    failures.append(
                        f"StartGrab without verified standby or physical synchronization "
                        f"at {line.timestamp}"
                    )
                start_sync_ready = False
                standby_start_ready = False
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
                if tail_active and not tail_completed:
                    failures.append(
                        f"StopGrab before tail completion at {line.timestamp}"
                    )
                tail_active = False
                tail_completed = False
                capture_open = False
                stop_pending = True
                background_capture = False
                continue

            if message == "capture gate closed standby=on":
                if not stop_pending:
                    failures.append(f"gate closed without StopGrab at {line.timestamp}")
                stop_pending = False
                continue

            if message.startswith("parameter reconfigure begin "):
                if not capture_open:
                    failures.append(
                        f"parameter reconfigure began while capture gate closed at {line.timestamp}"
                    )
                capture_open = False
                parameter_sync_ready = False
                continue

            if message.startswith("parameter reconfigure complete "):
                if "gate=open warm=True" not in message:
                    failures.append(
                        f"parameter reconfigure reopened before warm at {line.timestamp}"
                    )
                if not parameter_sync_ready:
                    failures.append(
                        f"parameter reconfigure reopened without phase synchronization "
                        f"at {line.timestamp}"
                    )
                parameter_sync_ready = False
                capture_open = True
                continue

            if message.startswith(
                ("parameter reconfigure failed ", "parameter reconfigure canceled ")
            ):
                capture_open = False
                continue

            first_set_match = re.match(
                r"capture first-set ready path=\S+ cams=[\d,]+ aligned=(True|False)",
                message,
            )
            if first_set_match:
                if first_set_match.group(1) != "True":
                    failures.append(
                        f"first accepted frame set is out of phase at {line.timestamp}"
                    )
                continue

            if message.startswith("capture tail begin "):
                if not capture_open:
                    failures.append(
                        f"tail drain began while capture gate closed at {line.timestamp}"
                    )
                tail_active = True
                tail_completed = False
                continue

            if message.startswith("capture tail complete pending="):
                tail_completed = True
                continue

            if message.startswith("capture tail timeout "):
                failures.append(f"tail drain timeout at {line.timestamp}: {message}")
                tail_completed = True
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
            f"startSyncs={start_syncs} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_drag_first_publish(self, session: FlowSession, report: CheckReport) -> None:
        if not session.dvt_enabled:
            report.add(
                self.domain, "F6.first-view", CheckStatus.NOT_COVERED,
                "記錄範圍為日常運行；請切到流程驗證後重跑",
            )
            return
        session = session.dvt_only()
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
