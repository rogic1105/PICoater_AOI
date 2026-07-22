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
        self._check_curve_liveness(session, report)
        self._check_row_presentation(session, report)
        if not any(
            line.message.startswith(("LC ", "IC ", "WF ", "ui:【開始抓取】"))
            for line in session.lines
        ):
            report.add(self.domain, "F0", CheckStatus.NOT_COVERED, "本 session 無監控操作")
            return report

        self._check_drag_first_publish(session, report)
        return report

    def _check_curve_liveness(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        if not any(
            line.message.startswith("columnCurve first-present ")
            for line in session.lines
        ):
            report.add(
                self.domain,
                "F2.curve-liveness",
                CheckStatus.NOT_COVERED,
                "session predates per-grab curve presentation instrumentation",
            )
            return

        active = None
        completed = []
        for line in session.lines:
            message = line.message
            if message.startswith("StartGrab"):
                if active is not None:
                    completed.append(active)
                active = {
                    "start": line,
                    "first_frame": None,
                    "refire": None,
                    "column": None,
                    "row": None,
                    "stop": None,
                }
                continue
            if active is None:
                continue
            if message == "viewRange refire reason=capture-start mode=WF" or \
                    message == "viewRange refire reason=capture-start mode=IC":
                active["refire"] = line
            elif "firstFrame " in message and active["first_frame"] is None:
                active["first_frame"] = line
            elif message.startswith("columnCurve first-present "):
                active["column"] = active["column"] or line
            elif message.startswith("rowCurve present after=mainImage "):
                active["row"] = active["row"] or line
            elif message == "StopGrab":
                active["stop"] = line
                completed.append(active)
                active = None

        if active is not None:
            completed.append(active)

        checked = 0
        failures = []
        worst_delay = 0.0
        for index, run in enumerate(completed, start=1):
            end = run["stop"] or (session.lines[-1] if session.lines else run["start"])
            duration = end.elapsed - run["start"].elapsed
            if duration < 3.0:
                continue
            checked += 1
            if run["refire"] is None:
                failures.append(f"grab#{index} missing capture-start view-range refire")
            first_frame = run["first_frame"]
            if first_frame is None:
                failures.append(f"grab#{index} missing firstFrame")
                continue
            for curve_name in ("column", "row"):
                presented = run[curve_name]
                if presented is None:
                    failures.append(f"grab#{index} missing {curve_name} curve")
                    continue
                delay = presented.elapsed - first_frame.elapsed
                worst_delay = max(worst_delay, delay)
                if delay < 0 or delay > 3.0:
                    failures.append(
                        f"grab#{index} {curve_name} curve delay={delay:.3f}s"
                    )

        if checked == 0:
            report.add(
                self.domain,
                "F2.curve-liveness",
                CheckStatus.NOT_COVERED,
                "no grab lasting at least 3 seconds",
            )
            return
        report.add(
            self.domain,
            "F2.curve-liveness",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"grabs={checked} worstFirstCurveDelay={worst_delay:.3f}s "
            f"failures={len(failures)}"
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
            report.add(
                self.domain,
                "F2.row-presentation",
                CheckStatus.NOT_COVERED,
                "本 session 無監控列曲線更新",
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
        phase_verified = False
        start_path_ready = False
        pending_start_path = None
        active_start_path = None
        gate_armed = False
        armed_expected = 0
        admitted_cameras = set()
        admitted_frames = {}
        frame_set_ready = False
        standby_phase_ready = False
        has_standby_phase_probe = any(
            line.message.startswith("acquisition standby phase ")
            for line in session.lines
        )
        clock_frequency_by_system = {}
        parameter_sync_ready = False
        parameter_reconfigure_pending = False
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
                system_num = int(sync_ready_match.group(4))
                frequency = int(sync_ready_match.group(6))
                if system_num >= 0 and frequency > 0:
                    clock_frequency_by_system[system_num] = frequency
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
                r"system=(-?\d+) cams=([\d,]+) spreadTicks=(-?\d+) "
                r"spreadMs=([0-9]+(?:\.[0-9]+)?) "
                r"limitMs=([0-9]+(?:\.[0-9]+)?) "
                r"measurable=(True|False) aligned=(True|False)",
                message,
            )
            if sync_phase_match:
                measurable = sync_phase_match.group(8) == "True"
                aligned = sync_phase_match.group(9) == "True"
                spread_ms = float(sync_phase_match.group(6))
                limit_ms = float(sync_phase_match.group(7))
                if aligned and (not measurable or spread_ms > limit_ms):
                    failures.append(
                        f"invalid aligned phase evidence at {line.timestamp}: "
                        f"measurable={measurable} spread={spread_ms} limit={limit_ms}"
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
                    phase_verified = phase_ok
                elif reason == "idle":
                    phase_verified = phase_ok
                elif reason.startswith("parameter:"):
                    parameter_sync_ready = phase_ok
                    phase_verified = phase_ok
                continue

            standby_phase_match = re.match(
                r"acquisition standby phase system=(-?\d+) cams=([\d,]+) "
                r"periodMs=([0-9]+(?:\.[0-9]+)?) "
                r"periodMismatchMs=([0-9]+(?:\.[0-9]+)?) "
                r"spreadTicks=(-?\d+) spreadMs=([0-9]+(?:\.[0-9]+)?) "
                r"limitMs=([0-9]+(?:\.[0-9]+)?) "
                r"measurable=(True|False) aligned=(True|False)",
                message,
            )
            if standby_phase_match:
                measurable = standby_phase_match.group(8) == "True"
                aligned = standby_phase_match.group(9) == "True"
                spread_ms = float(standby_phase_match.group(6))
                limit_ms = float(standby_phase_match.group(7))
                standby_phase_ready = (
                    measurable and aligned and spread_ms <= limit_ms
                )
                if aligned and not standby_phase_ready:
                    failures.append(
                        f"invalid standby phase evidence at {line.timestamp}: "
                        f"measurable={measurable} spread={spread_ms} limit={limit_ms}"
                    )
                continue

            if message.startswith("acquisition phase invalidated "):
                phase_verified = False
                continue

            start_path_match = re.match(
                r"acquisition start path=(full-sync|verified-standby) cams=(\d+)",
                message,
            )
            if start_path_match:
                pending_start_path = start_path_match.group(1)
                if pending_start_path == "full-sync" and not start_sync_ready:
                    failures.append(
                        f"full-sync start path without phase proof at {line.timestamp}"
                    )
                if pending_start_path == "verified-standby" and not phase_verified:
                    failures.append(
                        f"verified standby used after invalidation at {line.timestamp}"
                    )
                if (
                    pending_start_path == "verified-standby"
                    and has_standby_phase_probe
                    and not standby_phase_ready
                ):
                    failures.append(
                        f"verified standby used without current phase proof "
                        f"at {line.timestamp}"
                    )
                standby_phase_ready = False
                start_path_ready = True
                continue

            if message.startswith(
                ("acquisition sync failed ", "capture synchronize failed ")
            ):
                failures.append(
                    f"acquisition synchronization failed at {line.timestamp}: {message}"
                )
                start_sync_ready = False
                parameter_sync_ready = False
                phase_verified = False
                continue

            if message == "ui:【取得背景】鈕":
                background_capture = True
                continue

            if message.startswith("StartGrab"):
                starts += 1
                if not start_path_ready and not start_sync_ready:
                    failures.append(
                        f"StartGrab without verified acquisition path "
                        f"at {line.timestamp}"
                    )
                active_start_path = pending_start_path or "legacy-full-sync"
                start_sync_ready = False
                start_path_ready = False
                pending_start_path = None
                gate_armed = False
                armed_expected = 0
                admitted_cameras = set()
                admitted_frames = {}
                frame_set_ready = False
                start_pending = True
                stop_pending = False
                plan_ready = background_capture
                continue

            if message.startswith("capture plan "):
                plan_ready = True
                continue

            gate_arm_match = re.match(
                r"capture gate arm path=(full-sync|verified-standby|parameter-sync) "
                r"cams=(\d+) marginMs=(\d+) targets=(.+)",
                message,
            )
            if gate_arm_match:
                arm_path = gate_arm_match.group(1)
                if arm_path == "parameter-sync":
                    if not parameter_reconfigure_pending or not parameter_sync_ready:
                        failures.append(
                            f"parameter gate armed without synchronization at {line.timestamp}"
                        )
                elif not start_pending or arm_path != active_start_path:
                    failures.append(
                        f"gate arm does not match active start at {line.timestamp}: "
                        f"arm={arm_path} active={active_start_path}"
                    )
                gate_armed = True
                armed_expected = int(gate_arm_match.group(2))
                admitted_cameras = set()
                admitted_frames = {}
                frame_set_ready = False
                continue

            gate_match = re.match(
                r"capture gate open cams=(\d+) warm=(True|False)"
                r"(?: path=(full-sync|verified-standby|parameter-sync))?",
                message,
            )
            if gate_match:
                expected = int(gate_match.group(1))
                warm = gate_match.group(2) == "True"
                gate_path = gate_match.group(3)
                is_parameter_gate = gate_path == "parameter-sync"
                if is_parameter_gate:
                    if not parameter_reconfigure_pending:
                        failures.append(
                            f"parameter gate open without reconfigure at {line.timestamp}"
                        )
                else:
                    if not start_pending:
                        failures.append(f"gate open without StartGrab at {line.timestamp}")
                    if not plan_ready:
                        failures.append(
                            f"gate opened before capture plan at {line.timestamp}"
                        )
                if gate_path is not None and (
                    not gate_armed or armed_expected != expected
                ):
                    failures.append(
                        f"gate opened without matching future-frame arm at {line.timestamp}"
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
                if not is_parameter_gate:
                    start_pending = False
                    plan_ready = False
                capture_open = True
                continue

            admitted_match = re.match(
                r"capture frame admitted path=(\S+) cam(\d+) system=(-?\d+) "
                r"tick=(-?\d+) target=(-?\d+)",
                message,
            )
            if admitted_match:
                camera_id = int(admitted_match.group(2))
                tick = int(admitted_match.group(4))
                target = int(admitted_match.group(5))
                if not capture_open or not gate_armed or tick < target:
                    failures.append(
                        f"invalid admitted frame for cam{camera_id} at {line.timestamp}"
                    )
                admitted_cameras.add(camera_id)
                admitted_frames[camera_id] = (
                    int(admitted_match.group(3)), tick
                )
                continue

            frame_set_phase_match = re.match(
                r"capture frame-set phase path=(\S+) system=(-?\d+) "
                r"cams=([\d,]+) spreadTicks=(-?\d+) "
                r"spreadMs=([0-9]+(?:\.[0-9]+)?) "
                r"limitMs=([0-9]+(?:\.[0-9]+)?) "
                r"measurable=(True|False) aligned=(True|False)",
                message,
            )
            if frame_set_phase_match:
                measurable = frame_set_phase_match.group(7) == "True"
                aligned = frame_set_phase_match.group(8) == "True"
                spread_ms = float(frame_set_phase_match.group(5))
                limit_ms = float(frame_set_phase_match.group(6))
                if not measurable or not aligned or spread_ms > limit_ms:
                    failures.append(
                        f"admitted frame-set phase out of range at {line.timestamp}: "
                        f"spread={spread_ms} limit={limit_ms}"
                    )
                continue

            frame_set_match = re.match(
                r"capture frame-set ready path=(\S+) cams=(\d+)", message
            )
            if frame_set_match:
                count = int(frame_set_match.group(2))
                if not gate_armed or count != armed_expected or len(admitted_cameras) != count:
                    failures.append(
                        f"frame-set ready without all admitted cameras at {line.timestamp}: "
                        f"admitted={len(admitted_cameras)} expected={armed_expected}"
                    )
                for system_num in sorted(
                    {system for system, _ in admitted_frames.values()}
                ):
                    ticks = [
                        tick
                        for system, tick in admitted_frames.values()
                        if system == system_num
                    ]
                    if len(ticks) <= 1:
                        continue
                    frequency = clock_frequency_by_system.get(system_num, 0)
                    if frequency <= 0:
                        failures.append(
                            f"frame-set phase clock unavailable for system{system_num} "
                            f"at {line.timestamp}"
                        )
                        continue
                    spread_ms = (max(ticks) - min(ticks)) * 1000.0 / frequency
                    if spread_ms > 5.0:
                        failures.append(
                            f"admitted frame-set spread={spread_ms:.3f}ms exceeds "
                            f"5.000ms for system{system_num} at {line.timestamp}"
                        )
                frame_set_ready = True
                continue

            phase_drift_match = re.match(
                r"capture phase drift deferred reason=(\S+) "
                r"gate=open next=resync",
                message,
            )
            if phase_drift_match:
                if not capture_open:
                    failures.append(
                        f"runtime phase drift reported while gate was closed "
                        f"at {line.timestamp}"
                    )
                phase_verified = False
                continue

            if message.startswith("capture phase drift gate=closed reason="):
                failures.append(
                    f"runtime phase drift truncated the active capture at {line.timestamp}"
                )
                capture_open = False
                continue

            if message == "StopGrab":
                stops += 1
                if (
                    gate_armed
                    and admitted_cameras
                    and len(admitted_cameras) == armed_expected
                    and not frame_set_ready
                ):
                    failures.append(
                        f"all cameras admitted without frame-set ready before "
                        f"{line.timestamp}"
                    )
                capture_open = False
                gate_armed = False
                admitted_cameras = set()
                admitted_frames = {}
                frame_set_ready = False
                stop_pending = True
                background_capture = False
                continue

            if message == "capture gate closed standby=on":
                if not stop_pending:
                    failures.append(f"gate closed without StopGrab at {line.timestamp}")
                stop_pending = False
                continue

            if message.startswith("rowCurve present after=mainImage ") and not capture_open:
                failures.append(
                    f"row curve presented while capture gate closed at {line.timestamp}"
                )
                continue

            if message.startswith("parameter reconfigure begin "):
                if not capture_open:
                    failures.append(
                        f"parameter reconfigure began while capture gate closed at {line.timestamp}"
                    )
                capture_open = False
                parameter_sync_ready = False
                parameter_reconfigure_pending = True
                gate_armed = False
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
                parameter_reconfigure_pending = False
                capture_open = True
                continue

            if message.startswith(
                ("parameter reconfigure failed ", "parameter reconfigure canceled ")
            ):
                capture_open = False
                parameter_reconfigure_pending = False
                continue

            first_frame_match = re.search(r"firstFrame cam(\d+) ", message)
            if first_frame_match:
                camera_id = int(first_frame_match.group(1))
                if not capture_open:
                    failures.append(
                        f"firstFrame while capture gate closed at {line.timestamp}"
                    )
                elif gate_armed and camera_id not in admitted_cameras:
                    failures.append(
                        f"firstFrame before future-frame admission for cam{camera_id} "
                        f"at {line.timestamp}"
                    )

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
