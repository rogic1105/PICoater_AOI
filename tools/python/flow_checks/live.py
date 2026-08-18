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
        self._check_time_stop_origin(session, report)
        self._check_waterfall_bootstrap(session, report)
        self._check_waterfall_first_band(session, report)
        self._check_capture_chart_reset(session, report)
        self._check_capture_view_refire(session, report)
        self._check_row_presentation(session, report)
        self._check_column_range_stability(session, report)
        self._check_wheel_zoom_floor(session, report)
        self._check_background_subtraction(session, report)
        self._check_background_capture_output(session, report)
        self._check_background_preview_row_chart(session, report)
        if not any(
            line.message.startswith(("LC ", "IC ", "WF ", "ui:【開始抓取】"))
            for line in session.lines
        ):
            report.add(self.domain, "F0", CheckStatus.NOT_COVERED, "本 session 無監控操作")
            return report

        self._check_drag_first_publish(session, report)
        return report

    def _check_column_range_stability(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        main_pattern = re.compile(
            r"^LC mainRange viewX=(?P<left>-?\d+(?:\.\d+)?)~"
            r"(?P<right>-?\d+(?:\.\d+)?)"
        )
        column_pattern = re.compile(
            r"^LC colRange source=(?P<source>view|data) "
            r"target=(?P<left>-?\d+(?:\.\d+)?)~(?P<right>-?\d+(?:\.\d+)?) "
            r"axis=(?P<axis_left>-?\d+(?:\.\d+)?)~(?P<axis_right>-?\d+(?:\.\d+)?)/"
            r"view=(?P<view_left>-?\d+(?:\.\d+)?)~(?P<view_right>-?\d+(?:\.\d+)?) "
            r"plot=(?P<plot_left>-?\d+(?:\.\d+)?)~(?P<plot_right>-?\d+(?:\.\d+)?)$"
        )
        current_main = None
        groups = {}
        latest_view_by_target = {}
        target_mismatches = []
        view_data_mismatches = []
        sample_count = 0
        paired_data_count = 0

        for line in session.lines:
            main_match = main_pattern.match(line.message)
            if main_match:
                current_main = (
                    float(main_match.group("left")),
                    float(main_match.group("right")),
                )
                continue
            match = column_pattern.match(line.message)
            if not match:
                continue
            target = (float(match.group("left")), float(match.group("right")))
            values = (
                float(match.group("view_left")),
                float(match.group("view_right")),
                float(match.group("plot_left")),
                float(match.group("plot_right")),
            )
            if current_main is not None and (
                abs(target[0] - current_main[0]) > 0.05
                or abs(target[1] - current_main[1]) > 0.05
            ):
                target_mismatches.append(
                    f"{line.timestamp} target={target[0]:.2f}~{target[1]:.2f} "
                    f"main={current_main[0]:.2f}~{current_main[1]:.2f}"
                )
            if match.group("source") == "view":
                latest_view_by_target[target] = values
                continue

            sample_count += 1
            groups.setdefault(target, []).append(values)
            expected = latest_view_by_target.get(target)
            if expected is not None:
                paired_data_count += 1
                differences = [abs(values[index] - expected[index]) for index in range(4)]
                if (
                    max(differences[0:2]) > 0.50
                    or max(differences[2:4]) > 0.05
                ):
                    view_data_mismatches.append(
                        f"{line.timestamp} target={target[0]:.2f}~{target[1]:.2f} "
                        f"dataView={values[0]:.2f}~{values[1]:.2f} "
                        f"expected={expected[0]:.2f}~{expected[1]:.2f}"
                    )

        if sample_count == 0:
            report.add(
                self.domain,
                "F2.column-range-stability",
                CheckStatus.NOT_COVERED,
                "no LC colRange evidence",
            )
            return

        failures = list(target_mismatches) + list(view_data_mismatches)
        tested_groups = 0
        worst_view_drift = 0.0
        worst_plot_drift = 0.0
        for target, values in groups.items():
            if len(values) < 3:
                continue
            tested_groups += 1
            # The first paint may freeze MSChart's InnerPlotPosition once. The
            # contract concerns repeated redraws after that one-time bootstrap.
            steady = values[1:]
            view_drift = max(
                max(item[index] for item in steady)
                - min(item[index] for item in steady)
                for index in (0, 1)
            )
            plot_drift = max(
                max(item[index] for item in steady)
                - min(item[index] for item in steady)
                for index in (2, 3)
            )
            worst_view_drift = max(worst_view_drift, view_drift)
            worst_plot_drift = max(worst_plot_drift, plot_drift)
            if view_drift > 0.50 or plot_drift > 0.05:
                failures.append(
                    f"target={target[0]:.2f}~{target[1]:.2f} "
                    f"viewDrift={view_drift:.2f}mm plotDrift={plot_drift:.2f}%"
                )

        if tested_groups == 0 and not failures:
            report.add(
                self.domain,
                "F2.column-range-stability",
                CheckStatus.NOT_COVERED,
                f"samples={sample_count}; no target has three redraw samples",
            )
            return

        status = CheckStatus.FAIL if failures else CheckStatus.PASS
        detail = (
            f"samples={sample_count} stableTargets={tested_groups} "
            f"pairedData={paired_data_count} "
            f"viewDriftMax={worst_view_drift:.2f}mm "
            f"plotDriftMax={worst_plot_drift:.2f}% "
            f"failures={len(failures)}"
        )
        if failures:
            detail += "; first=" + failures[0]
        report.add(self.domain, "F2.column-range-stability", status, detail)

    def _check_background_subtraction(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        bind_pattern = re.compile(
            r"^background bind cam(?P<cam>\d+) "
            r"mode=(?P<mode>standard|single) "
            r"source=(?P<source>\S+) status=(?P<status>ready|failed|skipped)"
        )
        apply_pattern = re.compile(
            r"^background apply cam(?P<cam>\d+) grab=(?P<grab>\S+) "
            r"mode=(?P<mode>standard|single) "
            r"source=(?P<source>precomputed|per-frame) width=(?P<width>\d+)$"
        )
        plan_pattern = re.compile(r"^capture plan grab=(?P<grab>\S+)\b")
        gate_pattern = re.compile(r"^capture gate open cams=(?P<cams>\d+)\b")

        bindings = {}
        pending_grab = None
        active_capture = None
        ready_binds = 0
        failed_binds = 0
        skipped_binds = 0
        applications = 0
        completed = 0
        blocked = 0
        failures = []

        for line in session.lines:
            message = line.message
            bind_match = bind_pattern.match(message)
            if bind_match:
                camera_id = int(bind_match.group("cam"))
                status = bind_match.group("status")
                if status == "ready":
                    bindings[camera_id] = bind_match.group("mode")
                    ready_binds += 1
                elif status == "failed":
                    failed_binds += 1
                    if "retained=False" in message:
                        bindings.pop(camera_id, None)
                else:
                    skipped_binds += 1
                    bindings.pop(camera_id, None)
                continue

            plan_match = plan_pattern.match(message)
            if plan_match:
                pending_grab = plan_match.group("grab")
                continue

            gate_match = gate_pattern.match(message)
            if gate_match and pending_grab is not None:
                active_capture = {
                    "grab": pending_grab,
                    "expected": int(gate_match.group("cams")),
                    "cameras": set(),
                }
                pending_grab = None
                continue

            apply_match = apply_pattern.match(message)
            if apply_match:
                applications += 1
                camera_id = int(apply_match.group("cam"))
                grab = apply_match.group("grab")
                mode = apply_match.group("mode")
                source = apply_match.group("source")
                width = int(apply_match.group("width"))
                expected_source = (
                    "precomputed" if mode == "standard" else "per-frame"
                )
                if source != expected_source or width <= 0:
                    failures.append(
                        f"{line.timestamp} cam{camera_id} mode={mode} "
                        f"source={source} width={width}"
                    )
                if bindings.get(camera_id) != mode:
                    failures.append(
                        f"{line.timestamp} cam{camera_id} applied {mode} "
                        f"without matching ready binding"
                    )
                if (
                    active_capture is not None
                    and grab == active_capture["grab"]
                ):
                    active_capture["cameras"].add(camera_id)
                continue

            if message.startswith(
                "capture start blocked reason=standard-background-not-ready"
            ):
                blocked += 1
                pending_grab = None
                active_capture = None
                continue

            if message == "StopGrab" and active_capture is not None:
                actual = len(active_capture["cameras"])
                expected = active_capture["expected"]
                if actual != expected:
                    failures.append(
                        f"{line.timestamp} grab={active_capture['grab']} "
                        f"background evidence {actual}/{expected}"
                    )
                completed += 1
                active_capture = None

        if ready_binds == 0 and failed_binds == 0 and applications == 0:
            report.add(
                self.domain,
                "F8.background-subtraction",
                CheckStatus.NOT_COVERED,
                "session predates standard-background runtime evidence",
            )
            return

        report.add(
            self.domain,
            "F8.background-subtraction",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"readyBinds={ready_binds} failedBinds={failed_binds} "
            f"skippedBinds={skipped_binds} "
            f"applies={applications} completedGrabs={completed} blocked={blocked} "
            f"failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_background_capture_output(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        active = False
        guard_product_output = False
        sample_seconds = None
        sample_completed = False
        begins = 0
        ends = 0
        failures = []

        for line in session.lines:
            message = line.message
            if message == "background capture begin output=disabled":
                begins += 1
                if active:
                    failures.append(f"{line.timestamp} nested background capture begin")
                active = True
                guard_product_output = True
                sample_seconds = None
                sample_completed = False
                continue

            if message.startswith("background capture end output=disabled result="):
                ends += 1
                if not active:
                    failures.append(f"{line.timestamp} background capture end without begin")
                if message.endswith("result=ok") and (
                    sample_seconds is None or not sample_completed
                ):
                    failures.append(
                        f"{line.timestamp} successful background capture lacks "
                        "complete timed sample evidence"
                    )
                active = False
                continue

            sample_start = re.match(
                r"^background capture sampling start duration=(\d+)s$",
                message,
            )
            if sample_start and active:
                sample_seconds = int(sample_start.group(1))
                continue

            sample_end = re.match(
                r"^background capture sampling complete durationMs=(\d+)\b",
                message,
            )
            if sample_end and active:
                duration_ms = int(sample_end.group(1))
                if sample_seconds is None:
                    failures.append(
                        f"{line.timestamp} sample complete without duration start"
                    )
                elif duration_ms < sample_seconds * 1000:
                    failures.append(
                        f"{line.timestamp} sample {duration_ms}ms "
                        f"< configured {sample_seconds * 1000}ms"
                    )
                sample_completed = True
                continue

            if message.startswith("capture plan grab="):
                guard_product_output = False

            if guard_product_output and "code=CaptureWriteFailure." in message:
                failures.append(
                    f"{line.timestamp} product capture write attempted during background sample"
                )

        if begins == 0 and ends == 0:
            report.add(
                self.domain,
                "F8.background-capture",
                CheckStatus.NOT_COVERED,
                "session predates non-product background capture evidence",
            )
            return

        if active:
            failures.append("background capture begin has no matching end")
        if begins != ends:
            failures.append(f"background capture lifecycle begin={begins} end={ends}")

        report.add(
            self.domain,
            "F8.background-capture",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"begins={begins} ends={ends} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_background_preview_row_chart(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        active = False
        cleared = False
        entries = 0
        failures = []

        for line in session.lines:
            message = line.message
            if message.startswith("EnterBackgroundPreview"):
                if active:
                    failures.append(
                        f"{line.timestamp} nested background preview entry"
                    )
                active = True
                cleared = False
                entries += 1
                continue

            if message == "background preview rowChart clear":
                if not active:
                    failures.append(
                        f"{line.timestamp} rowChart clear outside background preview"
                    )
                cleared = True
                continue

            if active and message.startswith("bgPreview push "):
                if not cleared:
                    failures.append(
                        f"{line.timestamp} background frame pushed before rowChart clear"
                    )
                continue

            if active and message.startswith("rowCurve present after=mainImage"):
                failures.append(
                    f"{line.timestamp} rowCurve presented during background preview"
                )
                continue

            if message.startswith("ExitBackgroundPreview"):
                if active and not cleared:
                    failures.append(
                        f"{line.timestamp} background preview exited without rowChart clear"
                    )
                active = False
                cleared = False

        if entries == 0:
            report.add(
                self.domain,
                "F8.background-preview-row",
                CheckStatus.NOT_COVERED,
                "session has no background preview entry",
            )
            return

        if active and not cleared:
            failures.append("active background preview has no rowChart clear")

        report.add(
            self.domain,
            "F8.background-preview-row",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"entries={entries} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_time_stop_origin(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        wait_pattern = re.compile(
            r"^grab stop waiting condition=Time configured=(?P<seconds>\d+)s "
            r"source=(?P<source>io|manual) grab=(?P<grab>\S+)$"
        )
        arm_pattern = re.compile(
            r"^grab stop armed condition=Time limit=(?P<seconds>\d+)s "
            r"configured=(?P<configured>\d+)s grace=0s "
            r"source=(?P<source>io|manual) start=first-set grab=(?P<grab>\S+)$"
        )

        active = None
        waits = 0
        arms = 0
        cancelled = 0
        failures = []

        for index, line in enumerate(session.lines):
            message = line.message
            wait_match = wait_pattern.match(message)
            if wait_match:
                waits += 1
                if active is not None:
                    failures.append(
                        f"{line.timestamp} overlapping Time wait "
                        f"{active['grab']}->{wait_match.group('grab')}"
                    )
                active = {
                    "grab": wait_match.group("grab"),
                    "seconds": int(wait_match.group("seconds")),
                    "source": wait_match.group("source"),
                    "first_set": None,
                }
                continue

            if message.startswith("capture first-set ready ") and active is not None:
                if "aligned=True" in message:
                    active["first_set"] = index
                continue

            arm_match = arm_pattern.match(message)
            if arm_match:
                arms += 1
                if active is None:
                    failures.append(
                        f"{line.timestamp} Time armed without waiting state"
                    )
                    continue
                if arm_match.group("grab") != active["grab"]:
                    failures.append(
                        f"{line.timestamp} Time grab mismatch "
                        f"{active['grab']}->{arm_match.group('grab')}"
                    )
                if active["first_set"] is None:
                    failures.append(
                        f"{line.timestamp} Time armed before aligned first-set"
                    )
                if (
                    int(arm_match.group("seconds")) != active["seconds"]
                    or int(arm_match.group("configured")) != active["seconds"]
                    or arm_match.group("source") != active["source"]
                ):
                    failures.append(
                        f"{line.timestamp} Time arm parameters changed while waiting"
                    )
                active = None
                continue

            if message == "StopGrab" and active is not None:
                cancelled += 1
                active = None

        if waits == 0 and arms == 0:
            report.add(
                self.domain,
                "F2.time-origin",
                CheckStatus.NOT_COVERED,
                "session predates first-set Time origin evidence",
            )
            return

        report.add(
            self.domain,
            "F2.time-origin",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"waits={waits} arms={arms} cancelled={cancelled} "
            f"failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_capture_head_guard(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        has_phase_guard = any(
            line.message.startswith("capture head guard ")
            for line in session.lines
        )
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
        head_approved = None
        opens = 0
        completed = 0
        rejected = 0
        rejection_pending_stop = False
        failures = []

        for line in session.lines:
            gate_match = re.match(r"capture gate open cams=(\d+)\b", line.message)
            if gate_match:
                if rejection_pending_stop:
                    failures.append(
                        f"new gate opened before rejected capture stopped "
                        f"at {line.timestamp}"
                    )
                expected = int(gate_match.group(1))
                dropped = set()
                head_approved = None
                rejection_pending_stop = False
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

            guard_match = re.match(
                r"capture head guard path=\S+ cams=([\d,]+) "
                r"aligned=(True|False)$",
                line.message,
            )
            if guard_match:
                guard_cameras = {
                    int(value) for value in guard_match.group(1).split(",")
                }
                aligned = guard_match.group(2) == "True"
                if expected is None:
                    failures.append(
                        f"head phase guard without open gate at {line.timestamp}"
                    )
                elif len(dropped) != expected:
                    failures.append(
                        f"head phase guard before all probes were dropped "
                        f"at {line.timestamp}: dropped={len(dropped)}/{expected}"
                    )
                elif dropped != guard_cameras:
                    failures.append(
                        f"head-drop cameras {sorted(dropped)} differ from phase probe "
                        f"{sorted(guard_cameras)} at {line.timestamp}"
                    )
                head_approved = aligned
                if not aligned:
                    rejected += 1
                    rejection_pending_stop = True
                continue

            if rejection_pending_stop and (
                "firstFrame " in line.message
                or line.message.startswith("capture csv ")
            ):
                failures.append(
                    f"product output after rejected head phase "
                    f"at {line.timestamp}: {line.message}"
                )
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
                elif has_phase_guard and head_approved is not True:
                    failures.append(
                        f"first accepted set before aligned head phase guard "
                        f"at {line.timestamp}"
                    )
                else:
                    completed += 1
                expected = None
                dropped = set()
                head_approved = None
                continue

            if line.message == "StopGrab":
                rejection_pending_stop = False
                expected = None
                dropped = set()
                head_approved = None

        if rejection_pending_stop:
            failures.append("rejected head phase was not followed by StopGrab")

        if completed == 0 and rejected == 0 and not failures:
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
            f"gateOpens={opens} completed={completed} rejected={rejected} "
            f"failures={len(failures)}"
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

    def _check_waterfall_first_band(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        pattern = re.compile(
            r"^WF band first generation=(\d+) seq=(-?\d+) "
            r"cams=([\d,]+) expected=([\d,]+) "
            r"ticks=(-?\d+)~(-?\d+) startRow=(\d+) height=(\d+) reason=(\w+)$"
        )
        samples = []
        failures = []
        for line in session.lines:
            if not line.message.startswith("WF band first "):
                continue
            match = pattern.match(line.message)
            if match is None:
                continue  # 舊版 log 沒有相機集合，不能據此判定。
            actual = {int(value) for value in match.group(3).split(",")}
            expected = {int(value) for value in match.group(4).split(",")}
            reason = match.group(9)
            samples.append((actual, expected, reason))
            if not expected or actual != expected or reason != "complete":
                failures.append(
                    f"{line.timestamp} first band cameras={sorted(actual)} "
                    f"expected={sorted(expected)} reason={reason}"
                )

        if not samples:
            report.add(
                self.domain,
                "F2.waterfall-first-band",
                CheckStatus.NOT_COVERED,
                "本 session 無新版瀑布第一列相機集合儀器",
            )
            return

        report.add(
            self.domain,
            "F2.waterfall-first-band",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(samples)} failures={len(failures)}"
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
        latest_direction_change = None
        presentations = 0
        rows = 0
        mapping_only_rows = 0
        failures = []

        for line in session.lines:
            if line.message.startswith("ui:設定[hee_VerticalDirection]="):
                latest_direction_change = line
                continue
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
            if (
                latest_direction_change is not None
                and line.elapsed - latest_direction_change.elapsed <= 2.0
                and (
                    latest_presentation is None
                    or latest_direction_change.elapsed
                    > latest_presentation.elapsed
                )
            ):
                mapping_only_rows += 1
                continue
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
            f"presentations={presentations} rowUpdates={rows} "
            f"mappingOnly={mapping_only_rows} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_capture_chart_reset(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        pending_reset = False
        resets = 0
        starts = 0
        failures = []

        for line in session.lines:
            message = line.message
            if message == "capture charts reset reason=start-grab":
                if pending_reset:
                    failures.append(
                        f"{line.timestamp} repeated chart reset before StartGrab"
                    )
                pending_reset = True
                resets += 1
                continue

            if message.startswith("StartGrab"):
                starts += 1
                if not pending_reset:
                    failures.append(
                        f"{line.timestamp} StartGrab without shared chart reset"
                    )
                pending_reset = False

        if starts == 0 and resets == 0:
            report.add(
                self.domain,
                "F2.chart-reset",
                CheckStatus.NOT_COVERED,
                "session predates shared capture chart-reset evidence",
            )
            return

        if pending_reset:
            failures.append("last chart reset has no matching StartGrab")

        report.add(
            self.domain,
            "F2.chart-reset",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"starts={starts} resets={resets} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
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
