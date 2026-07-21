"""Data/report-tab (D-series) flow validators."""

import re

from .core import (
    CheckReport,
    CheckStatus,
    FlowSession,
    assess_ui_responsiveness,
    grab_id,
)


class DataFlowValidator:
    domain = "DATA"

    @staticmethod
    def _covered(session: FlowSession) -> bool:
        prefixes = (
            "DT ",
            "ui:【報表序號】",
            "ui:【明細列表】",
            "ui:【序號範圍-",
            "ui:【期間-",
            "ui:【良率",
            "ui:【讀取資料】鈕（Data）",
        )
        return any(line.message.startswith(prefixes) for line in session.lines)

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        if not self._covered(session):
            report.add(self.domain, "D0", CheckStatus.NOT_COVERED, "本 session 無報表操作")
            return report

        self._check_statistics_snapshot(session, report)
        self._check_single_selection(session, report)
        self._check_single_curve_policy(session, report)
        self._check_single_curve(session, report)
        self._check_single_fit(session, report)
        self._check_curve_summary_writes(session, report)
        self._check_single_row_curve(session, report)
        self._check_list_ownership(session, report)
        self._check_range_policy(session, report)
        self._check_range_debounce(session, report)
        self._check_range_list_preview(session, report)
        self._check_range_preview(session, report)
        self._check_y_scale_toggle(session, report)
        self._check_ui_stall(session, report)
        return report

    def _check_statistics_snapshot(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        load_positions = [
            index for index, line in enumerate(session.lines)
            if line.message == "ui:【讀取資料】鈕（Data）"
        ]
        snapshot_pattern = re.compile(
            r"^DT stats snapshot csv=(\d+) records=(\d+) grabs=(\d+) ms=(\d+)$"
        )
        snapshots = []
        invalid = []
        for index, line in enumerate(session.lines):
            if not line.message.startswith("DT stats snapshot"):
                continue
            match = snapshot_pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            csv_count, record_count, grab_count, elapsed_ms = map(
                int, match.groups()
            )
            valid_counts = (
                record_count >= grab_count
                and (csv_count > 0 or (record_count == 0 and grab_count == 0))
            )
            if not valid_counts:
                invalid.append(
                    f"{line.timestamp} csv={csv_count} records={record_count} grabs={grab_count}"
                )
            snapshots.append((index, csv_count, record_count, grab_count, elapsed_ms))

        if not snapshots:
            report.add(
                self.domain,
                "D1.snapshot",
                CheckStatus.NOT_COVERED,
                "舊版 log 無一次式統計 snapshot 儀器",
            )
            return

        missing = 0
        for position, load_index in enumerate(load_positions):
            next_load = (
                load_positions[position + 1]
                if position + 1 < len(load_positions)
                else len(session.lines)
            )
            if not any(load_index < item[0] < next_load for item in snapshots):
                missing += 1

        worst_ms = max(item[4] for item in snapshots)
        ok = not invalid and missing == 0
        report.add(
            self.domain,
            "D1.snapshot",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"讀取={len(load_positions)} snapshot={len(snapshots)} "
            f"缺少={missing} 格式/計數錯誤={len(invalid)} 最慢={worst_ms}ms"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )

    def _check_single_selection(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.selected", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        missing = []
        cache_hits = 0
        scans = 0
        for position, (line_index, selected_id) in enumerate(intents):
            next_index = intents[position + 1][0] if position + 1 < len(intents) else len(session.lines)
            matching = [
                line.message
                for line in session.lines[line_index + 1 : next_index]
                if line.message.startswith(f"DT selected {selected_id} ")
            ]
            if not matching:
                missing.append(selected_id)
                continue
            if "stats=cache" in matching[-1]:
                cache_hits += 1
            elif "stats=scan" in matching[-1]:
                scans += 1

        report.add(
            self.domain,
            "D3.selected",
            CheckStatus.PASS if not missing and scans <= 1 else CheckStatus.FAIL,
            f"intent={len(intents)} cache={cache_hits} scan={scans} 缺終態={len(missing)}"
            + (f"；首筆 {missing[0]}" if missing else ""),
        )

    def _check_single_curve(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.curve", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        pattern = re.compile(
            r"^DT curve load (\d{6}-\d{6}) captures=(\d+) "
            r"source=shared storage=(summary|bins|memory-summary|memory-bins) "
            r"configMs=(\d+) waitMs=(\d+) pathMs=(\d+) mergeMs=(\d+) "
            r"summaryMs=(\d+) (?:points=(\d+) )?drawMs=(\d+) totalMs=(\d+)$"
        )
        invalid = []
        storage = {"summary": 0, "bins": 0, "memory-summary": 0, "memory-bins": 0}
        has_current_instrument = any(
            line.message.startswith("DT curve load ") and " source=" in line.message
            for line in session.lines
        )

        loads = [
            (index, line.message)
            for index, line in enumerate(session.lines)
            if line.message.startswith("DT curve load ")
            and not line.message.startswith("DT curve load policy ")
        ]
        for _, message in loads:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue
            storage[match.group(3)] += 1

        final_index, final_id = intents[-1]
        final_loaded = any(
            index > final_index and message.startswith(f"DT curve load {final_id} ")
            for index, message in loads
        )
        stale = sum(
            1 for line in session.lines
            if line.message.startswith("DT curve stale-drop ")
        )

        if not final_loaded or invalid:
            status = CheckStatus.FAIL
        elif not has_current_instrument:
            status = CheckStatus.NOT_COVERED
        else:
            status = CheckStatus.PASS
        report.add(
            self.domain,
            "D3.curve",
            status,
            f"intent={len(intents)} applied={len(loads)} stale={stale} "
            f"summary={storage['summary']} bins={storage['bins']} "
            f"memory={storage['memory-summary'] + storage['memory-bins']} "
            f"final={'ok' if final_loaded else 'missing'} 格式錯誤={len(invalid)}"
            + ("；舊版儀器無 source 分段" if not has_current_instrument else ""),
        )

    def _check_single_curve_policy(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT curve load policy ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.curve-policy",
                CheckStatus.NOT_COVERED,
                "舊版 log 無 curve load policy 儀器",
            )
            return

        expected = (
            "DT curve load policy latest-only shared-loader "
            "entries=512 maxMB=256 scale=merged-only"
        )
        unique = sorted(set(lines))
        ok = len(lines) == 1 and unique == [expected]
        report.add(
            self.domain,
            "D3.curve-policy",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"行數={len(lines)} 實際={' | '.join(unique)}",
        )

    def _check_single_fit(self, session: FlowSession, report: CheckReport) -> None:
        if not session.dvt_enabled:
            report.add(
                self.domain, "D3.fit", CheckStatus.NOT_COVERED,
                "記錄範圍為日常運行；請切到流程驗證後重跑",
            )
            return
        session = session.dvt_only()
        number = r"-?\d+(?:\.\d+)?"
        pattern = re.compile(
            r"^RV prefit (\d{6}-\d{6}) content=(\d+)x(\d+) "
            r"viewport=(\d+)x(\d+) viewX=(-?\d+)~(-?\d+) "
            r"viewY=(-?\d+)~(-?\d+)$"
        )
        report_fit_pattern = re.compile(
            rf"^DT prefit (\d{{6}}-\d{{6}}) content=(\d+)x(\d+) "
            rf"viewX=({number})~({number}) viewY=({number})~({number}) "
            r"source=main-geometry$"
        )
        fits = []
        invalid = []
        report_fits = []
        invalid_report_fits = []
        for index, line in enumerate(session.lines):
            if line.message.startswith("DT prefit "):
                report_match = report_fit_pattern.match(line.message)
                if not report_match:
                    invalid_report_fits.append(line.message)
                    continue
                width = int(report_match.group(2))
                height = int(report_match.group(3))
                view = tuple(float(report_match.group(group)) for group in range(4, 8))
                if width <= 0 or height <= 0 or view[0] == view[1] or view[2] == view[3]:
                    invalid_report_fits.append(line.message)
                    continue
                report_fits.append((index, report_match.group(1), width, height, view))
                continue
            if not line.message.startswith("RV prefit "):
                continue
            match = pattern.match(line.message)
            if not match:
                invalid.append(line.message)
                continue
            values = tuple(map(int, match.groups()[1:]))
            if min(values[:4]) <= 0 or values[4] == values[5] or values[6] == values[7]:
                invalid.append(line.message)
                continue
            fits.append((index, match.group(1), values[0], values[1]))

        if not fits and not invalid and not report_fits and not invalid_report_fits:
            report.add(
                self.domain, "D3.fit", CheckStatus.NOT_COVERED,
                "舊版 log 無報表／回顧預排版儀器",
            )
            return

        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        final_ok = True
        if intents:
            final_index, final_id = intents[-1]
            final_ok = any(index > final_index and item_id == final_id
                           for index, item_id, _, _, _ in report_fits)

        actual = []
        ordering_failures = []
        curve_ordering_failures = []
        missing_prefit = []
        missing_paints = []
        late_paints = []
        chart_drifts = []
        report_chart_drifts = []
        report_prefit_ordering_failures = []
        active = None
        lod_pattern = re.compile(r"^RV lodRebind merge (\d+)x(\d+)")
        paint_pattern = re.compile(
            r"^RV prefitPaint (\d{6}-\d{6}) chart=(col|row) after=\d+ms "
        )
        apply_pattern = re.compile(
            rf"^RV prefitApply (\d{{6}}-\d{{6}}) after=\d+ms visible=(True|False) "
            rf"col=axis=({number})~({number})/view=({number})~({number}) "
            rf"row=axis=({number})~({number})/view=({number})~({number})$"
        )
        chart_range_pattern = re.compile(
            rf"^RV chartRange (\d{{6}}-\d{{6}}|-) chart=(col|row) "
            rf"axis=({number})~({number})/view=({number})~({number})$"
        )
        report_chart_range_pattern = re.compile(
            rf"^DT chartRange (\d{{6}}-\d{{6}}|-) chart=(col|row) "
            rf"axis=({number})~({number})/view=({number})~({number})$"
        )
        has_main_range_edges = any(
            line.message.startswith("RV mainRange ") for line in session.lines
        )
        has_chart_range_edges = any(
            line.message.startswith("RV chartRange ") for line in session.lines
        )
        last_curve_paths = {}
        last_prefit = {}
        last_layout = {}
        last_report_intent = {}
        last_report_prefit = {}
        report_chart_ranges = []
        for index, line in enumerate(session.lines):
            if line.message.startswith("ui:【報表序號】"):
                last_report_intent[grab_id(line.message)] = index
            elif line.message.startswith("DT prefit "):
                report_match = report_fit_pattern.match(line.message)
                if report_match:
                    last_report_prefit[report_match.group(1)] = index
            elif line.message.startswith("DT curve load "):
                item_id = grab_id(line.message)
                intent_index = last_report_intent.get(item_id)
                prefit_index = last_report_prefit.get(item_id)
                if (intent_index is not None and
                        (prefit_index is None or not intent_index < prefit_index < index)):
                    report_prefit_ordering_failures.append(item_id)
            elif line.message.startswith("RV curves paths "):
                last_curve_paths[grab_id(line.message)] = index
            elif line.message.startswith("RV prefit "):
                last_prefit[grab_id(line.message)] = index
            elif line.message.startswith("RV layout intent "):
                last_layout[grab_id(line.message)] = index
            elif (line.message.startswith("RV curves ")
                  and not line.message.startswith(("RV curves paths ", "RV curves stale-drop "))):
                item_id = grab_id(line.message)
                path_index = last_curve_paths.get(item_id)
                prefit_index = last_prefit.get(item_id)
                layout_index = last_layout.get(item_id)
                if (path_index is None or prefit_index is None or layout_index is None
                        or not path_index < prefit_index < layout_index < index):
                    curve_ordering_failures.append(item_id)

            report_match = report_chart_range_pattern.match(line.message)
            if report_match:
                report_chart_ranges.append((
                    index, report_match.group(1), report_match.group(2),
                    tuple(float(report_match.group(group)) for group in range(3, 7)),
                ))

            if line.message.startswith("RV loadGrab begin "):
                active = {
                    "id": grab_id(line.message),
                    "begin": index,
                    "prefit": None,
                    "lod": None,
                    "push": None,
                    "visible": None,
                    "paints": {},
                    "apply": None,
                    "chart_ranges": [],
                }
                continue
            if active and line.message.startswith("RV prefit "):
                match = pattern.match(line.message)
                if match and match.group(1) == active["id"]:
                    active["prefit"] = (
                        index, int(match.group(2)), int(match.group(3))
                    )
                else:
                    invalid.append(line.message)
                continue
            match = paint_pattern.match(line.message)
            if active and match:
                if match.group(1) == active["id"]:
                    active["paints"][match.group(2)] = index
                else:
                    invalid.append(line.message)
                continue
            match = apply_pattern.match(line.message)
            if active and match:
                if match.group(1) == active["id"]:
                    active["visible"] = match.group(2) == "True"
                    active["apply"] = {
                        "index": index,
                        "col": tuple(float(match.group(group)) for group in range(3, 7)),
                        "row": tuple(float(match.group(group)) for group in range(7, 11)),
                    }
                else:
                    invalid.append(line.message)
                continue
            match = chart_range_pattern.match(line.message)
            if active and match:
                if match.group(1) in (active["id"], "-"):
                    active["chart_ranges"].append((
                        index, match.group(2),
                        tuple(float(match.group(group)) for group in range(3, 7)),
                    ))
                else:
                    invalid.append(line.message)
                continue
            match = lod_pattern.match(line.message)
            if active and match and active["lod"] is None:
                active["lod"] = (index, int(match.group(1)), int(match.group(2)))
            if active and line.message.startswith("RV pushFrames ") and active["push"] is None:
                active["push"] = index
            if line.message.startswith(("RV loadGrab done ", "RV loadGrab stale-drop ")):
                completed = line.message.startswith("RV loadGrab done ")
                if active and active["lod"]:
                    lod_index, width, height = active["lod"]
                    actual.append((active["id"], width, height))
                if active and completed and (active["lod"] or active["push"]):
                    deadlines = [item[0] for item in [active["lod"]] if item]
                    if active["push"] is not None:
                        deadlines.append(active["push"])
                    deadline = min(deadlines)
                    if active["prefit"] is None:
                        missing_prefit.append(active["id"])
                    elif active["prefit"][0] >= deadline:
                        ordering_failures.append(active["id"])

                    if active["visible"] is True:
                        absent = {"col", "row"} - set(active["paints"])
                        if absent:
                            missing_paints.append(
                                f"{active['id']}:{','.join(sorted(absent))}"
                            )
                        for chart, paint_index in active["paints"].items():
                            if paint_index >= deadline:
                                late_paints.append(f"{active['id']}:{chart}")
                    elif active["visible"] is None:
                        missing_paints.append(f"{active['id']}:prefitApply")

                    if active["apply"] is not None:
                        apply_index = active["apply"]["index"]
                        for range_index, chart, view in active["chart_ranges"]:
                            if range_index <= apply_index:
                                continue
                            expected_view = active["apply"][chart]
                            if any(abs(actual_value - expected_value) > 0.05
                                   for actual_value, expected_value in zip(view, expected_view)):
                                chart_drifts.append(
                                    f"{active['id']}:{chart} "
                                    f"axis/view {expected_view}->{view}"
                                )
                active = None

        report_edges_ok = not intents
        if intents:
            final_intent_index, final_intent_id = intents[-1]
            final_ranges = [
                (chart, state) for index, item_id, chart, state in report_chart_ranges
                if index > final_intent_index and item_id == final_intent_id
            ]
            report_edges_ok = {chart for chart, _ in final_ranges} >= {"col", "row"}
            for chart in ("col", "row"):
                states = [state for item_chart, state in final_ranges if item_chart == chart]
                if states and any(
                    any(abs(actual_value - expected_value) > 0.05
                        for actual_value, expected_value in zip(state, states[0]))
                    for state in states[1:]
                ):
                    report_chart_drifts.append(f"{final_intent_id}:{chart}")

        predicted = {item_id: (width, height) for _, item_id, width, height in fits}
        mismatches = [
            f"{item_id}:{predicted[item_id][0]}x{predicted[item_id][1]}!={width}x{height}"
            for item_id, width, height in actual
            if item_id in predicted and predicted[item_id] != (width, height)
        ]
        review_edges_ok = not fits or (has_main_range_edges and has_chart_range_edges)
        ok = (
            not invalid and not invalid_report_fits and final_ok and not mismatches
            and not ordering_failures and not curve_ordering_failures and not missing_prefit
            and not missing_paints and not late_paints
            and review_edges_ok and not chart_drifts
            and report_edges_ok and not report_chart_drifts
            and not report_prefit_ordering_failures
        )
        report.add(
            self.domain,
            "D3.fit",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"rvPrefit={len(fits)} dtPrefit={len(report_fits)} actual={len(actual)} "
            f"final={'ok' if final_ok else 'missing'} "
            f"格式錯誤={len(invalid) + len(invalid_report_fits)} 尺寸矛盾={len(mismatches)} "
            f"圖片順序錯誤={len(ordering_failures)} Curve順序錯誤={len(curve_ordering_failures)} "
            f"報表順序錯誤={len(report_prefit_ordering_failures)} "
            f"缺prefit={len(missing_prefit)} "
            f"缺paint={len(missing_paints)} paint過晚={len(late_paints)} "
            f"mainEdge={'yes' if has_main_range_edges else 'no'} "
            f"chartEdge={'yes' if has_chart_range_edges else 'no'} "
            f"reportEdge={'yes' if report_edges_ok else 'no'} "
            f"二次跳位={len(chart_drifts)} 報表跳位={len(report_chart_drifts)}"
            + (f"；首筆 {mismatches[0]}" if mismatches else ""),
        )

    def _check_single_row_curve(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.row-curve", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        pattern = re.compile(
            r"^DT row curve load (\d{6}-\d{6}) source=shared "
            r"storage=(summary|bins|memory-summary|memory-bins) "
            r"points=(\d+) pitch=([0-9.]+)mm$"
        )
        invalid = []
        loaded = 0
        unavailable = 0
        terminals = []
        for index, line in enumerate(session.lines):
            if line.message.startswith("DT row curve load ") or line.message.startswith("DT row curve missing "):
                terminals.append((index, line.message))
            if line.message.startswith("DT row curve missing "):
                unavailable += 1
                continue
            if not line.message.startswith("DT row curve load "):
                continue
            match = pattern.match(line.message)
            if not match or int(match.group(3)) <= 0 or float(match.group(4)) <= 0:
                invalid.append(line.message)
                continue
            loaded += 1

        final_index, final_id = intents[-1]
        final_terminal = any(
            index > final_index and message.startswith((
                f"DT row curve load {final_id} ",
                f"DT row curve missing {final_id}"))
            for index, message in terminals
        )
        status = CheckStatus.PASS if final_terminal and not invalid else CheckStatus.FAIL
        report.add(
            self.domain,
            "D3.row-curve",
            status,
            f"intent={len(intents)} loaded={loaded} legacy/missing={unavailable} "
            f"final={'ok' if final_terminal else 'missing'} 格式錯誤={len(invalid)}",
        )

    def _check_curve_summary_writes(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT curve summary ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.summary-write",
                CheckStatus.NOT_COVERED,
                "本 session 沒有由 bins 重建匯總",
            )
            return

        pattern = re.compile(
            r"^DT curve summary (\d{6}-\d{6}) "
            r"write=(queued|ok|failed|dropped|evicted|skip-incomplete) "
            r"captures=(\d+) merged=(\d+) ms=(\d+)"
            r"(?: reason=(idle|pressure))?$"
        )
        counts = {
            "queued": 0,
            "ok": 0,
            "failed": 0,
            "dropped": 0,
            "evicted": 0,
            "skip-incomplete": 0,
        }
        invalid = []
        for message in lines:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue
            status = match.group(2)
            captures = int(match.group(3))
            merged = int(match.group(4))
            reason = match.group(6)
            counts[status] += 1
            if status in ("queued", "ok") and captures != merged:
                invalid.append(message)
            if status in ("ok", "failed") and reason not in ("idle", "pressure"):
                invalid.append(message)

        bad = counts["failed"] + counts["dropped"] + counts["evicted"]
        ok = not invalid and bad == 0
        report.add(
            self.domain,
            "D3.summary-write",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"queued={counts['queued']} ok={counts['ok']} "
            f"failed={counts['failed']} dropped={counts['dropped']} "
            f"evicted={counts['evicted']} incomplete={counts['skip-incomplete']} "
            f"格式錯誤={len(invalid)}"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )

    def _check_list_ownership(self, session: FlowSession, report: CheckReport) -> None:
        last_intent = None
        last_intent_time = None
        violations = []
        reloads = 0
        for line in session.lines:
            message = line.message
            if message.startswith("ui:"):
                last_intent = message
                last_intent_time = line.elapsed
            elif message.startswith("DT list reload"):
                reloads += 1
                if (
                    last_intent
                    and last_intent.startswith("ui:【報表序號】")
                    and last_intent_time is not None
                    and line.elapsed - last_intent_time <= 2
                ):
                    violations.append(f"{line.timestamp} {message}")
        report.add(
            self.domain,
            "D3.list-keep",
            CheckStatus.PASS if not violations else CheckStatus.FAIL,
            f"list reload={reloads}；由單片選擇觸發={len(violations)}"
            + (f"；首筆 {violations[0]}" if violations else ""),
        )

    def _check_ui_stall(
        self, session: FlowSession, report: CheckReport, limit_ms: int = 1000
    ) -> None:
        data_times = [
            line.elapsed
            for line in session.lines
            if (
                line.message.startswith(
                    ("ui:【報表", "ui:【明細列表】", "ui:【序號範圍-", "ui:【期間-", "ui:【良率")
                )
                or (
                    line.message.startswith("DT ")
                    and not line.message.startswith(("DT curve load policy ", "DT range policy "))
                )
            )
        ]
        if not data_times:
            report.add(self.domain, "U.stall", CheckStatus.NOT_COVERED, "無可量測的報表互動")
            return
        assessment = assess_ui_responsiveness(session, data_times, limit_ms)
        report.add(
            self.domain,
            "U.stall",
            CheckStatus.PASS if assessment.passed else CheckStatus.FAIL,
            assessment.detail(limit_ms),
        )

    def _check_range_debounce(self, session: FlowSession, report: CheckReport) -> None:
        range_intents = 0
        settles = 0
        waiting_for_settle = False
        premature = []
        for line in session.lines:
            message = line.message
            if message.startswith("ui:【序號範圍-"):
                range_intents += 1
                waiting_for_settle = True
            elif message == "DT range settle → refresh":
                settles += 1
                waiting_for_settle = False
            elif waiting_for_settle and message.startswith(("DT list reload", "DT curve candidates")):
                premature.append(f"{line.timestamp} {message}")

        if range_intents == 0:
            report.add(self.domain, "D3.range-settle", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        ok = not premature and settles > 0 and settles <= range_intents
        report.add(
            self.domain,
            "D3.range-settle",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={range_intents} settle={settles} settle前重算={len(premature)}"
            + (f"；首筆 {premature[0]}" if premature else ""),
        )

    def _check_range_policy(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT range policy ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.range-policy",
                CheckStatus.NOT_COVERED,
                "舊版 log 無 range policy 儀器",
            )
            return

        expected = "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic"
        unique = sorted(set(lines))
        ok = len(lines) == 1 and unique == [expected]
        report.add(
            self.domain,
            "D3.range-policy",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"行數={len(lines)} 實際={' | '.join(unique)}",
        )

    def _check_range_preview(self, session: FlowSession, report: CheckReport) -> None:
        intents = sum(
            1 for line in session.lines
            if line.message.startswith("ui:【序號範圍-")
        )
        pattern = re.compile(
            r"^DT range preview apply gen=(\d+) range=(\d{6}-\d{6})~(\d{6}-\d{6}) "
            r"loadMs=(\d+) drawMs=(\d+) meanRows=(\d+) maxRows=(\d+) "
            r"method=(top-maxcmean|mixed|even) coverage=(\d+)/(\d+) rankedCams=(\d+)/(\d+)"
            r" index=(\d+)/(\d+)$"
        )
        generations = []
        invalid = []
        for line in session.lines:
            if not line.message.startswith("DT range preview apply"):
                continue
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            generations.append(int(match.group(1)))

        if intents == 0:
            report.add(self.domain, "D3.range-preview", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        if not generations:
            report.add(
                self.domain,
                "D3.range-preview",
                CheckStatus.FAIL,
                f"intent={intents}，沒有 Curve 預覽上畫",
            )
            return

        monotonic = all(
            current > previous
            for previous, current in zip(generations, generations[1:])
        )
        list_generations = []
        for line in session.lines:
            match = re.match(r"^DT range list preview gen=(\d+) ", line.message)
            if match:
                list_generations.append(int(match.group(1)))
        final_caught_up = not list_generations or generations[-1] >= list_generations[-1]
        ok = not invalid and monotonic and final_caught_up
        report.add(
            self.domain,
            "D3.range-preview",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={intents} apply={len(generations)} generation={generations[0]}~{generations[-1]} "
            f"倒退上畫={0 if monotonic else 1} 最終追上={1 if final_caught_up else 0} "
            f"格式錯誤={len(invalid)}",
        )

    def _check_range_list_preview(self, session: FlowSession, report: CheckReport) -> None:
        intents = sum(
            1 for line in session.lines
            if line.message.startswith("ui:【序號範圍-")
        )
        pattern = re.compile(
            r"^DT range list preview gen=(\d+) range=(\d{6}-\d{6})~(\d{6}-\d{6}) "
            r"rows=(\d+) ms=(\d+) source=index$"
        )
        generations = []
        invalid = []
        worst_ms = 0
        for line in session.lines:
            if not line.message.startswith("DT range list preview"):
                continue
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            generations.append(int(match.group(1)))
            worst_ms = max(worst_ms, int(match.group(5)))

        if intents == 0:
            report.add(self.domain, "D3.list-preview", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        if not generations:
            report.add(
                self.domain,
                "D3.list-preview",
                CheckStatus.FAIL,
                f"intent={intents}，沒有 List 預覽套用",
            )
            return

        monotonic = all(
            current > previous
            for previous, current in zip(generations, generations[1:])
        )
        ok = not invalid and monotonic
        report.add(
            self.domain,
            "D3.list-preview",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={intents} apply={len(generations)} generation={generations[0]}~{generations[-1]} "
            f"最大={worst_ms}ms 過期套用={0 if monotonic else 1} 格式錯誤={len(invalid)}",
        )

    def _check_y_scale_toggle(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(
            r"^ui:【良率圖-(年|月|日)】→ Y軸=(Auto|Fixed) "
            r"setting=(Auto|Fixed) override=(Auto|Fixed|off)$"
        )
        lines = [
            line for line in session.lines if line.message.startswith("ui:【良率圖-")
        ]
        if not lines:
            report.add(self.domain, "D5.y-scale", CheckStatus.NOT_COVERED, "無良率圖 Y 軸點擊")
            return

        invalid = []
        for line in lines:
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            _, effective, setting, scale_override = match.groups()
            if scale_override == "off":
                valid = effective == setting
            else:
                valid = effective == scale_override and scale_override != setting
            if not valid:
                invalid.append(
                    f"{line.timestamp} effective={effective} setting={setting} override={scale_override}"
                )

        report.add(
            self.domain,
            "D5.y-scale",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"點擊={len(lines)} 狀態矛盾={len(invalid)}"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )
