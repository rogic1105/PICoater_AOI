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
            "ui:【篩選異常】",
            "ui:【讀取資料】鈕（Data）",
        )
        return any(line.message.startswith(prefixes) for line in session.lines)

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        if not self._covered(session):
            report.add(self.domain, "D0", CheckStatus.NOT_COVERED, "本 session 無報表操作")
            return report

        self._check_statistics_snapshot(session, report)
        self._check_column_verdict_index(session, report)
        self._check_column_verdicts(session, report)
        self._check_column_chart_verdict_alignment(session, report)
        self._check_column_verdict_clicks(session, report)
        self._check_single_selection(session, report)
        self._check_single_curve_policy(session, report)
        self._check_single_curve(session, report)
        self._check_cross_tab_curve_reuse(session, report)
        self._check_single_fit(session, report)
        self._check_curve_summary_writes(session, report)
        self._check_single_row_curve(session, report)
        self._check_list_ownership(session, report)
        self._check_virtual_list(session, report)
        self._check_range_policy(session, report)
        self._check_range_debounce(session, report)
        self._check_range_list_preview(session, report)
        self._check_range_preview(session, report)
        self._check_y_scale_toggle(session, report)
        self._check_fail_filter(session, report)
        self._check_ui_stall(session, report)
        return report

    def _check_column_verdict_index(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        pattern = re.compile(
            r"^DT verdict index apply=ok gen=(\d+) summaries=(\d+) bins=(\d+) "
            r"missing=(\d+)/(\d+) cams=(\d+) verdicts=(\d+) ms=(\d+)$"
        )
        candidates = [
            line.message for line in session.lines
            if line.message.startswith("DT verdict index apply=")
        ]
        if not candidates:
            report.add(
                self.domain, "D1.verdict-index", CheckStatus.NOT_COVERED,
                "No full-list column verdict index evidence",
            )
            return

        invalid = []
        for message in candidates:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue
            _, summaries, bins, missing, requested, cameras, verdicts, _ = map(
                int, match.groups()
            )
            if summaries + bins + missing != requested or verdicts != cameras:
                invalid.append(message)

        report.add(
            self.domain,
            "D1.verdict-index",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"runs={len(candidates)} invalid={len(invalid)}"
            + (f" first={invalid[0]}" if invalid else ""),
        )

    def _check_column_verdicts(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        pattern = re.compile(
            r"^DT verdict (?P<grab>\d{6}-\d{6}) cam=(?P<cam>\d+) "
            r"(?:mode=(?P<mode>mean|max|both) )?"
            r"mean=(?P<mean>\d+(?:\.\d+)?)/(?P<mean_threshold>\d+(?:\.\d+)?) "
            r"(?:enabled=(?P<mean_enabled>[01]) )?"
            r"max=(?P<max>\d+(?:\.\d+)?)/(?P<max_threshold>\d+(?:\.\d+)?) "
            r"(?:enabled=(?P<max_enabled>[01]) )?"
            r"result=(?P<result>pass|fail) "
            r"cause=(?P<cause>none|mean|max|both) "
            r"source=(?:visible-merged-curve|merged-curve)$"
        )
        candidates = [
            line.message for line in session.lines
            if line.message.startswith("DT verdict ")
            and not line.message.startswith(("DT verdict index ", "DT verdict click "))
        ]
        if not candidates:
            report.add(
                self.domain, "D1.verdict", CheckStatus.NOT_COVERED,
                "No selected column verdict evidence",
            )
            return

        invalid = []
        causes = {"none": 0, "mean": 0, "max": 0, "both": 0}
        for message in candidates:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue

            mode = match.group("mode") or "both"
            mean_enabled = match.group("mean_enabled") != "0"
            max_enabled = match.group("max_enabled") != "0"
            expected_flags = {
                "mean": (True, False),
                "max": (False, True),
                "both": (True, True),
            }[mode]
            mean_failed = mean_enabled and float(match.group("mean")) > float(
                match.group("mean_threshold"))
            max_failed = max_enabled and float(match.group("max")) > float(
                match.group("max_threshold"))
            expected_cause = (
                "both" if mean_failed and max_failed else
                "mean" if mean_failed else
                "max" if max_failed else
                "none"
            )
            expected_result = "fail" if mean_failed or max_failed else "pass"
            actual_cause = match.group("cause")
            causes[actual_cause] += 1
            if ((mean_enabled, max_enabled) != expected_flags or
                    match.group("result") != expected_result or
                    actual_cause != expected_cause):
                invalid.append(message)

        report.add(
            self.domain,
            "D1.verdict",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"rows={len(candidates)} causes="
            f"none:{causes['none']}/mean:{causes['mean']}/"
            f"max:{causes['max']}/both:{causes['both']} invalid={len(invalid)}"
            + (f" first={invalid[0]}" if invalid else ""),
        )

    def _check_column_chart_verdict_alignment(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        display_pattern = re.compile(
            r"^DT curve display (?P<grab>\d{6}-\d{6}) "
            r"mode=(?P<mode>mean|max|both) "
            r"mean=(?P<mean>\d+(?:\.\d+)?)/(?P<mean_threshold>\d+(?:\.\d+)?) "
            r"max=(?P<max>\d+(?:\.\d+)?)/(?P<max_threshold>\d+(?:\.\d+)?) "
            r"scale=(?P<scale>\d+(?:\.\d+)?) points=(?P<points>\d+)$"
        )
        verdict_pattern = re.compile(
            r"^DT verdict (?P<grab>\d{6}-\d{6}) cam=\d+ "
            r"mode=(?P<mode>mean|max|both) "
            r"mean=(?P<mean>\d+(?:\.\d+)?)/\d+(?:\.\d+)? enabled=[01] "
            r"max=(?P<max>\d+(?:\.\d+)?)/\d+(?:\.\d+)? enabled=[01] "
        )
        messages = [line.message for line in session.lines]
        display_indices = [
            index for index, message in enumerate(messages)
            if message.startswith("DT curve display ")
        ]
        if not display_indices:
            report.add(
                self.domain, "D1.chart-verdict", CheckStatus.NOT_COVERED,
                "No chart display peak evidence",
            )
            return

        invalid = []
        compared = 0
        for position, start in enumerate(display_indices):
            end = display_indices[position + 1] if position + 1 < len(display_indices) else len(messages)
            display = display_pattern.match(messages[start])
            if not display:
                invalid.append(messages[start])
                continue
            rows = []
            for message in messages[start + 1:end]:
                match = verdict_pattern.match(message)
                if match and match.group("grab") == display.group("grab"):
                    rows.append(match)
            if not rows:
                continue

            compared += 1
            chart_mean = float(display.group("mean"))
            chart_max = float(display.group("max"))
            verdict_mean = max(float(row.group("mean")) for row in rows)
            verdict_max = max(float(row.group("max")) for row in rows)
            if (display.group("mode") != rows[0].group("mode") or
                    abs(chart_mean - verdict_mean) > 0.0002 or
                    abs(chart_max - verdict_max) > 0.0002):
                invalid.append(
                    f"{display.group('grab')} chart={chart_mean:.4f}/{chart_max:.4f} "
                    f"verdict={verdict_mean:.4f}/{verdict_max:.4f}"
                )

        status = CheckStatus.PASS if compared > 0 and not invalid else (
            CheckStatus.NOT_COVERED if compared == 0 and not invalid else CheckStatus.FAIL
        )
        report.add(
            self.domain, "D1.chart-verdict", status,
            f"compared={compared} invalid={len(invalid)}"
            + (f" first={invalid[0]}" if invalid else ""),
        )

    def _check_column_verdict_clicks(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        row_pattern = re.compile(
            r"^DT verdict click (?P<grab>\d{6}-\d{6}) cam=(?P<cam>\d+) "
            r"mode=(?P<mode>mean|max|both) "
            r"mean=(?P<mean>nan|\d+(?:\.\d+)?)/(?P<mean_threshold>\d+(?:\.\d+)?) "
            r"enabled=(?P<mean_enabled>[01]) "
            r"max=(?P<max>nan|\d+(?:\.\d+)?)/(?P<max_threshold>\d+(?:\.\d+)?) "
            r"enabled=(?P<max_enabled>[01]) "
            r"result=(?P<result>pass|fail|unknown) "
            r"cause=(?P<cause>none|mean|max|both) "
            r"list=(?P<list>pass|fail|unknown) "
            r"source=(?P<source>visible-curve-index|curve-index|missing)$"
        )
        done_pattern = re.compile(
            r"^DT verdict click done (?P<grab>\d{6}-\d{6}) cams=(?P<cams>\d+)$"
        )
        candidates = [
            line.message for line in session.lines
            if line.message.startswith("DT verdict click ")
        ]
        if not candidates:
            report.add(
                self.domain, "D1.verdict-click", CheckStatus.NOT_COVERED,
                "No report-list column verdict audit evidence",
            )
            return

        pending = {}
        invalid = []
        runs = 0
        rows = 0
        for message in candidates:
            done = done_pattern.match(message)
            if done:
                grab = done.group("grab")
                expected_cameras = int(done.group("cams"))
                audited = pending.pop(grab, [])
                runs += 1
                if (len(audited) != expected_cameras or
                        sorted(audited) != list(range(1, expected_cameras + 1))):
                    invalid.append(message)
                continue

            match = row_pattern.match(message)
            if not match:
                invalid.append(message)
                continue

            rows += 1
            grab = match.group("grab")
            pending.setdefault(grab, []).append(int(match.group("cam")))
            mode = match.group("mode")
            mean_enabled = match.group("mean_enabled") == "1"
            max_enabled = match.group("max_enabled") == "1"
            expected_flags = {
                "mean": (True, False),
                "max": (False, True),
                "both": (True, True),
            }[mode]
            if (mean_enabled, max_enabled) != expected_flags:
                invalid.append(message)
                continue

            mean_value = match.group("mean")
            max_value = match.group("max")
            mean_failed = (
                mean_enabled and mean_value != "nan" and
                float(mean_value) > float(match.group("mean_threshold"))
            )
            max_failed = (
                max_enabled and max_value != "nan" and
                float(max_value) > float(match.group("max_threshold"))
            )
            has_data = (
                (mean_enabled and mean_value != "nan") or
                (max_enabled and max_value != "nan")
            )
            expected_cause = (
                "both" if mean_failed and max_failed else
                "mean" if mean_failed else
                "max" if max_failed else
                "none"
            )
            expected_result = (
                "unknown" if not has_data else
                "fail" if mean_failed or max_failed else "pass"
            )
            list_matches = (
                expected_result == "unknown" or
                match.group("list") == expected_result
            )
            if (match.group("result") != expected_result or
                    match.group("cause") != expected_cause or
                    not list_matches):
                invalid.append(message)

        invalid.extend(f"missing done: {grab}" for grab in pending)
        report.add(
            self.domain,
            "D1.verdict-click",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"runs={runs} rows={rows} invalid={len(invalid)}"
            + (f" first={invalid[0]}" if invalid else ""),
        )

    def _check_cross_tab_curve_reuse(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        shares = {}
        for index, line in enumerate(session.lines):
            if line.message.startswith("DT curve share "):
                shares.setdefault(grab_id(line.message), []).append(index)

        syncs = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("DT review sync apply ")
        ]
        exercised = []
        failures = []
        for sequence, (index, item_id) in enumerate(syncs):
            prior_shares = [position for position in shares.get(item_id, []) if position < index]
            if not prior_shares:
                continue
            exercised.append(item_id)
            end = syncs[sequence + 1][0] if sequence + 1 < len(syncs) else len(session.lines)
            window = session.lines[index + 1:end]
            completion = next(
                (
                    position for position, line in enumerate(window)
                    if line.message.startswith(("RV loadGrab done ", "RV loadGrab stale-drop "))
                    and grab_id(line.message) == item_id
                ),
                None,
            )
            if completion is not None:
                window = window[:completion + 1]
            reuse = any(
                line.message == f"RV loadGrab curves=reuse source=Data {item_id}"
                for line in window
            )
            duplicate = any(
                line.message == f"RV loadGrab curves=load source=bin {item_id}"
                or line.message.startswith(f"RV curves paths {item_id} ")
                for line in window
            )
            if not reuse or duplicate:
                failures.append(
                    f"{item_id} reuse={'yes' if reuse else 'no'} "
                    f"duplicate={'yes' if duplicate else 'no'}"
                )

        if not exercised:
            report.add(
                self.domain,
                "D3.review-reuse",
                CheckStatus.NOT_COVERED,
                "沒有『報表曲線已完成後切到回顧』的同序號案例",
            )
            return
        report.add(
            self.domain,
            "D3.review-reuse",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"cases={len(exercised)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

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
            "entries=512 maxMB=256 scale=merged-only minCycleMs=33"
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
        transition_chart_drifts = []
        report_chart_drifts = []
        report_prefit_ordering_failures = []
        active = None
        lod_pattern = re.compile(r"^RV lodRebind merge (\d+)x(\d+)")
        push_pattern = re.compile(r"^RV pushFrames .*feedScale=(\d+),")
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
        pending_review_fit = None
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

            transition_apply_match = apply_pattern.match(line.message)
            if transition_apply_match:
                pending_review_fit = {
                    "id": transition_apply_match.group(1),
                    "col": tuple(
                        float(transition_apply_match.group(group))
                        for group in range(3, 7)
                    ),
                    "row": tuple(
                        float(transition_apply_match.group(group))
                        for group in range(7, 11)
                    ),
                }
            elif line.message == "RV fit(record-change)":
                pending_review_fit = None
            else:
                transition_range_match = chart_range_pattern.match(line.message)
                if (pending_review_fit and transition_range_match and
                        transition_range_match.group(1) in (
                            pending_review_fit["id"], "-"
                        )):
                    chart = transition_range_match.group(2)
                    view = tuple(
                        float(transition_range_match.group(group))
                        for group in range(3, 7)
                    )
                    expected_view = pending_review_fit[chart]
                    if any(
                        abs(actual_value - expected_value) > 0.05
                        for actual_value, expected_value in zip(view, expected_view)
                    ):
                        transition_chart_drifts.append(
                            f"{pending_review_fit['id']}:{chart} "
                            f"axis/view {expected_view}->{view}"
                        )

            if line.message.startswith("RV loadGrab begin "):
                active = {
                    "id": grab_id(line.message),
                    "begin": index,
                    "prefit": None,
                    "lod": None,
                    "push": None,
                    "feed_scale": None,
                    "visible": None,
                    "paints": {},
                    "apply": None,
                    "chart_ranges": [],
                    "keep_curves": False,
                    "push_keep": False,
                }
                continue
            if active and line.message == (
                    f"RV loadGrab curves=keep source=display {active['id']}"):
                active["keep_curves"] = True
                continue
            if active and line.message.startswith("RV prefit "):
                match = pattern.match(line.message)
                if match and match.group(1) == active["id"]:
                    active["prefit"] = (
                        index, int(match.group(2)), int(match.group(3))
                    )
                # Curve fast-path may advance while an older debounced image
                # load is still active. Only matching evidence belongs here.
                continue
            match = paint_pattern.match(line.message)
            if active and match:
                if match.group(1) == active["id"]:
                    active["paints"][match.group(2)] = index
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
                continue
            match = chart_range_pattern.match(line.message)
            if active and match:
                if match.group(1) in (active["id"], "-"):
                    active["chart_ranges"].append((
                        index, match.group(2),
                        tuple(float(match.group(group)) for group in range(3, 7)),
                    ))
                continue
            match = lod_pattern.match(line.message)
            if active and match and active["lod"] is None:
                active["lod"] = (index, int(match.group(1)), int(match.group(2)))
            push_match = push_pattern.match(line.message)
            if active and push_match and active["push"] is None:
                active["push"] = index
                active["feed_scale"] = int(push_match.group(1))
                active["push_keep"] = "chartView=keep" in line.message
            if line.message.startswith(("RV loadGrab done ", "RV loadGrab stale-drop ")):
                completed = line.message.startswith("RV loadGrab done ")
                if active and active["lod"]:
                    lod_index, width, height = active["lod"]
                    actual.append((
                        active["id"], width, height,
                        active["feed_scale"] or 5,
                    ))
                if active and completed and (active["lod"] or active["push"]):
                    image_variant_only = active["keep_curves"] and active["push_keep"]
                    if image_variant_only:
                        active = None
                        continue
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
            # WinForms can synchronously repaint one chart while the native ComboBox
            # selection message is still unwinding, a few milliseconds before the
            # managed intent line. Only state edges for the final ID, after the most
            # recent different-ID intent, belong to the final selection burst.
            final_window_start = max(
                (
                    index for index, item_id in intents[:-1]
                    if item_id != final_intent_id
                ),
                default=-1,
            )
            final_mode_end = next(
                (
                    index for index in range(final_intent_index + 1, len(session.lines))
                    if session.lines[index].message.startswith((
                        "ui:【序號範圍-", "ui:【期間-", "ui:【明細列表】同列再點 "
                    ))
                ),
                len(session.lines),
            )
            final_ranges = [
                (chart, state) for index, item_id, chart, state in report_chart_ranges
                if final_window_start < index < final_mode_end and item_id == final_intent_id
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
        mismatches = []
        for item_id, width, height, feed_scale in actual:
            if item_id not in predicted:
                continue
            predicted_width, predicted_height = predicted[item_id]
            # Prefit dimensions are based on the normal /5 archive image. A thumbnail or
            # Hessian standard map may use another pixel density, but its physical extent
            # must remain identical after applying the published feedScale.
            predicted_physical_width = predicted_width * 5
            predicted_physical_height = predicted_height * 5
            actual_physical_width = width * feed_scale
            actual_physical_height = height * feed_scale
            if feed_scale == 5:
                same_physical_size = (
                    predicted_width == width and predicted_height == height
                )
            else:
                same_physical_size = (
                    abs(predicted_physical_width - actual_physical_width)
                    <= max(1, predicted_physical_width) * 0.005
                    and abs(predicted_physical_height - actual_physical_height)
                    <= max(1, predicted_physical_height) * 0.005
                )
            if not same_physical_size:
                mismatches.append(
                    f"{item_id}:{predicted_width}x{predicted_height}@5"
                    f"!={width}x{height}@{feed_scale}"
                )
        review_edges_ok = not fits or (has_main_range_edges and has_chart_range_edges)
        ok = (
            not invalid and not invalid_report_fits and final_ok and not mismatches
            and not ordering_failures and not curve_ordering_failures and not missing_prefit
            and not missing_paints and not late_paints
            and review_edges_ok and not chart_drifts and not transition_chart_drifts
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
            f"transitionDrift={len(transition_chart_drifts)} "
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

    def _check_virtual_list(self, session: FlowSession, report: CheckReport) -> None:
        fallbacks = [
            line for line in session.lines
            if line.message.startswith("DT list virtual fallback ")
        ]
        report.add(
            self.domain,
            "D2.virtual-list",
            CheckStatus.PASS if not fallbacks else CheckStatus.FAIL,
            f"fallbacks={len(fallbacks)}"
            + (f" first={fallbacks[0].timestamp} {fallbacks[0].message}" if fallbacks else ""),
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

        expected = (
            "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic "
            "curveSamples=50 "
            "curveCacheEntries=2048 curveCacheMB=256"
        )
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
            r"^DT range preview apply gen=(\d+) latest=(\d+) "
            r"range=(\d{6}-\d{6})~(\d{6}-\d{6}) "
            r"loadMs=(\d+) drawMs=(\d+) meanRows=(\d+) maxRows=(\d+) "
            r"method=(top-maxcmean|mixed|even) coverage=(\d+)/(\d+) rankedCams=(\d+)/(\d+)"
            r" index=(\d+)/(\d+) cache=(\d+)/(\d+) "
            r"hmCoverage=(\d+)/(\d+) hmCurrent=([0-9]+(?:\.[0-9]+)?) "
            r"sampleLimit=(\d+)$"
        )
        applies = []
        invalid = []
        for line in session.lines:
            if line.message.startswith("DT range preview apply"):
                match = pattern.match(line.message)
                if not match:
                    invalid.append(f"{line.timestamp} apply 格式錯誤")
                    continue
                generation = int(match.group(1))
                latest = int(match.group(2))
                hm_rows = int(match.group(18))
                total_rows = int(match.group(19))
                current_hm = float(match.group(20))
                sample_limit = int(match.group(21))
                if (generation > latest or sample_limit != 50 or
                        hm_rows > total_rows or current_hm <= 0):
                    invalid.append(
                        f"{line.timestamp} gen={generation} latest={latest} "
                        f"hm={hm_rows}/{total_rows} current={current_hm} "
                        f"sampleLimit={sample_limit}"
                    )
                applies.append((generation, latest, sample_limit))
            elif line.message.startswith("DT range preview stale-drop"):
                invalid.append(f"{line.timestamp} monotonic 模式不應 stale-drop")

        if intents == 0:
            report.add(self.domain, "D3.range-preview", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        if not applies:
            report.add(
                self.domain,
                "D3.range-preview",
                CheckStatus.FAIL,
                f"intent={intents}，沒有 Curve 預覽上畫",
            )
            return

        generations = [item[0] for item in applies]
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
        lagged_applies = sum(1 for generation, latest, _ in applies if generation != latest)
        sustained_scroll = intents >= 100
        backpressure_visible = (
            not sustained_scroll or
            (len(generations) >= 2 and
             lagged_applies >= 1 and
             len(generations) < intents)
        )
        ok = (
            not invalid and monotonic and final_caught_up and
            backpressure_visible
        )
        report.add(
            self.domain,
            "D3.range-preview",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={intents} apply={len(generations)} 跳讀上畫={lagged_applies} "
            f"generation={generations[0]}~{generations[-1]} "
            f"倒退上畫={0 if monotonic else 1} 最終追上={1 if final_caught_up else 0} "
            f"大量滾動守門={1 if backpressure_visible else 0} "
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

    def _check_fail_filter(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(
            r"^ui:【篩選異常】→ (只顯示異常|顯示全部) "
            r"dataOptions=(\d+) rangeOptions=(\d+) "
            r"selected=(empty|\d{6}-\d{6}) "
            r"range=(empty|\d{6}-\d{6}~\d{6}-\d{6})$"
        )
        lines = [
            line for line in session.lines
            if line.message.startswith("ui:【篩選異常】")
        ]
        if not lines:
            report.add(self.domain, "D5.fail-filter", CheckStatus.NOT_COVERED, "未切換異常篩選")
            return

        invalid = []
        previous_mode = None
        for line in lines:
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤或缺少範圍證據")
                continue
            mode, data_count_text, range_count_text, selected, range_text = match.groups()
            data_count = int(data_count_text)
            range_count = int(range_count_text)
            if data_count != range_count:
                invalid.append(
                    f"{line.timestamp} dataOptions={data_count} rangeOptions={range_count}"
                )
            if (range_count == 0) != (range_text == "empty"):
                invalid.append(
                    f"{line.timestamp} rangeOptions={range_count} range={range_text}"
                )
            if (data_count == 0) != (selected == "empty"):
                invalid.append(
                    f"{line.timestamp} dataOptions={data_count} selected={selected}"
                )
            if previous_mode == mode:
                invalid.append(f"{line.timestamp} toggle 未換狀態：{mode}")
            previous_mode = mode

        report.add(
            self.domain,
            "D5.fail-filter",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"切換={len(lines)} 範圍矛盾={len(invalid)}"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )
