"""Review-tab (R-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession, grab_id


ROW_RE = re.compile(
    r"rowChart dir=(?P<dir>\w+) .*?total=(?P<total>[-\d.]+)mm .*?"
    r"dataPhys (?P<p0>[-\d.]+)~(?P<p1>[-\d.]+)mm "
    r"dataChart (?P<c0>[-\d.]+)~(?P<c1>[-\d.]+)"
)


class ReviewFlowValidator:
    domain = "REVIEW"

    @staticmethod
    def _covered(session: FlowSession) -> bool:
        prefixes = (
            "RV ",
            "ui:【單片序號】",
            "ui:【讀取資料】",
            "ui:【時段導航】",
        )
        return any(line.message.startswith(prefixes) for line in session.lines)

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        if not self._covered(session):
            report.add(self.domain, "R0", CheckStatus.NOT_COVERED, "本 session 無回顧操作")
            return report

        self._check_curves_follow(session, report)
        self._check_ui_stall(session, report)
        self._check_reload_jumps_to_newest(session, report)
        self._check_period_dedup(session, report)
        self._check_period_single_flight(session, report)
        self._check_curve_single_flight(session, report)
        self._check_drag_first_publish(session, report)
        self._check_direction(session, report)
        return report

    def _check_curves_follow(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (line.elapsed, grab_id(line.message))
            for line in session.lines
            if line.message.startswith("ui:【單片序號】")
        ]
        if not intents:
            report.add(self.domain, "R2.curves", CheckStatus.NOT_COVERED, "無單片序號操作")
            report.add(self.domain, "R2.token", CheckStatus.NOT_COVERED, "無單片序號操作")
            report.add(self.domain, "R2.lifecycle", CheckStatus.NOT_COVERED, "無單片序號操作")
            return

        last_time, last_id = intents[-1]
        curves_ok = [
            grab_id(line.message)
            for line in session.lines
            if line.elapsed >= last_time
            and line.message.startswith("RV curves ")
            and "stale-drop" not in line.message
            and "paths" not in line.message
        ]
        report.add(
            self.domain,
            "R2.curves",
            CheckStatus.PASS if last_id in curves_ok else CheckStatus.FAIL,
            f"最後 intent={last_id}；其後成功 curves={curves_ok or '無'}",
        )

        done_lines = [
            (line.elapsed, grab_id(line.message))
            for line in session.lines
            if line.message.startswith("RV loadGrab done")
        ]
        dones = [current for _, current in done_lines]
        matching_dones = [
            current for elapsed, current in done_lines if elapsed >= last_time and current == last_id
        ]
        report.add(
            self.domain,
            "R2.token",
            CheckStatus.PASS if matching_dones else CheckStatus.FAIL,
            f"intent={last_id}；其後同 ID done={len(matching_dones)}",
        )

        begins = [
            grab_id(line.message)
            for line in session.lines
            if line.message.startswith("RV loadGrab begin")
        ]
        open_loads = []
        stale_count = 0
        for line in session.lines:
            message = line.message
            if message.startswith("RV loadGrab begin"):
                open_loads.append(grab_id(message))
            elif message.startswith(("RV loadGrab done", "RV loadGrab stale-drop")):
                current = grab_id(message)
                if current in open_loads:
                    open_loads.remove(current)
                if message.startswith("RV loadGrab stale-drop"):
                    stale_count += 1
        report.add(
            self.domain,
            "R2.lifecycle",
            CheckStatus.PASS if len(open_loads) <= 1 else CheckStatus.FAIL,
            f"begin={len(begins)} done={len(dones)} stale={stale_count} 未結束={open_loads or 0}",
        )

    def _check_ui_stall(
        self, session: FlowSession, report: CheckReport, limit_ms: int = 1000
    ) -> None:
        review_times = [
            line.elapsed
            for line in session.lines
            if line.message.startswith(
                ("RV ", "ui:【單片序號】", "ui:【讀取資料】", "ui:【時段導航】")
            )
        ]
        stalls = []
        for line in session.lines:
            if not line.message.startswith("[UiStall]"):
                continue
            if not any(event_time - 1 <= line.elapsed <= event_time + 3 for event_time in review_times):
                continue
            match = re.search(r"\[UiStall\]\s+(\d+)ms（(.*)）", line.message)
            if match:
                stalls.append((int(match.group(1)), match.group(2)))
        worst = max(stalls) if stalls else (0, "")
        report.add(
            self.domain,
            "U.stall",
            CheckStatus.PASS if worst[0] <= limit_ms else CheckStatus.FAIL,
            f"最大={worst[0]}ms（{worst[1]}）；>{limit_ms}ms 共 "
            f"{sum(1 for duration, _ in stalls if duration > limit_ms)} 次",
        )

    def _check_reload_jumps_to_newest(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        reload_times = [
            line.elapsed
            for line in session.lines
            if line.message.startswith("ui:【讀取資料】鈕（Review）")
        ]
        if len(reload_times) < 2:
            report.add(
                self.domain,
                "R1.reload-latest",
                CheckStatus.NOT_COVERED,
                f"讀取資料 {len(reload_times)} 次；需要至少 2 次",
            )
            return

        details = []
        for reload_time in reload_times[1:]:
            seen = [
                current
                for line in session.lines
                if line.elapsed < reload_time
                for current in [grab_id(line.message)]
                if current
            ]
            newest_seen = max(seen) if seen else None
            after_begin = next(
                (
                    grab_id(line.message)
                    for line in session.lines
                    if line.elapsed >= reload_time
                    and line.message.startswith("RV loadGrab begin")
                ),
                None,
            )
            if newest_seen and after_begin and after_begin < newest_seen:
                details.append(f"{after_begin} < {newest_seen}")
        report.add(
            self.domain,
            "R1.reload-latest",
            CheckStatus.FAIL if details else CheckStatus.PASS,
            "; ".join(details) if details else "第 2 次起皆未退回舊序號",
        )

    def _check_period_dedup(self, session: FlowSession, report: CheckReport) -> None:
        loads = [
            line.message
            for line in session.lines
            if line.message.startswith("RV period load ")
        ]
        if not loads:
            report.add(self.domain, "R3.dedup", CheckStatus.NOT_COVERED, "無時段導航")
            return

        duplicate_count = 0
        longest_run = 1
        run = 1
        for index in range(1, len(loads)):
            if loads[index] == loads[index - 1]:
                run += 1
                duplicate_count += 1
                longest_run = max(longest_run, run)
            else:
                run = 1
        report.add(
            self.domain,
            "R3.dedup",
            CheckStatus.PASS if duplicate_count == 0 else CheckStatus.FAIL,
            f"重複={duplicate_count}；最長同點連發 x{longest_run}",
        )

    def _check_curve_single_flight(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        active_id = None
        overlaps = []
        started = 0
        for line in session.lines:
            message = line.message
            if message.startswith("RV curves paths "):
                current = grab_id(message)
                started += 1
                if active_id is not None:
                    overlaps.append(f"{active_id}->{current}")
                active_id = current
            elif message.startswith("RV curves stale-drop "):
                if grab_id(message) == active_id:
                    active_id = None
            elif message.startswith("RV curves ") and "paths" not in message:
                if grab_id(message) == active_id:
                    active_id = None
        if started == 0:
            report.add(self.domain, "R2.single-flight", CheckStatus.NOT_COVERED, "無曲線讀取")
            return
        report.add(
            self.domain,
            "R2.single-flight",
            CheckStatus.PASS if not overlaps else CheckStatus.FAIL,
            f"啟動={started}；重疊={len(overlaps)}"
            + (f"；首例 {overlaps[0]}" if overlaps else ""),
        )

    def _check_period_single_flight(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        active = None
        starts = 0
        overlaps = []
        open_periods = []
        for line in session.lines:
            message = line.message
            if message.startswith("RV period begin "):
                period = message[len("RV period begin "):]
                starts += 1
                if active is not None:
                    overlaps.append(f"{active}->{period}")
                active = period
                open_periods.append(period)
            elif message.startswith(("RV period done ", "RV period stale-drop ")):
                prefix = "RV period done " if message.startswith("RV period done ") else "RV period stale-drop "
                period = message[len(prefix):]
                if period in open_periods:
                    open_periods.remove(period)
                if active == period:
                    active = None

        if starts == 0:
            report.add(self.domain, "R3.single-flight", CheckStatus.NOT_COVERED, "無新版 period begin/done 儀器")
            return
        report.add(
            self.domain,
            "R3.single-flight",
            CheckStatus.PASS if not overlaps and not open_periods else CheckStatus.FAIL,
            f"啟動={starts}；重疊={len(overlaps)}；未結束={open_periods or 0}"
            + (f"；首例 {overlaps[0]}" if overlaps else ""),
        )

    def _check_direction(
        self, session: FlowSession, report: CheckReport, tolerance_mm: float = 15.0
    ) -> None:
        bad = []
        checked = 0
        for line in session.lines:
            message = line.message
            if " rowChart dir=" not in message or "dataChart" not in message:
                continue
            match = ROW_RE.search(message)
            if not match:
                continue
            checked += 1
            direction = match.group("dir")
            total = float(match.group("total"))
            phys_low, phys_high = float(match.group("p0")), float(match.group("p1"))
            chart_low, chart_high = float(match.group("c0")), float(match.group("c1"))
            effective_tolerance = max(tolerance_mm, abs(total) * 0.001)
            if direction == "TopToBottom":
                expected_low, expected_high = total - phys_high, total - phys_low
            else:
                expected_low, expected_high = phys_low, phys_high
            if (
                abs(chart_low - expected_low) > effective_tolerance
                or abs(chart_high - expected_high) > effective_tolerance
            ):
                bad.append(message[:100])

        if checked == 0:
            report.add(self.domain, "R2.direction", CheckStatus.NOT_COVERED, "無 rowChart 快照")
            return
        report.add(
            self.domain,
            "R2.direction",
            CheckStatus.PASS if not bad else CheckStatus.FAIL,
            f"檢查={checked}；違規={len(bad)}" + (f"；首例 {bad[0]}" if bad else ""),
        )

    def _check_drag_first_publish(self, session: FlowSession, report: CheckReport) -> None:
        starts = 0
        active = None
        failures = []
        for line in session.lines:
            message = line.message
            if message == "RV drag(start)":
                starts += 1
                active = False
            elif message == "RV drag(view-published)" and active is False:
                active = True
            elif message.startswith("RV viewEdges") and active is not None:
                if active is not True:
                    failures.append(f"{line.elapsed:.3f}s")
                active = None
        if active is False:
            failures.append("未結束")
        if starts == 0:
            report.add(self.domain, "R4.first-view", CheckStatus.NOT_COVERED, "無回顧主畫面拖曳")
            return
        report.add(
            self.domain,
            "R4.first-view",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"拖曳={starts}；首位移未發布={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
