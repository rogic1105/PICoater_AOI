"""Mura alarm (M-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class MuraFlowValidator:
    domain = "MURA"

    _exceed_pattern = re.compile(
        r"^⚠ MURA 超標（(?P<axis>[vh])）"
        r"mean=(?P<mean>-?\d+(?:\.\d+)?)/max=(?P<max>-?\d+(?:\.\d+)?)"
        r"（thr (?P<th_mean>-?\d+(?:\.\d+)?)/(?P<th_max>-?\d+(?:\.\d+)?)，"
        r"(?P<io>IO已連線|IO未連線→僅畫面警告|IO暫停中→僅畫面警告)）$"
    )
    _recover_pattern = re.compile(r"^MURA 恢復（(?P<axis>[vh])）$")
    _health_pattern = re.compile(
        r"^\[OutputHealth\] (?P<action>raise|resolve) "
        r"code=MuraExceed\.(?P<axis>[vh])\b"
    )
    _pause_setting_pattern = re.compile(
        r"^set:\[MuraDetectPaused\]=(?P<paused>True|False)$"
    )

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        covered = any(
            line.message.startswith(
                (
                    "⚠ MURA 超標",
                    "MURA 恢復",
                    "ui:【暫停Mura檢測】",
                    "set:[MuraDetectPaused]",
                    "[OutputHealth] raise code=MuraExceed.",
                    "[OutputHealth] resolve code=MuraExceed.",
                )
            )
            for line in session.lines
        )
        if not covered:
            report.add(self.domain, "M0", CheckStatus.NOT_COVERED, "本 session 無 MURA 邊緣或暫停操作")
            return report

        self._check_edges(session, report)
        self._check_health_pairs(session, report)
        self._check_pause(session, report)
        return report

    def _check_edges(self, session: FlowSession, report: CheckReport) -> None:
        active = {"v": False, "h": False}
        edge_count = 0
        failures = []
        for line in session.lines:
            message = line.message
            if (
                message.startswith(("StartGrab", "StopGrab"))
                or self._pause_setting_pattern.match(message)
            ):
                active["v"] = active["h"] = False
                continue

            exceed = self._exceed_pattern.match(message)
            if exceed:
                edge_count += 1
                axis = exceed.group("axis")
                if active[axis]:
                    failures.append(f"{line.timestamp} {axis} 重複超標")
                active[axis] = True
                continue

            recover = self._recover_pattern.match(message)
            if recover:
                edge_count += 1
                axis = recover.group("axis")
                if not active[axis]:
                    failures.append(f"{line.timestamp} {axis} 未超標卻恢復")
                active[axis] = False

        if edge_count == 0:
            report.add(self.domain, "M1.edges", CheckStatus.NOT_COVERED, "只有暫停操作，無超標/恢復邊緣")
            return
        report.add(
            self.domain,
            "M1.edges",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={edge_count} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_health_pairs(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        edges = []
        for index, line in enumerate(session.lines):
            exceed = self._exceed_pattern.match(line.message)
            recover = self._recover_pattern.match(line.message)
            if exceed:
                edges.append((index, line, "raise", exceed.group("axis")))
            elif recover:
                edges.append((index, line, "resolve", recover.group("axis")))

        if not edges:
            report.add(
                self.domain,
                "M1.health",
                CheckStatus.NOT_COVERED,
                "無 MURA 邊緣可核對 OutputHealth",
            )
            return

        health_lines = [
            (index, line, self._health_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if self._health_pattern.match(line.message)
        ]
        if not health_lines:
            report.add(
                self.domain,
                "M1.health",
                CheckStatus.NOT_COVERED,
                "舊版 log 尚無 MuraExceed OutputHealth 儀器",
            )
            return

        failures = []
        consumed = set()
        for index, line, expected_action, axis in edges:
            matched = False
            for candidate_index in range(index + 1, len(session.lines)):
                if candidate_index in consumed:
                    continue
                candidate = session.lines[candidate_index]
                if candidate.elapsed - line.elapsed > 1.0:
                    break
                health = self._health_pattern.match(candidate.message)
                if (
                    health
                    and health.group("action") == expected_action
                    and health.group("axis") == axis
                ):
                    matched = True
                    consumed.add(candidate_index)
                    break
            if not matched:
                failures.append(
                    f"{line.timestamp} {axis} 缺 OutputHealth {expected_action}"
                )

        report.add(
            self.domain,
            "M1.health",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={len(edges)} unmatched={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_pause(self, session: FlowSession, report: CheckReport) -> None:
        clicks = [
            (index, line)
            for index, line in enumerate(session.lines)
            if line.message == "ui:【暫停Mura檢測】鈕"
        ]
        if not clicks:
            report.add(self.domain, "M1.pause", CheckStatus.NOT_COVERED, "未操作 MURA 暫停")
            return

        failures = []
        paused = None
        for sequence, (index, line) in enumerate(clicks):
            next_click = (
                clicks[sequence + 1][0]
                if sequence + 1 < len(clicks)
                else len(session.lines)
            )
            window = session.lines[index + 1:next_click]
            setting = next(
                (
                    (item, match)
                    for item in window
                    for match in [self._pause_setting_pattern.match(item.message)]
                    if match and item.elapsed - line.elapsed <= 3.0
                ),
                None,
            )
            if setting is None:
                failures.append(f"{line.timestamp} 缺 set:[MuraDetectPaused]")
                continue

            setting_line, match = setting
            next_paused = match.group("paused") == "True"
            if paused is not None and next_paused == paused:
                failures.append(
                    f"{setting_line.timestamp} 暫停值未切換（{next_paused}）"
                )
            paused = next_paused
            if paused and not any(
                item.message == "MURA 暫停 → 清除 DO1"
                and 0 <= item.elapsed - setting_line.elapsed <= 3.0
                for item in window
            ):
                failures.append(f"{setting_line.timestamp} 暫停後缺清除 DO1")

        report.add(
            self.domain,
            "M1.pause",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"clicks={len(clicks)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
