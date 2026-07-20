"""Settings (S-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession, grab_id


class SettingsFlowValidator:
    domain = "SETTINGS"

    _setting_pattern = re.compile(
        r"^(?P<source>ui:設定|set:)\[(?P<name>[^\]]+)\]"
        r"(?:(?:=(?P<value>.+))|(?: → (?P<arrow>.+)))$"
    )
    _row_pattern = re.compile(
        r"^(?:LC|RV) row (?:rowChart|rowView) "
        r"dir=(?P<direction>TopToBottom|BottomToTop)\b"
    )

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        settings = [
            (index, line, self._setting_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith(("ui:設定[", "set:["))
            and not line.message.startswith("set:[顯示基線]")
        ]
        if not settings:
            report.add(self.domain, "S0", CheckStatus.NOT_COVERED, "本 session 無設定變更")
            return report

        self._check_format(settings, report)
        self._check_review_enhance(session, settings, report)
        self._check_direction_refresh(session, settings, report)
        return report

    def _check_format(self, settings, report: CheckReport) -> None:
        failures = []
        for _, line, match in settings:
            if not match:
                failures.append(f"{line.timestamp} {line.message}")
                continue
            name = match.group("name")
            value = match.group("value") or match.group("arrow")
            if not name.strip() or not value.strip():
                failures.append(f"{line.timestamp} 屬性名或值為空")

        report.add(
            self.domain,
            "S0.format",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(settings)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_review_enhance(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hd_EnableReviewEnhance"
        ]
        if not changes:
            report.add(
                self.domain,
                "S2.review-enhance",
                CheckStatus.NOT_COVERED,
                "未切換回顧強化",
            )
            return

        failures = []
        exercised = 0
        for sequence, (index, line, _) in enumerate(changes):
            previous_done = next(
                (
                    item
                    for item in reversed(session.lines[:index])
                    if item.message.startswith("RV loadGrab done ")
                ),
                None,
            )
            if previous_done is None:
                continue

            current_id = grab_id(previous_done.message)
            next_change = (
                changes[sequence + 1][0]
                if sequence + 1 < len(changes)
                else len(session.lines)
            )
            window = session.lines[index + 1:next_change]
            begin_pos = next(
                (
                    pos
                    for pos, item in enumerate(window)
                    if item.message.startswith("RV loadGrab begin ")
                    and grab_id(item.message) == current_id
                ),
                None,
            )
            exercised += 1
            if begin_pos is None:
                failures.append(
                    f"{line.timestamp} grab={current_id} 缺 RV loadGrab begin"
                )
                continue
            if not any(
                item.message.startswith("RV loadGrab done ")
                and grab_id(item.message) == current_id
                for item in window[begin_pos + 1:]
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 缺 RV loadGrab done"
                )

        if exercised == 0:
            report.add(
                self.domain,
                "S2.review-enhance",
                CheckStatus.NOT_COVERED,
                f"changes={len(changes)}；當時尚無已載入回顧序號",
            )
            return
        report.add(
            self.domain,
            "S2.review-enhance",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} exercised={exercised} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_direction_refresh(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hee_VerticalDirection"
        ]
        if not changes:
            report.add(
                self.domain,
                "S3.direction",
                CheckStatus.NOT_COVERED,
                "未切換上下方向",
            )
            return

        failures = []
        for sequence, (index, line, match) in enumerate(changes):
            expected = match.group("value") or match.group("arrow")
            next_change = (
                changes[sequence + 1][0]
                if sequence + 1 < len(changes)
                else len(session.lines)
            )
            window = session.lines[index + 1:next_change]
            row_directions = [
                row_match.group("direction")
                for item in window
                for row_match in [self._row_pattern.match(item.message)]
                if row_match
            ]
            if expected not in row_directions:
                found = ",".join(sorted(set(row_directions))) or "none"
                failures.append(
                    f"{line.timestamp} expected={expected} rowDirections={found}"
                )

        report.add(
            self.domain,
            "S3.direction",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
