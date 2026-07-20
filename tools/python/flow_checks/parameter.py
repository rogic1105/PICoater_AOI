"""Camera-parameter (P-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class ParameterFlowValidator:
    domain = "PARAM"

    _intent_pattern = re.compile(
        r"^ui:【相機參數】(?P<scope>All|cam[1-7]) "
        r"(?P<parameter>Exp|LineRate|Height)(?P<all>All)?=(?P<value>\d+)$"
    )
    _stall_pattern = re.compile(r"^\[UiStall\]\s+(?P<ms>\d+)ms")

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        intents = [
            (index, line, self._intent_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【相機參數】")
        ]

        self._check_startup(session, intents, report)
        if not intents:
            report.add(self.domain, "P1.intent", CheckStatus.NOT_COVERED, "本 session 無使用者調整相機參數")
            report.add(self.domain, "P1.responsiveness", CheckStatus.NOT_COVERED, "無調參操作可量測")
            return report

        self._check_intent_format(intents, report)
        self._check_responsiveness(session, intents, report)
        return report

    def _check_startup(
        self, session: FlowSession, intents, report: CheckReport
    ) -> None:
        allocation_begin = next(
            (
                index
                for index, line in enumerate(session.lines)
                if line.message.startswith("AllocateCameras begin")
            ),
            None,
        )
        allocation_done = next(
            (
                index
                for index, line in enumerate(session.lines)
                if line.message.startswith("AllocateCameras done")
            ),
            None,
        )
        if allocation_begin is None:
            report.add(
                self.domain,
                "P1.startup",
                CheckStatus.NOT_COVERED,
                "本 session 無相機配置窗口",
            )
            return

        if allocation_done is None:
            boundary_elapsed = float("inf")
        else:
            # Initial control values used to arm a 1-second debounce which fired
            # just after AllocateCameras done. Keep that tail inside the startup
            # quiet window so the historical regression remains machine-visible.
            boundary_elapsed = session.lines[allocation_done].elapsed + 1.0
        leaked = [
            line
            for _, line, _ in intents
            if line.elapsed <= boundary_elapsed
        ]
        report.add(
            self.domain,
            "P1.startup",
            CheckStatus.PASS if not leaked else CheckStatus.FAIL,
            f"initIntents={len(leaked)}"
            + (f"；首例 {leaked[0].timestamp} {leaked[0].message}" if leaked else ""),
        )

    def _check_intent_format(self, intents, report: CheckReport) -> None:
        failures = []
        for _, line, match in intents:
            if not match:
                failures.append(f"{line.timestamp} {line.message}")
                continue
            all_scope = match.group("scope") == "All"
            all_suffix = match.group("all") == "All"
            if all_scope != all_suffix:
                failures.append(f"{line.timestamp} scope/param 不一致")

        report.add(
            self.domain,
            "P1.intent",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"intents={len(intents)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_responsiveness(
        self, session: FlowSession, intents, report: CheckReport
    ) -> None:
        failures = []
        worst = 0
        for index, intent, _ in intents:
            for line in session.lines[index + 1:]:
                if line.elapsed - intent.elapsed > 5.0:
                    break
                stall = self._stall_pattern.match(line.message)
                if not stall:
                    continue
                duration = int(stall.group("ms"))
                worst = max(worst, duration)
                if duration > 1000:
                    failures.append(
                        f"{intent.timestamp} 後 UiStall={duration}ms"
                    )

        report.add(
            self.domain,
            "P1.responsiveness",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"intents={len(intents)} worstStall={worst}ms failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
