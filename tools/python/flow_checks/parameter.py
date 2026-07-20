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
    _complete_pattern = re.compile(
        r"^parameter reconfigure complete scope=(?P<scope>All|cam[1-7]) "
        r"gate=open warm=True$"
    )
    _fast_begin_pattern = re.compile(
        r"^exposure live apply begin scope=(?P<scope>All|cam[1-7]) gate=open$"
    )
    _fast_complete_pattern = re.compile(
        r"^exposure live apply complete scope=(?P<scope>All|cam[1-7]) "
        r"gate=(?P<gate>open|closed) elapsedMs=(?P<elapsed>\d+)$"
    )
    _fast_failed_pattern = re.compile(
        r"^exposure live apply failed scope=(?P<scope>All|cam[1-7]) "
        r"gate=(?P<gate>open|closed) error=(?P<error>\w+)$"
    )
    _blocked_pattern = re.compile(
        r"^parameter change blocked scope=(?P<scope>All|cam[1-7]) "
        r"param=(?P<parameter>LineRate|LineRateAll|Height|HeightAll) "
        r"reason=GrabActive$"
    )

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        intents = [
            (index, line, self._intent_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【相機參數】")
        ]

        self._check_startup(session, intents, report)
        self._check_live_policy(session, report)
        if not intents:
            report.add(self.domain, "P1.intent", CheckStatus.NOT_COVERED, "本 session 無使用者調整相機參數")
            report.add(self.domain, "P1.responsiveness", CheckStatus.NOT_COVERED, "無調參操作可量測")
            report.add(self.domain, "P1.synchronization", CheckStatus.NOT_COVERED, "無調參操作可量測")
            return report

        self._check_intent_format(intents, report)
        self._check_responsiveness(session, intents, report)
        self._check_synchronization(session, intents, report)
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

    def _check_live_policy(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        capturing = False
        covered = 0
        forbidden = []
        blocked = 0

        for line in session.lines:
            message = line.message
            if message.startswith("capture gate open "):
                capturing = True
            elif message == "StopGrab":
                capturing = False
            elif self._complete_pattern.match(message):
                capturing = True

            intent = self._intent_pattern.match(message)
            if capturing and intent:
                covered += 1
                if intent.group("parameter") != "Exp":
                    forbidden.append(f"{line.timestamp} {message}")

            if self._blocked_pattern.match(message):
                covered += 1
                blocked += 1

        if covered == 0:
            report.add(
                self.domain,
                "P1.live-policy",
                CheckStatus.NOT_COVERED,
                "本 session 未在 Grab 中操作相機參數",
            )
            return

        report.add(
            self.domain,
            "P1.live-policy",
            CheckStatus.PASS if not forbidden else CheckStatus.FAIL,
            f"covered={covered} blocked={blocked} forbiddenIntents={len(forbidden)}"
            + (f"；首例 {forbidden[0]}" if forbidden else ""),
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

    def _check_synchronization(
        self, session: FlowSession, intents, report: CheckReport
    ) -> None:
        capturing = False
        live_intents = []
        for index, line in enumerate(session.lines):
            message = line.message
            if message.startswith("capture gate open "):
                capturing = True
            elif message == "StopGrab":
                capturing = False
            elif self._complete_pattern.match(message):
                capturing = True

            if (
                capturing
                and message.startswith("ui:【相機參數】")
                and self._intent_pattern.match(message)
                and self._intent_pattern.match(message).group("parameter") == "Exp"
            ):
                live_intents.append((index, line, self._intent_pattern.match(message)))

        if not live_intents:
            report.add(
                self.domain,
                "P1.synchronization",
                CheckStatus.NOT_COVERED,
                "沒有在 capture gate 開啟期間調參",
            )
            return

        failures = []
        completed = 0
        interrupted = 0
        for intent_index, intent_line, intent_match in live_intents:
            scope = intent_match.group("scope")
            begin_index = next(
                (
                    index
                    for index in range(intent_index + 1, len(session.lines))
                    if self._fast_begin_pattern.match(session.lines[index].message)
                ),
                None,
            )
            if begin_index is None:
                failures.append(f"{intent_line.timestamp} {scope} 缺 exposure fast begin")
                continue

            begin_match = self._fast_begin_pattern.match(
                session.lines[begin_index].message
            )
            if begin_match.group("scope") != scope:
                failures.append(
                    f"{intent_line.timestamp} scope={scope} 但 begin={begin_match.group('scope')}"
                )
                continue

            terminal_index = next(
                (
                    index
                    for index in range(begin_index + 1, len(session.lines))
                    if session.lines[index].message.startswith(
                        (
                            "exposure live apply complete ",
                            "exposure live apply failed ",
                        )
                    )
                ),
                None,
            )
            if terminal_index is None:
                failures.append(f"{intent_line.timestamp} {scope} 缺 complete/failed")
                continue

            segment = session.lines[begin_index : terminal_index + 1]
            terminal = session.lines[terminal_index]
            complete_match = self._fast_complete_pattern.match(terminal.message)
            failed_match = self._fast_failed_pattern.match(terminal.message)
            terminal_match = complete_match or failed_match
            if not terminal_match:
                failures.append(
                    f"{intent_line.timestamp} {scope} 未恢復：{terminal.message}"
                )
                continue
            if terminal_match.group("scope") != scope:
                failures.append(
                    f"{intent_line.timestamp} scope={scope} "
                    f"但 terminal={terminal_match.group('scope')}"
                )
                continue

            stopped = any(line.message == "StopGrab" for line in segment)
            forbidden = [
                line.message
                for line in segment
                if line.message.startswith(
                    (
                        "parameter reconfigure ",
                        "acquisition sync begin reason=parameter:",
                        "parameter sequence reset ",
                    )
                )
                or (
                    not stopped
                    and line.message.startswith("capture gate closed ")
                )
            ]
            if forbidden:
                failures.append(
                    f"{intent_line.timestamp} {scope} fast path 出現重配置：{forbidden[0]}"
                )
                continue
            if failed_match:
                failures.append(
                    f"{intent_line.timestamp} {scope} 套用失敗：{terminal.message}"
                )
                continue
            if complete_match.group("gate") != ("closed" if stopped else "open"):
                failures.append(
                    f"{intent_line.timestamp} {scope} gate 狀態與 StopGrab 不一致"
                )
                continue
            elapsed_ms = int(complete_match.group("elapsed"))
            if elapsed_ms > 5000:
                failures.append(
                    f"{intent_line.timestamp} {scope} exposure apply={elapsed_ms}ms > 5000ms"
                )
                continue

            completed += 1
            if stopped:
                interrupted += 1

        report.add(
            self.domain,
            "P1.synchronization",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"liveIntents={len(live_intents)} completed={completed} "
            f"interrupted={interrupted} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
