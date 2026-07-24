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
    _hardware_applied_pattern = re.compile(
        r"^parameter hardware applied scope=(?P<scope>cam[1-7]) "
        r"param=(?P<parameter>LineRate|Height) "
        r"requested=(?P<requested>[0-9.]+) applied=(?P<applied>[0-9.]+)$"
    )
    _ui_apply_complete_pattern = re.compile(
        r"^parameter ui apply complete scope=(?P<scope>All|cam[1-7]) "
        r"param=(?P<parameter>LineRate|LineRateAll|Height|HeightAll) "
        r"value=(?P<value>[0-9.]+)$"
    )
    _ui_apply_failed_pattern = re.compile(
        r"^parameter ui apply failed scope=(?P<scope>All|cam[1-7]) "
        r"param=(?P<parameter>LineRate|LineRateAll|Height|HeightAll) "
        r"value=(?P<value>[0-9.]+)(?: error=\w+)?$"
    )
    _queue_flush_begin_pattern = re.compile(
        r"^parameter queue flush begin reason=GrabStart pending=(?P<pending>\d+)$"
    )
    _queue_flush_complete_pattern = re.compile(
        r"^parameter queue flush complete reason=GrabStart success=(?P<success>True|False)$"
    )
    _start_lock_pattern = re.compile(
        r"^parameter controls lock reason=GrabStart state=(?P<state>on|off)$"
    )
    _deferred_timing_pattern = re.compile(
        r"^parameter queue deferred scope=(?P<scope>All|cam[1-7]) "
        r"param=(?P<parameter>LineRate|LineRateAll|Height|HeightAll) "
        r"until=GrabStop value=(?P<value>\d+)$"
    )
    _post_stop_begin_pattern = re.compile(
        r"^parameter post-stop apply begin pending=(?P<pending>\d+)$"
    )
    _post_stop_complete_pattern = re.compile(
        r"^parameter post-stop apply complete success=(?P<success>True|False) "
        r"bindings=(?P<bindings>\d+)$"
    )
    _reconfigure_begin_pattern = re.compile(
        r"^parameter reconfigure begin scope=(?P<scope>All|cam[1-7]) "
        r"gate=closed targets=(?P<targets>\d+)$"
    )
    _reconfigure_complete_closed_pattern = re.compile(
        r"^parameter reconfigure complete scope=(?P<scope>All|cam[1-7]) "
        r"gate=closed warm=True$"
    )
    _frame_period_pattern = re.compile(
        r"^acquisition sync rate reason=parameter:(?P<scope>All|cam[1-7]) "
        r"attempt=(?P<attempt>\d+) cam(?P<camera>[1-7]) "
        r"expectedMs=(?P<expected>[0-9.]+) actualMs=(?P<actual>[0-9.]+) "
        r"toleranceMs=(?P<tolerance>[0-9.]+) aligned=(?P<aligned>True|False)$"
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
        self._check_queue_before_grab(session, report)
        self._check_deferred_timing(session, report)
        self._check_frame_period_evidence(session, report)
        self._check_start_transition(session, report)
        if not intents:
            report.add(self.domain, "P1.intent", CheckStatus.NOT_COVERED, "本 session 無使用者調整相機參數")
            report.add(self.domain, "P1.responsiveness", CheckStatus.NOT_COVERED, "無調參操作可量測")
            report.add(self.domain, "P1.synchronization", CheckStatus.NOT_COVERED, "無調參操作可量測")
            report.add(self.domain, "P1.applied-truth", CheckStatus.NOT_COVERED, "無調參操作可量測")
            return report

        self._check_intent_format(intents, report)
        self._check_responsiveness(session, intents, report)
        self._check_synchronization(session, intents, report)
        self._check_applied_truth(session, intents, report)
        return report

    def _check_queue_before_grab(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        begins = [
            (index, line, self._queue_flush_begin_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if self._queue_flush_begin_pattern.match(line.message)
        ]
        if not begins:
            report.add(
                self.domain,
                "P1.queue-before-grab",
                CheckStatus.NOT_COVERED,
                "本 session 的 Grab Start 前沒有待套用參數",
            )
            return

        failures = []
        for index, line, begin in begins:
            terminal = next(
                (
                    candidate
                    for candidate in session.lines[index + 1 :]
                    if self._queue_flush_complete_pattern.match(candidate.message)
                    or candidate.message.startswith("StartGrab")
                ),
                None,
            )
            if terminal is None:
                failures.append(f"{line.timestamp} flush 無 terminal")
                continue
            complete = self._queue_flush_complete_pattern.match(terminal.message)
            if not complete:
                failures.append(f"{line.timestamp} StartGrab 超車 parameter flush")
                continue
            if complete.group("success") != "True":
                failures.append(f"{line.timestamp} parameter flush 失敗")

        report.add(
            self.domain,
            "P1.queue-before-grab",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"flushes={len(begins)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_deferred_timing(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        deferred = [
            (index, line)
            for index, line in enumerate(session.lines)
            if self._deferred_timing_pattern.match(line.message)
        ]
        begins = [
            (index, line)
            for index, line in enumerate(session.lines)
            if self._post_stop_begin_pattern.match(line.message)
        ]
        if not deferred and not begins:
            report.add(
                self.domain,
                "P1.deferred-timing",
                CheckStatus.NOT_COVERED,
                "本 session 未在 Grab 中調整線掃或高度",
            )
            return

        failures = []
        for index, line in deferred:
            stop_index = next(
                (
                    candidate
                    for candidate in range(index + 1, len(session.lines))
                    if session.lines[candidate].message == "StopGrab"
                ),
                None,
            )
            if stop_index is None:
                continue
            begin_index = next(
                (
                    candidate
                    for candidate in range(stop_index + 1, len(session.lines))
                    if self._post_stop_begin_pattern.match(
                        session.lines[candidate].message
                    )
                    or session.lines[candidate].message.startswith("StartGrab")
                ),
                None,
            )
            if begin_index is None or not self._post_stop_begin_pattern.match(
                session.lines[begin_index].message
            ):
                failures.append(f"{line.timestamp} StopGrab 後未開始套用 timing 參數")

        for index, line in begins:
            terminal_index = next(
                (
                    candidate
                    for candidate in range(index + 1, len(session.lines))
                    if self._post_stop_complete_pattern.match(
                        session.lines[candidate].message
                    )
                    or session.lines[candidate].message.startswith("StartGrab")
                ),
                None,
            )
            if terminal_index is None:
                failures.append(f"{line.timestamp} post-stop apply 沒有 terminal")
                continue
            terminal = session.lines[terminal_index]
            complete = self._post_stop_complete_pattern.match(terminal.message)
            if complete is None:
                failures.append(f"{line.timestamp} StartGrab 超車 timing 參數套用")
                continue
            if complete.group("success") != "True":
                failures.append(f"{line.timestamp} post-stop timing 參數套用失敗")

            segment = session.lines[index + 1 : terminal_index]
            io_starts = [
                candidate
                for candidate in segment
                if candidate.message == "io:DI START 上升緣 → 開始抓取"
            ]
            if io_starts and not any(
                candidate.message
                == "IO grab rejected busy=off reason=timing-parameter-busy"
                for candidate in segment
            ):
                failures.append(f"{line.timestamp} timing 套用期間 IO High 未被拒絕")

        report.add(
            self.domain,
            "P1.deferred-timing",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"deferred={len(deferred)} applies={len(begins)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_frame_period_evidence(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        begins = [
            (index, line, match)
            for index, line in enumerate(session.lines)
            if (match := self._reconfigure_begin_pattern.match(line.message))
        ]
        if not begins:
            report.add(
                self.domain,
                "P1.frame-period",
                CheckStatus.NOT_COVERED,
                "本 session 沒有停止狀態 timing 參數重配",
            )
            return

        failures = []
        verified = 0
        for begin_index, begin_line, begin in begins:
            terminal_index = next(
                (
                    index
                    for index in range(begin_index + 1, len(session.lines))
                    if self._reconfigure_complete_closed_pattern.match(
                        session.lines[index].message
                    )
                    or session.lines[index].message.startswith(
                        (
                            "parameter reconfigure failed ",
                            "parameter reconfigure canceled ",
                        )
                    )
                    or self._reconfigure_begin_pattern.match(
                        session.lines[index].message
                    )
                ),
                None,
            )
            if terminal_index is None:
                failures.append(f"{begin_line.timestamp} timing 重配沒有 terminal")
                continue

            terminal = session.lines[terminal_index]
            if not self._reconfigure_complete_closed_pattern.match(terminal.message):
                continue

            latest_by_camera = {}
            for line in session.lines[begin_index + 1 : terminal_index]:
                match = self._frame_period_pattern.match(line.message)
                if match and match.group("scope") == begin.group("scope"):
                    latest_by_camera[int(match.group("camera"))] = match

            target_count = int(begin.group("targets"))
            if len(latest_by_camera) != target_count:
                failures.append(
                    f"{begin_line.timestamp} timing 重配缺實際幀週期 "
                    f"expectedCams={target_count} measured={len(latest_by_camera)}"
                )
                continue

            mismatched = [
                camera
                for camera, match in latest_by_camera.items()
                if match.group("aligned") != "True"
            ]
            if mismatched:
                failures.append(
                    f"{begin_line.timestamp} 實際幀週期不符 cams="
                    + ",".join(str(camera) for camera in sorted(mismatched))
                )
                continue
            verified += 1

        report.add(
            self.domain,
            "P1.frame-period",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"reconfigures={len(begins)} verified={verified} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_applied_truth(
        self, session: FlowSession, intents, report: CheckReport
    ) -> None:
        relevant = [
            (index, line, match)
            for index, line, match in intents
            if match and match.group("parameter") in ("LineRate", "Height")
        ]
        if not relevant:
            report.add(
                self.domain,
                "P1.applied-truth",
                CheckStatus.NOT_COVERED,
                "本 session 無線掃或高度調整",
            )
            return

        failures = []
        verified = 0
        used_terminals = set()
        used_hardware = set()

        for intent_index, intent_line, intent in relevant:
            scope = intent.group("scope")
            parameter = intent.group("parameter")
            requested = float(intent.group("value"))
            tolerance = max(5.0, requested * 0.02) if parameter == "LineRate" else 0.0

            terminal = None
            terminal_index = None
            for index in range(intent_index + 1, len(session.lines)):
                if index in used_terminals:
                    continue
                message = session.lines[index].message
                complete = self._ui_apply_complete_pattern.match(message)
                failed = self._ui_apply_failed_pattern.match(message)
                candidate = complete or failed
                if candidate is None:
                    continue
                candidate_parameter = candidate.group("parameter").replace("All", "")
                if candidate.group("scope") != scope or candidate_parameter != parameter:
                    continue
                if abs(float(candidate.group("value")) - requested) > tolerance:
                    continue
                terminal = (complete is not None, session.lines[index])
                terminal_index = index
                break

            if terminal is not None:
                used_terminals.add(terminal_index)
                if terminal[0]:
                    verified += 1
                else:
                    failures.append(
                        f"{intent_line.timestamp} {parameter} 套用失敗：{terminal[1].message}"
                    )
                continue

            # Older traces do not have the UI terminal line. Match the measured hardware
            # value by parameter and request instead of stopping at an unrelated intent.
            matching_hardware = None
            mismatch_hardware = None
            blocked_line = None
            for index in range(intent_index + 1, len(session.lines)):
                message = session.lines[index].message
                blocked = self._blocked_pattern.match(message)
                if blocked is not None:
                    blocked_parameter = blocked.group("parameter").replace("All", "")
                    if blocked.group("scope") == scope and blocked_parameter == parameter:
                        blocked_line = session.lines[index]
                        break

                applied = self._hardware_applied_pattern.match(message)
                if applied is None or applied.group("parameter") != parameter:
                    continue
                if scope != "All" and applied.group("scope") != scope:
                    continue
                if index in used_hardware:
                    continue
                logged_requested = float(applied.group("requested"))
                actual = float(applied.group("applied"))
                if abs(logged_requested - requested) <= tolerance:
                    if abs(actual - requested) <= tolerance:
                        matching_hardware = (index, applied)
                        break
                    if mismatch_hardware is None:
                        mismatch_hardware = applied

            if blocked_line is not None:
                failures.append(
                    f"{intent_line.timestamp} {parameter} 套用失敗：{blocked_line.message}"
                )
            elif matching_hardware is not None:
                used_hardware.add(matching_hardware[0])
                verified += 1
            elif mismatch_hardware is not None:
                failures.append(
                    f"{intent_line.timestamp} {parameter} 不一致："
                    f"{mismatch_hardware.group('scope')} requested={requested:g} "
                    f"applied={float(mismatch_hardware.group('applied')):g}"
                )
            else:
                failures.append(
                    f"{intent_line.timestamp} {parameter} 缺 hardware applied"
                )

        report.add(
            self.domain,
            "P1.applied-truth",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"intents={len(relevant)} verified={verified} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_start_transition(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        locks = [
            (index, line, self._start_lock_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if self._start_lock_pattern.match(line.message)
        ]
        failures = []

        if locks:
            for index, line, match in locks:
                if match.group("state") != "on":
                    continue
                off_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, len(session.lines))
                        if session.lines[candidate].message
                        == "parameter controls lock reason=GrabStart state=off"
                    ),
                    None,
                )
                if off_index is None:
                    failures.append(f"{line.timestamp} start lock 未釋放")
                    continue
                sync_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, off_index)
                        if session.lines[candidate].message.startswith(
                            "acquisition sync begin reason=start "
                        )
                    ),
                    None,
                )
                if sync_index is None:
                    continue
                timing_intents = [
                    candidate
                    for candidate in session.lines[sync_index + 1 : off_index]
                    if (
                        (intent := self._intent_pattern.match(candidate.message))
                        and intent.group("parameter") in ("LineRate", "Height")
                    )
                ]
                if timing_intents:
                    failures.append(
                        f"{line.timestamp} 啟動期間仍接受 {timing_intents[0].message}"
                    )
        else:
            # Compatibility probe for traces from before the explicit start lock existed.
            for index, line in enumerate(session.lines):
                if not line.message.startswith("acquisition sync begin reason=start "):
                    continue
                start_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, len(session.lines))
                        if session.lines[candidate].message.startswith("StartGrab")
                    ),
                    None,
                )
                if start_index is None:
                    continue
                timing_intent = next(
                    (
                        candidate
                        for candidate in session.lines[index + 1 : start_index]
                        if (
                            (intent := self._intent_pattern.match(candidate.message))
                            and intent.group("parameter") in ("LineRate", "Height")
                        )
                    ),
                    None,
                )
                if timing_intent is not None:
                    failures.append(
                        f"{line.timestamp} 啟動同步期間接受 {timing_intent.message}"
                    )

        report.add(
            self.domain,
            "P1.start-transition",
            (CheckStatus.PASS if locks and not failures else
             CheckStatus.FAIL if failures else CheckStatus.NOT_COVERED),
            f"locks={sum(1 for _, _, match in locks if match.group('state') == 'on')} "
            f"failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

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
        deferred = 0

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

            if capturing and self._deferred_timing_pattern.match(message):
                covered += 1
                deferred += 1

            if capturing and self._hardware_applied_pattern.match(message):
                applied = self._hardware_applied_pattern.match(message)
                if applied.group("parameter") in ("LineRate", "Height"):
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
            f"covered={covered} deferred={deferred} blocked={blocked} "
            f"forbidden={len(forbidden)}"
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
