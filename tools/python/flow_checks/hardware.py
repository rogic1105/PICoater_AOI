"""Hardware and external-service (H-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class HardwareFlowValidator:
    domain = "HARDWARE"

    _edge_pattern = re.compile(
        r"^(?:⚠ )?(IO|光源|儲存分享) (?:未連線（開機基線）|斷線|恢復連線)$"
    )
    _camera_pattern = re.compile(r"^(?:⚠ 相機離線|相機在線) \d+→\d+/\d+$")

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        covered = any(
            self._edge_pattern.match(line.message)
            or self._camera_pattern.match(line.message)
            or line.message.startswith(("儲存程式 heartbeat ", "⚠ 儲存程式 heartbeat "))
            or line.message.startswith("[RemoteCopy] ")
            or line.message.startswith(
                (
                    "IO controller ",
                    "IO poll state ",
                    "io:DI START ",
                    "IO grab ",
                    "IO START edge=",
                    "grab stop armed condition=IoSignal ",
                    "auto:",
                )
            )
            for line in session.lines
        )
        if not covered:
            report.add(self.domain, "H0", CheckStatus.NOT_COVERED, "本 session 無硬體狀態邊緣")
            return report

        self._check_connection_edges(session, report)
        self._check_storage_heartbeat(session, report)
        self._check_remote_copy_recovery(session, report)
        self._check_camera_edges(session, report)
        self._check_io_controller_lifecycle(session, report)
        self._check_io_poll_health(session, report)
        self._check_io_grab_outcomes(session, report)
        self._check_io_stop_policy(session, report)
        return report

    def _check_remote_copy_recovery(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        queued = [
            (index, line)
            for index, line in enumerate(session.lines)
            if line.message.startswith("[RemoteCopy] pending queued ")
        ]
        retries = [
            (index, line)
            for index, line in enumerate(session.lines)
            if line.message.startswith("[RemoteCopy] retry pending ")
        ]
        evidence = sorted(queued + retries, key=lambda item: item[0])
        if not evidence:
            report.add(
                self.domain,
                "H1.remote-copy-recovery",
                CheckStatus.NOT_COVERED,
                "本 session 無 SMB 中斷待傳資料",
            )
            return

        last_pending_index = evidence[-1][0]
        unavailable = [
            line for line in session.lines[: last_pending_index + 1]
            if line.message.startswith("[RemoteCopy] remote share unavailable:")
        ]
        accepted = [
            (index, line)
            for index, line in enumerate(session.lines)
            if index > last_pending_index
            and line.message
            == "[RemoteCopy] remote share accepted (write verified)"
        ]
        drained = [
            (index, line)
            for index, line in enumerate(session.lines)
            if index > last_pending_index
            and line.message.startswith("[RemoteCopy] backlog drained: ")
        ]

        failures = []
        if not unavailable:
            failures.append("待傳重試前缺 remote share unavailable")
        if not accepted:
            failures.append("最後重試後缺 remote share accepted")
        if not drained:
            failures.append("最後重試後缺 backlog drained")
        if accepted and drained and drained[0][0] < accepted[0][0]:
            failures.append("backlog drained 早於 remote share accepted")

        report.add(
            self.domain,
            "H1.remote-copy-recovery",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"queued={len(queued)} retries={len(retries)} unavailable={len(unavailable)} "
            f"acceptedAfter={len(accepted)} drainedAfter={len(drained)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_io_poll_health(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(
            r"^IO poll state attempts=(\d+) successes=(\d+) snapshots=(\d+) "
            r"connected=(True|False) state=(\w+)$"
        )
        samples = [
            (line, pattern.match(line.message))
            for line in session.lines
            if line.message.startswith("IO poll state ")
        ]
        if not samples:
            report.add(
                self.domain,
                "H1.io-poll",
                CheckStatus.NOT_COVERED,
                "no stable IO polling snapshot in this session",
            )
            return

        failures = []
        previous = None
        for line, match in samples:
            if match is None:
                failures.append(f"{line.timestamp} malformed snapshot")
                continue

            attempts = int(match.group(1))
            successes = int(match.group(2))
            snapshots = int(match.group(3))
            connected = match.group(4) == "True"
            current = (attempts, successes, snapshots)

            if not connected:
                failures.append(f"{line.timestamp} controller disconnected")
            if attempts != successes or successes != snapshots:
                failures.append(
                    f"{line.timestamp} counts diverged "
                    f"attempts={attempts} successes={successes} snapshots={snapshots}"
                )
            if previous is not None and any(
                current[index] <= previous[index] for index in range(3)
            ):
                failures.append(
                    f"{line.timestamp} counters did not advance "
                    f"previous={previous} current={current}"
                )
            previous = current

        report.add(
            self.domain,
            "H1.io-poll",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(samples)} invalid={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_io_controller_lifecycle(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(r"^IO controller (start|stop) generation=(\d+)")
        lines = [(line, pattern.match(line.message)) for line in session.lines]
        lines = [(line, match) for line, match in lines if match]
        if not lines:
            report.add(self.domain, "H1.io-lifecycle", CheckStatus.NOT_COVERED, "舊版或本 session 無 IO controller 生命週期")
            return

        active = None
        failures = []
        for line, match in lines:
            action = match.group(1)
            generation = int(match.group(2))
            if action == "start":
                if active is not None:
                    failures.append(f"{line.timestamp} generation={generation} 啟動時 generation={active} 尚未停止")
                active = generation
            else:
                if active != generation:
                    failures.append(f"{line.timestamp} stop generation={generation} 但 active={active}")
                active = None

        report.add(
            self.domain,
            "H1.io-lifecycle",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"events={len(lines)} invalid={len(failures)}" + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_io_grab_outcomes(self, session: FlowSession, report: CheckReport) -> None:
        request_message = "io:DI START 上升緣 → 抓取請求"
        legacy_message = "io:DI START 上升緣 → 開始抓取"
        has_request_probe = any(
            line.message == request_message for line in session.lines
        )
        starts = [
            index for index, line in enumerate(session.lines)
            if line.message == (
                request_message if has_request_probe else legacy_message
            )
        ]
        if not starts:
            report.add(self.domain, "H3.io-grab", CheckStatus.NOT_COVERED, "本 session 無 IO START Grab")
            return

        failures = []
        tail_active = False
        for line in session.lines:
            if line.message.startswith("capture tail begin "):
                tail_active = True
                continue
            if line.message == "StopGrab":
                tail_active = False
                continue
            if (
                tail_active
                and line.message
                == "IO grab accepted busy=on state=already-grabbing"
            ):
                failures.append(
                    f"{line.timestamp} tail drain was accepted as already-grabbing"
                )

        for position, start_index in enumerate(starts):
            end = starts[position + 1] if position + 1 < len(starts) else len(session.lines)
            outcomes = [
                line.message for line in session.lines[start_index + 1:end]
                if line.message.startswith(("IO grab accepted busy=on", "IO grab rejected busy=off"))
            ]
            if len(outcomes) != 1:
                failures.append(
                    f"{session.lines[start_index].timestamp} outcome={len(outcomes)}（應恰一）"
                )

        report.add(
            self.domain,
            "H3.io-grab",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"requests={len(starts)} probe={'request' if has_request_probe else 'legacy'} "
            f"invalid={len(failures)}" + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_io_stop_policy(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(
            r"^IO grab stop (?P<action>accepted|ignored) "
            r"reason=(?P<reason>StartLow|PlcAliveLost|CommunicationLost) "
            r"stopCondition=(?P<condition>IoSignal|Time|Height) "
            r"(?:(?:drainTail=(?P<drain>True|False))|(?:captureContinues=True))$"
        )
        lines = [
            (line, pattern.match(line.message))
            for line in session.lines
            if line.message.startswith("IO grab stop ")
        ]
        fixed_low_message = (
            "IO START edge=Low stopOnLow=False "
            "action=continue-fixed-target"
        )
        fixed_lows = [
            (index, line)
            for index, line in enumerate(session.lines)
            if line.message == fixed_low_message
        ]
        arm_pattern = re.compile(
            r"^grab stop armed condition=IoSignal limit=io-low "
            r"configured=\d+s grace=unused source=(?:io|manual) grab=.*$"
        )
        io_arms = [
            line
            for line in session.lines
            if line.message.startswith("grab stop armed condition=IoSignal ")
        ]
        io_timer_stops = [
            line
            for line in session.lines
            if line.message.startswith("auto:")
            and "condition=IoSignal" in line.message
        ]
        if not lines and not fixed_lows and not io_arms and not io_timer_stops:
            report.add(
                self.domain,
                "H4.io-stop-policy",
                CheckStatus.NOT_COVERED,
                "本 session 無 IO 停止要求",
            )
            return

        failures = []
        for line in io_arms:
            if arm_pattern.match(line.message) is None:
                failures.append(
                    f"{line.timestamp} IO arm must use limit=io-low and grace=unused"
                )
        for line in io_timer_stops:
            failures.append(
                f"{line.timestamp} IO mode was terminated by a timer"
            )

        for line, match in lines:
            if match is None:
                failures.append(f"{line.timestamp} 格式錯誤")
                continue
            action = match.group("action")
            reason = match.group("reason")
            condition = match.group("condition")
            drain = match.group("drain")
            if action == "accepted":
                expected_drain = "True" if reason == "StartLow" else "False"
                if condition != "IoSignal" or drain != expected_drain:
                    failures.append(
                        f"{line.timestamp} accepted condition={condition} "
                        f"reason={reason} drain={drain}"
                    )
            elif condition not in ("Time", "Height"):
                failures.append(
                    f"{line.timestamp} ignored condition={condition}"
                )

        request_pattern = re.compile(
            r"^IO grab request stopCondition=(?P<condition>IoSignal|Time|Height) "
            r"stopOnLow=(?P<stop>True|False)$"
        )
        for index, line in fixed_lows:
            prior_request = next(
                (
                    request_pattern.match(candidate.message)
                    for candidate in reversed(session.lines[:index])
                    if request_pattern.match(candidate.message)
                ),
                None,
            )
            if (
                prior_request is None
                or prior_request.group("condition") not in ("Time", "Height")
                or prior_request.group("stop") != "False"
            ):
                failures.append(
                    f"{line.timestamp} fixed Low 無有效的 Time/Height request"
                )

        report.add(
            self.domain,
            "H4.io-stop-policy",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"requests={len(lines)} fixedLow={len(fixed_lows)} "
            f"arms={len(io_arms)} timerStops={len(io_timer_stops)} "
            f"invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_connection_edges(self, session: FlowSession, report: CheckReport) -> None:
        last = {}
        count = 0
        failures = []
        for line in session.lines:
            match = self._edge_pattern.match(line.message)
            if not match:
                continue
            count += 1
            name = match.group(1)
            connected = line.message.endswith("恢復連線")
            if name in last and last[name] == connected:
                failures.append(f"{line.timestamp} {name} 連續重複狀態")
            last[name] = connected

        if count == 0:
            report.add(self.domain, "H1.edges", CheckStatus.NOT_COVERED, "無 IO/光源/儲存分享轉變")
            return
        report.add(
            self.domain,
            "H1.edges",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={count} duplicate={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_storage_heartbeat(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line for line in session.lines
            if line.message.startswith(("儲存程式 heartbeat ", "⚠ 儲存程式 heartbeat "))
        ]
        if not lines:
            report.add(self.domain, "H1.heartbeat", CheckStatus.NOT_COVERED, "舊版或未設定儲存 heartbeat")
            return

        last = None
        failures = []
        for line in lines:
            alive = line.message.startswith("儲存程式 heartbeat 恢復 ")
            valid = (
                alive and " pid=" in line.message and " age=" in line.message
            ) or (
                not alive and line.message.startswith("⚠ 儲存程式 heartbeat 未回報 reason=")
            )
            if not valid:
                failures.append(f"{line.timestamp} 格式錯誤")
            elif last is not None and last == alive:
                failures.append(f"{line.timestamp} 連續重複狀態")
            last = alive

        report.add(
            self.domain,
            "H1.heartbeat",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={len(lines)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_camera_edges(self, session: FlowSession, report: CheckReport) -> None:
        lines = [line for line in session.lines if self._camera_pattern.match(line.message)]
        if not lines:
            report.add(self.domain, "H2.camera-count", CheckStatus.NOT_COVERED, "無相機在線數轉變")
            return
        report.add(
            self.domain,
            "H2.camera-count",
            CheckStatus.PASS,
            f"cameraEdges={len(lines)}",
        )
