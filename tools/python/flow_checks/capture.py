"""Capture/storage (C-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession, grab_id


class CaptureFlowValidator:
    domain = "CAPTURE"
    _health_raise = re.compile(
        r"^\[OutputHealth\] raise code=(?P<code>\S+) "
        r"severity=(?P<severity>Notice|OutputFault|Critical) "
        r"message=(?P<message>.*)$"
    )
    _health_resolve = re.compile(
        r"^\[OutputHealth\] resolve code=(?P<code>\S+) "
        r"message=(?P<message>.*)$"
    )
    _health_state = re.compile(
        r"^\[OutputHealth\] state "
        r"(?P<old>Normal|Notice|OutputFault|Critical) -> "
        r"(?P<new>Normal|Notice|OutputFault|Critical) "
        r"code=(?P<code>\S+) active=(?P<active>True|False)$"
    )
    _health_ack = re.compile(r"^\[OutputHealth\] ack codes=(?P<codes>\S+)$")
    _remote_release = re.compile(
        r"^capture remote release grab=(?P<grab>\d{6}-\d{6}) "
        r"files=(?P<files>\d+) bytes=(?P<bytes>\d+)$"
    )

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        plans = [line for line in session.lines if line.message.startswith("capture plan ")]
        records = [
            line for line in session.lines
            if line.message.startswith("capture csv firstRecord ")
        ]
        archive_appends = [
            line for line in session.lines
            if line.message.startswith("capture archive append ")
        ]
        csv_lines = [
            line for line in session.lines if line.message.startswith("capture csv ")
        ]
        configs = [
            line for line in session.lines
            if line.message.startswith("capture csv cfg ")
        ]
        if not plans and not csv_lines:
            report.add(self.domain, "C0", CheckStatus.NOT_COVERED, "本 session 無存檔/檢測輸出")
        else:
            self._check_capture_plan(plans, report)
            self._check_config_snapshots(configs, report)
            self._check_first_records(plans, archive_appends, records, report)
        self._check_write_integrity(plans, archive_appends, session, report)
        self._check_delivery_release(plans, archive_appends, session, report)
        self._check_output_health(session, report)
        return report

    def _check_capture_plan(self, plans, report: CheckReport) -> None:
        legacy = (
            " files=",
            "_raw.jpg",
            "_proc_v.jpg",
            "_proc_h.jpg",
            "_mean_v.bin",
            "_max_v.bin",
            "_mean_h.bin",
            "_max_h.bin",
        )
        failures = []
        ids = set()
        for line in plans:
            message = line.message
            current_id = grab_id(message)
            if current_id:
                ids.add(current_id)
            old = [token for token in legacy if token in message]
            if not current_id or " root=" not in message or " imageDir=" not in message or " csv=" not in message:
                failures.append(f"{line.timestamp} 欄位不完整")
            elif f" archive={current_id}.acap" not in message or old:
                failures.append(
                    f"{line.timestamp} archive=invalid legacy={','.join(old) or '-'}"
                )

        if not plans:
            report.add(self.domain, "C1.plan", CheckStatus.NOT_COVERED, "舊版 log 無 capture plan")
            return
        report.add(
            self.domain,
            "C1.plan",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"plans={len(plans)} grabs={len(ids)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_config_snapshots(self, configs, report: CheckReport) -> None:
        failures = []
        pattern = re.compile(
            r"^capture csv cfg path=.+ speed=(?P<speed>[-+0-9.]+) "
            r"lr=(?P<line_rate>[-+0-9.]+) HM="
        )
        for line in configs:
            match = pattern.match(line.message)
            if match is None:
                failures.append(f"{line.timestamp} 缺 speed/lr")
                continue
            try:
                if float(match.group("speed")) <= 0 or float(match.group("line_rate")) <= 0:
                    failures.append(f"{line.timestamp} speed/lr 必須 > 0")
            except ValueError:
                failures.append(f"{line.timestamp} speed/lr 格式錯誤")

        if not configs:
            report.add(
                self.domain,
                "C2.cfg-scale",
                CheckStatus.NOT_COVERED,
                "本 session 未寫入新版 #CFG",
            )
            return
        report.add(
            self.domain,
            "C2.cfg-scale",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"configs={len(configs)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_first_records(self, plans, archive_appends, records, report: CheckReport) -> None:
        plan_positions = {}
        for index, line in enumerate(plans):
            current_id = grab_id(line.message)
            if current_id:
                plan_positions[current_id] = line.elapsed

        append_positions = {}
        for line in archive_appends:
            current_id = grab_id(line.message)
            if current_id:
                append_positions.setdefault(current_id, []).append(line.elapsed)

        failures = []
        seen = set()
        required_fields = (" path=", " file=", " verdict=", " peak=", " rowPeak=", " maxCMean=")
        for line in records:
            current_id = grab_id(line.message)
            if not current_id or any(field not in line.message for field in required_fields):
                failures.append(f"{line.timestamp} 格式不完整")
                continue
            if current_id in seen:
                failures.append(f"{line.timestamp} grab={current_id} 重複 firstRecord")
            seen.add(current_id)
            if current_id not in plan_positions or plan_positions[current_id] > line.elapsed:
                failures.append(f"{line.timestamp} grab={current_id} 缺先行 capture plan")

        for line in records:
            current_id = grab_id(line.message)
            if current_id and (
                current_id not in append_positions
                or not any(elapsed <= line.elapsed for elapsed in append_positions[current_id])
            ):
                failures.append(f"{line.timestamp} grab={current_id} missing archive append")

        if not records:
            report.add(self.domain, "C2.first-record", CheckStatus.NOT_COVERED, "本 session 無成功存檔首筆")
            return
        report.add(
            self.domain,
            "C2.first-record",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"firstRecords={len(records)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_write_integrity(
        self, plans, archive_appends, session: FlowSession, report: CheckReport
    ) -> None:
        if not plans and not archive_appends:
            report.add(
                self.domain,
                "C3.write-integrity",
                CheckStatus.NOT_COVERED,
                "session did not run capture persistence",
            )
            return

        failures = []
        for line in session.lines:
            match = self._health_raise.match(line.message)
            if match and match.group("code").startswith("CaptureWriteFailure."):
                failures.append(
                    f"{line.timestamp} {match.group('code')} {match.group('message')}"
                )

        report.add(
            self.domain,
            "C3.write-integrity",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"archiveAppends={len(archive_appends)} writeFailures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_delivery_release(
        self, plans, archive_appends, session: FlowSession, report: CheckReport
    ) -> None:
        indexed = list(enumerate(session.lines))
        plan_positions = []
        for index, line in indexed:
            if not line.message.startswith("capture plan "):
                continue
            current_id = grab_id(line.message)
            if current_id:
                plan_positions.append((index, current_id))

        stopped_grabs = set()
        for position, (plan_index, current_id) in enumerate(plan_positions):
            next_plan_index = (
                plan_positions[position + 1][0]
                if position + 1 < len(plan_positions)
                else len(session.lines)
            )
            if any(
                line.message == "StopGrab"
                for _, line in indexed[plan_index + 1:next_plan_index]
            ):
                stopped_grabs.add(current_id)

        events = {}
        for index, line in indexed:
            message = line.message
            if message.startswith("capture save drain begin "):
                kind = "begin"
            elif message.startswith("capture save drain done "):
                kind = "done"
            elif message.startswith("capture remote release "):
                kind = "release"
            else:
                continue
            current_id = grab_id(message)
            if current_id:
                events.setdefault(current_id, {"begin": [], "done": [], "release": []})[
                    kind
                ].append((index, line))

        covered_grabs = stopped_grabs | set(events)
        if not covered_grabs:
            report.add(
                self.domain,
                "C3.delivery-release",
                CheckStatus.NOT_COVERED,
                "session did not complete a capture stop",
            )
            return

        append_positions = {}
        for index, line in indexed:
            if not line.message.startswith("capture archive append "):
                continue
            current_id = grab_id(line.message)
            if current_id:
                append_positions.setdefault(current_id, []).append(index)

        plan_index_by_grab = {current_id: index for index, current_id in plan_positions}
        stop_positions = [
            index for index, line in indexed if line.message == "StopGrab"
        ]
        failures = []
        for current_id in sorted(covered_grabs):
            current = events.get(
                current_id, {"begin": [], "done": [], "release": []})
            counts = {kind: len(current[kind]) for kind in ("begin", "done", "release")}
            if any(count != 1 for count in counts.values()):
                failures.append(
                    f"grab={current_id} event-count "
                    f"begin={counts['begin']} done={counts['done']} release={counts['release']}"
                )
                continue

            begin_index, _ = current["begin"][0]
            done_index, _ = current["done"][0]
            release_index, release_line = current["release"][0]
            if not begin_index < done_index < release_index:
                failures.append(
                    f"grab={current_id} order begin={begin_index} "
                    f"done={done_index} release={release_index}"
                )

            plan_index = plan_index_by_grab.get(current_id, -1)
            if not any(plan_index < stop_index < begin_index for stop_index in stop_positions):
                failures.append(f"grab={current_id} drain has no preceding StopGrab")

            late_appends = [
                index for index in append_positions.get(current_id, []) if index > done_index
            ]
            if late_appends:
                failures.append(
                    f"grab={current_id} archive append after drain done count={len(late_appends)}"
                )

            release_match = self._remote_release.match(release_line.message)
            if release_match is None:
                failures.append(f"grab={current_id} malformed remote release")
            elif append_positions.get(current_id):
                files = int(release_match.group("files"))
                byte_count = int(release_match.group("bytes"))
                if files != 2 or byte_count <= 0:
                    failures.append(
                        f"grab={current_id} release files={files} bytes={byte_count}; expected ACAP+CSV"
                    )

        report.add(
            self.domain,
            "C3.delivery-release",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"stoppedGrabs={len(covered_grabs)} invalid={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_output_health(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        lines = [
            line for line in session.lines
            if line.message.startswith("[OutputHealth]")
        ]
        if not lines:
            report.add(
                self.domain,
                "C4.output-health",
                CheckStatus.NOT_COVERED,
                "本 session 無產出健康度狀態轉變",
            )
            return

        severity_rank = {
            "Normal": 0,
            "Notice": 1,
            "OutputFault": 2,
            "Critical": 3,
        }
        incidents = {}
        sequence = 0
        displayed = {
            "severity": "Normal",
            "code": "none",
            "active": False,
            "message": "",
        }
        pending_state = None
        failures = []
        event_count = 0
        state_count = 0

        def selected_snapshot():
            if not incidents:
                return {
                    "severity": "Normal",
                    "code": "none",
                    "active": False,
                    "message": "",
                }
            selected = max(
                incidents.values(),
                key=lambda item: (
                    severity_rank[item["severity"]],
                    1 if item["active"] else 0,
                    item["sequence"],
                ),
            )
            return {
                "severity": selected["severity"],
                "code": selected["code"],
                "active": selected["active"],
                "message": selected["message"],
            }

        for line in lines:
            message = line.message
            state_match = self._health_state.match(message)
            if pending_state is not None and state_match is None:
                failures.append(
                    f"{line.timestamp} 前一事件缺 state："
                    f"{displayed['severity']} -> {pending_state['severity']} "
                    f"code={pending_state['code']} active={pending_state['active']}"
                )
                displayed = pending_state
                pending_state = None

            match = self._health_raise.match(message)
            if match:
                event_count += 1
                code = match.group("code")
                severity = match.group("severity")
                health_message = match.group("message")
                existing = incidents.get(code)
                if (
                    existing is not None
                    and existing["active"]
                    and existing["severity"] == severity
                    and existing["message"] == health_message
                ):
                    failures.append(
                        f"{line.timestamp} code={code} 同內容未轉變卻重複 raise"
                    )
                    continue
                sequence += 1
                incidents[code] = {
                    "code": code,
                    "severity": severity,
                    "message": health_message,
                    "active": True,
                    "sequence": sequence,
                }
                expected = selected_snapshot()
                if expected != displayed:
                    pending_state = expected
                continue

            match = self._health_resolve.match(message)
            if match:
                event_count += 1
                code = match.group("code")
                existing = incidents.get(code)
                if existing is None or not existing["active"]:
                    failures.append(f"{line.timestamp} code={code} 無 active 來源卻 resolve")
                    continue
                sequence += 1
                existing["active"] = False
                existing["message"] = match.group("message")
                existing["sequence"] = sequence
                expected = selected_snapshot()
                if expected != displayed:
                    pending_state = expected
                continue

            match = self._health_ack.match(message)
            if match:
                event_count += 1
                codes = [code for code in match.group("codes").split(",") if code]
                if len(codes) != 1:
                    failures.append(
                        f"{line.timestamp} 一次確認包含 {len(codes)} 個 code；每個問題必須個別確認"
                    )
                invalid = [
                    code for code in codes
                    if code not in incidents or incidents[code]["active"]
                ]
                if invalid:
                    failures.append(
                        f"{line.timestamp} active/未知事件被 ack={','.join(invalid)}"
                    )
                for code in codes:
                    if code not in invalid:
                        incidents.pop(code, None)
                expected = selected_snapshot()
                if expected != displayed:
                    pending_state = expected
                continue

            if state_match:
                state_count += 1
                if pending_state is None:
                    failures.append(f"{line.timestamp} 無事件來源卻出現 state")
                    continue
                old = state_match.group("old")
                new = state_match.group("new")
                code = state_match.group("code")
                active = state_match.group("active") == "True"
                if old != displayed["severity"]:
                    failures.append(
                        f"{line.timestamp} state old={old}，預期 {displayed['severity']}"
                    )
                if (
                    new != pending_state["severity"]
                    or code != pending_state["code"]
                    or active != pending_state["active"]
                ):
                    failures.append(
                        f"{line.timestamp} state={new}/{code}/{active}，預期 "
                        f"{pending_state['severity']}/{pending_state['code']}/"
                        f"{pending_state['active']}"
                    )
                displayed = pending_state
                pending_state = None
                continue

            failures.append(f"{line.timestamp} 格式錯誤：{message}")

        if pending_state is not None:
            failures.append(
                "檔尾缺 state："
                f"{displayed['severity']} -> {pending_state['severity']} "
                f"code={pending_state['code']} active={pending_state['active']}"
            )

        report.add(
            self.domain,
            "C4.output-health",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"events={event_count} states={state_count} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
