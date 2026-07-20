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

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        plans = [line for line in session.lines if line.message.startswith("capture plan ")]
        records = [
            line for line in session.lines
            if line.message.startswith("capture csv firstRecord ")
        ]
        csv_lines = [
            line for line in session.lines if line.message.startswith("capture csv ")
        ]
        if not plans and not csv_lines:
            report.add(self.domain, "C0", CheckStatus.NOT_COVERED, "本 session 無存檔/檢測輸出")
        else:
            self._check_capture_plan(plans, report)
            self._check_first_records(plans, records, report)
        self._check_output_health(session, report)
        return report

    def _check_capture_plan(self, plans, report: CheckReport) -> None:
        required = (
            "_raw.jpg",
            "_proc_c.jpg",
            "_proc_r.jpg",
            "_mean_c.bin",
            "_max_c.bin",
            "_mean_r.bin",
            "_max_r.bin",
        )
        legacy = (
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
            missing = [token for token in required if token not in message]
            old = [token for token in legacy if token in message]
            if not current_id or " root=" not in message or " imageDir=" not in message or " csv=" not in message:
                failures.append(f"{line.timestamp} 欄位不完整")
            elif missing or old:
                failures.append(
                    f"{line.timestamp} missing={','.join(missing) or '-'} legacy={','.join(old) or '-'}"
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

    def _check_first_records(self, plans, records, report: CheckReport) -> None:
        plan_positions = {}
        for index, line in enumerate(plans):
            current_id = grab_id(line.message)
            if current_id:
                plan_positions[current_id] = line.elapsed

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
