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
        configs = [
            line for line in session.lines
            if line.message.startswith("capture csv cfg ")
        ]
        finalizes = [
            line for line in session.lines
            if line.message.startswith("capture finalize ")
        ]
        if not plans and not csv_lines:
            report.add(self.domain, "C0", CheckStatus.NOT_COVERED, "本 session 無存檔/檢測輸出")
        else:
            self._check_capture_plan(plans, report)
            self._check_config_snapshots(configs, report)
            self._check_first_records(plans, records, report)
            self._check_capture_finalize(plans, finalizes, report)
        self._check_output_health(session, report)
        return report

    def _check_capture_plan(self, plans, report: CheckReport) -> None:
        required = (
            " archive=",
            ".acap",
            " assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r",
            " preview=1920x1080x3",
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

    def _check_capture_finalize(self, plans, finalizes, report: CheckReport) -> None:
        planned_ids = {
            current_id
            for line in plans
            for current_id in [grab_id(line.message)]
            if current_id
        }
        completed_ids = set()
        failures = []
        pattern = re.compile(
            r"^capture finalize grab=(?P<grab>\S+) "
            r"archive=(?P<archive>.+\.acap) "
            r"atlas=(?P<atlas>\d+) atlasBytes=(?P<bytes>\d+) "
            r"remoteFiles=(?P<remote>\d+)$"
        )
        for line in finalizes:
            if line.message.startswith("capture finalize failed "):
                failures.append(f"{line.timestamp} {line.message}")
                continue
            match = pattern.match(line.message)
            if match is None:
                failures.append(f"{line.timestamp} 欄位不完整")
                continue
            current_id = match.group("grab")
            completed_ids.add(current_id)
            if current_id not in planned_ids:
                failures.append(f"{line.timestamp} grab={current_id} 缺 capture plan")
            if int(match.group("atlas")) != 3 or int(match.group("bytes")) <= 0:
                failures.append(
                    f"{line.timestamp} grab={current_id} atlas="
                    f"{match.group('atlas')} bytes={match.group('bytes')}"
                )

        if not finalizes:
            report.add(
                self.domain,
                "C3.finalize",
                CheckStatus.NOT_COVERED,
                "本 session 沒有完成 Stop 後封裝收尾",
            )
            return
        missing = sorted(planned_ids - completed_ids)
        if missing:
            failures.append("缺完成：" + ",".join(missing[:3]))
        report.add(
            self.domain,
            "C3.finalize",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"planned={len(planned_ids)} finalized={len(completed_ids)} "
            f"invalid={len(failures)}"
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
