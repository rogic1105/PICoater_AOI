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
        report_caches = [
            line for line in session.lines
            if line.message.startswith("capture report cache ")
        ]
        layout_lines = [
            line for line in session.lines
            if line.message.startswith("capture layout ")
        ]
        if not plans and not csv_lines:
            report.add(self.domain, "C0", CheckStatus.NOT_COVERED, "本 session 無存檔/檢測輸出")
        else:
            self._check_capture_plan(plans, report)
            self._check_config_snapshots(configs, report)
            self._check_first_records(plans, records, report)
            self._check_capture_finalize(plans, finalizes, report)
            self._check_report_cache(plans, report_caches, report)
        self._check_final_layout(plans, finalizes, layout_lines, report)
        self._check_output_health(session, report)
        return report

    def _check_capture_plan(self, plans, report: CheckReport) -> None:
        required = (
            " archive=",
            ".acap",
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
            if " hessianScale=" in message:
                if " assets=raw|proc_c|proc_r|hessian_c|hessian_r|mean_c|max_c|mean_r|max_r" not in message:
                    missing.append("hessian-standard-assets")
            elif " assets=raw|proc_c|proc_r|mean_c|max_c|mean_r|max_r" not in message:
                missing.append("capture-assets")
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

    def _check_report_cache(self, plans, cache_lines, report: CheckReport) -> None:
        if not cache_lines:
            report.add(
                self.domain,
                "C3.report-cache",
                CheckStatus.NOT_COVERED,
                "session has no new capture report-cache evidence",
            )
            return

        planned_ids = {
            current_id
            for line in plans
            for current_id in [grab_id(line.message)]
            if current_id
        }
        pattern = re.compile(
            r"^capture report cache grab=(?P<grab>\S+) "
            r"summary=(?P<summary>queued|failed|skip-incomplete) "
            r"peakIndex=(?P<index>ok|failed|skip-incomplete) "
            r"captures=(?P<captures>\d+) merged=(?P<merged>\d+) "
            r"align=(?P<align>tick|filename|none) ms=(?P<ms>\d+)$"
        )
        seen = set()
        failures = []
        for line in cache_lines:
            match = pattern.match(line.message)
            if match is None:
                failures.append(f"{line.timestamp} malformed report-cache line")
                continue
            current_id = match.group("grab")
            seen.add(current_id)
            captures = int(match.group("captures"))
            merged = int(match.group("merged"))
            if current_id not in planned_ids:
                failures.append(f"{line.timestamp} grab={current_id} missing capture plan")
            if match.group("summary") != "queued" or match.group("index") != "ok":
                failures.append(
                    f"{line.timestamp} grab={current_id} summary={match.group('summary')} "
                    f"peakIndex={match.group('index')}"
                )
            if captures <= 0 or merged != captures:
                failures.append(
                    f"{line.timestamp} grab={current_id} captures={captures} merged={merged}"
                )

        missing = sorted(planned_ids - seen)
        if missing:
            failures.append("missing=" + ",".join(missing[:3]))
        report.add(
            self.domain,
            "C3.report-cache",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"planned={len(planned_ids)} cached={len(seen)} invalid={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
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

    def _check_final_layout(
        self, plans, finalizes, layout_lines, report: CheckReport
    ) -> None:
        if not layout_lines:
            report.add(
                self.domain,
                "C2.final-layout",
                CheckStatus.NOT_COVERED,
                "此紀錄由舊版程式產生，沒有 grab 最終布局探針",
            )
            return

        number = r"[-+]?\d+(?:\.\d+)?"
        values = (
            rf"ops=(?P<ops>{number}(?:\|{number}){{6}}) "
            rf"start=(?P<start>{number}(?:\|{number}){{6}}) "
            rf"speed=(?P<speed>{number}) "
            rf"head=(?P<head>{number}) tail=(?P<tail>{number})"
        )
        pending_pattern = re.compile(
            r"^capture layout pending grab=(?P<grab>\S+) "
            r"setting=(?P<setting>\S+) "
            r"apply=(?P<apply>display-now\+stop-final|stop-final)$"
        )
        final_pattern = re.compile(
            rf"^capture layout final grab=(?P<grab>\S+) {values} path=.+$"
        )
        applied_pattern = re.compile(
            rf"^capture layout applied grab=(?P<grab>\S+) timing=stop "
            rf"{values} render=(?P<render>once|already-applied) source=unchanged$"
        )

        pending = {}
        finals = {}
        applied = {}
        failures = []

        for line in layout_lines:
            match = pending_pattern.match(line.message)
            if match:
                pending.setdefault(match.group("grab"), []).append(line)
                continue
            match = final_pattern.match(line.message)
            if match:
                current_id = match.group("grab")
                if current_id in finals:
                    failures.append(
                        f"{line.timestamp} grab={current_id} 重複 final"
                    )
                finals[current_id] = (line, match)
                continue
            match = applied_pattern.match(line.message)
            if match:
                current_id = match.group("grab")
                if current_id in applied:
                    failures.append(
                        f"{line.timestamp} grab={current_id} 重複 applied"
                    )
                applied[current_id] = (line, match)
                continue
            failures.append(f"{line.timestamp} 最終布局行格式錯誤")

        finalized_ids = {
            current_id
            for line in finalizes
            if not line.message.startswith("capture finalize failed ")
            for current_id in [grab_id(line.message)]
            if current_id
        }
        missing_final = sorted(finalized_ids - set(finals))
        if missing_final:
            failures.append("已封裝但缺 final=" + ",".join(missing_final[:3]))

        for current_id, pending_lines in pending.items():
            final_item = finals.get(current_id)
            applied_item = applied.get(current_id)
            if final_item is None:
                failures.append(f"grab={current_id} 有 pending 但缺 final")
                continue
            if applied_item is None:
                failures.append(f"grab={current_id} 有 pending 但缺 applied")
                continue

            final_line, final_match = final_item
            applied_line, applied_match = applied_item
            if final_line.elapsed < pending_lines[-1].elapsed:
                failures.append(f"grab={current_id} final 早於最後 pending")
            if applied_line.elapsed < final_line.elapsed:
                failures.append(f"grab={current_id} applied 早於 final")

            for field in ("ops", "start", "speed", "head", "tail"):
                if final_match.group(field) != applied_match.group(field):
                    failures.append(
                        f"grab={current_id} final/applied {field} 不一致"
                    )
                    break

        unexpected_applied = sorted(set(applied) - set(pending))
        if unexpected_applied:
            failures.append(
                "沒有 pending 卻 applied=" + ",".join(unexpected_applied[:3])
            )

        report.add(
            self.domain,
            "C2.final-layout",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"final={len(finals)} pendingGrabs={len(pending)} "
            f"applied={len(applied)} invalid={len(failures)}"
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
