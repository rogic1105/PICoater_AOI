"""Data/report-tab (D-series) flow validators."""

import re

from .core import CheckReport, CheckStatus, FlowSession, grab_id


class DataFlowValidator:
    domain = "DATA"

    @staticmethod
    def _covered(session: FlowSession) -> bool:
        prefixes = (
            "DT ",
            "ui:【報表序號】",
            "ui:【明細列表】",
            "ui:【序號範圍-",
            "ui:【期間-",
            "ui:【良率",
            "ui:【讀取資料】鈕（Data）",
        )
        return any(line.message.startswith(prefixes) for line in session.lines)

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        if not self._covered(session):
            report.add(self.domain, "D0", CheckStatus.NOT_COVERED, "本 session 無報表操作")
            return report

        self._check_single_selection(session, report)
        self._check_list_ownership(session, report)
        self._check_range_debounce(session, report)
        self._check_y_scale_toggle(session, report)
        self._check_ui_stall(session, report)
        return report

    def _check_single_selection(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.selected", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        missing = []
        cache_hits = 0
        scans = 0
        for position, (line_index, selected_id) in enumerate(intents):
            next_index = intents[position + 1][0] if position + 1 < len(intents) else len(session.lines)
            matching = [
                line.message
                for line in session.lines[line_index + 1 : next_index]
                if line.message.startswith(f"DT selected {selected_id} ")
            ]
            if not matching:
                missing.append(selected_id)
                continue
            if "stats=cache" in matching[-1]:
                cache_hits += 1
            elif "stats=scan" in matching[-1]:
                scans += 1

        report.add(
            self.domain,
            "D3.selected",
            CheckStatus.PASS if not missing else CheckStatus.FAIL,
            f"intent={len(intents)} cache={cache_hits} scan={scans} 缺終態={len(missing)}"
            + (f"；首筆 {missing[0]}" if missing else ""),
        )

    def _check_list_ownership(self, session: FlowSession, report: CheckReport) -> None:
        last_intent = None
        last_intent_time = None
        violations = []
        reloads = 0
        for line in session.lines:
            message = line.message
            if message.startswith("ui:"):
                last_intent = message
                last_intent_time = line.elapsed
            elif message.startswith("DT list reload"):
                reloads += 1
                if (
                    last_intent
                    and last_intent.startswith("ui:【報表序號】")
                    and last_intent_time is not None
                    and line.elapsed - last_intent_time <= 2
                ):
                    violations.append(f"{line.timestamp} {message}")
        report.add(
            self.domain,
            "D3.list-keep",
            CheckStatus.PASS if not violations else CheckStatus.FAIL,
            f"list reload={reloads}；由單片選擇觸發={len(violations)}"
            + (f"；首筆 {violations[0]}" if violations else ""),
        )

    def _check_ui_stall(
        self, session: FlowSession, report: CheckReport, limit_ms: int = 1000
    ) -> None:
        data_times = [
            line.elapsed
            for line in session.lines
            if line.message.startswith(("DT ", "ui:【報表", "ui:【明細列表】", "ui:【序號範圍-", "ui:【期間-", "ui:【良率"))
        ]
        if not data_times:
            report.add(self.domain, "U.stall", CheckStatus.NOT_COVERED, "無可量測的報表互動")
            return
        stalls = []
        for line in session.lines:
            if not line.message.startswith("[UiStall]"):
                continue
            if not any(event_time - 1 <= line.elapsed <= event_time + 3 for event_time in data_times):
                continue
            match = re.search(r"\[UiStall\]\s+(\d+)ms（(.*)）", line.message)
            if match:
                stalls.append((int(match.group(1)), match.group(2)))
        worst = max(stalls) if stalls else (0, "")
        report.add(
            self.domain,
            "U.stall",
            CheckStatus.PASS if worst[0] <= limit_ms else CheckStatus.FAIL,
            f"最大={worst[0]}ms（{worst[1]}）；>{limit_ms}ms 共 "
            f"{sum(1 for duration, _ in stalls if duration > limit_ms)} 次",
        )

    def _check_range_debounce(self, session: FlowSession, report: CheckReport) -> None:
        range_intents = 0
        settles = 0
        waiting_for_settle = False
        premature = []
        for line in session.lines:
            message = line.message
            if message.startswith("ui:【序號範圍-"):
                range_intents += 1
                waiting_for_settle = True
            elif message == "DT range settle → refresh":
                settles += 1
                waiting_for_settle = False
            elif waiting_for_settle and message.startswith(("DT list reload", "DT curve candidates")):
                premature.append(f"{line.timestamp} {message}")

        if range_intents == 0:
            report.add(self.domain, "D3.range-settle", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        ok = not premature and settles > 0 and settles <= range_intents
        report.add(
            self.domain,
            "D3.range-settle",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={range_intents} settle={settles} settle前重算={len(premature)}"
            + (f"；首筆 {premature[0]}" if premature else ""),
        )

    def _check_y_scale_toggle(self, session: FlowSession, report: CheckReport) -> None:
        pattern = re.compile(
            r"^ui:【良率圖-(年|月|日)】→ Y軸=(Auto|Fixed) "
            r"setting=(Auto|Fixed) override=(Auto|Fixed|off)$"
        )
        lines = [
            line for line in session.lines if line.message.startswith("ui:【良率圖-")
        ]
        if not lines:
            report.add(self.domain, "D5.y-scale", CheckStatus.NOT_COVERED, "無良率圖 Y 軸點擊")
            return

        invalid = []
        for line in lines:
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            _, effective, setting, scale_override = match.groups()
            if scale_override == "off":
                valid = effective == setting
            else:
                valid = effective == scale_override and scale_override != setting
            if not valid:
                invalid.append(
                    f"{line.timestamp} effective={effective} setting={setting} override={scale_override}"
                )

        report.add(
            self.domain,
            "D5.y-scale",
            CheckStatus.PASS if not invalid else CheckStatus.FAIL,
            f"點擊={len(lines)} 狀態矛盾={len(invalid)}"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )
