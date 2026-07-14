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
        self._check_single_curve(session, report)
        self._check_list_ownership(session, report)
        self._check_range_policy(session, report)
        self._check_range_debounce(session, report)
        self._check_range_list_preview(session, report)
        self._check_range_preview(session, report)
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
            CheckStatus.PASS if not missing and scans <= 1 else CheckStatus.FAIL,
            f"intent={len(intents)} cache={cache_hits} scan={scans} 缺終態={len(missing)}"
            + (f"；首筆 {missing[0]}" if missing else ""),
        )

    def _check_single_curve(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.curve", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        pattern = re.compile(
            r"^DT curve load (\d{6}-\d{6}) captures=(\d+) "
            r"source=(disk|prefetch|cache) storage=(summary|bins) "
            r"configMs=(\d+) waitMs=(\d+) pathMs=(\d+) mergeMs=(\d+) "
            r"summaryMs=(\d+) (?:points=(\d+) )?drawMs=(\d+) totalMs=(\d+)$"
        )
        missing = []
        invalid = []
        sources = {"disk": 0, "prefetch": 0, "cache": 0}
        storage = {"summary": 0, "bins": 0}
        has_current_instrument = any(
            line.message.startswith("DT curve load ") and " source=" in line.message
            for line in session.lines
        )

        for position, (line_index, selected_id) in enumerate(intents):
            next_index = intents[position + 1][0] if position + 1 < len(intents) else len(session.lines)
            matching = [
                line.message
                for line in session.lines[line_index + 1 : next_index]
                if line.message.startswith(f"DT curve load {selected_id} ")
            ]
            if not matching:
                missing.append(selected_id)
                continue
            if not has_current_instrument:
                continue
            match = pattern.match(matching[-1])
            if not match:
                invalid.append(matching[-1])
                continue
            sources[match.group(3)] += 1
            storage[match.group(4)] += 1

        if missing or invalid:
            status = CheckStatus.FAIL
        elif not has_current_instrument:
            status = CheckStatus.NOT_COVERED
        else:
            status = CheckStatus.PASS
        report.add(
            self.domain,
            "D3.curve",
            status,
            f"intent={len(intents)} disk={sources['disk']} prefetch={sources['prefetch']} "
            f"summary={storage['summary']} bins={storage['bins']} "
            f"cache={sources['cache']} 缺Curve={len(missing)} 格式錯誤={len(invalid)}"
            + (f"；首筆 {missing[0]}" if missing else "")
            + ("；舊版儀器無 source 分段" if not has_current_instrument else ""),
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

    def _check_range_policy(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT range policy ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.range-policy",
                CheckStatus.NOT_COVERED,
                "舊版 log 無 range policy 儀器",
            )
            return

        expected = "DT range policy listMs=33 curveMs=80 settleMs=150 curveMode=monotonic"
        unique = sorted(set(lines))
        ok = len(lines) == 1 and unique == [expected]
        report.add(
            self.domain,
            "D3.range-policy",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"行數={len(lines)} 實際={' | '.join(unique)}",
        )

    def _check_range_preview(self, session: FlowSession, report: CheckReport) -> None:
        intents = sum(
            1 for line in session.lines
            if line.message.startswith("ui:【序號範圍-")
        )
        pattern = re.compile(
            r"^DT range preview apply gen=(\d+) range=(\d{6}-\d{6})~(\d{6}-\d{6}) "
            r"loadMs=(\d+) drawMs=(\d+) meanRows=(\d+) maxRows=(\d+) "
            r"method=(top-maxcmean|mixed|even) coverage=(\d+)/(\d+) rankedCams=(\d+)/(\d+)"
            r" index=(\d+)/(\d+)$"
        )
        generations = []
        invalid = []
        for line in session.lines:
            if not line.message.startswith("DT range preview apply"):
                continue
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            generations.append(int(match.group(1)))

        if intents == 0:
            report.add(self.domain, "D3.range-preview", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        if not generations:
            report.add(
                self.domain,
                "D3.range-preview",
                CheckStatus.FAIL,
                f"intent={intents}，沒有 Curve 預覽上畫",
            )
            return

        monotonic = all(
            current > previous
            for previous, current in zip(generations, generations[1:])
        )
        list_generations = []
        for line in session.lines:
            match = re.match(r"^DT range list preview gen=(\d+) ", line.message)
            if match:
                list_generations.append(int(match.group(1)))
        final_caught_up = not list_generations or generations[-1] >= list_generations[-1]
        ok = not invalid and monotonic and final_caught_up
        report.add(
            self.domain,
            "D3.range-preview",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={intents} apply={len(generations)} generation={generations[0]}~{generations[-1]} "
            f"倒退上畫={0 if monotonic else 1} 最終追上={1 if final_caught_up else 0} "
            f"格式錯誤={len(invalid)}",
        )

    def _check_range_list_preview(self, session: FlowSession, report: CheckReport) -> None:
        intents = sum(
            1 for line in session.lines
            if line.message.startswith("ui:【序號範圍-")
        )
        pattern = re.compile(
            r"^DT range list preview gen=(\d+) range=(\d{6}-\d{6})~(\d{6}-\d{6}) "
            r"rows=(\d+) ms=(\d+) source=index$"
        )
        generations = []
        invalid = []
        worst_ms = 0
        for line in session.lines:
            if not line.message.startswith("DT range list preview"):
                continue
            match = pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            generations.append(int(match.group(1)))
            worst_ms = max(worst_ms, int(match.group(5)))

        if intents == 0:
            report.add(self.domain, "D3.list-preview", CheckStatus.NOT_COVERED, "無序號範圍滾動")
            return
        if not generations:
            report.add(
                self.domain,
                "D3.list-preview",
                CheckStatus.FAIL,
                f"intent={intents}，沒有 List 預覽套用",
            )
            return

        monotonic = all(
            current > previous
            for previous, current in zip(generations, generations[1:])
        )
        ok = not invalid and monotonic
        report.add(
            self.domain,
            "D3.list-preview",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"intent={intents} apply={len(generations)} generation={generations[0]}~{generations[-1]} "
            f"最大={worst_ms}ms 過期套用={0 if monotonic else 1} 格式錯誤={len(invalid)}",
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
