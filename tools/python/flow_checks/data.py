"""Data/report-tab (D-series) flow validators."""

import re

from .core import (
    CheckReport,
    CheckStatus,
    FlowSession,
    assess_ui_responsiveness,
    grab_id,
)


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

        self._check_statistics_snapshot(session, report)
        self._check_single_selection(session, report)
        self._check_single_curve_policy(session, report)
        self._check_single_curve(session, report)
        self._check_curve_summary_writes(session, report)
        self._check_single_row_curve(session, report)
        self._check_list_ownership(session, report)
        self._check_range_policy(session, report)
        self._check_range_debounce(session, report)
        self._check_range_list_preview(session, report)
        self._check_range_preview(session, report)
        self._check_y_scale_toggle(session, report)
        self._check_ui_stall(session, report)
        return report

    def _check_statistics_snapshot(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        load_positions = [
            index for index, line in enumerate(session.lines)
            if line.message == "ui:【讀取資料】鈕（Data）"
        ]
        snapshot_pattern = re.compile(
            r"^DT stats snapshot csv=(\d+) records=(\d+) grabs=(\d+) ms=(\d+)$"
        )
        snapshots = []
        invalid = []
        for index, line in enumerate(session.lines):
            if not line.message.startswith("DT stats snapshot"):
                continue
            match = snapshot_pattern.match(line.message)
            if not match:
                invalid.append(f"{line.timestamp} 格式錯誤")
                continue
            csv_count, record_count, grab_count, elapsed_ms = map(
                int, match.groups()
            )
            valid_counts = (
                record_count >= grab_count
                and (csv_count > 0 or (record_count == 0 and grab_count == 0))
            )
            if not valid_counts:
                invalid.append(
                    f"{line.timestamp} csv={csv_count} records={record_count} grabs={grab_count}"
                )
            snapshots.append((index, csv_count, record_count, grab_count, elapsed_ms))

        if not snapshots:
            report.add(
                self.domain,
                "D1.snapshot",
                CheckStatus.NOT_COVERED,
                "舊版 log 無一次式統計 snapshot 儀器",
            )
            return

        missing = 0
        for position, load_index in enumerate(load_positions):
            next_load = (
                load_positions[position + 1]
                if position + 1 < len(load_positions)
                else len(session.lines)
            )
            if not any(load_index < item[0] < next_load for item in snapshots):
                missing += 1

        worst_ms = max(item[4] for item in snapshots)
        ok = not invalid and missing == 0
        report.add(
            self.domain,
            "D1.snapshot",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"讀取={len(load_positions)} snapshot={len(snapshots)} "
            f"缺少={missing} 格式/計數錯誤={len(invalid)} 最慢={worst_ms}ms"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
        )

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
            r"source=shared storage=(summary|bins|memory-summary|memory-bins) "
            r"configMs=(\d+) waitMs=(\d+) pathMs=(\d+) mergeMs=(\d+) "
            r"summaryMs=(\d+) (?:points=(\d+) )?drawMs=(\d+) totalMs=(\d+)$"
        )
        invalid = []
        storage = {"summary": 0, "bins": 0, "memory-summary": 0, "memory-bins": 0}
        has_current_instrument = any(
            line.message.startswith("DT curve load ") and " source=" in line.message
            for line in session.lines
        )

        loads = [
            (index, line.message)
            for index, line in enumerate(session.lines)
            if line.message.startswith("DT curve load ")
            and not line.message.startswith("DT curve load policy ")
        ]
        for _, message in loads:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue
            storage[match.group(3)] += 1

        final_index, final_id = intents[-1]
        final_loaded = any(
            index > final_index and message.startswith(f"DT curve load {final_id} ")
            for index, message in loads
        )
        stale = sum(
            1 for line in session.lines
            if line.message.startswith("DT curve stale-drop ")
        )

        if not final_loaded or invalid:
            status = CheckStatus.FAIL
        elif not has_current_instrument:
            status = CheckStatus.NOT_COVERED
        else:
            status = CheckStatus.PASS
        report.add(
            self.domain,
            "D3.curve",
            status,
            f"intent={len(intents)} applied={len(loads)} stale={stale} "
            f"summary={storage['summary']} bins={storage['bins']} "
            f"memory={storage['memory-summary'] + storage['memory-bins']} "
            f"final={'ok' if final_loaded else 'missing'} 格式錯誤={len(invalid)}"
            + ("；舊版儀器無 source 分段" if not has_current_instrument else ""),
        )

    def _check_single_curve_policy(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT curve load policy ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.curve-policy",
                CheckStatus.NOT_COVERED,
                "舊版 log 無 curve load policy 儀器",
            )
            return

        expected = (
            "DT curve load policy latest-only shared-loader "
            "entries=512 maxMB=256 scale=merged-only"
        )
        unique = sorted(set(lines))
        ok = len(lines) == 1 and unique == [expected]
        report.add(
            self.domain,
            "D3.curve-policy",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"行數={len(lines)} 實際={' | '.join(unique)}",
        )

    def _check_single_row_curve(self, session: FlowSession, report: CheckReport) -> None:
        intents = [
            (index, grab_id(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith("ui:【報表序號】")
        ]
        if not intents:
            report.add(self.domain, "D3.row-curve", CheckStatus.NOT_COVERED, "無報表序號操作")
            return

        pattern = re.compile(
            r"^DT row curve load (\d{6}-\d{6}) source=shared "
            r"storage=(summary|bins|memory-summary|memory-bins) "
            r"points=(\d+) pitch=([0-9.]+)mm$"
        )
        invalid = []
        loaded = 0
        unavailable = 0
        terminals = []
        for index, line in enumerate(session.lines):
            if line.message.startswith("DT row curve load ") or line.message.startswith("DT row curve missing "):
                terminals.append((index, line.message))
            if line.message.startswith("DT row curve missing "):
                unavailable += 1
                continue
            if not line.message.startswith("DT row curve load "):
                continue
            match = pattern.match(line.message)
            if not match or int(match.group(3)) <= 0 or float(match.group(4)) <= 0:
                invalid.append(line.message)
                continue
            loaded += 1

        final_index, final_id = intents[-1]
        final_terminal = any(
            index > final_index and message.startswith((
                f"DT row curve load {final_id} ",
                f"DT row curve missing {final_id}"))
            for index, message in terminals
        )
        status = CheckStatus.PASS if final_terminal and not invalid else CheckStatus.FAIL
        report.add(
            self.domain,
            "D3.row-curve",
            status,
            f"intent={len(intents)} loaded={loaded} legacy/missing={unavailable} "
            f"final={'ok' if final_terminal else 'missing'} 格式錯誤={len(invalid)}",
        )

    def _check_curve_summary_writes(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        lines = [
            line.message for line in session.lines
            if line.message.startswith("DT curve summary ")
        ]
        if not lines:
            report.add(
                self.domain,
                "D3.summary-write",
                CheckStatus.NOT_COVERED,
                "本 session 沒有由 bins 重建匯總",
            )
            return

        pattern = re.compile(
            r"^DT curve summary (\d{6}-\d{6}) "
            r"write=(queued|ok|failed|dropped|evicted|skip-incomplete) "
            r"captures=(\d+) merged=(\d+) ms=(\d+)"
            r"(?: reason=(idle|pressure))?$"
        )
        counts = {
            "queued": 0,
            "ok": 0,
            "failed": 0,
            "dropped": 0,
            "evicted": 0,
            "skip-incomplete": 0,
        }
        invalid = []
        for message in lines:
            match = pattern.match(message)
            if not match:
                invalid.append(message)
                continue
            status = match.group(2)
            captures = int(match.group(3))
            merged = int(match.group(4))
            reason = match.group(6)
            counts[status] += 1
            if status in ("queued", "ok") and captures != merged:
                invalid.append(message)
            if status in ("ok", "failed") and reason not in ("idle", "pressure"):
                invalid.append(message)

        bad = counts["failed"] + counts["dropped"] + counts["evicted"]
        ok = not invalid and bad == 0
        report.add(
            self.domain,
            "D3.summary-write",
            CheckStatus.PASS if ok else CheckStatus.FAIL,
            f"queued={counts['queued']} ok={counts['ok']} "
            f"failed={counts['failed']} dropped={counts['dropped']} "
            f"evicted={counts['evicted']} incomplete={counts['skip-incomplete']} "
            f"格式錯誤={len(invalid)}"
            + (f"；首筆 {invalid[0]}" if invalid else ""),
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
            if (
                line.message.startswith(
                    ("ui:【報表", "ui:【明細列表】", "ui:【序號範圍-", "ui:【期間-", "ui:【良率")
                )
                or (
                    line.message.startswith("DT ")
                    and not line.message.startswith(("DT curve load policy ", "DT range policy "))
                )
            )
        ]
        if not data_times:
            report.add(self.domain, "U.stall", CheckStatus.NOT_COVERED, "無可量測的報表互動")
            return
        assessment = assess_ui_responsiveness(session, data_times, limit_ms)
        report.add(
            self.domain,
            "U.stall",
            CheckStatus.PASS if assessment.passed else CheckStatus.FAIL,
            assessment.detail(limit_ms),
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
