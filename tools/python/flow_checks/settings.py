"""Settings (S-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession, grab_id


class SettingsFlowValidator:
    domain = "SETTINGS"

    _setting_pattern = re.compile(
        r"^(?P<source>ui:設定|set:)\[(?P<name>[^\]]+)\]"
        r"(?:(?:=(?P<value>.+))|(?: → (?P<arrow>.+)))$"
    )
    _row_pattern = re.compile(
        r"^(?:LC|RV) row (?:rowChart|rowView) "
        r"dir=(?P<direction>TopToBottom|BottomToTop)\b"
    )
    _route_pattern = re.compile(
        r"^setting route (?P<name>\S+) "
        r"owner=(?P<owner>\w+) effects=(?P<effects>[\w+]+)$"
    )
    _capture_policy_settings = {
        "AniloxRootPath",
        "EnableAutoCapture",
        "SaveOriginalBmp",
        "dc_HessianMaxFactorV",
        "de_RidgeSigma",
        "eb_RidgeDir",
    }

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        settings = [
            (index, line, self._setting_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith(("ui:設定[", "set:["))
            and not line.message.startswith("set:[顯示基線]")
        ]
        if not settings:
            report.add(self.domain, "S0", CheckStatus.NOT_COVERED, "本 session 無設定變更")
            return report

        self._check_format(settings, report)
        self._check_routes(session, settings, report)
        self._check_review_enhance(session, settings, report)
        self._check_live_enhance(session, settings, report)
        self._check_enhance_heatmap(session, settings, report)
        self._check_direction_refresh(session, settings, report)
        return report

    def _check_format(self, settings, report: CheckReport) -> None:
        failures = []
        for _, line, match in settings:
            if not match:
                failures.append(f"{line.timestamp} {line.message}")
                continue
            name = match.group("name")
            value = match.group("value") or match.group("arrow")
            if not name.strip() or not value.strip():
                failures.append(f"{line.timestamp} 屬性名或值為空")

        report.add(
            self.domain,
            "S0.format",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(settings)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_routes(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        failures = []
        for index, line, setting_match in settings:
            if not setting_match:
                continue
            expected_name = setting_match.group("name")
            if index + 1 >= len(session.lines):
                failures.append(f"{line.timestamp} {expected_name} 缺 route")
                continue

            route_line = session.lines[index + 1]
            route_match = self._route_pattern.match(route_line.message)
            if not route_match or route_match.group("name") != expected_name:
                failures.append(
                    f"{line.timestamp} {expected_name} 下一行不是同名 route"
                )
                continue

            effects = set(route_match.group("effects").split("+"))
            has_capture_policy = "CapturePolicy" in effects
            expects_capture_policy = expected_name in self._capture_policy_settings
            if has_capture_policy != expects_capture_policy:
                failures.append(
                    f"{line.timestamp} {expected_name} CapturePolicy={has_capture_policy}"
                )

        report.add(
            self.domain,
            "S0.route",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(settings)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_review_enhance(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hd_EnableReviewEnhance"
        ]
        if not changes:
            report.add(
                self.domain,
                "S2.review-enhance",
                CheckStatus.NOT_COVERED,
                "未切換回顧強化",
            )
            return

        failures = []
        exercised = 0
        for sequence, (index, line, _) in enumerate(changes):
            previous_view = next(
                (
                    item
                    for item in reversed(session.lines[:index])
                    if item.message.startswith(("RV loadGrab done ", "RV period load "))
                ),
                None,
            )
            if previous_view is None:
                continue

            next_change = (
                changes[sequence + 1][0]
                if sequence + 1 < len(changes)
                else len(session.lines)
            )
            window = session.lines[index + 1:next_change]
            exercised += 1
            if previous_view.message.startswith("RV period load "):
                period_match = re.match(
                    r"^RV period load (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}) ",
                    previous_view.message,
                )
                if period_match is None or not any(
                    item.message.startswith(
                        f"RV period load {period_match.group(1)} "
                    )
                    for item in window
                ):
                    failures.append(
                        f"{line.timestamp} period 缺同時點 RV period load"
                    )
                if not any(
                    item.message == "RV period curves=keep source=display"
                    for item in window
                ):
                    failures.append(
                        f"{line.timestamp} period 強化不應重畫 Curve"
                    )
                if not any(
                    item.message.startswith("RV pushFrames ")
                    and "chartView=keep" in item.message
                    for item in window
                ):
                    failures.append(
                        f"{line.timestamp} period 強化未保留 chart view"
                    )
                continue

            current_id = grab_id(previous_view.message)
            begin_pos = next(
                (
                    pos
                    for pos, item in enumerate(window)
                    if item.message.startswith("RV loadGrab begin ")
                    and grab_id(item.message) == current_id
                ),
                None,
            )
            if begin_pos is None:
                failures.append(
                    f"{line.timestamp} grab={current_id} 缺 RV loadGrab begin"
                )
                continue
            if not any(
                item.message.startswith("RV loadGrab done ")
                and grab_id(item.message) == current_id
                for item in window[begin_pos + 1:]
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 缺 RV loadGrab done"
                )
                continue
            load_window = window[begin_pos + 1:]
            if not any(
                item.message ==
                f"RV loadGrab curves=keep source=display {current_id}"
                for item in load_window
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 強化未保留既有 Curve"
                )
            if any(
                item.message.startswith(f"RV prefit {current_id} ")
                for item in load_window
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 強化不應重新 prefit"
                )
            if any(
                item.message.startswith("RV loadGrab curves=load source=")
                and item.message.endswith(f" {current_id}")
                for item in load_window
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 強化不應重讀 Curve"
                )
            if not any(
                item.message.startswith("RV pushFrames ")
                and "chartView=keep" in item.message
                for item in load_window
            ):
                failures.append(
                    f"{line.timestamp} grab={current_id} 強化未保留 chart view"
                )

        if exercised == 0:
            report.add(
                self.domain,
                "S2.review-enhance",
                CheckStatus.NOT_COVERED,
                f"changes={len(changes)}；當時尚無已載入回顧序號",
            )
            return
        report.add(
            self.domain,
            "S2.review-enhance",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} exercised={exercised} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_live_enhance(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hc_EnableMuraEnhance"
        ]
        if not changes:
            report.add(
                self.domain,
                "S4.live-enhance",
                CheckStatus.NOT_COVERED,
                "未切換監控強化",
            )
            return
        if not any(
            line.message.startswith("live enhance enabled=")
            for line in session.lines
        ):
            report.add(
                self.domain,
                "S4.live-enhance",
                CheckStatus.NOT_COVERED,
                "舊版 log 無監控強化持續狀態儀器",
            )
            return

        pattern = re.compile(
            r"^live enhance enabled=(True|False) direction=(raw|column|row) "
            r"cams=(\d+) scope=all-cameras waterfallHistory=preserved$"
        )
        failures = []
        exercised = 0
        for sequence, (index, line, match) in enumerate(changes):
            next_change = (
                changes[sequence + 1][0]
                if sequence + 1 < len(changes)
                else len(session.lines)
            )
            expected = (match.group("value") or match.group("arrow")).strip()
            state_line = next(
                (
                    item for item in session.lines[index + 1:next_change]
                    if item.message.startswith("live enhance enabled=")
                ),
                None,
            )
            if state_line is None:
                failures.append(f"{line.timestamp} 缺持續狀態行")
                continue
            state_match = pattern.match(state_line.message)
            if not state_match:
                failures.append(f"{state_line.timestamp} 格式錯誤")
                continue
            cameras = int(state_match.group(3))
            if cameras == 0:
                continue
            exercised += 1
            if state_match.group(1).lower() != expected.lower():
                failures.append(
                    f"{line.timestamp} 設定={expected} 實際={state_match.group(1)}"
                )
            layer = state_match.group(2)
            enabled = state_match.group(1).lower() == "true"
            if (enabled and layer == "raw") or (not enabled and layer != "raw"):
                failures.append(
                    f"{state_line.timestamp} enabled={enabled} 與 direction={layer} 矛盾"
                )
            waterfall_lines = [
                item for item in session.lines[index + 1:next_change]
                if item.message.startswith("WF layer ")
            ]
            for waterfall_line in waterfall_lines:
                waterfall_match = re.match(
                    r"^WF layer (raw|column|row)->(raw|column|row) "
                    r"writeRow=(\d+) history=preserved$",
                    waterfall_line.message,
                )
                if waterfall_match is None:
                    failures.append(f"{waterfall_line.timestamp} WF layer 格式錯誤")
                elif waterfall_match.group(2) != layer:
                    failures.append(
                        f"{waterfall_line.timestamp} WF={waterfall_match.group(2)} "
                        f"但 manager={layer}"
                    )

        if exercised == 0 and not failures:
            report.add(
                self.domain,
                "S4.live-enhance",
                CheckStatus.NOT_COVERED,
                f"changes={len(changes)}；切換時沒有已配置相機",
            )
            return
        report.add(
            self.domain,
            "S4.live-enhance",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} exercised={exercised} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_enhance_heatmap(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hda_EnhanceHeatmap"
        ]
        if not changes:
            report.add(
                self.domain,
                "S5.enhance-heatmap",
                CheckStatus.NOT_COVERED,
                "未切換強化熱力圖",
            )
            return

        pattern = re.compile(
            r"^enhance heatmap mode=(Off|Cold|Warm|BlueYellowRed) "
            r"live=(cold|warm|blue-yellow-red|gray) review=(cold|warm|blue-yellow-red|gray) "
            r"scope=main-only data=unchanged$"
        )
        failures = []
        for index, line, match in changes:
            expected = (match.group("value") or match.group("arrow")).strip()
            state_index = index + 2  # intent 後固定為 setting route，再來必須是顯示狀態行
            if state_index >= len(session.lines):
                failures.append(f"{line.timestamp} 缺熱力圖狀態行")
                continue
            state_line = session.lines[state_index]
            state_match = pattern.match(state_line.message)
            if state_match is None:
                failures.append(f"{state_line.timestamp} 熱力圖狀態行缺失或格式錯誤")
                continue
            mode = state_match.group(1)
            if mode.lower() != expected.lower():
                failures.append(
                    f"{line.timestamp} 設定={expected} 實際={mode}"
                )
            if mode == "Off" and (
                state_match.group(2) != "gray" or state_match.group(3) != "gray"
            ):
                failures.append(f"{state_line.timestamp} 關閉後主畫面仍有熱力圖")
            expected_map = {
                "Cold": "cold",
                "Warm": "warm",
                "BlueYellowRed": "blue-yellow-red",
            }.get(mode)
            if expected_map is not None:
                for target, actual in (
                    ("live", state_match.group(2)),
                    ("review", state_match.group(3)),
                ):
                    if actual not in ("gray", expected_map):
                        failures.append(
                            f"{state_line.timestamp} {target}={actual} 與 mode={mode} 不一致"
                        )

        report.add(
            self.domain,
            "S5.enhance-heatmap",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_direction_refresh(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hee_VerticalDirection"
        ]
        if not changes:
            report.add(
                self.domain,
                "S3.direction",
                CheckStatus.NOT_COVERED,
                "未切換上下方向",
            )
            return

        if not session.dvt_enabled:
            report.add(
                self.domain, "S3.direction", CheckStatus.NOT_COVERED,
                "記錄範圍為日常運行；請切到流程驗證後重跑",
            )
            return

        session = session.dvt_only()
        settings = [
            (index, line, self._setting_pattern.match(line.message))
            for index, line in enumerate(session.lines)
            if line.message.startswith(("ui:設定[", "set:["))
            and not line.message.startswith("set:[顯示基線]")
        ]
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "hee_VerticalDirection"
        ]
        if not changes:
            report.add(
                self.domain, "S3.direction", CheckStatus.NOT_COVERED,
                "流程驗證模式期間未切換上下方向",
            )
            return

        failures = []
        for sequence, (index, line, match) in enumerate(changes):
            expected = match.group("value") or match.group("arrow")
            next_change = (
                changes[sequence + 1][0]
                if sequence + 1 < len(changes)
                else len(session.lines)
            )
            window = session.lines[index + 1:next_change]
            row_directions = [
                row_match.group("direction")
                for item in window
                for row_match in [self._row_pattern.match(item.message)]
                if row_match
            ]
            if expected not in row_directions:
                found = ",".join(sorted(set(row_directions))) or "none"
                failures.append(
                    f"{line.timestamp} expected={expected} rowDirections={found}"
                )

        report.add(
            self.domain,
            "S3.direction",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
