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
    _live_apply_pattern = re.compile(
        r"^live inspection apply setting=(?P<name>\S+) "
        r"hm=(?P<hm_c>\d+(?:\.\d+)?)/(?P<hm_r>\d+(?:\.\d+)?) "
        r"thresholdC=(?P<mean_c>\d+(?:\.\d+)?)/(?P<max_c>\d+(?:\.\d+)?) "
        r"thresholdR=(?P<mean_r>\d+(?:\.\d+)?)/(?P<max_r>\d+(?:\.\d+)?) "
        r"mode=(?P<mode>Mean|Max|Both) direction=(?P<direction>Vertical|Horizontal|Both) "
        r"action=(?P<action>normalization-latest|refresh)$"
    )
    _row_metric_pattern = re.compile(
        r"^(?P<scope>LC|RV|DT) row metrics "
        r"mean=(?P<mean>True|False) max=(?P<max>True|False)$"
    )
    _live_curve_applied_pattern = re.compile(
        r"^live curve applied setting=(?P<name>\S+) generation=(?P<generation>\d+) "
        r"hm=(?P<hm_c>\d+(?:\.\d+)?)/(?P<hm_r>\d+(?:\.\d+)?) "
        r"colMeanPeak=(?P<col_mean>\d+(?:\.\d+)?) "
        r"colMaxPeak=(?P<col_max>\d+(?:\.\d+)?) "
        r"rowMeanPeak=(?P<row_mean>\d+(?:\.\d+)?) "
        r"rowMaxPeak=(?P<row_max>\d+(?:\.\d+)?) "
        r"rowAction=(?P<row_action>rescale-current|replace-current|none) "
        r"rowWrite=(?P<row_write_before>\d+)->(?P<row_write_after>\d+)$"
    )
    _live_image_scale_pattern = re.compile(
        r"^live image scale source=(?P<source>[\w-]+) "
        r"captureHm=(?P<capture>\d+(?:\.\d+)?) "
        r"currentHm=(?P<hm_c>\d+(?:\.\d+)?)/(?P<hm_r>\d+(?:\.\d+)?) "
        r"scale=(?P<scale_c>\d+(?:\.\d+)?)/(?P<scale_r>\d+(?:\.\d+)?)$"
    )
    _live_pixel_curve_probe_pattern = re.compile(
        r"^live pixel-curve probe cam(?P<camera>\d+) tick=(?P<tick>\d+) "
        r"axis=(?P<axis>C|R) captureHm=(?P<capture>\d+(?:\.\d+)?) "
        r"currentHm=(?P<current>\d+(?:\.\d+)?) "
        r"sourceGain=(?P<source_gain>\d+(?:\.\d+)?) "
        r"imagePeak=(?P<image>\d+(?:\.\d+)?) "
        r"curveMeanPeak=(?P<mean>\d+(?:\.\d+)?) "
        r"curveMaxPeak=(?P<maximum>\d+(?:\.\d+)?) "
        r"delta=(?P<delta>\d+(?:\.\d+)?) "
        r"sourceImageMax=(?P<source_image>\d+(?:\.\d+)?) "
        r"sourceCurveMax=(?P<source_curve>\d+(?:\.\d+)?) "
        r"verdict=(?P<verdict>match|mismatch) "
        r"reason=(?P<reason>none|quantized-zero|max-delta)$"
    )
    _review_normalization_queued_pattern = re.compile(
        r"^RV normalization queued generation=(?P<generation>\d+) setting=(?P<name>\S+)$"
    )
    _review_normalization_settle_pattern = re.compile(
        r"^RV normalization settle generation=(?P<generation>\d+) setting=(?P<name>\S+) "
        r"hm=(?P<hm_c>\d+(?:\.\d+)?)/(?P<hm_r>\d+(?:\.\d+)?)$"
    )
    _live_row_scale_pattern = re.compile(
        r"^live row normalize captureHm=(?P<capture>\d+(?:\.\d+)?) "
        r"rowHm=(?P<row>\d+(?:\.\d+)?) ratio=(?P<ratio>\d+(?:\.\d+)?)$"
    )
    _report_curve_refresh_pattern = re.compile(
        r"^DT curve refresh (?P<grab>\S+) "
        r"reason=setting-(?P<setting>\S+) "
        r"column=(?P<column>True|False) row=(?P<row>True|False) "
        r"source=memory preserveRange=True "
        r"rangeDelta=(?P<range_delta>\d+(?:\.\d+)?)$"
    )
    _hessian_standard_pattern = re.compile(
        r"^RV hessian standard (?P<grab>\S+) dir=(?P<direction>C|R) "
        r"gain=(?P<gain>\d+(?:\.\d+)?) scale=(?P<scale>\d+) "
        r"sampleMin=(?P<minimum>\d+) sampleMax=(?P<maximum>\d+) "
        r"sampleMean=(?P<mean>\d+(?:\.\d+)?)$"
    )
    _live_stimulus_pattern = re.compile(
        r"^live inspection stimulus brightness=(?P<brightness>\d+) "
        r"direction=(?P<direction>col|row) "
        r"mean=(?P<mean>\d+(?:\.\d+)?) max=(?P<max>\d+(?:\.\d+)?) "
        r"threshold=(?P<th_mean>\d+(?:\.\d+)?)/(?P<th_max>\d+(?:\.\d+)?) "
        r"mode=(?P<mode>Mean|Max|Both) verdict=(?P<verdict>O|X) "
        r"source=light-surrogate-not-mura$"
    )
    _live_inspection_settings = {
        "dc_HessianMaxFactorV",
        "dd_HessianMaxFactorH",
        "eb_RidgeDir",
        "eca_ColumnCurveMode",
        "ec_ErrorValueMeanV",
        "ed_ErrorValueMaxV",
        "ee_ErrorValueMeanH",
        "ef_ErrorValueMaxH",
    }
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
        self._check_live_inspection_settings(session, settings, report)
        self._check_curve_metric_visibility(session, settings, report)
        self._check_live_normalization_output(session, settings, report)
        self._check_live_pixel_curve_consistency(session, report)
        self._check_live_row_scale(session, report)
        self._check_report_curve_refresh(session, report)
        self._check_hessian_standard_gain(session, report)
        self._check_live_inspection_stimulus(session, report)
        self._check_review_enhance(session, settings, report)
        self._check_live_enhance(session, settings, report)
        self._check_enhance_heatmap(session, settings, report)
        self._check_display_crop(session, settings, report)
        self._check_live_layout(session, settings, report)
        self._check_direction_refresh(session, settings, report)
        return report

    def _check_curve_metric_visibility(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") == "eca_ColumnCurveMode"
        ]
        if not changes:
            report.add(
                self.domain, "S1.curve-metrics", CheckStatus.NOT_COVERED,
                "no column/row curve metric mode change in this session",
            )
            return

        failures = []
        for index, line, match in changes:
            value = (match.group("value") or match.group("arrow") or "").strip()
            if value == "Mean" or "平均" in value:
                expected = (True, False)
            elif value == "Max" or "最大" in value:
                expected = (False, True)
            elif value == "Both" or "兩者" in value:
                expected = (True, True)
            else:
                failures.append(f"{line.timestamp} unknown curve mode={value}")
                continue

            found = {}
            for candidate in session.lines[index + 1:]:
                if candidate.message.startswith(("ui:設定[", "set:[")):
                    break
                parsed = self._row_metric_pattern.match(candidate.message)
                if parsed:
                    found[parsed.group("scope")] = (
                        parsed.group("mean") == "True",
                        parsed.group("max") == "True",
                    )

            for scope in ("LC", "RV", "DT"):
                if scope not in found:
                    failures.append(f"{line.timestamp} {scope} row metric evidence missing")
                elif found[scope] != expected:
                    failures.append(
                        f"{line.timestamp} {scope} row metrics={found[scope]} expected={expected}"
                    )

        report.add(
            self.domain,
            "S1.curve-metrics",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_live_pixel_curve_consistency(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        samples = []
        for line in session.lines:
            match = self._live_pixel_curve_probe_pattern.match(line.message)
            if match:
                samples.append(match)

        if not samples:
            report.add(
                self.domain,
                "S1.live-pixel-curve",
                CheckStatus.NOT_COVERED,
                "no DVT pixel/curve sample was captured during Grab",
            )
            return

        failures = []
        axes = set()
        for sample in samples:
            axis = sample.group("axis")
            axes.add(axis)
            delta = float(sample.group("delta"))
            verdict = sample.group("verdict")
            source_image = float(sample.group("source_image"))
            source_curve = float(sample.group("source_curve"))
            information_lost = source_image == 0.0 and source_curve > 0.0
            expected = (
                "match"
                if not information_lost and delta <= (1.0 / 255.0) + 0.0001
                else "mismatch"
            )
            if verdict != expected:
                failures.append(
                    f"cam{sample.group('camera')} axis={axis} "
                    f"verdict={verdict} expected={expected} delta={delta:.4f} "
                    f"reason={sample.group('reason')}"
                )
            elif verdict == "mismatch":
                failures.append(
                    f"cam{sample.group('camera')} axis={axis} "
                    f"image={sample.group('image')} curveMax={sample.group('maximum')} "
                    f"delta={delta:.4f}"
                )

        missing_axes = {"C", "R"} - axes
        if missing_axes:
            failures.append("missing axes=" + ",".join(sorted(missing_axes)))

        report.add(
            self.domain,
            "S1.live-pixel-curve",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(samples)} axes={','.join(sorted(axes))} "
            f"failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

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

    def _check_live_inspection_settings(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") in self._live_inspection_settings
        ]
        if not changes:
            report.add(
                self.domain, "S1.live-apply", CheckStatus.NOT_COVERED,
                "未調整監控檢測標準",
            )
            return

        failures = []
        for index, line, match in changes:
            name = match.group("name")
            route_match = (
                self._route_pattern.match(session.lines[index + 1].message)
                if index + 1 < len(session.lines) else None
            )
            if route_match is None or "LiveInspectionCurves" not in set(
                route_match.group("effects").split("+")
            ):
                failures.append(f"{line.timestamp} {name} 缺 LiveInspectionCurves route")
                continue

            apply_match = None
            for candidate in session.lines[index + 2:index + 20]:
                if candidate.message.startswith(("ui:設定[", "set:[")):
                    break
                parsed = self._live_apply_pattern.match(candidate.message)
                if parsed and parsed.group("name") == name:
                    apply_match = parsed
                    break
            if apply_match is None:
                failures.append(f"{line.timestamp} {name} 缺 live inspection apply")
                continue

            expected_action = (
                "normalization-latest"
                if name in {"dc_HessianMaxFactorV", "dd_HessianMaxFactorH"}
                else "refresh"
            )
            if apply_match.group("action") != expected_action:
                failures.append(
                    f"{line.timestamp} {name} action={apply_match.group('action')}"
                )

            value = (match.group("value") or match.group("arrow")).strip()
            numeric_fields = {
                "dc_HessianMaxFactorV": "hm_c",
                "dd_HessianMaxFactorH": "hm_r",
                "ec_ErrorValueMeanV": "mean_c",
                "ed_ErrorValueMaxV": "max_c",
                "ee_ErrorValueMeanH": "mean_r",
                "ef_ErrorValueMaxH": "max_r",
            }
            field = numeric_fields.get(name)
            if field is not None:
                try:
                    if abs(float(value) - float(apply_match.group(field))) > 0.0001:
                        failures.append(
                            f"{line.timestamp} {name} intent={value} "
                            f"applied={apply_match.group(field)}"
                        )
                except ValueError:
                    failures.append(f"{line.timestamp} {name} 非數值 intent={value}")

        report.add(
            self.domain,
            "S1.live-apply",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_live_normalization_output(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        normalization_names = {
            "dc_HessianMaxFactorV",
            "dd_HessianMaxFactorH",
        }
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") in normalization_names
        ]
        if not changes:
            report.add(
                self.domain, "S1.live-normalization-output",
                CheckStatus.NOT_COVERED,
                "no live normalization change in this session",
            )
            return

        applied = []
        image_scales = []
        review_queued = []
        review_settled = []
        latest_hm_c = None
        latest_hm_r = None
        failures = []
        for line in session.lines:
            apply_match = self._live_apply_pattern.match(line.message)
            if apply_match:
                latest_hm_c = float(apply_match.group("hm_c"))
                latest_hm_r = float(apply_match.group("hm_r"))
                continue

            curve_match = self._live_curve_applied_pattern.match(line.message)
            if curve_match:
                applied.append(curve_match)
                if latest_hm_c is None or latest_hm_r is None:
                    failures.append("curve output appeared before live inspection apply")
                    continue
                actual_hm_c = float(curve_match.group("hm_c"))
                actual_hm_r = float(curve_match.group("hm_r"))
                if (abs(actual_hm_c - latest_hm_c) > 0.0001 or
                        abs(actual_hm_r - latest_hm_r) > 0.0001):
                    failures.append(
                        "stale curve output "
                        f"hm={actual_hm_c:g}/{actual_hm_r:g} "
                        f"latest={latest_hm_c:g}/{latest_hm_r:g}"
                    )
                if (curve_match.group("row_action") == "rescale-current"):
                    write_before = int(curve_match.group("row_write_before"))
                    write_after = int(curve_match.group("row_write_after"))
                    if write_before != write_after:
                        failures.append(
                            "waterfall row normalization appended data "
                            f"write={write_before}->{write_after}"
                        )
                continue

            image_match = self._live_image_scale_pattern.match(line.message)
            if image_match:
                image_scales.append(image_match)
                if image_match.group("source") != "adaptive-standard-half":
                    failures.append(
                        f"image source={image_match.group('source')} expected=adaptive-standard-half"
                    )
                capture = float(image_match.group("capture"))
                for axis in ("c", "r"):
                    current = float(image_match.group(f"hm_{axis}"))
                    scale = float(image_match.group(f"scale_{axis}"))
                    expected = current if current > 0 else 1.0
                    if abs(scale - expected) <= 0.0002:
                        continue
                    failures.append(
                        f"image axis={axis.upper()} scale={scale:g} expected={expected:.4f}"
                    )

            queued_match = self._review_normalization_queued_pattern.match(line.message)
            if queued_match:
                review_queued.append(queued_match)
                continue
            settle_match = self._review_normalization_settle_pattern.match(line.message)
            if settle_match:
                review_settled.append(settle_match)

        if review_queued:
            final_queued = review_queued[-1]
            final_generation = int(final_queued.group("generation"))
            matching = [
                match for match in review_settled
                if int(match.group("generation")) == final_generation
            ]
            if not matching:
                failures.append(
                    f"review final generation={final_generation} did not settle"
                )
            elif latest_hm_c is not None and latest_hm_r is not None:
                final_settle = matching[-1]
                if (abs(float(final_settle.group("hm_c")) - latest_hm_c) > 0.0001 or
                        abs(float(final_settle.group("hm_r")) - latest_hm_r) > 0.0001):
                    failures.append("review final settle used stale normalization")

        if not applied and not image_scales and not review_queued:
            report.add(
                self.domain, "S1.live-normalization-output",
                CheckStatus.NOT_COVERED,
                "setting route was exercised but no live frame/curve output was measured",
            )
            return

        report.add(
            self.domain,
            "S1.live-normalization-output",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} curveOutputs={len(applied)} "
            f"imageScales={len(image_scales)} reviewQueued={len(review_queued)} "
            f"reviewSettled={len(review_settled)} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
        )

    def _check_live_row_scale(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        matches = [
            self._live_row_scale_pattern.match(line.message)
            for line in session.lines
        ]
        matches = [match for match in matches if match]
        if not matches:
            report.add(
                self.domain, "S1.row-normalize", CheckStatus.NOT_COVERED,
                "無監控列 Curve 正規值樣本",
            )
            return

        failures = []
        for match in matches:
            capture = float(match.group("capture"))
            row = float(match.group("row"))
            ratio = float(match.group("ratio"))
            expected = capture * row if capture > 0 and row > 0 else 1.0
            if abs(ratio - expected) > 0.0002:
                failures.append(
                    f"capture={capture} row={row} ratio={ratio} expected={expected:.4f}"
                )
        report.add(
            self.domain,
            "S1.row-normalize",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(matches)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_report_curve_refresh(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        matches = [
            self._report_curve_refresh_pattern.match(line.message)
            for line in session.lines
        ]
        matches = [match for match in matches if match]
        if not matches:
            report.add(
                self.domain, "S1.report-preserve-range", CheckStatus.NOT_COVERED,
                "報表單序號未在已載入狀態變更檢出標準",
            )
            return

        expected_axes = {
            "dc_HessianMaxFactorV": ("True", "False"),
            "dd_HessianMaxFactorH": ("False", "True"),
            "ec_ErrorValueMeanV": ("True", "False"),
            "ed_ErrorValueMaxV": ("True", "False"),
            "ee_ErrorValueMeanH": ("False", "True"),
            "ef_ErrorValueMaxH": ("False", "True"),
        }
        failures = []
        for match in matches:
            expected = expected_axes.get(match.group("setting"))
            if expected is None:
                continue
            actual = (match.group("column"), match.group("row"))
            if actual != expected:
                failures.append(
                    f"{match.group('setting')} column/row={actual} expected={expected}"
                )
            if float(match.group("range_delta")) > 0.01:
                failures.append(
                    f"{match.group('setting')} rangeDelta={match.group('range_delta')}mm"
                )
        report.add(
            self.domain,
            "S1.report-preserve-range",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(matches)} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_hessian_standard_gain(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        samples = []
        for line in session.lines:
            match = self._hessian_standard_pattern.match(line.message)
            if match:
                samples.append(match)

        comparable = 0
        failures = []
        grouped = {}
        for match in samples:
            key = (match.group("grab"), match.group("direction"))
            grouped.setdefault(key, []).append(match)

        for key, observations in grouped.items():
            for previous, current in zip(observations, observations[1:]):
                previous_gain = float(previous.group("gain"))
                current_gain = float(current.group("gain"))
                if abs(current_gain - previous_gain) <= 0.000001:
                    continue

                comparable += 1
                previous_mean = float(previous.group("mean"))
                current_mean = float(current.group("mean"))
                previous_max = int(previous.group("maximum"))
                current_max = int(current.group("maximum"))
                increasing = current_gain > previous_gain
                mean_ok = current_mean + 0.001 >= previous_mean if increasing \
                    else current_mean <= previous_mean + 0.001
                max_ok = current_max >= previous_max if increasing \
                    else current_max <= previous_max
                if not mean_ok or not max_ok:
                    failures.append(
                        f"{key[0]}/{key[1]} gain={previous_gain:g}->{current_gain:g} "
                        f"mean={previous_mean:.3f}->{current_mean:.3f} "
                        f"max={previous_max}->{current_max}"
                    )

        if comparable == 0:
            report.add(
                self.domain, "S1.hessian-image-gain", CheckStatus.NOT_COVERED,
                "同一序號、同一欄／列方向需要至少兩個不同正規值的強化圖樣本",
            )
            return

        report.add(
            self.domain,
            "S1.hessian-image-gain",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"samples={len(samples)} comparisons={comparable} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_live_inspection_stimulus(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        samples = [
            match
            for line in session.lines
            for match in [self._live_stimulus_pattern.match(line.message)]
            if match
        ]
        levels = sorted({int(match.group("brightness")) for match in samples})
        complete_levels = [
            level for level in levels
            if {match.group("direction") for match in samples
                if int(match.group("brightness")) == level} == {"col", "row"}
        ]
        if len(complete_levels) < 2:
            report.add(
                self.domain, "S1.light-stimulus", CheckStatus.NOT_COVERED,
                "需要至少兩個亮度且每個亮度都有欄／列樣本；此測試不是正式 Mura 模擬",
            )
            return

        failures = []
        for match in samples:
            mean = float(match.group("mean"))
            maximum = float(match.group("max"))
            th_mean = float(match.group("th_mean"))
            th_max = float(match.group("th_max"))
            mode = match.group("mode")
            expected_fail = (
                (mode != "Max" and mean > th_mean) or
                (mode != "Mean" and maximum > th_max)
            )
            expected = "X" if expected_fail else "O"
            if match.group("verdict") != expected:
                failures.append(
                    f"brightness={match.group('brightness')} "
                    f"direction={match.group('direction')} "
                    f"verdict={match.group('verdict')} expected={expected}"
                )

        changed_directions = set()
        for direction in ("col", "row"):
            by_level = {
                int(match.group("brightness")): (
                    float(match.group("mean")), float(match.group("max")))
                for match in samples if match.group("direction") == direction
            }
            values = [by_level[level] for level in complete_levels if level in by_level]
            if values and any(
                abs(value[0] - values[0][0]) >= 0.00005 or
                abs(value[1] - values[0][1]) >= 0.00005
                for value in values[1:]
            ):
                changed_directions.add(direction)
        if changed_directions != {"col", "row"}:
            failures.append(
                "亮度切換後欄／列 Curve 未同時產生可量測變化：" +
                ",".join(sorted(changed_directions))
            )

        report.add(
            self.domain,
            "S1.light-stimulus",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"levels={complete_levels} samples={len(samples)} failures={len(failures)}；"
            "光源僅為檢測標準替代刺激，不代表真實 Mura 檢出率"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_review_enhance(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        variant_view_pattern = re.compile(
            r"^RV variantView keep .* maxDelta=(missing|\d+(?:\.\d+)?)$"
        )

        def variant_view_failure(lines):
            matches = [
                variant_view_pattern.match(item.message)
                for item in lines
                if item.message.startswith("RV variantView keep ")
            ]
            matches = [match for match in matches if match is not None]
            if not matches:
                return "missing physical viewport preservation evidence"
            value = matches[-1].group(1)
            if value == "missing" or float(value) > 0.1:
                return f"physical viewport drift maxDelta={value}mm"
            return None

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
                variant_failure = variant_view_failure(window)
                if variant_failure:
                    failures.append(f"{line.timestamp} period {variant_failure}")
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
            done_pos = next(
                (
                    pos
                    for pos in range(begin_pos + 1, len(window))
                    if window[pos].message.startswith("RV loadGrab done ")
                    and grab_id(window[pos].message) == current_id
                ),
                None,
            )
            if done_pos is None:
                failures.append(
                    f"{line.timestamp} grab={current_id} 缺 RV loadGrab done"
                )
                continue
            # Only the matching reload belongs to this setting transition. Later drags,
            # zooms, or navigation may legitimately publish a new main range.
            load_window = window[begin_pos + 1:done_pos + 1]
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
            variant_failure = variant_view_failure(load_window)
            if variant_failure:
                failures.append(
                    f"{line.timestamp} grab={current_id} {variant_failure}"
                )
            if any(item.message.startswith("RV mainRange ") for item in load_window):
                failures.append(
                    f"{line.timestamp} grab={current_id} 圖片版本切換不應發布新視野"
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
            transition_lines = session.lines[index + 1:next_change]
            for transition_index, waterfall_line in enumerate(transition_lines):
                if not waterfall_line.message.startswith("WF layer "):
                    continue
                waterfall_match = re.match(
                    r"^WF layer (raw|column|row)->(raw|column|row) "
                    r"writeRow=(\d+) history=preserved$",
                    waterfall_line.message,
                )
                if waterfall_match is None:
                    failures.append(f"{waterfall_line.timestamp} WF layer 格式錯誤")
                    continue

                paired_state = next(
                    (
                        item for item in transition_lines[transition_index + 1:]
                        if item.message.startswith("live enhance enabled=")
                    ),
                    None,
                )
                paired_match = (
                    pattern.match(paired_state.message)
                    if paired_state is not None
                    else None
                )
                if paired_match is None:
                    failures.append(
                        f"{waterfall_line.timestamp} WF layer 後缺對應持續狀態行"
                    )
                elif waterfall_match.group(2) != paired_match.group(2):
                    failures.append(
                        f"{waterfall_line.timestamp} WF={waterfall_match.group(2)} "
                        f"但 manager={paired_match.group(2)}"
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
            r"^enhance heatmap mode=(Off|Cold|Warm|BlueYellowRed|Green) "
            r"live=(cold|warm|blue-yellow-red|green|gray) review=(cold|warm|blue-yellow-red|green|gray) "
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
                "Green": "green",
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

    def _check_display_crop(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") in ("cb_CropHead", "cc_CropTail")
        ]
        if not changes:
            report.add(
                self.domain,
                "S6.display-crop",
                CheckStatus.NOT_COVERED,
                "未調整顯示去頭／去尾",
            )
            return
        if not session.dvt_enabled:
            report.add(
                self.domain,
                "S6.display-crop",
                CheckStatus.NOT_COVERED,
                "記錄範圍為日常運行；請切到流程驗證後重跑",
            )
            return

        state_pattern = re.compile(
            r"^displayCrop head=(?P<head>\d+(?:\.\d+)?) "
            r"tail=(?P<tail>\d+(?:\.\d+)?) "
            r"scope=main\+column-chart data=unchanged "
            r"waterfallHistory=preserved$"
        )
        actual_pattern = re.compile(
            r"^displayCrop applied head=(?P<head>\d+(?:\.\d+)?) "
            r"tail=(?P<tail>\d+(?:\.\d+)?) "
            r"mode=(?P<mode>IC|WF) "
            r"content=(?P<width>\d+)x(?P<height>\d+) "
            r"zoom=(?P<zoom>\d+(?:\.\d+)?) "
            r"fit=(?P<fit>True|False) frames=dynamic$"
        )
        pending_pattern = re.compile(
            r"^capture layout pending grab=(?P<grab>\S+) "
            r"setting=(?P<setting>\S+) "
            r"apply=(?P<apply>display-now\+stop-final|stop-final)$"
        )
        applied_pattern = re.compile(
            r"^capture layout applied grab=(?P<grab>\S+) timing=stop "
            r"ops=\S+ start=\S+ speed=\S+ "
            r"head=(?P<head>\d+(?:\.\d+)?) "
            r"tail=(?P<tail>\d+(?:\.\d+)?) "
            r"render=(?P<render>once|already-applied) source=unchanged$"
        )
        failures = []
        blank_changes = 0
        deferred = {}
        setting_indices = [item[0] for item in settings]
        for index, line, match in changes:
            next_change = next(
                (
                    setting_index
                    for setting_index in setting_indices
                    if setting_index > index
                ),
                len(session.lines),
            )
            pending_line = next(
                (
                    item
                    for item in session.lines[index + 2:next_change]
                    if item.message.startswith("capture layout pending ")
                ),
                None,
            )
            if pending_line is not None:
                pending_match = pending_pattern.match(pending_line.message)
                if (
                    pending_match is None
                    or pending_match.group("setting") != match.group("name")
                    or pending_match.group("apply") != "display-now+stop-final"
                ):
                    failures.append(f"{line.timestamp} 延後布局狀態行格式錯誤")
                    continue
                deferred.setdefault(pending_match.group("grab"), []).append(
                    (index, line, match, pending_line)
                )

            state_line = next(
                (
                    item
                    for item in session.lines[index + 2:next_change]
                    if item.message.startswith("displayCrop head=")
                ),
                None,
            )
            if state_line is None:
                failures.append(f"{line.timestamp} 缺顯示裁切狀態行")
                continue
            state_match = state_pattern.match(state_line.message)
            if state_match is None:
                failures.append(f"{state_line.timestamp} 顯示裁切狀態行缺失或格式錯誤")
                continue

            try:
                expected = float(
                    (match.group("value") or match.group("arrow")).strip()
                )
            except ValueError:
                failures.append(f"{line.timestamp} Crop 值無法解析")
                continue
            actual = float(
                state_match.group(
                    "head" if match.group("name") == "cb_CropHead" else "tail"
                )
            )
            if abs(actual - expected) > 0.011:
                failures.append(
                    f"{line.timestamp} 設定={expected:g} 實際={actual:g}"
                )
            actual_line = next(
                (
                    item
                    for item in session.lines[index + 2:next_change]
                    if item.message.startswith("displayCrop applied ")
                ),
                None,
            )
            actual_match = actual_pattern.match(actual_line.message) if actual_line else None
            if actual_match is None:
                failures.append(f"{line.timestamp} 缺實際畫布 Crop 後置條件")
            elif (
                int(actual_match.group("width")) <= 0
                or int(actual_match.group("height")) <= 0
            ):
                if pending_line is not None:
                    failures.append(
                        f"{line.timestamp} Grab 中 Crop 後置畫布仍為空"
                    )
                else:
                    blank_changes += 1

        for current_id, items in deferred.items():
            last_pending = items[-1][3]
            applied_line = next(
                (
                    item
                    for item in session.lines
                    if item.elapsed >= last_pending.elapsed
                    and item.message.startswith(
                        f"capture layout applied grab={current_id} "
                    )
                ),
                None,
            )
            if applied_line is None:
                failures.append(f"grab={current_id} 延後布局未在 Stop 套用")
                continue
            applied_match = applied_pattern.match(applied_line.message)
            if applied_match is None:
                failures.append(f"{applied_line.timestamp} applied 格式錯誤")
                continue

            final_line = next(
                (
                    item
                    for item in session.lines
                    if item.elapsed >= last_pending.elapsed
                    and item.message.startswith(
                        f"capture layout final grab={current_id} "
                    )
                ),
                None,
            )
            if final_line is None:
                failures.append(f"grab={current_id} 延後布局缺 final")
                continue

            actual_line = next(
                (
                    item
                    for item in session.lines
                    if item.elapsed >= (
                        last_pending.elapsed
                        if applied_match.group("render") == "already-applied"
                        else final_line.elapsed
                    )
                    and item.elapsed <= applied_line.elapsed
                    and item.message.startswith("displayCrop applied ")
                ),
                None,
            )
            actual_match = actual_pattern.match(actual_line.message) if actual_line else None
            if (
                actual_match is None
                or int(actual_match.group("width")) <= 0
                or int(actual_match.group("height")) <= 0
            ):
                failures.append(
                    f"grab={current_id} 延後布局缺實際畫布 Crop 後置條件"
                )
            elif (
                abs(float(actual_match.group("head")) - float(applied_match.group("head"))) > 0.011
                or abs(float(actual_match.group("tail")) - float(applied_match.group("tail"))) > 0.011
            ):
                failures.append(
                    f"grab={current_id} 畫布 Crop 與最終布局不一致"
                )

            early_crop = next(
                (
                    item
                    for item in session.lines
                    if last_pending.elapsed <= item.elapsed < final_line.elapsed
                    and item.message.startswith("displayCrop head=")
                ),
                None,
            )
            if (
                early_crop is not None
                and applied_match.group("render") == "once"
            ):
                failures.append(
                    f"{early_crop.timestamp} Grab 中提前套用顯示裁切"
                )

            last_by_name = {}
            for _, line, match, _ in items:
                last_by_name[match.group("name")] = (line, match)
            for name, (line, match) in last_by_name.items():
                try:
                    expected = float(
                        (match.group("value") or match.group("arrow")).strip()
                    )
                except ValueError:
                    failures.append(f"{line.timestamp} Crop 值無法解析")
                    continue
                field = "head" if name == "cb_CropHead" else "tail"
                actual = float(applied_match.group(field))
                if abs(actual - expected) > 0.011:
                    failures.append(
                        f"grab={current_id} 最後{name}={expected:g} "
                        f"applied={actual:g}"
                    )

        report.add(
            self.domain,
            "S6.display-crop",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} deferredGrabs={len(deferred)} "
            f"blankCanvas={blank_changes} failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_live_layout(
        self, session: FlowSession, settings, report: CheckReport
    ) -> None:
        ops_names = (
            "ab_OpsCam1", "ac_OpsCam2", "ad_OpsCam3", "ae_OpsCam4",
            "af_OpsCam5", "ag_OpsCam6", "ah_OpsCam7",
        )
        start_names = (
            "bb_StartCam1", "bc_StartCam2", "bd_StartCam3", "be_StartCam4",
            "bf_StartCam5", "bg_StartCam6", "bh_StartCam7",
        )
        name_to_slot = {
            name: ("ops", index) for index, name in enumerate(ops_names)
        }
        name_to_slot.update(
            {name: ("start", index) for index, name in enumerate(start_names)}
        )
        changes = [
            (index, line, match)
            for index, line, match in settings
            if match and match.group("name") in name_to_slot
        ]
        if not changes:
            report.add(
                self.domain,
                "S7.live-layout",
                CheckStatus.NOT_COVERED,
                "No OPS/Start setting change in this session.",
            )
            return
        if not session.dvt_enabled:
            report.add(
                self.domain,
                "S7.live-layout",
                CheckStatus.NOT_COVERED,
                "OPS/Start state snapshots require DVT logging.",
            )
            return

        state_pattern = re.compile(
            r"^displayLayout applied setting=(?P<setting>\S+) "
            r"refGrid=cam1 ops=(?P<ops>\S+) start=(?P<start>\S+) "
            r"speed=\S+ head=\S+ tail=\S+ "
            r"scope=main\+column-chart source=unchanged$"
        )
        setting_indices = [item[0] for item in settings]
        failures = []
        for index, line, match in changes:
            name = match.group("name")
            next_change = next(
                (
                    setting_index
                    for setting_index in setting_indices
                    if setting_index > index
                ),
                len(session.lines),
            )
            state_line = next(
                (
                    item
                    for item in session.lines[index + 2:next_change]
                    if item.message.startswith("displayLayout applied ")
                ),
                None,
            )
            state_match = state_pattern.match(state_line.message) if state_line else None
            if state_match is None:
                failures.append(f"{line.timestamp} {name} missing displayLayout state")
                continue
            if state_match.group("setting") != name:
                failures.append(
                    f"{line.timestamp} {name} paired with {state_match.group('setting')}"
                )
                continue

            try:
                ops = [float(value) for value in state_match.group("ops").split("|")]
                starts = [float(value) for value in state_match.group("start").split("|")]
                expected = float(
                    (match.group("value") or match.group("arrow")).strip()
                )
            except ValueError:
                failures.append(f"{line.timestamp} {name} has invalid numeric state")
                continue
            if len(ops) != 7 or len(starts) != 7 or any(value <= 0 for value in ops):
                failures.append(f"{state_line.timestamp} invalid seven-camera layout")
                continue

            field, slot = name_to_slot[name]
            actual = ops[slot] if field == "ops" else starts[slot]
            if abs(actual - expected) > 0.00011:
                failures.append(
                    f"{line.timestamp} {name} expected={expected:g} applied={actual:g}"
                )

            pending_line = next(
                (
                    item
                    for item in session.lines[index + 2:next_change]
                    if item.message.startswith("capture layout pending ")
                ),
                None,
            )
            if pending_line is not None and (
                f"setting={name}" not in pending_line.message
                or "apply=display-now+stop-final" not in pending_line.message
            ):
                failures.append(f"{pending_line.timestamp} {name} was deferred during Grab")

        report.add(
            self.domain,
            "S7.live-layout",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"changes={len(changes)} failures={len(failures)}"
            + (f"; first={failures[0]}" if failures else ""),
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
