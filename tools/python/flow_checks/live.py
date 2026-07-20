"""Live-display (F-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class LiveFlowValidator:
    domain = "LIVE"

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        self._check_camera_initialization(session, report)
        if not any(
            line.message.startswith(("LC ", "IC ", "WF ", "ui:【開始抓取】"))
            for line in session.lines
        ):
            report.add(self.domain, "F0", CheckStatus.NOT_COVERED, "本 session 無監控操作")
            return report

        self._check_drag_first_publish(session, report)
        return report

    def _check_camera_initialization(
        self, session: FlowSession, report: CheckReport
    ) -> None:
        begins = [
            index
            for index, line in enumerate(session.lines)
            if line.message.startswith("AllocateCameras begin")
        ]
        if not begins:
            report.add(self.domain, "F1.init", CheckStatus.NOT_COVERED, "本 session 無相機配置")
            return

        failures = []
        covered = 0
        worst_allocation_stall = 0
        worst_stall = 0
        for sequence, begin_index in enumerate(begins, start=1):
            end_index = begins[sequence] if sequence < len(begins) else len(session.lines)
            lines = session.lines[begin_index:end_index]
            if not any("camera init phase=" in line.message for line in lines):
                continue

            covered += 1
            acquisition = [
                line for line in lines
                if re.match(r"camera init cam=\d+ phase=acquisition ", line.message)
            ]
            processing = [
                line for line in lines
                if re.match(r"camera init cam=\d+ phase=processing ", line.message)
            ]
            acquisition_done = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=acquisition done ")
                ),
                None,
            )
            processing_begin = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=processing begin ")
                ),
                None,
            )
            processing_done = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init phase=processing done ")
                ),
                None,
            )
            summary = next(
                (
                    line for line in lines
                    if line.message.startswith("camera init summary ")
                ),
                None,
            )

            required = (acquisition_done, processing_begin, processing_done, summary)
            if any(line is None for line in required):
                failures.append(f"配置#{sequence} 階段行不完整")
                continue

            expected_match = re.search(r"cams=(\d+)", acquisition_done.message)
            expected = int(expected_match.group(1)) if expected_match else -1
            if len(acquisition) != expected or len(processing) != expected:
                failures.append(
                    f"配置#{sequence} cams={expected} acquisition={len(acquisition)} "
                    f"processing={len(processing)}"
                )

            allocation_calls = []
            for line in processing:
                match = re.search(r"\ballocCalls=(\d+)\b", line.message)
                allocation_calls.append(int(match.group(1)) if match else -1)
            if any(count != 2 for count in allocation_calls):
                failures.append(
                    f"配置#{sequence} processing allocCalls={allocation_calls}"
                )

            if expected > 0:
                acquisition_threads = {line.thread for line in acquisition}
                processing_threads = {line.thread for line in processing}
                if (
                    len(acquisition_threads) != 1
                    or len(processing_threads) != 1
                    or acquisition_threads == processing_threads
                ):
                    failures.append(
                        f"配置#{sequence} acquisitionThreads={sorted(acquisition_threads)} "
                        f"processingThreads={sorted(processing_threads)}"
                    )

            processing_window = [
                line
                for line in lines
                if processing_begin.elapsed <= line.elapsed <= processing_done.elapsed
            ]
            stalls = []
            for line in processing_window:
                match = re.match(r"\[UiStall\]\s+(\d+)ms", line.message)
                if match:
                    stalls.append(int(match.group(1)))
            if stalls:
                worst_stall = max(worst_stall, max(stalls))
                failures.append(
                    f"配置#{sequence} processing 期間 UiStall 最大 {max(stalls)}ms"
                )

            allocation_stalls = []
            for line in lines:
                # WM_TIMER is low priority, so the stall line can be emitted just after
                # the allocation summary even though the measured gap occurred inside it.
                if summary.elapsed + 1 < line.elapsed:
                    continue
                match = re.match(r"\[UiStall\]\s+(\d+)ms", line.message)
                if match:
                    allocation_stalls.append(int(match.group(1)))
            if allocation_stalls:
                worst_allocation_stall = max(
                    worst_allocation_stall, max(allocation_stalls)
                )
                if max(allocation_stalls) > 1000:
                    failures.append(
                        f"配置#{sequence} UI stall 最大 {max(allocation_stalls)}ms"
                    )

        if covered == 0:
            report.add(
                self.domain,
                "F1.init",
                CheckStatus.NOT_COVERED,
                f"配置={len(begins)}；舊版 log 無 phase 儀器",
            )
            return

        report.add(
            self.domain,
            "F1.init",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"配置={covered}；failures={len(failures)}；"
            f"allocationWorstStall={worst_allocation_stall}ms；"
            f"processingWorstStall={worst_stall}ms"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_drag_first_publish(self, session: FlowSession, report: CheckReport) -> None:
        starts = []
        failures = []
        active = {"IC": None, "WF": None}

        for line in session.lines:
            message = line.message
            for prefix in active:
                if message == f"{prefix} drag(start)":
                    starts.append((prefix, line.elapsed))
                    active[prefix] = False
                elif message == f"{prefix} drag(view-published)" and active[prefix] is False:
                    active[prefix] = True
                elif message.startswith(f"{prefix} viewEdges") and active[prefix] is not None:
                    if active[prefix] is not True:
                        failures.append(f"{prefix}@{line.elapsed:.3f}s")
                    active[prefix] = None

        for prefix, state in active.items():
            if state is False:
                failures.append(f"{prefix}@未結束")

        if not starts:
            report.add(self.domain, "F6.first-view", CheckStatus.NOT_COVERED, "無主畫面拖曳")
            return
        report.add(
            self.domain,
            "F6.first-view",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"拖曳={len(starts)}；首位移未發布={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
