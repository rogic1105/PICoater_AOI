"""Capture/storage (C-series) flow validators."""

from __future__ import annotations

from .core import CheckReport, CheckStatus, FlowSession, grab_id


class CaptureFlowValidator:
    domain = "CAPTURE"

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
            return report

        self._check_capture_plan(plans, report)
        self._check_first_records(plans, records, report)
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
