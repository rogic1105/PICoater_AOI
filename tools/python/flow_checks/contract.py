"""Global invariants that apply to every flow domain."""

from .core import CheckReport, CheckStatus, FlowSession


class GlobalContractValidator:
    domain = "GLOBAL"

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        if not session.lines:
            report.add(self.domain, "G0", CheckStatus.NOT_COVERED, "沒有 [Flow] 行")
            return report

        violations = [
            line for line in session.lines if "契約違規" in line.message
        ]
        if violations:
            first = violations[0]
            report.add(
                self.domain,
                "G1",
                CheckStatus.FAIL,
                f"契約違規 {len(violations)} 行；首筆 {first.timestamp} {first.message}",
            )
        else:
            report.add(self.domain, "G1", CheckStatus.PASS, "未出現契約違規行")
        self._check_shutdown(session, report)
        return report

    def _check_shutdown(self, session: FlowSession, report: CheckReport) -> None:
        closing = [
            index
            for index, line in enumerate(session.lines)
            if line.message == "ui:關閉程式"
        ]
        if not closing:
            report.add(self.domain, "G2.shutdown", CheckStatus.NOT_COVERED, "本 session 未正常關閉")
            return

        failures = []
        has_camera_allocation = any(
            line.message.startswith("AllocateCameras begin")
            for line in session.lines
        )
        for sequence, close_index in enumerate(closing, start=1):
            tail = session.lines[close_index + 1:]
            complete_index = next(
                (
                    index
                    for index, line in enumerate(tail)
                    if line.message == "shutdown resources released"
                ),
                None,
            )
            if complete_index is None:
                failures.append(f"關閉#{sequence} 缺 shutdown resources released")
                continue

            if has_camera_allocation and not any(
                line.message.startswith("FreeCameras")
                for line in tail[:complete_index]
            ):
                failures.append(f"關閉#{sequence} 缺 FreeCameras")

        report.add(
            self.domain,
            "G2.shutdown",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"關閉={len(closing)}；failures={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )
