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
        return report
