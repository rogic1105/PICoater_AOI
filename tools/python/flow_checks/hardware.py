"""Hardware and external-service (H-series) flow validators."""

from __future__ import annotations

import re

from .core import CheckReport, CheckStatus, FlowSession


class HardwareFlowValidator:
    domain = "HARDWARE"

    _edge_pattern = re.compile(
        r"^(?:⚠ )?(IO|光源|儲存分享) (?:未連線（開機基線）|斷線|恢復連線)$"
    )
    _camera_pattern = re.compile(r"^(?:⚠ 相機離線|相機在線) \d+→\d+/\d+$")

    def validate(self, session: FlowSession) -> CheckReport:
        report = CheckReport()
        covered = any(
            self._edge_pattern.match(line.message)
            or self._camera_pattern.match(line.message)
            or line.message.startswith(("儲存程式 heartbeat ", "⚠ 儲存程式 heartbeat "))
            for line in session.lines
        )
        if not covered:
            report.add(self.domain, "H0", CheckStatus.NOT_COVERED, "本 session 無硬體狀態邊緣")
            return report

        self._check_connection_edges(session, report)
        self._check_storage_heartbeat(session, report)
        self._check_camera_edges(session, report)
        return report

    def _check_connection_edges(self, session: FlowSession, report: CheckReport) -> None:
        last = {}
        count = 0
        failures = []
        for line in session.lines:
            match = self._edge_pattern.match(line.message)
            if not match:
                continue
            count += 1
            name = match.group(1)
            connected = line.message.endswith("恢復連線")
            if name in last and last[name] == connected:
                failures.append(f"{line.timestamp} {name} 連續重複狀態")
            last[name] = connected

        if count == 0:
            report.add(self.domain, "H1.edges", CheckStatus.NOT_COVERED, "無 IO/光源/儲存分享轉變")
            return
        report.add(
            self.domain,
            "H1.edges",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={count} duplicate={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_storage_heartbeat(self, session: FlowSession, report: CheckReport) -> None:
        lines = [
            line for line in session.lines
            if line.message.startswith(("儲存程式 heartbeat ", "⚠ 儲存程式 heartbeat "))
        ]
        if not lines:
            report.add(self.domain, "H1.heartbeat", CheckStatus.NOT_COVERED, "舊版或未設定儲存 heartbeat")
            return

        last = None
        failures = []
        for line in lines:
            alive = line.message.startswith("儲存程式 heartbeat 恢復 ")
            valid = (
                alive and " pid=" in line.message and " age=" in line.message
            ) or (
                not alive and line.message.startswith("⚠ 儲存程式 heartbeat 未回報 reason=")
            )
            if not valid:
                failures.append(f"{line.timestamp} 格式錯誤")
            elif last is not None and last == alive:
                failures.append(f"{line.timestamp} 連續重複狀態")
            last = alive

        report.add(
            self.domain,
            "H1.heartbeat",
            CheckStatus.PASS if not failures else CheckStatus.FAIL,
            f"edges={len(lines)} invalid={len(failures)}"
            + (f"；首例 {failures[0]}" if failures else ""),
        )

    def _check_camera_edges(self, session: FlowSession, report: CheckReport) -> None:
        lines = [line for line in session.lines if self._camera_pattern.match(line.message)]
        if not lines:
            report.add(self.domain, "H2.camera-count", CheckStatus.NOT_COVERED, "無相機在線數轉變")
            return
        report.add(
            self.domain,
            "H2.camera-count",
            CheckStatus.PASS,
            f"cameraEdges={len(lines)}",
        )
