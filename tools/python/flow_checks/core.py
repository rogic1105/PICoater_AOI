"""Shared parsing, session, and reporting primitives for flow-log checks."""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_LOG_DIR = Path(r"D:\Anilox\Logs")
FLOW_RE = re.compile(
    r"\[Flow\]\s+(?P<ts>\d{2}:\d{2}:\d{2}\.\d{3})\s+T\s*(?P<thread>\d+)\s+(?P<msg>.*)$"
)
REMOTE_COPY_TRACE_RE = re.compile(
    r"^\s*AniloxRoll\.Monitor\.exe (?:Information|Warning|Error): \d+ : "
    r"(?P<msg>\[RemoteCopy\]\s+.*)$"
)
GRABID_RE = re.compile(r"\b(\d{6}-\d{6})\b")
UI_STALL_RE = re.compile(r"^\[UiStall\]\s+(?P<ms>\d+)ms(?:（(?P<gc>.*)）)?")
UI_PING_RE = re.compile(r"^\[UiPing\]\s+(?P<ms>\d+)ms")
LOG_MODE_RE = re.compile(
    r"^log mode=(Operational|FlowVerification|FullDiagnostic)$"
)


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


def parse_ts(ts: str) -> float:
    hour, minute, second = ts.split(":")
    return int(hour) * 3600 + int(minute) * 60 + float(second)


def grab_id(message: str) -> Optional[str]:
    match = GRABID_RE.search(message)
    return match.group(1) if match else None


@dataclass(frozen=True)
class FlowLine:
    elapsed: float
    timestamp: str
    thread: int
    message: str


@dataclass
class FlowSession:
    path: Path
    lines: List[FlowLine]

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "FlowSession":
        source = Path(path)
        lines: List[FlowLine] = []
        last_elapsed = 0.0
        last_timestamp = "00:00:00.000"
        with source.open(encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                match = FLOW_RE.search(raw)
                if match:
                    timestamp = match.group("ts")
                    last_elapsed = parse_ts(timestamp)
                    last_timestamp = timestamp
                    lines.append(
                        FlowLine(
                            elapsed=last_elapsed,
                            timestamp=timestamp,
                            thread=int(match.group("thread")),
                            message=match.group("msg").strip(),
                        )
                    )
                    continue

                trace_match = REMOTE_COPY_TRACE_RE.search(raw)
                if trace_match:
                    lines.append(
                        FlowLine(
                            elapsed=last_elapsed,
                            timestamp=last_timestamp,
                            thread=0,
                            message=trace_match.group("msg").strip(),
                        )
                    )
        return cls(source, lines)

    @property
    def label(self) -> str:
        return self.path.stem

    @property
    def recording_mode(self) -> str:
        modes = []
        for line in self.lines:
            match = LOG_MODE_RE.match(line.message)
            if match and (not modes or modes[-1] != match.group(1)):
                modes.append(match.group(1))
        if modes:
            return " -> ".join(modes)
        # 舊版沒有分級，等同所有 DVT 探針皆開啟。
        return "LegacyAll"

    @property
    def dvt_enabled(self) -> bool:
        if self.recording_mode == "LegacyAll":
            return True
        return any(
            LOG_MODE_RE.match(line.message)
            and LOG_MODE_RE.match(line.message).group(1) != "Operational"
            for line in self.lines
        )

    def dvt_only(self) -> "FlowSession":
        """Return lines recorded while DVT-or-higher mode was active.

        A legacy trace had no mode marker and all of its probes were unconditional.
        """
        if self.recording_mode == "LegacyAll":
            return self
        enabled = False
        selected = []
        for line in self.lines:
            match = LOG_MODE_RE.match(line.message)
            if match:
                enabled = match.group(1) != "Operational"
            if enabled:
                selected.append(line)
        return FlowSession(self.path, selected)


@dataclass(frozen=True)
class UiResponsivenessAssessment:
    worst_timer_gap_ms: int
    over_limit_count: int
    hard_block_count: int
    timer_starvation_count: int
    worst_correlated_ping_ms: int
    correlated_stack_count: int
    gc_observed_count: int

    @property
    def passed(self) -> bool:
        return self.hard_block_count == 0

    def detail(self, limit_ms: int) -> str:
        return (
            f"最大Timer={self.worst_timer_gap_ms}ms；>{limit_ms}ms "
            f"真阻塞={self.hard_block_count} 計時器飢餓={self.timer_starvation_count}；"
            f"最大關聯UiPing={self.worst_correlated_ping_ms}ms "
            f"UiStack={self.correlated_stack_count} GC伴隨={self.gc_observed_count}"
        )


def assess_ui_responsiveness(
    session: FlowSession,
    activity_times: Sequence[float],
    limit_ms: int = 1000,
    activity_before_s: float = 1.0,
    activity_after_s: float = 3.0,
) -> UiResponsivenessAssessment:
    """Distinguish a blocked UI thread from a starved low-priority WM_TIMER.

    UiStall alone only says that the WinForms timer fired late. A matching UiStack,
    or a sufficiently large BeginInvoke round trip in the same interval, is required
    before the delay is classified as a synchronous UI block.
    """
    stalls = []
    for line in session.lines:
        match = UI_STALL_RE.match(line.message)
        if not match:
            continue
        if not any(
            event_time - activity_before_s <= line.elapsed <= event_time + activity_after_s
            for event_time in activity_times
        ):
            continue
        stalls.append((line, int(match.group("ms")), match.group("gc") or ""))

    pings = []
    stacks = []
    for line in session.lines:
        ping_match = UI_PING_RE.match(line.message)
        if ping_match:
            pings.append((line.elapsed, int(ping_match.group("ms"))))
        elif line.message.startswith("[UiStack]"):
            stacks.append(line.elapsed)

    hard_blocks = 0
    timer_starvations = 0
    worst_ping = 0
    correlated_stacks = 0
    gc_observed = 0
    for line, duration_ms, gc_text in stalls:
        if duration_ms <= limit_ms:
            continue
        interval_start = line.elapsed - duration_ms / 1000.0 - 0.25
        interval_end = line.elapsed + 0.25
        related_pings = [
            duration
            for elapsed, duration in pings
            if interval_start <= elapsed <= interval_end
        ]
        related_stacks = sum(
            interval_start <= elapsed <= interval_end for elapsed in stacks
        )
        worst_related_ping = max(related_pings) if related_pings else 0
        worst_ping = max(worst_ping, worst_related_ping)
        correlated_stacks += related_stacks
        if re.search(r"GC[012]\+[1-9]\d*", gc_text):
            gc_observed += 1

        # A ping must consume at least half the timer gap (minimum 200 ms,
        # capped at the contract limit) to corroborate a blocking stall.
        ping_block_ms = max(200, min(limit_ms, duration_ms // 2))
        if related_stacks > 0 or worst_related_ping >= ping_block_ms:
            hard_blocks += 1
        else:
            timer_starvations += 1

    return UiResponsivenessAssessment(
        worst_timer_gap_ms=max((duration for _, duration, _ in stalls), default=0),
        over_limit_count=sum(duration > limit_ms for _, duration, _ in stalls),
        hard_block_count=hard_blocks,
        timer_starvation_count=timer_starvations,
        worst_correlated_ping_ms=worst_ping,
        correlated_stack_count=correlated_stacks,
        gc_observed_count=gc_observed,
    )


class CheckStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_COVERED = "NOT COVERED"


@dataclass(frozen=True)
class CheckResult:
    domain: str
    rule: str
    status: CheckStatus
    detail: str


@dataclass
class CheckReport:
    results: List[CheckResult] = field(default_factory=list)

    def add(self, domain: str, rule: str, status: CheckStatus, detail: str) -> None:
        self.results.append(CheckResult(domain, rule, status, detail))

    def extend(self, results: Iterable[CheckResult]) -> None:
        self.results.extend(results)

    @property
    def has_failures(self) -> bool:
        return any(result.status is CheckStatus.FAIL for result in self.results)

    def count(self, status: CheckStatus) -> int:
        return sum(result.status is status for result in self.results)

    def dump(self) -> None:
        for result in self.results:
            print(
                f"[{result.status.value}] {result.domain}/{result.rule}: {result.detail}"
            )


def resolve_log_paths(
    explicit: Sequence[str] | None = None,
    log_dir: os.PathLike[str] | str = DEFAULT_LOG_DIR,
    date: Optional[str] = None,
    latest: bool = False,
) -> List[Path]:
    if explicit:
        paths = [Path(item) for item in explicit]
    else:
        root = Path(log_dir)
        if date:
            date_key = date.replace("-", "")
            pattern = str(root / f"trace-{date_key}_*.log")
        else:
            pattern = str(root / "trace-*.log")
        paths = [Path(item) for item in glob.glob(pattern)]

    paths = [path for path in paths if path.is_file()]
    paths.sort(key=lambda path: (path.stat().st_mtime, str(path)))
    if latest or (not explicit and not date):
        return paths[-1:] if paths else []
    return paths
