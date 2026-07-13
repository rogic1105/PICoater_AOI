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
GRABID_RE = re.compile(r"\b(\d{6}-\d{6})\b")


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
        with source.open(encoding="utf-8", errors="replace") as stream:
            for raw in stream:
                match = FLOW_RE.search(raw)
                if not match:
                    continue
                timestamp = match.group("ts")
                lines.append(
                    FlowLine(
                        elapsed=parse_ts(timestamp),
                        timestamp=timestamp,
                        thread=int(match.group("thread")),
                        message=match.group("msg").strip(),
                    )
                )
        return cls(source, lines)

    @property
    def label(self) -> str:
        return self.path.stem


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
