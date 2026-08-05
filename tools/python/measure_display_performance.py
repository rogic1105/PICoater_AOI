#!/usr/bin/env python3
"""Measure review image loading and live layout presentation from Flow logs.

This is a DVT measurement tool, not a unit or stress test. It separates:
  - repository catalog loading (Read Data),
  - thumbnail switching while scrolling,
  - settled full-resolution image loading,
  - live waterfall layout remap and actual LOD presentation.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from flow_checks.core import FlowSession, configure_stdout, resolve_log_paths


REPO_RE = re.compile(r"^RV repo scan root=.*\bms=(?P<ms>\d+)$")
THUMB_RE = re.compile(
    r"^RV thumbnail done (?P<grab>\S+) total=(?P<total>\d+)ms "
    r"decode=(?P<decode>\d+)ms .* source=(?P<source>atlas|frames) "
    r"(?:cache=(?P<cache>cold|join|hit) )?"
)
PREFETCH_RE = re.compile(
    r"^RV prefetch ready center=\S+ neighbor=\S+ "
    r"thumbnail=(?P<cache>cold|join|hit) total=(?P<total>\d+)ms$"
)
SELECT_RE = re.compile(r"^ui:【單片序號】→ (?P<grab>\d{6}-\d{6})$")
FULL_RE = re.compile(r"^RV loadGrab done \S+.*?[（(](?P<ms>\d+)ms[）)]$")
LAYOUT_RE = re.compile(
    r"^WF layout remap storage=per-camera historyRows=(?P<history>\d+) "
    r"virtual=(?P<width>\d+)x(?P<height>\d+) slots=(?P<slots>\S*) "
    r"ms=(?P<ms>\d+)$"
)
LAYOUT_PRESENT_RE = re.compile(
    r"^WF layout presented storage=per-camera historyRows=(?P<history>\d+) "
    r"virtual=(?P<width>\d+)x(?P<height>\d+) latency=(?P<ms>\d+)ms$"
)
SLOT_RE = re.compile(
    r"^(?P<camera>\d+):(?P<source>\d+)@(?P<x>-?\d+)\+(?P<width>\d+)$"
)


@dataclass(frozen=True)
class Measurement:
    name: str
    values: List[int]
    limit_ms: int

    @property
    def passed(self) -> bool:
        return bool(self.values) and percentile(self.values, 95) <= self.limit_ms


@dataclass(frozen=True)
class DisplayPerformance:
    repository_ms: List[int]
    thumbnail_ms: List[int]
    thumbnail_all_ms: List[int]
    thumbnail_decode_ms: List[int]
    thumbnail_sources: List[str]
    thumbnail_cache_accesses: List[str]
    prefetch_ready_ms: List[int]
    full_image_ms: List[int]
    layout_schedule_ms: List[int]
    layout_present_ms: List[int]
    layout_integrity_errors: List[str]
    preserved_history_layouts: int


def percentile(values: Iterable[int], percent: int) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    rank = max(0, ((len(ordered) * percent + 99) // 100) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def parse_session(session: FlowSession) -> DisplayPerformance:
    repository_ms: List[int] = []
    thumbnail_ms: List[int] = []
    thumbnail_all_ms: List[int] = []
    thumbnail_decode_ms: List[int] = []
    thumbnail_sources: List[str] = []
    thumbnail_cache_accesses: List[str] = []
    prefetch_ready_ms: List[int] = []
    full_image_ms: List[int] = []
    layout_schedule_ms: List[int] = []
    layout_present_ms: List[int] = []
    layout_errors: List[str] = []
    preserved = 0
    repo_started_at = None
    current_grab_id = None

    for line in session.lines:
        message = line.message
        match = SELECT_RE.match(message)
        if match:
            current_grab_id = match.group("grab")
            continue
        if message.startswith("RV repo scan begin root="):
            repo_started_at = line.elapsed
            continue
        match = REPO_RE.match(message)
        if match:
            repository_ms.append(int(match.group("ms")))
            repo_started_at = None
            continue
        if message.startswith("RV repo scan root="):
            if repo_started_at is not None:
                repository_ms.append(
                    max(0, int(round((line.elapsed - repo_started_at) * 1000)))
                )
            repo_started_at = None
            continue
        match = THUMB_RE.match(message)
        if match:
            total_ms = int(match.group("total"))
            thumbnail_all_ms.append(total_ms)
            if current_grab_id is None or match.group("grab") == current_grab_id:
                thumbnail_ms.append(total_ms)
            thumbnail_decode_ms.append(int(match.group("decode")))
            thumbnail_sources.append(match.group("source"))
            if match.group("cache"):
                thumbnail_cache_accesses.append(match.group("cache"))
            continue
        match = PREFETCH_RE.match(message)
        if match:
            prefetch_ready_ms.append(int(match.group("total")))
            continue
        match = FULL_RE.match(message)
        if match:
            full_image_ms.append(int(match.group("ms")))
            continue
        match = LAYOUT_RE.match(message)
        if match:
            layout_schedule_ms.append(int(match.group("ms")))
            history_rows = int(match.group("history"))
            virtual_width = int(match.group("width"))
            virtual_height = int(match.group("height"))
            if history_rows > 0:
                preserved += 1
            if virtual_width <= 0 or virtual_height <= 0:
                layout_errors.append(f"{line.timestamp}: invalid virtual size")
            raw_slots = match.group("slots")
            slots = [item for item in raw_slots.split("|") if item]
            if not slots:
                layout_errors.append(f"{line.timestamp}: no camera slots")
                continue
            seen = set()
            for raw_slot in slots:
                slot = SLOT_RE.match(raw_slot)
                if slot is None:
                    layout_errors.append(f"{line.timestamp}: invalid slot {raw_slot}")
                    continue
                camera = int(slot.group("camera"))
                source_width = int(slot.group("source"))
                x = int(slot.group("x"))
                width = int(slot.group("width"))
                if camera in seen:
                    layout_errors.append(f"{line.timestamp}: duplicate camera {camera}")
                seen.add(camera)
                if source_width <= 0 or width <= 0 or x < 0 or x + width > virtual_width:
                    layout_errors.append(f"{line.timestamp}: out-of-range slot {raw_slot}")
            continue
        match = LAYOUT_PRESENT_RE.match(message)
        if match:
            layout_present_ms.append(int(match.group("ms")))

    return DisplayPerformance(
        repository_ms=repository_ms,
        thumbnail_ms=thumbnail_ms,
        thumbnail_all_ms=thumbnail_all_ms,
        thumbnail_decode_ms=thumbnail_decode_ms,
        thumbnail_sources=thumbnail_sources,
        thumbnail_cache_accesses=thumbnail_cache_accesses,
        prefetch_ready_ms=prefetch_ready_ms,
        full_image_ms=full_image_ms,
        layout_schedule_ms=layout_schedule_ms,
        layout_present_ms=layout_present_ms,
        layout_integrity_errors=layout_errors,
        preserved_history_layouts=preserved,
    )


def describe(values: List[int]) -> str:
    if not values:
        return "no samples"
    return (
        f"n={len(values)} min={min(values)}ms "
        f"p50={percentile(values, 50)}ms p95={percentile(values, 95)}ms "
        f"max={max(values)}ms"
    )


def print_measurement(measurement: Measurement, strict: bool) -> bool:
    if not measurement.values:
        print(f"[NOT COVERED] {measurement.name}: no samples")
        return not strict
    status = "PASS" if measurement.passed else "FAIL"
    print(
        f"[{status}] {measurement.name}: {describe(measurement.values)} "
        f"limit(p95)<={measurement.limit_ms}ms"
    )
    return measurement.passed


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(
        description="Measure review read/switch speed and live layout presentation."
    )
    parser.add_argument("logs", nargs="*", help="trace log paths")
    parser.add_argument("--log-dir", default=r"D:\Anilox\Logs")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--strict", action="store_true", help="missing samples fail")
    parser.add_argument("--repo-limit-ms", type=int, default=2000)
    parser.add_argument("--thumbnail-limit-ms", type=int, default=250)
    parser.add_argument("--full-limit-ms", type=int, default=1500)
    parser.add_argument("--layout-limit-ms", type=int, default=250)
    args = parser.parse_args()

    paths = resolve_log_paths(
        args.logs or None,
        log_dir=args.log_dir,
        latest=args.latest or not args.logs,
    )
    if not paths:
        print("No trace logs found.")
        return 2

    passed = True
    for path in paths:
        session = FlowSession.load(path)
        result = parse_session(session)
        print(f"\n=== {Path(path).name} ===")
        passed &= print_measurement(
            Measurement("Read Data catalog", result.repository_ms, args.repo_limit_ms),
            args.strict,
        )
        passed &= print_measurement(
            Measurement(
                "Fast image switch (current selection)",
                result.thumbnail_ms,
                args.thumbnail_limit_ms,
            ),
            args.strict,
        )
        print(f"[INFO] All thumbnail completions: {describe(result.thumbnail_all_ms)}")
        print(f"[INFO] Thumbnail decode: {describe(result.thumbnail_decode_ms)}")
        if result.thumbnail_sources:
            print(
                f"[INFO] Thumbnail source: atlas={result.thumbnail_sources.count('atlas')} "
                f"frames={result.thumbnail_sources.count('frames')}"
            )
        if result.thumbnail_cache_accesses:
            print(
                "[INFO] Thumbnail cache: "
                f"hit={result.thumbnail_cache_accesses.count('hit')} "
                f"join={result.thumbnail_cache_accesses.count('join')} "
                f"cold={result.thumbnail_cache_accesses.count('cold')}"
            )
        print(f"[INFO] Adjacent prefetch: {describe(result.prefetch_ready_ms)}")
        passed &= print_measurement(
            Measurement("Settled full image", result.full_image_ms, args.full_limit_ms),
            args.strict,
        )
        passed &= print_measurement(
            Measurement("Layout visible", result.layout_present_ms, args.layout_limit_ms),
            args.strict,
        )
        print(f"[INFO] Layout scheduling: {describe(result.layout_schedule_ms)}")
        if result.layout_schedule_ms:
            layout_ok = not result.layout_integrity_errors
            print(
                f"[{'PASS' if layout_ok else 'FAIL'}] Layout structure: "
                f"remaps={len(result.layout_schedule_ms)} "
                f"historyPreserved={result.preserved_history_layouts} "
                f"errors={len(result.layout_integrity_errors)}"
            )
            if result.layout_integrity_errors:
                print(f"  first: {result.layout_integrity_errors[0]}")
            passed &= layout_ok
        elif args.strict:
            print("[FAIL] Layout structure: no remap samples")
            passed = False
        else:
            print("[NOT COVERED] Layout structure: no remap samples")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
