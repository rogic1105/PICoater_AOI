#!/usr/bin/env python3
"""Measure IO HIGH to acquisition/display milestones from a PICoater trace."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta


FLOW_RE = re.compile(
    r"^\[Flow\]\s+(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+T\s*\d+\s+(?P<message>.*)$"
)
CAM_RE = re.compile(r"firstFrame cam(?P<cam>\d+)")
CONNECTED_RE = re.compile(r"capture gate open cams=(?P<count>\d+)")
PLAN_RE = re.compile(r"capture plan grab=(?P<grab>\S+).* csv=(?P<csv>\S+)")
FILE_RE = re.compile(r"(?P<stamp>\d{8}_\d{6}\.\d{3})-(?P<cam>\d+)$")


def latest_trace() -> str:
    paths = glob.glob(r"D:\Anilox\Logs\trace-*.log")
    if not paths:
        raise FileNotFoundError(r"No trace found under D:\Anilox\Logs")
    return max(paths, key=os.path.getmtime)


def elapsed_ms(start: datetime | None, end: datetime | None) -> str:
    if start is None or end is None:
        return "--"
    return f"{(end - start).total_seconds() * 1000.0:.0f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?", default=None)
    args = parser.parse_args()
    path = args.trace or latest_trace()

    cycles = []
    current = None
    day = datetime(2000, 1, 1)
    previous = None

    with open(path, "r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            match = FLOW_RE.match(raw.rstrip())
            if not match:
                continue
            clock = datetime.strptime(match.group("time"), "%H:%M:%S.%f")
            now = day.replace(
                hour=clock.hour,
                minute=clock.minute,
                second=clock.second,
                microsecond=clock.microsecond,
            )
            if previous is not None and now < previous:
                day += timedelta(days=1)
                now += timedelta(days=1)
            previous = now
            message = match.group("message")

            if "io:DI START 上升緣 → 開始抓取" in message:
                if current is not None:
                    cycles.append(current)
                current = {
                    "high": now,
                    "cold_begin": None,
                    "cold_armed": None,
                    "start": None,
                    "gate": None,
                    "stop": None,
                    "stop_source": None,
                    "expected": None,
                    "grab": None,
                    "csv": None,
                    "frames": {},
                }
                continue
            if current is None:
                continue

            if message.startswith("capture cold-start begin"):
                current["cold_begin"] = now
            elif message.startswith("capture cold-start armed"):
                current["cold_armed"] = now
            elif message.startswith("StartGrab"):
                current["start"] = now
            elif message.startswith("capture gate open"):
                current["gate"] = now
                count = CONNECTED_RE.search(message)
                if count:
                    current["expected"] = int(count.group("count"))
            elif message.startswith("capture plan"):
                plan = PLAN_RE.search(message)
                if plan:
                    current["grab"] = plan.group("grab")
                    current["csv"] = plan.group("csv")
            elif "io:DI START 下降緣 → 停止抓取" in message:
                current["stop"] = now
                current["stop_source"] = "IO LOW"
            elif message.startswith("auto:抓取上限到時"):
                current["stop"] = now
                current["stop_source"] = "watchdog"
            else:
                cam = CAM_RE.search(message)
                if cam:
                    current["frames"].setdefault(int(cam.group("cam")), now)

    if current is not None:
        cycles.append(current)

    csv_rows = {}
    for csv_path in {cycle["csv"] for cycle in cycles if cycle["csv"]}:
        by_grab = defaultdict(list)
        with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as stream:
            rows = (line for line in stream if not line.startswith("#CFG"))
            for row in csv.DictReader(rows):
                by_grab[row.get("Id", "")].append(row)
        csv_rows[csv_path] = by_grab

    print(f"trace: {path}")
    print(
        "cycle  stopSource  highDuration  high->armed  high->gate  "
        "high->allFirstFrames  savedPerCam"
    )
    for index, cycle in enumerate(cycles, 1):
        frames = list(cycle["frames"].values())
        all_first = max(frames) if frames else None
        saved = defaultdict(int)
        rows = csv_rows.get(cycle["csv"], {}).get(cycle["grab"], [])
        for row in rows:
            file_match = FILE_RE.match(row.get("FileName", ""))
            if file_match:
                saved[int(file_match.group("cam"))] += 1
        saved_text = ",".join(
            f"cam{cam}={saved[cam]}" for cam in sorted(saved)
        ) or "--"
        print(
            f"{index:>5}  "
            f"{(cycle['stop_source'] or '--'):>10}  "
            f"{elapsed_ms(cycle['high'], cycle['stop']):>12}  "
            f"{elapsed_ms(cycle['high'], cycle['cold_armed']):>11}  "
            f"{elapsed_ms(cycle['high'], cycle['gate']):>10}  "
            f"{elapsed_ms(cycle['high'], all_first):>20}  "
            f"{saved_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
