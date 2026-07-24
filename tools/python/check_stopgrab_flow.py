#!/usr/bin/env python3
import argparse
import glob
import os
import re
import sys


DEFAULT_TRACE_GLOB = r"D:\Anilox\Logs\trace-*.log"
FLOW_RE = re.compile(r"\[Flow\].*?\sT\s*\d+\s+(?P<msg>.*)$")

BOUNDARY_PATTERNS = (
    "ui:",
    "io:DI START 上升緣",
    "StartGrab",
    "AllocateCameras begin",
    "FreeCameras",
    "ApplyMainDisplayMode",
    "wheelZoom",
    "drag(",
)

FORBIDDEN_PATTERNS = (
    "firstFrame",
    "LC row ",
    "capture csv ",
    "capture plan ",
    "ImageDisplayView",
    "Waterfall",
    "EnsureImageDisplay",
    "EnableWaterfall",
    "SwitchMainDisplay",
    "lodRebind",
    "autoFit",
    "clearFrame",
    "pushFrames",
    "IC ",
    "WF ",
)

ALLOWED_AFTER_STOP = (
    "capture gate closed standby=on",
    "drop drainedFrame after StopGrab cam",
)

FINAL_TAIL_ROW_PATTERNS = (
    "LC row ",
    "rowCurve present after=mainImage",
)


def latest_trace():
    matches = glob.glob(DEFAULT_TRACE_GLOB)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def parse_flow_lines(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if "[Flow]" not in line:
                continue
            m = FLOW_RE.search(line)
            msg = m.group("msg").strip() if m else line.strip()
            rows.append((lineno, msg))
    return rows


def is_boundary(msg):
    if msg == "StopGrab":
        return False
    return any(p in msg for p in BOUNDARY_PATTERNS)


def is_allowed(msg):
    return any(p in msg for p in ALLOWED_AFTER_STOP)


def is_forbidden(msg):
    return any(p in msg for p in FORBIDDEN_PATTERNS)


def check(rows):
    violations = []
    stop_windows = 0
    in_stop_window = False
    gate_closed = False
    tail_completed = False
    allow_final_tail_row = False
    final_tail_row_presentations = 0

    for lineno, msg in rows:
        if msg.startswith("capture tail complete pending="):
            tail_completed = True
            continue

        if msg == "StopGrab":
            if in_stop_window and not gate_closed:
                violations.append((lineno, "previous StopGrab has no capture gate closed"))
            stop_windows += 1
            in_stop_window = True
            gate_closed = False
            allow_final_tail_row = tail_completed
            final_tail_row_presentations = 0
            tail_completed = False
            continue

        if not in_stop_window:
            continue

        if is_boundary(msg):
            if not gate_closed:
                violations.append((lineno, "StopGrab has no capture gate closed"))
            in_stop_window = False
            allow_final_tail_row = False
            continue

        if msg == "capture gate closed standby=on":
            gate_closed = True
            continue

        if is_allowed(msg):
            continue

        if any(pattern in msg for pattern in FINAL_TAIL_ROW_PATTERNS):
            if not allow_final_tail_row:
                violations.append((lineno, msg))
                continue
            if "rowCurve present after=mainImage" in msg:
                final_tail_row_presentations += 1
                if final_tail_row_presentations > 1:
                    violations.append((lineno, "more than one final tail row presentation"))
            continue

        if is_forbidden(msg):
            violations.append((lineno, msg))

    if in_stop_window and not gate_closed:
        violations.append((0, "last StopGrab has no capture gate closed"))

    return stop_windows, violations


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Check the StopGrab capture gate. One coalesced row-Curve presentation "
            "from a completed IO tail frame is allowed; new frames/CSV/display work is rejected."
        )
    )
    ap.add_argument("trace", nargs="?", help="trace-*.log path. Defaults to latest D:\\Anilox\\Logs\\trace-*.log")
    args = ap.parse_args()

    path = args.trace or latest_trace()
    if not path:
        print(f"ERROR no trace found: {DEFAULT_TRACE_GLOB}", file=sys.stderr)
        return 2
    if not os.path.exists(path):
        print(f"ERROR trace not found: {path}", file=sys.stderr)
        return 2

    rows = parse_flow_lines(path)
    stop_windows, violations = check(rows)
    print(f"trace={path}")
    print(f"stop_windows={stop_windows}")

    if violations:
        print("FAIL post-StopGrab flow violations:")
        for lineno, msg in violations:
            print(f"  line {lineno}: {msg}")
        return 1

    print("PASS StopGrab gate closed; only the optional final IO-tail row Curve appeared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
