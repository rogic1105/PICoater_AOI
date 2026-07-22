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
    "StartGrab",
    "AllocateCameras begin",
    "FreeCameras",
    "ApplyMainDisplayMode",
)

FORBIDDEN_PATTERNS = (
    "firstFrame",
    "LC row ",
    "capture csv ",
    "capture archive append ",
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
    "rowCurve present after=mainImage ",
)

ALLOWED_AFTER_STOP = (
    "capture gate closed standby=on",
    "drop drainedFrame after StopGrab cam",
    "capture save skipped closedSession ",
    "display capture quiesce mode=WF",
)

ALLOWED_DURING_SAVE_DRAIN = (
    "capture archive append ",
    "capture csv ",
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
    active_capture_plan = False
    requires_delivery = False
    drain_started = False
    drain_done = False
    released = False

    def finish_window(lineno):
        nonlocal in_stop_window
        if not gate_closed:
            violations.append((lineno, "StopGrab has no capture gate closed"))
        if requires_delivery:
            if not drain_started:
                violations.append((lineno, "capture save drain did not start"))
            elif not drain_done:
                violations.append((lineno, "capture save drain did not finish"))
            if not released:
                violations.append((lineno, "staged capture was not released for remote delivery"))
        in_stop_window = False

    for lineno, msg in rows:
        if msg == "StopGrab":
            if in_stop_window:
                finish_window(lineno)
            stop_windows += 1
            in_stop_window = True
            gate_closed = False
            requires_delivery = active_capture_plan
            active_capture_plan = False
            drain_started = False
            drain_done = False
            released = False
            continue

        if not in_stop_window:
            if msg.startswith(("StartGrab", "AllocateCameras begin")):
                active_capture_plan = False
            if msg.startswith("capture plan "):
                active_capture_plan = True
            continue

        if is_boundary(msg):
            finish_window(lineno)
            if msg.startswith(("StartGrab", "AllocateCameras begin")):
                active_capture_plan = False
            continue

        if msg == "capture gate closed standby=on":
            gate_closed = True
            continue

        if msg.startswith("capture save drain begin "):
            if not gate_closed:
                violations.append((lineno, "capture save drain started before capture gate closed"))
            drain_started = True
            continue

        if msg.startswith("capture save drain done "):
            if not drain_started:
                violations.append((lineno, "capture save drain finished before it started"))
            drain_done = True
            continue

        if msg.startswith("capture remote release "):
            if not drain_done:
                violations.append((lineno, "remote delivery released before save drain finished"))
            released = True
            continue

        if is_allowed(msg):
            continue

        if any(pattern in msg for pattern in ALLOWED_DURING_SAVE_DRAIN):
            if not requires_delivery or drain_done:
                violations.append((lineno, msg))
            continue

        if is_forbidden(msg):
            violations.append((lineno, msg))

    if in_stop_window:
        finish_window(0)

    return stop_windows, violations


def main():
    ap = argparse.ArgumentParser(
        description="Check StopGrab gate, save-drain, remote-release, and post-stop display flow."
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

    print("PASS StopGrab gate, save drain, remote release, and post-stop flow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
