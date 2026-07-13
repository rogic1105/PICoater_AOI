#!/usr/bin/env python3
"""Compatibility entry point for Review (R-series) flow checks."""

import argparse
import sys

from flow_checks.core import CheckStatus, FlowSession, configure_stdout, resolve_log_paths
from flow_checks.review import ReviewFlowValidator


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="檢查最新或指定的回顧 flow log")
    parser.add_argument("log", nargs="?", help="trace log；省略時使用 D:\\Anilox\\Logs 最新一份")
    args = parser.parse_args()

    paths = resolve_log_paths([args.log] if args.log else None, latest=not args.log)
    if not paths:
        print("找不到可讀取的 trace log")
        return 2

    session = FlowSession.load(paths[0])
    print(f"log={session.path}（[Flow] 行數={len(session.lines)}）")
    report = ReviewFlowValidator().validate(session)
    report.dump()
    print(
        "結果："
        + ("有 FAIL（見上）" if report.has_failures else "無 FAIL")
        + f"；NOT COVERED={report.count(CheckStatus.NOT_COVERED)}"
    )
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
