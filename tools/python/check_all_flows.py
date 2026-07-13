#!/usr/bin/env python3
"""Run every registered flow validator over one session or a full day of trace logs."""

import argparse
import sys

from flow_checks.core import CheckReport, CheckStatus, FlowSession, configure_stdout, resolve_log_paths
from flow_checks.registry import PENDING_DOMAINS, VALIDATORS


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="全天 flow-log DVT 總入口")
    parser.add_argument("logs", nargs="*", help="指定一或多個 trace log")
    parser.add_argument("--date", help="掃描指定日期全部 session，例如 2026-07-13")
    parser.add_argument("--log-dir", default=r"D:\Anilox\Logs", help="trace log 目錄")
    parser.add_argument("--latest", action="store_true", help="只檢查最新 session")
    args = parser.parse_args()

    if args.logs and args.date:
        parser.error("指定 logs 與 --date 只能擇一")

    paths = resolve_log_paths(
        args.logs or None,
        log_dir=args.log_dir,
        date=args.date,
        latest=args.latest or (not args.logs and not args.date),
    )
    if not paths:
        print("找不到符合條件的 trace log")
        return 2

    print("已掛載 validator：" + ", ".join(validator.domain for validator in VALIDATORS))
    print("尚待自動化：" + ", ".join(PENDING_DOMAINS))

    all_results = CheckReport()
    failed_sessions = 0
    for path in paths:
        session = FlowSession.load(path)
        session_report = CheckReport()
        for validator in VALIDATORS:
            session_report.extend(validator.validate(session).results)

        print(f"\n=== {session.label} | [Flow] {len(session.lines)} 行 ===")
        session_report.dump()
        all_results.extend(session_report.results)
        if session_report.has_failures:
            failed_sessions += 1

    print("\n=== 全天摘要 ===")
    print(
        f"sessions={len(paths)} failSessions={failed_sessions} "
        f"PASS={all_results.count(CheckStatus.PASS)} "
        f"FAIL={all_results.count(CheckStatus.FAIL)} "
        f"NOT_COVERED={all_results.count(CheckStatus.NOT_COVERED)}"
    )
    print("覆蓋狀態：PARTIAL（尚待自動化的 domain 見上方清單）")
    return 1 if all_results.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
