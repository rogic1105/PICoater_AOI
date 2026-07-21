#!/usr/bin/env python3
"""Run every registered flow validator over one session or a full day of trace logs."""

import argparse
import sys

from flow_checks.core import (
    CheckReport,
    CheckStatus,
    FlowSession,
    configure_stdout,
    resolve_log_paths,
)
from flow_checks.registry import PENDING_DOMAINS, VALIDATORS


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="檢查單一 session 或全天的 flow-log DVT")
    parser.add_argument("logs", nargs="*", help="一個或多個 trace log")
    parser.add_argument("--date", help="檢查指定日期全部 session，例如 2026-07-20")
    parser.add_argument("--log-dir", default=r"D:\Anilox\Logs", help="trace log 目錄")
    parser.add_argument("--latest", action="store_true", help="只檢查最新 session")
    args = parser.parse_args()

    if args.logs and args.date:
        parser.error("不可同時指定 logs 與 --date")

    paths = resolve_log_paths(
        args.logs or None,
        log_dir=args.log_dir,
        date=args.date,
        latest=args.latest or (not args.logs and not args.date),
    )
    if not paths:
        print("找不到可讀取的 trace log")
        return 2

    print("已掛載 validators：" + ", ".join(validator.domain for validator in VALIDATORS))
    print("尚待自動化：" + (", ".join(PENDING_DOMAINS) if PENDING_DOMAINS else "無"))

    all_results = CheckReport()
    failed_sessions = 0
    for path in paths:
        session = FlowSession.load(path)
        session_report = CheckReport()
        for validator in VALIDATORS:
            session_report.extend(validator.validate(session).results)

        print(
            f"\n=== {session.label} | [Flow] {len(session.lines)} 行 "
            f"| 記錄範圍={session.recording_mode} ==="
        )
        if not session.dvt_enabled:
            session_report.add(
                "CONTRACT",
                "LOG-MODE",
                CheckStatus.NOT_COVERED,
                "本 session 使用日常運行記錄；座標／預排版等 DVT 規則不做完整判定",
            )
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
    if PENDING_DOMAINS:
        print("自動化範圍：PARTIAL（仍有未掛 validator 的 domain）")
    else:
        print(
            "自動化範圍：FULL（所有已登記 domain 均有 validator；"
            "NOT COVERED 仍表示該 session 未操作）"
        )
    return 1 if all_results.has_failures else 0


if __name__ == "__main__":
    sys.exit(main())
