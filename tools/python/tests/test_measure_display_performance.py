"""Tests for the display-performance Flow-log measurement parser."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flow_checks.core import FlowLine, FlowSession
from measure_display_performance import parse_session, percentile


def session(*messages: str) -> FlowSession:
    return FlowSession(
        Path("synthetic.log"),
        [
            FlowLine(float(index), f"00:00:{index:02d}.000", 1, message)
            for index, message in enumerate(messages)
        ],
    )


class DisplayPerformanceTests(unittest.TestCase):
    def test_extracts_review_and_layout_measurements(self):
        result = parse_session(
            session(
                "RV repo scan root=D:\\Anilox\\Captures files=10 "
                "csvRecords=10 csvArchives=1 archiveFallback=0 legacy=0 ms=320",
                "RV thumbnail done 260804-090000 total=12ms decode=8ms "
                "images=7 ratio=0.1 source=atlas cache=hit atlas=1920x1080",
                "RV prefetch ready center=260804-090000 neighbor=260804-085959 "
                "thumbnail=cold total=63ms",
                "RV loadGrab done 260804-090000（288ms）",
                "WF layout remap storage=per-camera historyRows=3000 "
                "virtual=6x1000 slots=1:2@0+2|2:2@4+2 ms=1",
                "WF layout presented storage=per-camera historyRows=3000 "
                "virtual=6x1000 latency=18ms",
            )
        )

        self.assertEqual([320], result.repository_ms)
        self.assertEqual([12], result.thumbnail_ms)
        self.assertEqual([12], result.thumbnail_all_ms)
        self.assertEqual([8], result.thumbnail_decode_ms)
        self.assertEqual(["hit"], result.thumbnail_cache_accesses)
        self.assertEqual([63], result.prefetch_ready_ms)
        self.assertEqual([288], result.full_image_ms)
        self.assertEqual([18], result.layout_present_ms)
        self.assertEqual(1, result.preserved_history_layouts)
        self.assertEqual([], result.layout_integrity_errors)

    def test_rejects_layout_slot_outside_virtual_width(self):
        result = parse_session(
            session(
                "WF layout remap storage=per-camera historyRows=1 "
                "virtual=4x1000 slots=1:2@3+2 ms=0"
            )
        )
        self.assertEqual(1, len(result.layout_integrity_errors))

    def test_legacy_repo_scan_uses_begin_end_elapsed_time(self):
        result = parse_session(
            FlowSession(
                Path("synthetic.log"),
                [
                    FlowLine(10.0, "00:00:10.000", 1, "RV repo scan begin root=D:\\A"),
                    FlowLine(12.5, "00:00:12.500", 1, "RV repo scan root=D:\\A files=9"),
                ],
            )
        )
        self.assertEqual([2500], result.repository_ms)

    def test_only_current_selection_counts_toward_switch_limit(self):
        result = parse_session(
            session(
                "ui:【單片序號】→ 260804-090000",
                "RV thumbnail begin 260804-090000",
                "ui:【單片序號】→ 260804-090001",
                "RV thumbnail done 260804-090000 total=400ms decode=8ms "
                "images=7 ratio=0.1 source=atlas atlas=1920x1080",
                "RV thumbnail done 260804-090001 total=80ms decode=8ms "
                "images=7 ratio=0.1 source=atlas atlas=1920x1080",
            )
        )

        self.assertEqual([80], result.thumbnail_ms)
        self.assertEqual([400, 80], result.thumbnail_all_ms)

    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(5, percentile([1, 2, 3, 4, 5], 95))
        self.assertEqual(3, percentile([1, 2, 3, 4, 5], 50))


if __name__ == "__main__":
    unittest.main()
