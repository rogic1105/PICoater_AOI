#!/usr/bin/env python3
"""Backfill MeanRPeak/MaxRPeak CSV indexes from row-curve MCBF bins.

Dry-run is the default. Pass one or more capture roots and --execute to replace
report CSV files atomically. Hard-linked bins are cached by NTFS file identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MEAN_SUFFIXES = ("_mean_r.bin", "_mean_h.bin", "_row_mean.bin")
MAX_SUFFIXES = ("_max_r.bin", "_max_h.bin", "_row_max.bin")
MEAN_COLUMN = 10
MAX_COLUMN = 11
REQUIRED_COLUMNS = 12


@dataclass
class Result:
    csv_files: int = 0
    records: int = 0
    covered: int = 0
    missing: int = 0
    changed_rows: int = 0
    changed_files: int = 0

    def add(self, other: "Result") -> None:
        self.csv_files += other.csv_files
        self.records += other.records
        self.covered += other.covered
        self.missing += other.missing
        self.changed_rows += other.changed_rows
        self.changed_files += other.changed_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill report row-curve peaks from MeanR/MaxR MCBF bins."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def report_csv_files(root: Path) -> List[Path]:
    return sorted(
        path
        for path in root.glob("*/*/*.csv")
        if len(path.stem) == 8 and path.stem.isdigit()
    )


def resolve_curve(base: Path, suffixes: Iterable[str]) -> Optional[Path]:
    for suffix in suffixes:
        path = Path(str(base) + suffix)
        if path.is_file():
            return path
    return None


def read_peak_normalized(
    path: Path, cache: Dict[Tuple[int, int, int, int], float]
) -> float:
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    cached = cache.get(key)
    if cached is not None:
        return cached

    with path.open("rb") as stream:
        prefix = stream.read(24)
        if len(prefix) < 16 or prefix[:4] != b"MCBF":
            raise RuntimeError(f"Invalid MCBF header: {path}")
        version = struct.unpack_from("<i", prefix, 4)[0]
        length_offset = 20 if version >= 2 else 12
        length = struct.unpack_from("<i", prefix, length_offset)[0]
        if length <= 0 or length > 200_000:
            raise RuntimeError(f"Invalid MCBF length {length}: {path}")
        payload_offset = length_offset + 4
        stream.seek(payload_offset)
        payload = stream.read(length * 4)
        if len(payload) != length * 4:
            raise RuntimeError(f"Truncated MCBF payload: {path}")

    peak = max(value[0] for value in struct.iter_unpack("<f", payload)) / 255.0
    cache[key] = peak
    return peak


def same_metric(text: str, value: float) -> bool:
    return text == f"{value:.6f}"


def rewrite_csv(
    root: Path,
    csv_path: Path,
    execute: bool,
    cache: Dict[Tuple[int, int, int, int], float],
) -> Result:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))

    result = Result(csv_files=1)
    file_changed = False
    for columns in rows:
        if not columns:
            continue
        if columns[0] == "Id":
            while len(columns) < REQUIRED_COLUMNS:
                columns.append("")
            if columns[MEAN_COLUMN] != "MeanRPeak" or columns[MAX_COLUMN] != "MaxRPeak":
                columns[MEAN_COLUMN] = "MeanRPeak"
                columns[MAX_COLUMN] = "MaxRPeak"
                file_changed = True
            continue
        if columns[0].startswith("#") or len(columns) < 2:
            continue

        result.records += 1
        file_name = columns[1]
        if len(file_name) < 8 or not file_name[:8].isdigit():
            result.missing += 1
            continue
        date_text = file_name[:8]
        base = root / date_text[:4] / date_text[:6] / date_text / file_name
        mean_path = resolve_curve(base, MEAN_SUFFIXES)
        max_path = resolve_curve(base, MAX_SUFFIXES)
        if mean_path is None or max_path is None:
            result.missing += 1
            continue

        mean_peak = read_peak_normalized(mean_path, cache)
        max_peak = read_peak_normalized(max_path, cache)
        while len(columns) < REQUIRED_COLUMNS:
            columns.append("")
        if not same_metric(columns[MEAN_COLUMN], mean_peak) or not same_metric(
            columns[MAX_COLUMN], max_peak
        ):
            columns[MEAN_COLUMN] = f"{mean_peak:.6f}"
            columns[MAX_COLUMN] = f"{max_peak:.6f}"
            result.changed_rows += 1
            file_changed = True
        result.covered += 1

    if file_changed:
        result.changed_files = 1
        if execute:
            temp = csv_path.with_suffix(csv_path.suffix + ".row-metrics.tmp")
            with temp.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerows(rows)
            os.replace(str(temp), str(csv_path))
    return result


def process_root(root: Path, execute: bool) -> Result:
    if not root.is_dir():
        raise RuntimeError(f"Capture root does not exist: {root}")
    csv_files = report_csv_files(root)
    if not csv_files:
        raise RuntimeError(f"No report CSV files found under: {root}")

    total = Result()
    cache: Dict[Tuple[int, int, int, int], float] = {}
    for index, csv_path in enumerate(csv_files, 1):
        current = rewrite_csv(root, csv_path, execute, cache)
        total.add(current)
        print(
            f"[{index}/{len(csv_files)}] {csv_path.stem} records={current.records:,} "
            f"covered={current.covered:,} missing={current.missing:,} "
            f"changed={current.changed_rows:,}",
            flush=True,
        )
    print(
        f"root={root.resolve()} csv={total.csv_files:,} records={total.records:,} "
        f"covered={total.covered:,} missing={total.missing:,} "
        f"changedRows={total.changed_rows:,} changedFiles={total.changed_files:,} "
        f"uniqueBins={len(cache):,} mode={'EXECUTE' if execute else 'DRY-RUN'}"
    )
    if execute:
        update_stress_marker(root, csv_files)
    return total


def update_stress_marker(root: Path, csv_files: List[Path]) -> None:
    marker_path = root / ".stress-capture-dataset.json"
    if not marker_path.is_file():
        return
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["curveMetricsVersion"] = 2
    marker["curveMetricsStatus"] = "complete"
    marker["curveMetricsCompletedCsv"] = [path.stem for path in csv_files]
    marker["curveMetricsUpdatedUtc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    temp = marker_path.with_suffix(marker_path.suffix + ".tmp")
    temp.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temp), str(marker_path))


def main() -> int:
    args = parse_args()
    grand = Result()
    for root in args.roots:
        grand.add(process_root(root, args.execute))
    if grand.missing:
        print(
            f"warning: {grand.missing:,} CSV rows have missing MeanR/MaxR bins and remain unknown",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
