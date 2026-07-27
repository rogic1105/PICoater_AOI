#!/usr/bin/env python3
"""Build an isolated legacy-loose dataset for compatibility UI stress tests.

The generated dataset uses seven camera records per grab. File contents are
backed by a rotating pool of NTFS hard links so the test exercises realistic
file counts and paths without duplicating the logical payload size. Production
Capture output uses Captures and .acap; this tool is only for exercising
the old loose-file reader.

Dry-run is the default. Pass --execute to create files.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FILE_SUFFIXES = (
    "_raw.jpg",
    "_proc_c.jpg",
    "_proc_r.jpg",
    "_mean_c.bin",
    "_max_c.bin",
    "_mean_r.bin",
    "_max_r.bin",
)
CSV_HEADER = (
    "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,"
    "GrabHeight,LineRateHz,ExposureUs,MaxCMean,MeanRPeak,MaxRPeak"
)
MARKER_NAME = ".stress-capture-dataset.json"
POOL_DIR_NAME = "._stress_link_pool"
CAMERA_COUNT = 7


@dataclass(frozen=True)
class Bucket:
    index: int
    date: datetime
    first_global_index: int
    grab_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an isolated 30k-style capture dataset. Dry-run by default."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Explicit legacy loose-file template root; production Captures is not modified.",
    )
    parser.add_argument("--output", default=r"D:\Anilox\StressCaptures_30000")
    parser.add_argument("--grabs", type=int, default=30_000)
    parser.add_argument("--months", type=int, default=30)
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument(
        "--links-per-pool-file",
        type=int,
        default=900,
        help="Keep below the NTFS per-file hard-link limit.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify every CSV reference and hard-link pool entry after generation.",
    )
    return parser.parse_args()


def add_months(value: datetime, count: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + count
    year, zero_month = divmod(month_index, 12)
    return datetime(year, zero_month + 1, 15, 8, 0, 0)


def build_buckets(start: datetime, grabs: int, months: int) -> List[Bucket]:
    base, remainder = divmod(grabs, months)
    buckets: List[Bucket] = []
    offset = 0
    for index in range(months):
        count = base + (1 if index < remainder else 0)
        buckets.append(Bucket(index, add_months(start, index), offset, count))
        offset += count
    return buckets


def find_complete_template_bases(source: Path) -> Dict[int, Path]:
    """Find complete current-format file families, newest files first."""
    raw_files = sorted(
        source.rglob("*_raw.jpg"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    result: Dict[int, Path] = {}
    camera_pattern = re.compile(r"-(\d+)$")
    for raw in raw_files:
        base_name = raw.name[: -len("_raw.jpg")]
        match = camera_pattern.search(base_name)
        if not match:
            continue
        camera_id = int(match.group(1))
        if camera_id in result:
            continue
        base = raw.parent / base_name
        if all(Path(str(base) + suffix).is_file() for suffix in FILE_SUFFIXES):
            result[camera_id] = base
    return result


def map_seven_cameras(templates: Dict[int, Path]) -> Dict[int, Path]:
    if not templates:
        raise RuntimeError("No complete raw/proc/curve template family was found.")
    ordered = [templates[key] for key in sorted(templates)]
    return {
        camera_id: ordered[(camera_id - 1) % len(ordered)]
        for camera_id in range(1, CAMERA_COUNT + 1)
    }


def cfg_line(timestamp: datetime) -> str:
    fields = ["#CFG", timestamp.isoformat(timespec="milliseconds")]
    fields.extend(f"Cam{i}_Ops=24.41" for i in range(1, 8))
    fields.extend(f"Cam{i}_Pos={(i - 1) * 345.0:.2f}" for i in range(1, 8))
    fields.extend(f"Cam{i}_GrabH=3000" for i in range(1, 8))
    fields.extend(f"Cam{i}_Exp=100.00" for i in range(1, 8))
    fields.extend(f"Cam{i}_Lr=3000.00" for i in range(1, 8))
    fields.extend(
        (
            "HessianMaxFactorV=1.0000",
            "HessianMaxFactorH=1.0000",
            "ErrorValueMeanV=0.2000",
            "ErrorValueMaxV=0.5000",
            "ErrorValueMeanH=0.2000",
            "ErrorValueMaxH=0.5000",
            "TrimHead=0.00",
            "TrimTail=0.00",
        )
    )
    return ",".join(fields)


def ensure_safe_output(source: Path, output: Path, execute: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise RuntimeError("Output must be an isolated sibling tree, not source or its parent/child.")
    if not execute:
        return
    if output.exists():
        entries = list(output.iterdir())
        if entries and not (output / MARKER_NAME).is_file():
            raise RuntimeError(
                f"Refusing non-empty unmarked output directory: {output}"
            )


def load_marker(output: Path) -> dict:
    marker = output / MARKER_NAME
    if not marker.is_file():
        return {"status": "building", "completedBuckets": []}
    return json.loads(marker.read_text(encoding="utf-8"))


def save_marker(output: Path, marker: dict) -> None:
    path = output / MARKER_NAME
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temp), str(path))


def pool_source(
    output: Path,
    template_base: Path,
    camera_id: int,
    suffix: str,
    group: int,
) -> Path:
    safe_suffix = suffix.lstrip("_").replace(".", "_")
    path = output / POOL_DIR_NAME / f"cam{camera_id}" / safe_suffix / f"part-{group:04d}{suffix}"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(template_base) + suffix, path)
    return path


def ensure_hard_link(source: Path, destination: Path) -> bool:
    if destination.exists():
        return False
    os.link(source, destination)
    return True


def load_mcbf_values(path: Path) -> List[float]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != b"MCBF":
        raise RuntimeError(f"Not an MCBF curve: {path}")
    version = struct.unpack_from("<i", raw, 4)[0]
    length_offset = 20 if version >= 2 else 12
    length = struct.unpack_from("<i", raw, length_offset)[0]
    payload_offset = length_offset + 4
    if length <= 0 or len(raw) < payload_offset + length * 4:
        raise RuntimeError(f"Invalid MCBF curve: {path}")
    return list(struct.unpack_from(f"<{length}f", raw, payload_offset))


def template_curve_metrics(
    camera_templates: Dict[int, Path]
) -> Dict[int, Tuple[float, float, float, float, float]]:
    result: Dict[int, Tuple[float, float, float, float, float]] = {}
    for camera_id, base_path in camera_templates.items():
        mean_values = load_mcbf_values(Path(str(base_path) + "_mean_c.bin"))
        max_values = load_mcbf_values(Path(str(base_path) + "_max_c.bin"))
        mean_r_values = load_mcbf_values(Path(str(base_path) + "_mean_r.bin"))
        max_r_values = load_mcbf_values(Path(str(base_path) + "_max_r.bin"))
        result[camera_id] = (
            max(mean_values) / 255.0,
            max(max_values) / 255.0,
            sum(max_values) / len(max_values) / 255.0,
            max(mean_r_values) / 255.0,
            max(max_r_values) / 255.0,
        )
    return result


def build_bucket(
    output: Path,
    bucket: Bucket,
    camera_templates: Dict[int, Path],
    camera_metrics: Dict[int, Tuple[float, float, float, float, float]],
    links_per_pool_file: int,
) -> Tuple[int, int]:
    date = bucket.date
    month_dir = output / date.strftime("%Y") / date.strftime("%Y%m")
    image_dir = month_dir / date.strftime("%Y%m%d")
    month_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    csv_path = month_dir / f"{date:%Y%m%d}.csv"
    ticks_path = image_dir / "_ticks.csv"
    csv_temp = csv_path.with_suffix(".csv.tmp")
    ticks_temp = ticks_path.with_suffix(".csv.tmp")

    linked = 0
    skipped = 0
    with csv_temp.open("w", encoding="utf-8", newline="\n") as csv_file, ticks_temp.open(
        "w", encoding="utf-8", newline="\n"
    ) as ticks_file:
        csv_file.write(cfg_line(date) + "\n")
        csv_file.write(CSV_HEADER + "\n")

        for local_index in range(bucket.grab_count):
            global_index = bucket.first_global_index + local_index
            capture_time = date + timedelta(seconds=local_index)
            grab_id = capture_time.strftime("%y%m%d-%H%M%S")
            base_stamp = capture_time.strftime("%Y%m%d_%H%M%S.000")
            frame_tick = (global_index + 1) * 1_000_000
            pool_group = global_index // links_per_pool_file

            for camera_id in range(1, CAMERA_COUNT + 1):
                file_name = f"{base_stamp}-{camera_id}"
                mean_peak, max_peak, max_c_mean, mean_r_peak, max_r_peak = camera_metrics[camera_id]
                max_exceed = 1 if max_peak > 0.5 else 0
                mean_exceed = 1 if mean_peak > 0.2 else 0
                csv_file.write(
                    f"{grab_id},{file_name},{max_exceed},{mean_exceed},"
                    f"{mean_peak:.4f},{max_peak:.4f},3000,3000.0,100.0,{max_c_mean:.6f},"
                    f"{mean_r_peak:.6f},{max_r_peak:.6f}\n"
                )
                ticks_file.write(f"{file_name},{frame_tick}\n")

                destination_base = image_dir / file_name
                template_base = camera_templates[camera_id]
                for suffix in FILE_SUFFIXES:
                    pooled = pool_source(
                        output, template_base, camera_id, suffix, pool_group
                    )
                    destination = Path(str(destination_base) + suffix)
                    if ensure_hard_link(pooled, destination):
                        linked += 1
                    else:
                        skipped += 1

    os.replace(str(csv_temp), str(csv_path))
    os.replace(str(ticks_temp), str(ticks_path))
    return linked, skipped


def logical_template_bytes(camera_templates: Dict[int, Path]) -> int:
    return sum(
        Path(str(camera_templates[camera_id]) + suffix).stat().st_size
        for camera_id in range(1, CAMERA_COUNT + 1)
        for suffix in FILE_SUFFIXES
    )


def format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TiB"


def verify_dataset(
    output: Path,
    expected_grabs: int,
    expected_months: int,
    links_per_pool_file: int,
) -> bool:
    marker = load_marker(output)
    failures: List[str] = []
    if marker.get("status") != "complete":
        failures.append(f"marker status is {marker.get('status')!r}, expected 'complete'")

    csv_files = sorted(output.glob("*/*/*.csv"))
    masks: Dict[str, int] = {}
    counts: Dict[str, int] = {}
    records = 0
    referenced_files = 0
    missing: List[str] = []
    duplicate_cameras = 0
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line or line[0] == "#" or line.startswith("Id,"):
                    continue
                columns = line.rstrip("\r\n").split(",")
                if len(columns) < 2:
                    failures.append(f"malformed CSV row in {csv_path}: {line[:80]!r}")
                    continue
                grab_id, file_name = columns[0], columns[1]
                try:
                    camera_id = int(file_name.rsplit("-", 1)[1])
                except (IndexError, ValueError):
                    failures.append(f"invalid camera suffix: {file_name}")
                    continue
                if not 1 <= camera_id <= CAMERA_COUNT:
                    failures.append(f"camera outside 1..7: {file_name}")
                    continue
                bit = 1 << (camera_id - 1)
                if masks.get(grab_id, 0) & bit:
                    duplicate_cameras += 1
                masks[grab_id] = masks.get(grab_id, 0) | bit
                counts[grab_id] = counts.get(grab_id, 0) + 1
                records += 1

                date_text = file_name[:8]
                image_dir = output / date_text[:4] / date_text[:6] / date_text
                for suffix in FILE_SUFFIXES:
                    referenced_files += 1
                    path = image_dir / (file_name + suffix)
                    if not path.is_file() and len(missing) < 20:
                        missing.append(str(path))

    full_mask = (1 << CAMERA_COUNT) - 1
    incomplete = [
        grab_id
        for grab_id, mask in masks.items()
        if mask != full_mask or counts.get(grab_id) != CAMERA_COUNT
    ]
    if len(csv_files) != expected_months:
        failures.append(f"CSV files={len(csv_files)}, expected={expected_months}")
    if len(masks) != expected_grabs:
        failures.append(f"grab IDs={len(masks)}, expected={expected_grabs}")
    if records != expected_grabs * CAMERA_COUNT:
        failures.append(
            f"CSV records={records}, expected={expected_grabs * CAMERA_COUNT}"
        )
    if incomplete:
        failures.append(f"incomplete seven-camera grabs={len(incomplete)} first={incomplete[:5]}")
    if duplicate_cameras:
        failures.append(f"duplicate camera records={duplicate_cameras}")
    if missing:
        failures.append(f"missing referenced files (first {len(missing)}): {missing}")

    pool_files = [path for path in (output / POOL_DIR_NAME).rglob("*") if path.is_file()]
    link_counts = [path.stat().st_nlink for path in pool_files]
    max_links = max(link_counts) if link_counts else 0
    if max_links > links_per_pool_file + 1:
        failures.append(
            f"pool hard-link count={max_links}, safe maximum={links_per_pool_file + 1}"
        )

    print(
        f"verify csvFiles={len(csv_files):,} grabIds={len(masks):,} "
        f"records={records:,} referencedFiles={referenced_files:,} "
        f"poolFiles={len(pool_files):,} maxPoolLinks={max_links:,}"
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return False
    print("verify PASS: every grab has CAM1..CAM7 and all referenced files exist")
    return True


def main() -> int:
    args = parse_args()
    if args.grabs <= 0 or args.months <= 0 or args.months > args.grabs:
        raise RuntimeError("--grabs and --months must be positive; months cannot exceed grabs.")
    if not 1 <= args.links_per_pool_file <= 900:
        raise RuntimeError("--links-per-pool-file must be between 1 and 900.")

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        raise RuntimeError(f"Source capture root does not exist: {source}")
    ensure_safe_output(source, output, args.execute)

    if args.verify:
        if not output.is_dir():
            raise RuntimeError(f"Output dataset does not exist: {output}")
        return 0 if verify_dataset(
            output, args.grabs, args.months, args.links_per_pool_file
        ) else 2

    start = datetime.strptime(args.start_month, "%Y-%m").replace(day=15, hour=8)
    buckets = build_buckets(start, args.grabs, args.months)
    discovered = find_complete_template_bases(source)
    camera_templates = map_seven_cameras(discovered)
    camera_metrics = template_curve_metrics(camera_templates)
    files_per_grab = CAMERA_COUNT * len(FILE_SUFFIXES)
    pool_groups = (args.grabs + args.links_per_pool_file - 1) // args.links_per_pool_file
    pool_files = pool_groups * files_per_grab
    logical_bytes = logical_template_bytes(camera_templates) * args.grabs
    pool_bytes = logical_template_bytes(camera_templates) * pool_groups

    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"source={source.resolve()}")
    print(f"output={output.resolve()}")
    print(f"grabs={args.grabs:,} months={args.months} cameras={CAMERA_COUNT}")
    print(f"csvRows={args.grabs * CAMERA_COUNT:,}")
    print(f"linkedFiles={args.grabs * files_per_grab:,} ({files_per_grab} per grab)")
    print(f"poolFiles={pool_files:,} linksPerPoolFile<={args.links_per_pool_file}")
    print(f"logicalPayload={format_bytes(logical_bytes)}")
    print(f"estimatedPhysicalPoolPayload={format_bytes(pool_bytes)} plus NTFS metadata")
    for camera_id in range(1, CAMERA_COUNT + 1):
        print(f"CAM{camera_id} template={camera_templates[camera_id]}")

    if not args.execute:
        print("dry-run complete; pass --execute to create the isolated dataset")
        return 0

    output.mkdir(parents=True, exist_ok=True)
    marker = load_marker(output)
    expected = {
        "source": str(source.resolve()),
        "grabs": args.grabs,
        "months": args.months,
        "startMonth": args.start_month,
        "cameraCount": CAMERA_COUNT,
        "suffixes": list(FILE_SUFFIXES),
        "linksPerPoolFile": args.links_per_pool_file,
    }
    for key, value in expected.items():
        existing = marker.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(f"Existing marker mismatch for {key}: {existing!r} != {value!r}")
        marker[key] = value
    completed = set(marker.get("completedBuckets", []))
    marker["status"] = "building"
    save_marker(output, marker)

    started = time.monotonic()
    total_linked = 0
    total_skipped = 0
    for bucket in buckets:
        key = bucket.date.strftime("%Y%m")
        if key in completed:
            print(f"[{bucket.index + 1}/{len(buckets)}] {key} already complete")
            continue
        bucket_start = time.monotonic()
        linked, skipped = build_bucket(
            output, bucket, camera_templates, camera_metrics, args.links_per_pool_file
        )
        total_linked += linked
        total_skipped += skipped
        completed.add(key)
        marker["completedBuckets"] = sorted(completed)
        marker["lastUpdatedUtc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        save_marker(output, marker)
        elapsed = time.monotonic() - bucket_start
        print(
            f"[{bucket.index + 1}/{len(buckets)}] {key} grabs={bucket.grab_count:,} "
            f"linked={linked:,} existing={skipped:,} elapsed={elapsed:.1f}s",
            flush=True,
        )

    marker["status"] = "complete"
    marker["completedUtc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    save_marker(output, marker)
    elapsed = time.monotonic() - started
    print(
        f"complete linked={total_linked:,} existing={total_skipped:,} "
        f"elapsed={elapsed:.1f}s output={output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted; rerun the same command to resume", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
