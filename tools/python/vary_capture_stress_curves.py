#!/usr/bin/env python3
"""Replace curve links in a marked stress dataset with visible variants.

The production capture tree is never accepted. The tool requires the isolated
dataset marker created by generate_capture_stress_dataset.py. Dry-run is the
default; pass --execute to relink curve bins and --verify to audit the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


MARKER_NAME = ".stress-capture-dataset.json"
VARIANT_DIR_NAME = "._stress_curve_variants"
CURVE_SUFFIXES = (
    "_mean_c.bin",
    "_max_c.bin",
    "_mean_r.bin",
    "_max_r.bin",
)
FORMULA_VERSION = 4
METRICS_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Give an isolated capture stress dataset distinguishable curves."
    )
    parser.add_argument("--output", default=r"D:\Anilox\StressCaptures_30000")
    parser.add_argument("--variants", type=int, default=32)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def load_marker(output: Path) -> dict:
    marker_path = output / MARKER_NAME
    if not marker_path.is_file():
        raise RuntimeError(f"Refusing unmarked dataset: {output}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "complete":
        raise RuntimeError(f"Dataset is not complete: status={marker.get('status')!r}")
    if int(marker.get("cameraCount", 0)) != 7:
        raise RuntimeError("Expected a seven-camera stress dataset.")
    source = Path(str(marker.get("source", ""))).resolve()
    if output.resolve() == source:
        raise RuntimeError("Refusing to modify the production source capture tree.")
    return marker


def save_marker(output: Path, marker: dict) -> None:
    path = output / MARKER_NAME
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(marker, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temp), str(path))


def report_csv_files(output: Path) -> List[Path]:
    return sorted(
        path
        for path in output.glob("*/*/*.csv")
        if len(path.stem) == 8 and path.stem.isdigit()
    )


def csv_records(csv_path: Path) -> Iterator[Tuple[str, str, int, Path]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(line for line in stream if line and line[0] != "#")
        header = next(reader, None)
        if not header or header[:2] != ["Id", "FileName"]:
            raise RuntimeError(f"Unexpected CSV header: {csv_path}")
        for columns in reader:
            if len(columns) < 2:
                continue
            grab_id, file_name = columns[0], columns[1]
            try:
                camera_id = int(file_name.rsplit("-", 1)[1])
            except (IndexError, ValueError) as exc:
                raise RuntimeError(f"Invalid camera name: {file_name}") from exc
            date_text = file_name[:8]
            image_dir = csv_path.parents[2] / date_text[:4] / date_text[:6] / date_text
            yield grab_id, file_name, camera_id, image_dir


def template_paths(output: Path, csv_files: Iterable[Path]) -> Dict[Tuple[int, str], Path]:
    result: Dict[Tuple[int, str], Path] = {}
    for csv_path in csv_files:
        for _, file_name, camera_id, image_dir in csv_records(csv_path):
            for suffix in CURVE_SUFFIXES:
                key = (camera_id, suffix)
                candidate = image_dir / (file_name + suffix)
                if key not in result and candidate.is_file():
                    result[key] = candidate
            if len(result) == 7 * len(CURVE_SUFFIXES):
                return result
    raise RuntimeError(
        f"Found {len(result)} curve template families, expected {7 * len(CURVE_SUFFIXES)}."
    )


def read_mcbf(path: Path) -> Tuple[bytes, int]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != b"MCBF":
        raise RuntimeError(f"Not an MCBF curve: {path}")
    version = struct.unpack_from("<i", raw, 4)[0]
    length_offset = 20 if version >= 2 else 12
    if len(raw) < length_offset + 4:
        raise RuntimeError(f"Truncated MCBF header: {path}")
    length = struct.unpack_from("<i", raw, length_offset)[0]
    payload_offset = length_offset + 4
    if length <= 0 or len(raw) < payload_offset + length * 4:
        raise RuntimeError(f"Invalid MCBF length: {path}")
    return raw[:payload_offset], length


def read_mcbf_values(path: Path) -> List[float]:
    raw = path.read_bytes()
    if len(raw) < 16 or raw[:4] != b"MCBF":
        raise RuntimeError(f"Not an MCBF curve: {path}")
    version = struct.unpack_from("<i", raw, 4)[0]
    length_offset = 20 if version >= 2 else 12
    if len(raw) < length_offset + 4:
        raise RuntimeError(f"Truncated MCBF header: {path}")
    length = struct.unpack_from("<i", raw, length_offset)[0]
    payload_offset = length_offset + 4
    if length <= 0 or len(raw) < payload_offset + length * 4:
        raise RuntimeError(f"Invalid MCBF length: {path}")
    return list(struct.unpack_from(f"<{length}f", raw, payload_offset))


def variant_path(
    output: Path, camera_id: int, suffix: str, variant: int
) -> Path:
    family = suffix.lstrip("_").replace(".bin", "")
    return (
        output
        / VARIANT_DIR_NAME
        / f"cam{camera_id}"
        / family
        / f"variant-{variant:02d}.bin"
    )


def curve_values(
    length: int, camera_id: int, suffix: str, variant: int, variants: int
) -> List[float]:
    is_max = suffix.startswith("_max")
    is_row = suffix.endswith("_r.bin")
    baseline = (24.0 if is_max else 10.0) + (variant % 8) * (3.0 if is_max else 1.5)
    amplitude = (105.0 if is_max else 42.0) + (variant // 8) * 12.0
    center_slot = (variant * 5 + camera_id * 3 + (11 if is_row else 0)) % variants
    center = (center_slot + 0.5) / variants
    width = 0.018 + (variant % 4) * 0.006
    phase = (camera_id * 0.7) + (variant * 0.25)
    result: List[float] = []
    if is_max:
        severity_scale = 0.55 if variant < variants // 2 else 1.0
    else:
        severity_scale = 0.55 if variant < variants // 2 else 0.44
    denominator = max(1, length - 1)
    for index in range(length):
        x = index / denominator
        peak = amplitude * math.exp(-0.5 * ((x - center) / width) ** 2)
        ripple = (5.0 if is_max else 2.0) * math.sin(x * math.tau * 3.0 + phase)
        result.append(
            max(0.0, min(250.0, (baseline + peak + ripple) * severity_scale))
        )
    return result


def upgrade_variant_pool_v1_to_v2(output: Path, variants: int) -> None:
    """Scale the lower half in place so every existing hard link sees v2."""
    for camera_id in range(1, 8):
        for suffix in CURVE_SUFFIXES:
            for variant in range(variants // 2):
                path = variant_path(output, camera_id, suffix, variant)
                raw = path.read_bytes()
                version = struct.unpack_from("<i", raw, 4)[0]
                length_offset = 20 if version >= 2 else 12
                length = struct.unpack_from("<i", raw, length_offset)[0]
                payload_offset = length_offset + 4
                values = struct.unpack_from(f"<{length}f", raw, payload_offset)
                payload = struct.pack(
                    f"<{length}f", *(value * 0.55 for value in values)
                )
                with path.open("r+b") as stream:
                    stream.seek(payload_offset)
                    stream.write(payload)
                    stream.truncate(payload_offset + len(payload))


def upgrade_variant_pool_v2_to_v3(output: Path, variants: int) -> None:
    """Keep Mean below its threshold so Max can be tested independently."""
    for camera_id in range(1, 8):
        for suffix in ("_mean_c.bin", "_mean_r.bin"):
            for variant in range(variants // 2, variants):
                path = variant_path(output, camera_id, suffix, variant)
                raw = path.read_bytes()
                version = struct.unpack_from("<i", raw, 4)[0]
                length_offset = 20 if version >= 2 else 12
                length = struct.unpack_from("<i", raw, length_offset)[0]
                payload_offset = length_offset + 4
                values = struct.unpack_from(f"<{length}f", raw, payload_offset)
                payload = struct.pack(
                    f"<{length}f", *(value * 0.55 for value in values)
                )
                with path.open("r+b") as stream:
                    stream.seek(payload_offset)
                    stream.write(payload)
                    stream.truncate(payload_offset + len(payload))


def upgrade_variant_pool_v3_to_v4(output: Path, variants: int) -> None:
    """Finish isolating Max by keeping every Mean variant below 0.2."""
    for camera_id in range(1, 8):
        for suffix in ("_mean_c.bin", "_mean_r.bin"):
            for variant in range(variants // 2, variants):
                path = variant_path(output, camera_id, suffix, variant)
                raw = path.read_bytes()
                version = struct.unpack_from("<i", raw, 4)[0]
                length_offset = 20 if version >= 2 else 12
                length = struct.unpack_from("<i", raw, length_offset)[0]
                payload_offset = length_offset + 4
                values = struct.unpack_from(f"<{length}f", raw, payload_offset)
                payload = struct.pack(
                    f"<{length}f", *(value * 0.8 for value in values)
                )
                with path.open("r+b") as stream:
                    stream.seek(payload_offset)
                    stream.write(payload)
                    stream.truncate(payload_offset + len(payload))


def build_variant_pool(
    output: Path,
    templates: Dict[Tuple[int, str], Path],
    variants: int,
) -> None:
    for camera_id in range(1, 8):
        for suffix in CURVE_SUFFIXES:
            header, length = read_mcbf(templates[(camera_id, suffix)])
            for variant in range(variants):
                destination = variant_path(output, camera_id, suffix, variant)
                if destination.is_file():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                values = curve_values(length, camera_id, suffix, variant, variants)
                temp = destination.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    stream.write(header)
                    stream.write(struct.pack(f"<{length}f", *values))
                os.replace(str(temp), str(destination))


def build_variant_metrics(
    output: Path, variants: int
) -> Dict[Tuple[int, int], Tuple[float, float, float, float, float]]:
    result: Dict[Tuple[int, int], Tuple[float, float, float, float, float]] = {}
    for camera_id in range(1, 8):
        for variant in range(variants):
            mean_values = read_mcbf_values(
                variant_path(output, camera_id, "_mean_c.bin", variant)
            )
            max_values = read_mcbf_values(
                variant_path(output, camera_id, "_max_c.bin", variant)
            )
            mean_r_values = read_mcbf_values(
                variant_path(output, camera_id, "_mean_r.bin", variant)
            )
            max_r_values = read_mcbf_values(
                variant_path(output, camera_id, "_max_r.bin", variant)
            )
            result[(camera_id, variant)] = (
                max(mean_values) / 255.0,
                max(max_values) / 255.0,
                sum(max_values) / len(max_values) / 255.0,
                max(mean_r_values) / 255.0,
                max(max_r_values) / 255.0,
            )
    return result


def rewrite_csv_metrics(
    csv_path: Path,
    grab_order: Dict[str, int],
    metrics: Dict[Tuple[int, int], Tuple[float, float, float, float, float]],
    variants: int,
) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    error_mean = 0.2
    error_max = 0.5
    updated = 0
    for columns in rows:
        if not columns:
            continue
        if columns[0] == "#CFG":
            for field in columns[2:]:
                if field.startswith("ErrorValueMeanV="):
                    error_mean = float(field.split("=", 1)[1])
                elif field.startswith("ErrorValueMaxV="):
                    error_max = float(field.split("=", 1)[1])
            continue
        if columns[0] == "Id":
            while len(columns) < 12:
                columns.append("")
            columns[10] = "MeanRPeak"
            columns[11] = "MaxRPeak"
            continue
        if len(columns) < 10:
            continue

        grab_id, file_name = columns[0], columns[1]
        camera_id = int(file_name.rsplit("-", 1)[1])
        variant = grab_order[grab_id] % variants
        mean_peak, max_peak, max_c_mean, mean_r_peak, max_r_peak = metrics[(camera_id, variant)]
        while len(columns) < 12:
            columns.append("")
        columns[2] = "1" if max_peak > error_max else "0"
        columns[3] = "1" if mean_peak > error_mean else "0"
        columns[4] = f"{mean_peak:.4f}"
        columns[5] = f"{max_peak:.4f}"
        columns[9] = f"{max_c_mean:.6f}"
        columns[10] = f"{mean_r_peak:.6f}"
        columns[11] = f"{max_r_peak:.6f}"
        updated += 1

    temp = csv_path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerows(rows)
    os.replace(str(temp), str(csv_path))
    return updated


def relink_dataset(
    output: Path,
    csv_files: List[Path],
    variants: int,
    marker: dict,
    metrics: Dict[Tuple[int, int], Tuple[float, float, float, float, float]],
) -> None:
    completed = set(marker.get("curveVariationCompletedCsv", []))
    metrics_completed = set(marker.get("curveMetricsCompletedCsv", []))
    global_grab_index = 0
    linked = 0
    existing = 0
    started = time.monotonic()
    for csv_index, csv_path in enumerate(csv_files):
        csv_key = csv_path.stem
        records = list(csv_records(csv_path))
        grab_order: Dict[str, int] = {}
        for grab_id, _, _, _ in records:
            if grab_id not in grab_order:
                grab_order[grab_id] = global_grab_index + len(grab_order)

        if csv_key not in completed:
            for grab_id, file_name, camera_id, image_dir in records:
                variant = grab_order[grab_id] % variants
                for suffix in CURVE_SUFFIXES:
                    source = variant_path(output, camera_id, suffix, variant)
                    destination = image_dir / (file_name + suffix)
                    if destination.exists() and os.path.samefile(source, destination):
                        existing += 1
                        continue
                    if destination.exists():
                        destination.unlink()
                    os.link(source, destination)
                    linked += 1

            completed.add(csv_key)
            marker["curveVariationCompletedCsv"] = sorted(completed)
            marker["curveVariationUpdatedUtc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            save_marker(output, marker)

        if csv_key not in metrics_completed:
            rewrite_csv_metrics(csv_path, grab_order, metrics, variants)
            metrics_completed.add(csv_key)
            marker["curveMetricsCompletedCsv"] = sorted(metrics_completed)
            marker["curveMetricsUpdatedUtc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            save_marker(output, marker)

        global_grab_index += len(grab_order)
        print(
            f"[{csv_index + 1}/{len(csv_files)}] {csv_key} grabs={len(grab_order):,} "
            f"linked={linked:,} existing={existing:,}",
            flush=True,
        )

    removed_summaries = 0
    for summary in output.glob("*/*/*/_curve_summary/*.mcsf"):
        summary.unlink()
        removed_summaries += 1
    marker["curveVariationStatus"] = "complete"
    marker["curveMetricsStatus"] = "complete"
    marker["curveVariationCompletedUtc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    save_marker(output, marker)
    print(
        f"complete linked={linked:,} existing={existing:,} "
        f"removedSummaries={removed_summaries:,} elapsed={time.monotonic() - started:.1f}s"
    )


def verify(output: Path, csv_files: List[Path], variants: int, marker: dict) -> bool:
    failures: List[str] = []
    if marker.get("curveVariationStatus") != "complete":
        failures.append(
            f"curveVariationStatus={marker.get('curveVariationStatus')!r}"
        )
    if marker.get("curveVariants") != variants:
        failures.append(f"curveVariants={marker.get('curveVariants')!r}")
    if marker.get("curveVariationFormulaVersion") != FORMULA_VERSION:
        failures.append(
            f"formulaVersion={marker.get('curveVariationFormulaVersion')!r}"
        )
    if marker.get("curveMetricsStatus") != "complete":
        failures.append(f"curveMetricsStatus={marker.get('curveMetricsStatus')!r}")
    if marker.get("curveMetricsVersion") != METRICS_VERSION:
        failures.append(f"curveMetricsVersion={marker.get('curveMetricsVersion')!r}")

    metrics = build_variant_metrics(output, variants)
    global_grab_index = 0
    checked = 0
    missing = 0
    wrong = 0
    metric_mismatches = 0
    for csv_path in csv_files:
        records = list(csv_records(csv_path))
        grab_order: Dict[str, int] = {}
        for grab_id, _, _, _ in records:
            if grab_id not in grab_order:
                grab_order[grab_id] = global_grab_index + len(grab_order)
        for grab_id, file_name, camera_id, image_dir in records:
            expected_variant = grab_order[grab_id] % variants
            for suffix in CURVE_SUFFIXES:
                expected = variant_path(
                    output, camera_id, suffix, expected_variant
                )
                actual = image_dir / (file_name + suffix)
                checked += 1
                if not actual.is_file():
                    missing += 1
                elif not os.path.samefile(expected, actual):
                    wrong += 1

        error_mean = 0.2
        error_max = 0.5
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            rows = csv.reader(stream)
            for columns in rows:
                if not columns:
                    continue
                if columns[0] == "#CFG":
                    for field in columns[2:]:
                        if field.startswith("ErrorValueMeanV="):
                            error_mean = float(field.split("=", 1)[1])
                        elif field.startswith("ErrorValueMaxV="):
                            error_max = float(field.split("=", 1)[1])
                    continue
                if columns[0] == "Id" or len(columns) < 12:
                    continue
                grab_id, file_name = columns[0], columns[1]
                camera_id = int(file_name.rsplit("-", 1)[1])
                variant = grab_order[grab_id] % variants
                mean_peak, max_peak, max_c_mean, mean_r_peak, max_r_peak = metrics[(camera_id, variant)]
                expected_values = (
                    1 if max_peak > error_max else 0,
                    1 if mean_peak > error_mean else 0,
                    mean_peak,
                    max_peak,
                    max_c_mean,
                    mean_r_peak,
                    max_r_peak,
                )
                actual_values = (
                    int(columns[2]),
                    int(columns[3]),
                    float(columns[4]),
                    float(columns[5]),
                    float(columns[9]),
                    float(columns[10]),
                    float(columns[11]),
                )
                if (
                    actual_values[0] != expected_values[0]
                    or actual_values[1] != expected_values[1]
                    or abs(actual_values[2] - expected_values[2]) > 0.00011
                    or abs(actual_values[3] - expected_values[3]) > 0.00011
                    or abs(actual_values[4] - expected_values[4]) > 0.0000011
                    or abs(actual_values[5] - expected_values[5]) > 0.0000011
                    or abs(actual_values[6] - expected_values[6]) > 0.0000011
                ):
                    metric_mismatches += 1
        global_grab_index += len(grab_order)

    pool_files = list((output / VARIANT_DIR_NAME).rglob("*.bin"))
    max_links = max((path.stat().st_nlink for path in pool_files), default=0)
    expected_pool_files = 7 * len(CURVE_SUFFIXES) * variants
    safe_max = math.ceil(global_grab_index / variants) + 1
    if len(pool_files) != expected_pool_files:
        failures.append(
            f"variant files={len(pool_files)}, expected={expected_pool_files}"
        )
    if max_links > safe_max:
        failures.append(f"max hard links={max_links}, safe maximum={safe_max}")
    if missing or wrong or metric_mismatches:
        failures.append(
            f"missing={missing} wrongVariant={wrong} csvMetricMismatch={metric_mismatches}"
        )

    print(
        f"verify grabs={global_grab_index:,} curves={checked:,} variants={variants} "
        f"poolFiles={len(pool_files):,} maxLinks={max_links:,} "
        f"csvMetricMismatch={metric_mismatches:,}"
    )
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return False
    print("verify PASS: adjacent grabs cycle through distinguishable curve variants")
    return True


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    if not 30 <= args.variants <= 64:
        raise RuntimeError("--variants must stay between 30 and 64 for NTFS link safety.")
    marker = load_marker(output)
    csv_files = report_csv_files(output)
    if not csv_files:
        raise RuntimeError(f"No report CSV files found under {output}")

    if args.verify:
        return 0 if verify(output, csv_files, args.variants, marker) else 2

    grab_count = int(marker.get("grabs", 0))
    max_links = math.ceil(grab_count / args.variants) + 1
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"output={output.resolve()}")
    print(f"grabs={grab_count:,} variants={args.variants} maxLinksPerVariant={max_links}")
    print(f"curveLinks={grab_count * 7 * len(CURVE_SUFFIXES):,}")
    if max_links > 1024:
        raise RuntimeError("Variant count is too small for the NTFS hard-link limit.")
    if not args.execute:
        print("dry-run complete; pass --execute to relink the isolated dataset")
        return 0

    existing_formula = marker.get("curveVariationFormulaVersion")
    if existing_formula not in (None, 1, 2, 3, FORMULA_VERSION):
        raise RuntimeError(
            f"Existing formula version {existing_formula} differs from {FORMULA_VERSION}."
        )
    existing_variants = marker.get("curveVariants")
    if existing_variants not in (None, args.variants):
        raise RuntimeError(
            f"Existing variant count {existing_variants} differs from {args.variants}."
        )
    if marker.get("curveMetricsVersion") != METRICS_VERSION:
        print(
            f"upgrade curve metrics v{marker.get('curveMetricsVersion')} -> v{METRICS_VERSION} "
            "(add MeanRPeak/MaxRPeak)"
        )
        marker["curveMetricsCompletedCsv"] = []
    if existing_formula == 1:
        print("upgrade curve formula v1 -> v2 (balanced pass/fail variants)")
        upgrade_variant_pool_v1_to_v2(output, args.variants)
        existing_formula = 2
    if existing_formula == 2:
        print("upgrade curve formula v2 -> v3 (Max-only threshold variants)")
        upgrade_variant_pool_v2_to_v3(output, args.variants)
        existing_formula = 3
    if existing_formula == 3:
        print("upgrade curve formula v3 -> v4 (Mean isolation)")
        upgrade_variant_pool_v3_to_v4(output, args.variants)
        marker["curveMetricsCompletedCsv"] = []

    marker["curveVariationFormulaVersion"] = FORMULA_VERSION
    marker["curveVariants"] = args.variants
    marker["curveVariationStatus"] = "building"
    marker["curveMetricsVersion"] = METRICS_VERSION
    marker["curveMetricsStatus"] = "building"
    save_marker(output, marker)

    templates = template_paths(output, csv_files)
    build_variant_pool(output, templates, args.variants)
    metrics = build_variant_metrics(output, args.variants)
    relink_dataset(output, csv_files, args.variants, marker, metrics)
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
