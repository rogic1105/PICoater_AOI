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
FORMULA_VERSION = 1


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
    denominator = max(1, length - 1)
    for index in range(length):
        x = index / denominator
        peak = amplitude * math.exp(-0.5 * ((x - center) / width) ** 2)
        ripple = (5.0 if is_max else 2.0) * math.sin(x * math.tau * 3.0 + phase)
        result.append(max(0.0, min(250.0, baseline + peak + ripple)))
    return result


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


def relink_dataset(output: Path, csv_files: List[Path], variants: int, marker: dict) -> None:
    completed = set(marker.get("curveVariationCompletedCsv", []))
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

    global_grab_index = 0
    checked = 0
    missing = 0
    wrong = 0
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
    if missing or wrong:
        failures.append(f"missing={missing} wrongVariant={wrong}")

    print(
        f"verify grabs={global_grab_index:,} curves={checked:,} variants={variants} "
        f"poolFiles={len(pool_files):,} maxLinks={max_links:,}"
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
    if existing_formula not in (None, FORMULA_VERSION):
        raise RuntimeError(
            f"Existing formula version {existing_formula} differs from {FORMULA_VERSION}."
        )
    existing_variants = marker.get("curveVariants")
    if existing_variants not in (None, args.variants):
        raise RuntimeError(
            f"Existing variant count {existing_variants} differs from {args.variants}."
        )
    marker["curveVariationFormulaVersion"] = FORMULA_VERSION
    marker["curveVariants"] = args.variants
    marker["curveVariationStatus"] = "building"
    save_marker(output, marker)

    templates = template_paths(output, csv_files)
    build_variant_pool(output, templates, args.variants)
    relink_dataset(output, csv_files, args.variants, marker)
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
