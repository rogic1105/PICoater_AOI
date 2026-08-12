#!/usr/bin/env python3
"""Resumably backfill neutral Hessian maps into existing ACAP archives.

Every changed archive is copied or rebuilt, validated, then atomically replaced.
Historical CSV stores RidgeSigma but not the background algorithm/version, so old
records are reconstructed with the single-frame background path and reported as such.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import shutil
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

FILE_MAGIC = b"PICACAP\0"
RECORD_MAGIC = b"AREC"
RAW_JPEG, HESSIAN_C, HESSIAN_R = 1, 14, 15
DEFAULT_RIDGE_SIGMA = 9.0


@dataclass
class Record:
    kind: int
    camera_id: int
    ticks: int
    base_name: str
    payload: bytes


@dataclass
class ArchiveData:
    path: Path
    grab_id: str
    records: list[Record]


@dataclass
class BackfillReport:
    root: str
    started: str
    apply: bool
    standard_scale: int
    replace_existing: bool
    archives_scanned: int = 0
    archives_updated: int = 0
    archives_complete: int = 0
    archives_without_raw: int = 0
    archives_failed: int = 0
    frames_scanned: int = 0
    frames_missing: int = 0
    frames_wrong_scale: int = 0
    frames_updated: int = 0
    maps_added: int = 0
    bytes_added: int = 0
    bytes_delta: int = 0
    failures: list[str] = field(default_factory=list)


class PipelineInput(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int),
                ("data", ctypes.c_void_p), ("stream", ctypes.c_void_p)]


class PipelineOutput(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int), ("height", ctypes.c_int),
        ("background_data", ctypes.c_void_p), ("mura_data", ctypes.c_void_p),
        ("ridge_data", ctypes.c_void_p), ("mura_curve_mean", ctypes.c_void_p),
        ("mura_curve_max", ctypes.c_void_p), ("mura_row_curve_mean", ctypes.c_void_p),
        ("mura_row_curve_max", ctypes.c_void_p), ("stream", ctypes.c_void_p),
        ("resize_width", ctypes.c_int), ("resize_height", ctypes.c_int),
        ("resized_raw", ctypes.c_void_p), ("resized_ridge", ctypes.c_void_p),
        ("resized_mura", ctypes.c_void_p), ("standard_width", ctypes.c_int),
        ("standard_height", ctypes.c_int),
        ("resized_hessian_column_half", ctypes.c_void_p),
        ("resized_hessian_row_half", ctypes.c_void_p)]


def _ptr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def read_archive(path: Path, verify_crc: bool = True) -> ArchiveData:
    data = path.read_bytes()
    if len(data) < 24 or data[:8] != FILE_MAGIC:
        raise ValueError("invalid ACAP header")
    version, = struct.unpack_from("<i", data, 8)
    grab_len, = struct.unpack_from("<i", data, 20)
    if version != 1 or grab_len <= 0 or 24 + grab_len > len(data):
        raise ValueError("invalid ACAP metadata")
    grab_id = data[24:24 + grab_len].decode("utf-8")
    pos, records = 24 + grab_len, []
    while pos < len(data):
        if pos + 36 > len(data) or data[pos:pos + 4] != RECORD_MAGIC:
            raise ValueError(f"truncated record at {pos}")
        rec_version, = struct.unpack_from("<i", data, pos + 4)
        kind = data[pos + 8]
        camera_id, ticks, name_len, payload_len, crc = struct.unpack_from("<iqiiI", data, pos + 12)
        payload_start = pos + 36 + name_len
        payload_end = payload_start + payload_len
        if rec_version != 1 or name_len <= 0 or payload_len <= 0 or payload_end > len(data):
            raise ValueError(f"invalid record at {pos}")
        name = data[pos + 36:payload_start].decode("utf-8")
        payload = data[payload_start:payload_end]
        if verify_crc and (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
            raise ValueError(f"CRC mismatch: {name} kind={kind}")
        records.append(Record(kind, camera_id, ticks, name, payload))
        pos = payload_end
    return ArchiveData(path, grab_id, records)


def encode_hsm(values: np.ndarray) -> bytes:
    values = np.ascontiguousarray(values, dtype=np.float16)
    height, width = values.shape
    pairs = values.view(np.uint8).reshape(-1, 2)
    shuffled = np.concatenate((pairs[:, 1], pairs[:, 0])).tobytes()
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(shuffled) + compressor.flush()
    return struct.pack("<IBBBBiiii", 0x314D5348, 1, 1, 1, 0,
                       width, height, values.nbytes, len(compressed)) + compressed


def validate_hsm(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24:
        raise ValueError("short HSM")
    magic, version, sample, compression, _, width, height, raw_len, compressed_len = \
        struct.unpack_from("<IBBBBiiii", payload)
    if (magic, version, sample, compression) != (0x314D5348, 1, 1, 1):
        raise ValueError("invalid HSM header")
    if width <= 0 or height <= 0 or raw_len != width * height * 2:
        raise ValueError("invalid HSM dimensions")
    compressed = payload[24:]
    if compressed_len != len(compressed) or len(zlib.decompress(compressed, -15)) != raw_len:
        raise ValueError("invalid HSM payload")
    return width, height


def append_record(stream, record: Record) -> int:
    name, payload = record.base_name.encode("utf-8"), record.payload
    stream.write(struct.pack("<4siB3xiqiiI", RECORD_MAGIC, 1, record.kind,
                             record.camera_id, record.ticks, len(name), len(payload),
                             zlib.crc32(payload) & 0xFFFFFFFF))
    stream.write(name)
    stream.write(payload)
    return len(payload)


def write_archive(path: Path, archive: ArchiveData, records: list[Record]) -> None:
    grab = archive.grab_id.encode("utf-8")
    with path.open("wb", buffering=1024 * 1024) as writer:
        writer.write(FILE_MAGIC + struct.pack("<iqi", 1, 0, len(grab)) + grab)
        for record in records:
            append_record(writer, record)
        writer.flush()
        os.fsync(writer.fileno())


def load_ridge_sigma_by_grab(root: Path) -> dict[str, float]:
    result = {}
    for csv_path in sorted(root.rglob("*.csv")):
        active = DEFAULT_RIDGE_SIGMA
        try:
            for line in csv_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                if line.startswith("#CFG,"):
                    for part in line.split(","):
                        if part.startswith("RidgeSigma="):
                            try:
                                value = float(part.split("=", 1)[1])
                                if value > 0:
                                    active = value
                            except ValueError:
                                pass
                elif line and not line.startswith("Id,"):
                    result[line.split(",", 1)[0].strip()] = active
        except OSError:
            pass
    return result


class NativePipeline:
    def __init__(self, dll_path: Path, standard_scale: int):
        dependencies = [dll_path.parent,
                        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
                        Path(r"C:\vcpkg\installed\x64-windows\bin")]
        self._dll_handles = []
        if hasattr(os, "add_dll_directory"):
            self._dll_handles = [os.add_dll_directory(str(p)) for p in dependencies if p.is_dir()]
        os.environ["PATH"] = os.pathsep.join(str(p) for p in dependencies) + os.pathsep + os.environ["PATH"]
        self._dll = ctypes.CDLL(str(dll_path))
        self._dll.TanukiPipeline_Create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._dll.TanukiPipeline_Create.restype = ctypes.c_void_p
        self._dll.TanukiPipeline_Process.argtypes = [ctypes.c_void_p, ctypes.POINTER(PipelineInput),
                                                      ctypes.c_char_p, ctypes.c_void_p,
                                                      ctypes.POINTER(PipelineOutput)]
        self._dll.TanukiPipeline_Process.restype = ctypes.c_int
        self._dll.TanukiPipeline_GetLastError.argtypes = [ctypes.c_void_p]
        self._dll.TanukiPipeline_GetLastError.restype = ctypes.c_char_p
        self._dll.TanukiPipeline_Destroy.argtypes = [ctypes.c_void_p]
        self._handle = self._dll.TanukiPipeline_Create(b"find_stream_ridgeline", None)
        if not self._handle:
            raise RuntimeError("TanukiPipeline_Create failed")
        self._scale = max(1, standard_scale)

    def close(self):
        if self._handle:
            self._dll.TanukiPipeline_Destroy(self._handle)
            self._handle = None

    def process(self, jpeg: bytes, ridge_sigma: float) -> tuple[np.ndarray, np.ndarray]:
        image = np.ascontiguousarray(np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"), dtype=np.uint8))
        height, width = image.shape
        background, mura, ridge = np.empty_like(image), np.empty_like(image), np.empty_like(image)
        cmean, cmax = np.empty(width, np.float32), np.empty(width, np.float32)
        rmean, rmax = np.empty(height, np.float32), np.empty(height, np.float32)
        sw, sh = max(1, width // self._scale), max(1, height // self._scale)
        standard_c, standard_r = np.empty((sh, sw), np.float16), np.empty((sh, sw), np.float16)
        native_input = PipelineInput(width, height, _ptr(image), None)
        output = PipelineOutput(width, height, _ptr(background), _ptr(mura), _ptr(ridge),
                                _ptr(cmean), _ptr(cmax), _ptr(rmean), _ptr(rmax), None,
                                0, 0, None, None, None, sw, sh, _ptr(standard_c), _ptr(standard_r))
        params = json.dumps({"bg_sigma_factor": 1.0, "ridge_sigma": ridge_sigma,
                             "hessian_max_factor": 2.0,
                             "ridge_mode": "vertical+horizontal"}, separators=(",", ":")).encode("ascii")
        result = self._dll.TanukiPipeline_Process(self._handle, ctypes.byref(native_input),
                                                  params, None, ctypes.byref(output))
        if result != 0:
            error = self._dll.TanukiPipeline_GetLastError(self._handle)
            raise RuntimeError(error.decode("utf-8", "replace") if error else f"native result {result}")
        return standard_c.copy(), standard_r.copy()


def expected_standard_dimensions(raw_payload: bytes, standard_scale: int) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw_payload)) as image:
        width, height = image.size
    return max(1, width // standard_scale), max(1, height // standard_scale)


def map_status(archive: ArchiveData, standard_scale: int) -> tuple[int, int, int]:
    by_base = {}
    for record in archive.records:
        by_base.setdefault(record.base_name, {})[record.kind] = record
    raw_count = missing_count = wrong_scale_count = 0
    for kinds in by_base.values():
        raw = kinds.get(RAW_JPEG)
        if raw is None:
            continue
        raw_count += 1
        column, row = kinds.get(HESSIAN_C), kinds.get(HESSIAN_R)
        if column is None or row is None:
            missing_count += 1
            continue
        expected = expected_standard_dimensions(raw.payload, standard_scale)
        if validate_hsm(column.payload) != expected or validate_hsm(row.payload) != expected:
            wrong_scale_count += 1
    return raw_count, missing_count, wrong_scale_count


def build_map_records(archive: ArchiveData, pipeline: NativePipeline,
                      ridge_sigma: float, replace_existing: bool) -> tuple[list[Record], int]:
    by_base = {}
    for record in archive.records:
        by_base.setdefault(record.base_name, {})[record.kind] = record
    additions, updated = [], 0
    for base_name in sorted(by_base):
        kinds = by_base[base_name]
        raw = kinds.get(RAW_JPEG)
        if raw is None:
            continue
        if not replace_existing and HESSIAN_C in kinds and HESSIAN_R in kinds:
            continue
        column, row = pipeline.process(raw.payload, ridge_sigma)
        if replace_existing or HESSIAN_C not in kinds:
            additions.append(Record(HESSIAN_C, raw.camera_id, raw.ticks, base_name, encode_hsm(column)))
        if replace_existing or HESSIAN_R not in kinds:
            additions.append(Record(HESSIAN_R, raw.camera_id, raw.ticks, base_name, encode_hsm(row)))
        updated += 1
    return additions, updated


def write_report(report: BackfillReport, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Hessian standard-map backfill", "", f"- Root: `{report.root}`",
             f"- Started: {report.started}", f"- Mode: {'apply' if report.apply else 'audit'}",
             f"- Standard-map scale: {report.standard_scale}x",
             f"- Replace existing maps: {report.replace_existing}",
             "- Reconstruction: raw JPEG + capture-time CSV RidgeSigma + single-frame background",
             "- Limitation: historical CSV has no background algorithm/version; old standard-background captures cannot be reproduced exactly.",
             "", "| Result | Count |", "|---|---:|",
             f"| Archives scanned | {report.archives_scanned} |",
             f"| Archives updated | {report.archives_updated} |",
             f"| Already complete | {report.archives_complete} |",
             f"| No raw source | {report.archives_without_raw} |",
             f"| Failed archives | {report.archives_failed} |",
             f"| Frames scanned | {report.frames_scanned} |",
             f"| Frames missing maps | {report.frames_missing} |",
             f"| Frames with wrong map scale | {report.frames_wrong_scale} |",
             f"| Frames updated | {report.frames_updated} |",
             f"| Maps added | {report.maps_added} |",
             f"| Map payload bytes written | {report.bytes_added} |",
             f"| Archive byte delta | {report.bytes_delta} |"]
    if report.failures:
        lines += ["", "## Failures", ""] + [f"- {item}" for item in report.failures]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_progress(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError:
        # A long-running backfill may outlive the terminal command that launched it.
        # Losing the progress pipe must not turn a committed archive into a failure.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\Anilox\Captures"))
    parser.add_argument("--dll", type=Path, default=Path(r"bin\x64\Release\tanuki_pipeline_api.dll"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--replace-existing", action="store_true",
                        help="Rebuild every raw frame map and remove old HSM records")
    parser.add_argument("--limit", type=int, default=0, help="Maximum archives to update; zero means all")
    parser.add_argument("--standard-scale", type=int, default=25,
                        help="Must match InspectionEngineConfig.DefaultHessianStandardMapScale")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = args.report_dir or Path("artifacts/hessian-standard-map-backfill") / stamp
    if args.replace_existing and not args.apply:
        parser.error("--replace-existing requires --apply")
    report = BackfillReport(str(root), time.strftime("%Y-%m-%d %H:%M:%S"), args.apply,
                            args.standard_scale, args.replace_existing)
    ridge_by_grab = load_ridge_sigma_by_grab(root)
    pipeline = NativePipeline(args.dll.resolve(), args.standard_scale) if args.apply else None
    attempted = 0
    try:
        archives = sorted(root.rglob("*.acap"))
        for index, archive_path in enumerate(archives, 1):
            report.archives_scanned += 1
            try:
                archive = read_archive(archive_path)
                raw_count, missing_count, wrong_scale_count = map_status(archive, args.standard_scale)
                report.frames_scanned += raw_count
                report.frames_missing += missing_count
                report.frames_wrong_scale += wrong_scale_count
                if not raw_count:
                    report.archives_without_raw += 1
                    continue
                if missing_count == 0 and wrong_scale_count == 0:
                    report.archives_complete += 1
                    continue
                if not args.apply:
                    continue
                if wrong_scale_count and not args.replace_existing:
                    raise ValueError("existing HSM scale differs; rerun with --replace-existing")
                if args.limit and attempted >= args.limit:
                    break
                attempted += 1
                additions, updated = build_map_records(
                    archive, pipeline, ridge_by_grab.get(archive.grab_id, DEFAULT_RIDGE_SIGMA),
                    args.replace_existing)
                temp_path = archive_path.with_name(archive_path.name + f".hsm-part-{os.getpid()}")
                if temp_path.exists():
                    temp_path.unlink()
                original_size = archive_path.stat().st_size
                if args.replace_existing:
                    retained = [record for record in archive.records
                                if record.kind not in (HESSIAN_C, HESSIAN_R)]
                    write_archive(temp_path, archive, retained + additions)
                    report.bytes_added += sum(len(record.payload) for record in additions)
                else:
                    shutil.copyfile(archive_path, temp_path)
                    with temp_path.open("ab", buffering=1024 * 1024) as writer:
                        for addition in additions:
                            report.bytes_added += append_record(writer, addition)
                        writer.flush()
                        os.fsync(writer.fileno())
                rebuilt = read_archive(temp_path)
                latest = {(record.base_name, record.kind): record for record in rebuilt.records}
                for addition in additions:
                    dimensions = validate_hsm(latest[(addition.base_name, addition.kind)].payload)
                    raw = next(record for record in rebuilt.records
                               if record.base_name == addition.base_name and record.kind == RAW_JPEG)
                    if dimensions != expected_standard_dimensions(raw.payload, args.standard_scale):
                        raise ValueError(f"wrong rebuilt HSM dimensions: {addition.base_name}")
                final_raw, final_missing, final_wrong = map_status(rebuilt, args.standard_scale)
                if final_raw != raw_count or final_missing or final_wrong:
                    raise ValueError("rebuilt archive did not pass map completeness/scale validation")
                report.bytes_delta += temp_path.stat().st_size - original_size
                os.replace(temp_path, archive_path)
                report.archives_updated += 1
                report.frames_updated += updated
                report.maps_added += len(additions)
                if report.archives_updated % 10 == 0:
                    write_progress(
                        f"[{index}/{len(archives)}] archives={report.archives_updated} "
                        f"frames={report.frames_updated}")
                    write_report(report, report_dir)
            except Exception as exc:
                report.archives_failed += 1
                report.failures.append(f"{archive_path}: {type(exc).__name__}: {exc}")
                for partial in archive_path.parent.glob(archive_path.name + ".hsm-part-*"):
                    try:
                        partial.unlink()
                    except OSError:
                        pass
        write_report(report, report_dir)
    finally:
        if pipeline:
            pipeline.close()
    write_progress(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    write_progress(f"Report: {report_dir.resolve()}")
    return 1 if report.archives_failed else 0


if __name__ == "__main__":
    sys.exit(main())
