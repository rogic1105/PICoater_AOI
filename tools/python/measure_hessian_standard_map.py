#!/usr/bin/env python3
"""Measure pre-normalization Hessian maps on a real archived raw frame.

This experiment reruns the native pipeline on one RawJpeg record, writes several
display normalizations, and reports standard-map compression versus legacy JPEG.
It does not modify capture data.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ArchiveFrame:
    archive: Path
    base_name: str
    raw_jpeg: bytes
    proc_c_jpeg: bytes | None
    proc_r_jpeg: bytes | None


class PipelineInput(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
    ]


class PipelineOutput(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("background_data", ctypes.c_void_p),
        ("mura_data", ctypes.c_void_p),
        ("ridge_data", ctypes.c_void_p),
        ("mura_curve_mean", ctypes.c_void_p),
        ("mura_curve_max", ctypes.c_void_p),
        ("mura_row_curve_mean", ctypes.c_void_p),
        ("mura_row_curve_max", ctypes.c_void_p),
        ("stream", ctypes.c_void_p),
        ("resize_width", ctypes.c_int),
        ("resize_height", ctypes.c_int),
        ("resized_raw", ctypes.c_void_p),
        ("resized_ridge", ctypes.c_void_p),
        ("resized_mura", ctypes.c_void_p),
        ("standard_width", ctypes.c_int),
        ("standard_height", ctypes.c_int),
        ("resized_hessian_column_half", ctypes.c_void_p),
        ("resized_hessian_row_half", ctypes.c_void_p),
    ]


def ptr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def read_frame(archive: Path) -> ArchiveFrame | None:
    data = archive.read_bytes()
    if len(data) < 24 or data[:8] != b"PICACAP\0":
        return None
    grab_len = struct.unpack_from("<i", data, 20)[0]
    pos = 24 + grab_len
    assets: dict[str, dict[int, bytes]] = {}
    while pos + 36 <= len(data) and data[pos : pos + 4] == b"AREC":
        version, = struct.unpack_from("<i", data, pos + 4)
        if version != 1:
            break
        kind = data[pos + 8]
        name_len, payload_len = struct.unpack_from("<ii", data, pos + 24)
        payload_start = pos + 36 + name_len
        payload_end = payload_start + payload_len
        if name_len <= 0 or payload_len <= 0 or payload_end > len(data):
            break
        name = data[pos + 36 : payload_start].decode("utf-8")
        assets.setdefault(name, {})[kind] = data[payload_start:payload_end]
        pos = payload_end
    for name, records in assets.items():
        if 1 in records:
            return ArchiveFrame(archive, name, records[1], records.get(2), records.get(3))
    return None


def newest_frame(root: Path) -> ArchiveFrame:
    archives = sorted(root.rglob("*.acap"), key=lambda p: p.stat().st_mtime, reverse=True)
    for archive in archives:
        frame = read_frame(archive)
        if frame is not None:
            return frame
    raise RuntimeError(f"No RawJpeg record found below {root}")


def run_pipeline(
    dll_path: Path, image: np.ndarray, standard_scale: int, iterations: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    dll_dir = dll_path.parent
    dependency_dirs = [
        dll_dir,
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
        Path(r"C:\vcpkg\installed\x64-windows\bin"),
    ]
    handles = []
    if hasattr(os, "add_dll_directory"):
        handles = [os.add_dll_directory(str(path)) for path in dependency_dirs if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join(str(path) for path in dependency_dirs) + os.pathsep + os.environ["PATH"]
    dll = ctypes.CDLL(str(dll_path))
    dll.TanukiPipeline_Create.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    dll.TanukiPipeline_Create.restype = ctypes.c_void_p
    dll.TanukiPipeline_Process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PipelineInput),
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(PipelineOutput),
    ]
    dll.TanukiPipeline_Process.restype = ctypes.c_int
    dll.TanukiPipeline_GetLastError.argtypes = [ctypes.c_void_p]
    dll.TanukiPipeline_GetLastError.restype = ctypes.c_char_p
    dll.TanukiPipeline_Destroy.argtypes = [ctypes.c_void_p]

    height, width = image.shape
    background = np.empty_like(image)
    mura = np.empty_like(image)
    ridge = np.empty_like(image)
    cmean = np.empty(width, dtype=np.float32)
    cmax = np.empty(width, dtype=np.float32)
    rmean = np.empty(height, dtype=np.float32)
    rmax = np.empty(height, dtype=np.float32)
    resized_raw = np.empty_like(image)
    resized_ridge = np.empty_like(image)
    resized_mura = np.empty_like(image)
    standard_width = max(1, width // standard_scale)
    standard_height = max(1, height // standard_scale)
    standard_c = np.empty((standard_height, standard_width), dtype=np.float16)
    standard_r = np.empty((standard_height, standard_width), dtype=np.float16)

    native_input = PipelineInput(width, height, ptr(image), None)
    native_output = PipelineOutput(
        width, height, ptr(background), ptr(mura), ptr(ridge),
        ptr(cmean), ptr(cmax), ptr(rmean), ptr(rmax), None,
        width, height, ptr(resized_raw), ptr(resized_ridge), ptr(resized_mura),
        standard_width, standard_height,
        ptr(standard_c), ptr(standard_r),
    )
    handle = dll.TanukiPipeline_Create(b"find_stream_ridgeline", None)
    if not handle:
        raise RuntimeError("TanukiPipeline_Create failed")
    params = b'{"bg_sigma_factor":1,"ridge_sigma":1,"hessian_max_factor":0.3,"ridge_mode":"vertical+horizontal"}'
    def process_once() -> float:
        started = time.perf_counter()
        result = dll.TanukiPipeline_Process(
            handle, ctypes.byref(native_input), params, None, ctypes.byref(native_output))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result != 0:
            error = dll.TanukiPipeline_GetLastError(handle)
            raise RuntimeError(error.decode("utf-8", "replace") if error else f"native result {result}")
        return elapsed_ms

    try:
        cold_ms = process_once()
        steady_times = [process_once() for _ in range(max(1, iterations))]
        steady_ms = float(np.median(np.asarray(steady_times, dtype=np.float64)))
    finally:
        dll.TanukiPipeline_Destroy(handle)
    return standard_c.copy(), standard_r.copy(), cold_ms, steady_ms


def compressed_hsm_size(values: np.ndarray) -> int:
    raw = values.view(np.uint8).reshape(-1, 2)
    shuffled = np.concatenate((raw[:, 1], raw[:, 0])).tobytes()
    return 24 + len(zlib.compress(shuffled, level=9))


def render(values: np.ndarray, maximum: float, heatmap: bool) -> Image.Image:
    gray = np.clip(values.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)
    if not heatmap:
        return Image.fromarray(gray)
    t = gray.astype(np.float32) / 255.0
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    first = t <= 0.5
    u = np.clip(t * 2.0, 0, 1)
    rgb[..., 2] = np.where(first, u * 255, (1 - (t - 0.5) * 2) * 255).astype(np.uint8)
    rgb[..., 1] = np.where(first, u * 255, 255).astype(np.uint8)
    rgb[..., 0] = np.where(first, 0, (t - 0.5) * 2 * 255).astype(np.uint8)
    rgb[gray == 0] = 0
    return Image.fromarray(rgb)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\Anilox\Captures"))
    parser.add_argument("--dll", type=Path, default=Path(r"bin\x64\Release\tanuki_pipeline_api.dll"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/hessian-standard-map"))
    parser.add_argument(
        "--standard-scale", type=int, default=5,
        help="Additional scale applied to the archived /5 raw JPEG; 5 models /25 from camera input.")
    parser.add_argument(
        "--project-frame-count", type=int, default=0,
        help="Optional frame count used to project the additional HSM storage.")
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Steady-state native iterations after the first cold process.")
    args = parser.parse_args()

    frame = newest_frame(args.root)
    image = np.asarray(Image.open(io.BytesIO(frame.raw_jpeg)).convert("L"), dtype=np.uint8)
    standard_c, standard_r, cold_ms, steady_ms = run_pipeline(
        args.dll.resolve(), np.ascontiguousarray(image),
        max(1, args.standard_scale), max(1, args.iterations))
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-scale{max(1, args.standard_scale)}"
    out_dir = args.out / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(out_dir / "raw.png")
    for axis, values in (("column", standard_c), ("row", standard_r)):
        for maximum in (0.15, 0.3, 0.6, 1.2):
            label = str(maximum).replace(".", "p")
            render(values, maximum, False).save(out_dir / f"{axis}-max-{label}.png")
            render(values, maximum, True).save(out_dir / f"{axis}-max-{label}-heat.png")

    raw_half_bytes = standard_c.nbytes + standard_r.nbytes
    compressed_c = compressed_hsm_size(standard_c)
    compressed_r = compressed_hsm_size(standard_r)
    compressed_total = compressed_c + compressed_r
    legacy_bytes = len(frame.proc_c_jpeg or b"") + len(frame.proc_r_jpeg or b"")
    projected = ""
    if args.project_frame_count > 0:
        projected_bytes = compressed_total * args.project_frame_count
        projected = (
            f"\n- Projected HSM storage for {args.project_frame_count:,} frames: "
            f"{projected_bytes / 1073741824:.2f} GiB\n")
    report = out_dir / "report.md"
    report.write_text(
        "# Hessian standard-map measurement\n\n"
        f"- Source: `{frame.archive}` / `{frame.base_name}`\n"
        f"- Saved-frame dimensions: {image.shape[1]} x {image.shape[0]}\n"
        f"- Standard-map dimensions: {standard_c.shape[1]} x {standard_c.shape[0]} "
        f"(camera-input scale ~= /{5 * max(1, args.standard_scale)})\n"
        f"- Native processing cold: {cold_ms:.1f} ms\n"
        f"- Native processing steady median: {steady_ms:.1f} ms ({max(1, args.iterations)} iterations)\n"
        f"- Standard C range: min={float(standard_c.min()):.6f}, "
        f"p99={float(np.percentile(standard_c.astype(np.float32), 99)):.6f}, max={float(standard_c.max()):.6f}\n"
        f"- Standard R range: min={float(standard_r.min()):.6f}, "
        f"p99={float(np.percentile(standard_r.astype(np.float32), 99)):.6f}, max={float(standard_r.max()):.6f}\n\n"
        "| Payload | Bytes | MiB |\n|---|---:|---:|\n"
        f"| Raw half C+R | {raw_half_bytes} | {raw_half_bytes / 1048576:.2f} |\n"
        f"| Compressed HSM C | {compressed_c} | {compressed_c / 1048576:.2f} |\n"
        f"| Compressed HSM R | {compressed_r} | {compressed_r / 1048576:.2f} |\n"
        f"| Compressed HSM C+R | {compressed_total} | {compressed_total / 1048576:.2f} |\n"
        f"| Existing processed JPEG C+R | {legacy_bytes} | {legacy_bytes / 1048576:.2f} |\n\n"
        f"Compression ratio vs raw half: {compressed_total / raw_half_bytes:.2%}\n"
        f"{projected}",
        encoding="utf-8",
    )
    print(report)
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
