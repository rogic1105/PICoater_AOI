"""
generate_mock_captures.py
=========================
將 mock BMP 影像轉換為 PICoater AOI 的存檔格式：
  - 每個 CAM 資料夾內的 BMP 垂直拼接為一張影像（一個序號）
  - JPG（縮小 1/5，quality=90）：_raw.jpg, _proc_v.jpg, _proc_h.jpg
  - .bin 曲線檔（MCBF 格式）：_mean_v, _max_v, _mean_h, _max_h
  - CSV 檢測紀錄（每日一個 .csv）

Pipeline（與 C# AniloxCamera.cs 一致）：
  1. 讀取同一 CAM 資料夾所有 BMP，垂直拼接成一張
  2. remove_column_background（全解析度）
  3. compute_hessian_ridge V/H（全解析度，sigma=9.0）
  4. INTER_AREA resize 1/5（raw / proc_v / proc_h）→ 存 JPG
  5. 從全解析度 hessian 計算曲線 → 存 MCBF .bin

Usage:
    cd tests/python_test
    python generate_mock_captures.py <input_dir> <output_dir>

Example:
    python generate_mock_captures.py "C:/Users/User/Downloads/mura" "../../artifacts/mock_captures"
"""

import os
import sys
import struct
import datetime
import cv2
import numpy as np

# 將 src/ 加入搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.image_processing import (
    remove_column_background,
    compute_hessian_ridge,
    img_reduction_resize_average_filter,
)

# ── 設定 ──
RESIZE_SCALE = 5       # 縮圖倍率（1/5）
JPG_QUALITY = 90       # JPEG 壓縮品質
HESSIAN_SIGMA = 9.0    # Hessian ridge sigma（與 picoater_compress.py 一致）
HESSIAN_MAX_FACTOR = 1.0
ERROR_VALUE_MEAN = 0.5
ERROR_VALUE_MAX = 0.8
OPS_UM = 25.0          # 解析度 (um/pixel)
LINE_RATE_HZ = 50000.0
EXPOSURE_US = 50.0


def parse_timestamp(filename):
    """從 BMP 檔名解析 datetime，格式: yyyyMMdd_HHmmss.fff.bmp"""
    base = os.path.splitext(filename)[0]
    dt = datetime.datetime.strptime(base, "%Y%m%d_%H%M%S.%f")
    return dt, base


def save_curve_bin(arr, scale, path):
    """
    寫入 MCBF .bin 格式：
    Header: magic(4)"MCBF" + version(4)=1 + scale_factor(4f) + array_length(4) + float[]
    """
    with open(path, "wb") as f:
        f.write(b"MCBF")
        f.write(struct.pack("<i", 1))               # version
        f.write(struct.pack("<f", float(scale)))     # scale_factor
        f.write(struct.pack("<i", len(arr)))          # array_length
        f.write(struct.pack(f"<{len(arr)}f", *arr))  # float array


def stitch_bmp_files(bmp_paths):
    """
    讀取多張 BMP，垂直翻轉後垂直拼接成一張影像。
    回傳：(stitched_image, total_height, width, first_datetime, first_ts_str)
    """
    images = []
    first_dt = None
    first_ts = None
    for p in bmp_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read: {p}")
        img = np.flip(img, axis=0)
        images.append(img)
        if first_dt is None:
            fn = os.path.basename(p)
            first_dt, first_ts = parse_timestamp(fn)

    stitched = np.vstack(images)
    return stitched, first_dt, first_ts


def process_cam(bmp_paths, cam_id, grab_id, output_root):
    """
    處理單台相機的所有 BMP（垂直拼接）→ JPG + .bin + CSV record data
    """
    n_files = len(bmp_paths)
    src_img, dt, ts_str = stitch_bmp_files(bmp_paths)
    h, w = src_img.shape

    # 目錄結構: {root}/{yyyy}/{yyyyMM}/{yyyyMMdd}/
    day_dir = os.path.join(
        output_root,
        dt.strftime("%Y"),
        dt.strftime("%Y%m"),
        dt.strftime("%Y%m%d"),
    )
    os.makedirs(day_dir, exist_ok=True)

    base_name = f"{ts_str}-{cam_id}"
    print(f"  [{dt.strftime('%H:%M:%S')}] {grab_id} CAM{cam_id}  "
          f"{n_files} BMPs stitched -> {base_name}  size={w}x{h}")

    # ── 1. 全解析度背景去除 ──
    bg_removed = remove_column_background(src_img)

    # ── 2. 全解析度 Hessian ridge detection ──
    res_v = compute_hessian_ridge(bg_removed, sigma=HESSIAN_SIGMA, mode='vertical')
    res_h = compute_hessian_ridge(bg_removed, sigma=HESSIAN_SIGMA, mode='horizontal')

    # ── 3. INTER_AREA resize 1/5 ──
    raw_resized = img_reduction_resize_average_filter(src_img, RESIZE_SCALE)
    proc_v_resized = img_reduction_resize_average_filter(res_v, RESIZE_SCALE)
    proc_h_resized = img_reduction_resize_average_filter(res_h, RESIZE_SCALE)

    rh, rw = raw_resized.shape[:2]

    # ── 4. 存 JPG ──
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY]
    cv2.imwrite(os.path.join(day_dir, base_name + "_raw.jpg"), raw_resized, encode_params)
    cv2.imwrite(os.path.join(day_dir, base_name + "_proc_v.jpg"), proc_v_resized, encode_params)
    cv2.imwrite(os.path.join(day_dir, base_name + "_proc_h.jpg"), proc_h_resized, encode_params)

    # ── 5. 從全解析度 hessian 計算曲線（與 C# AniloxCamera.cs 一致）──
    col_mean_v = np.mean(res_v, axis=0).astype(np.float32)  # (W,) 全解析度
    col_max_v = np.max(res_v, axis=0).astype(np.float32)    # (W,) 全解析度
    row_mean_h = np.mean(res_h, axis=1).astype(np.float32)  # (H,) 全解析度
    row_max_h = np.max(res_h, axis=1).astype(np.float32)    # (H,) 全解析度

    # ── 6. 存 .bin 曲線檔（scale_factor = RESIZE_SCALE，與 C# _saveResizeScale 一致）──
    save_curve_bin(col_mean_v.tolist(), RESIZE_SCALE, os.path.join(day_dir, base_name + "_mean_v.bin"))
    save_curve_bin(col_max_v.tolist(), RESIZE_SCALE, os.path.join(day_dir, base_name + "_max_v.bin"))
    save_curve_bin(row_mean_h.tolist(), RESIZE_SCALE, os.path.join(day_dir, base_name + "_mean_h.bin"))
    save_curve_bin(row_max_h.tolist(), RESIZE_SCALE, os.path.join(day_dir, base_name + "_max_h.bin"))

    # CSV 用的 peak 值（從全解析度 V 曲線取最大值 / 255，與 C# _lastMeanPeak 計算一致）
    mean_peak = float(col_mean_v.max()) / 255.0
    max_peak = float(col_max_v.max()) / 255.0

    print(f"           jpg={rw}x{rh}  meanPeak={mean_peak:.4f}  maxPeak={max_peak:.4f}")

    max_exceed = 1 if max_peak > ERROR_VALUE_MAX else 0
    mean_exceed = 1 if mean_peak > ERROR_VALUE_MEAN else 0

    return {
        "dt": dt,
        "grab_id": grab_id,
        "file_name": base_name,
        "max_exceed": max_exceed,
        "mean_exceed": mean_exceed,
        "mean_peak": mean_peak,
        "max_peak": max_peak,
        "grab_height": h,
        "image_width": w,
    }


def write_csv(records, output_root):
    """
    寫入每日 CSV：{root}/{yyyy}/{yyyyMM}/{yyyyMMdd}.csv
    含 #CFG 行 + header + data rows
    """
    by_day = {}
    for r in records:
        day_key = r["dt"].strftime("%Y%m%d")
        by_day.setdefault(day_key, []).append(r)

    header = "Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,GrabHeight,LineRateHz,ExposureUs"

    for day_key, day_records in sorted(by_day.items()):
        dt = day_records[0]["dt"]
        csv_dir = os.path.join(output_root, dt.strftime("%Y"), dt.strftime("%Y%m"))
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, day_key + ".csv")

        with open(csv_path, "w", encoding="utf-8") as f:
            # #CFG 行
            cfg_ts = dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
            cfg_parts = [f"#CFG,{cfg_ts}"]
            # Cam_Pos = 等間隔，間距 = ops_um * image_width (um → mm)
            cam_width_mm = OPS_UM * day_records[0]["image_width"] / 1000.0
            for c in range(1, 8):
                cfg_parts.append(f"Cam{c}_Ops={OPS_UM:.2f}")
            for c in range(1, 8):
                pos_mm = (c - 1) * cam_width_mm
                cfg_parts.append(f"Cam{c}_Pos={pos_mm:.2f}")
            cfg_parts.append(f"HessianMaxFactor={HESSIAN_MAX_FACTOR:.4f}")
            cfg_parts.append(f"ErrorValueMean={ERROR_VALUE_MEAN:.4f}")
            cfg_parts.append(f"ErrorValueMax={ERROR_VALUE_MAX:.4f}")
            f.write(",".join(cfg_parts) + "\n")

            f.write(header + "\n")

            for r in sorted(day_records, key=lambda x: x["dt"]):
                f.write(
                    f'{r["grab_id"]},{r["file_name"]},'
                    f'{r["max_exceed"]},{r["mean_exceed"]},'
                    f'{r["mean_peak"]:.4f},{r["max_peak"]:.4f},'
                    f'{r["grab_height"]},{LINE_RATE_HZ:.1f},{EXPOSURE_US:.1f}\n'
                )

        print(f"  CSV: {csv_path}  ({len(day_records)} records)")


def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_mock_captures.py <input_dir> <output_dir>")
        print('Example: python generate_mock_captures.py "C:/Users/User/Downloads/mura" "../../artifacts/mock_captures"')
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"OPS:    {OPS_UM} um/pixel")
    print()

    # 掃描所有相機資料夾，每個資料夾 = 一個序號（BMP 垂直拼接）
    cam_dirs = sorted(
        [d for d in os.listdir(input_dir) if d.upper().startswith("CAM")],
        key=lambda x: int(x.upper().replace("CAM", "")),
    )

    # 收集每台相機的 BMP 列表
    cam_groups = []
    for cam_dir in cam_dirs:
        cam_id = int(cam_dir.upper().replace("CAM", ""))
        cam_path = os.path.join(input_dir, cam_dir)
        bmp_files = sorted(
            [f for f in os.listdir(cam_path) if f.lower().endswith(".bmp")]
        )
        if bmp_files:
            cam_groups.append({
                "cam_id": cam_id,
                "bmp_paths": [os.path.join(cam_path, f) for f in bmp_files],
                "n_files": len(bmp_files),
            })

    total_bmps = sum(g["n_files"] for g in cam_groups)
    print(f"Found {len(cam_groups)} cameras, {total_bmps} BMP files total")
    for g in cam_groups:
        print(f"  CAM{g['cam_id']}: {g['n_files']} BMPs -> 1 stitched grab")
    print()

    # 處理每台相機（一個資料夾 = 一個序號）
    records = []
    for i, group in enumerate(cam_groups):
        # GrabId = 首筆擷取時間 yyMMdd-HHmmss
        first_bmp = os.path.basename(group["bmp_paths"][0])
        first_dt, _ = parse_timestamp(first_bmp)
        grab_id = first_dt.strftime("%y%m%d-%H%M%S")
        record = process_cam(
            group["bmp_paths"],
            group["cam_id"],
            grab_id,
            output_dir,
        )
        records.append(record)

    print()
    write_csv(records, output_dir)

    print()
    total_pass = sum(1 for r in records if r["max_exceed"] == 0 and r["mean_exceed"] == 0)
    total_fail = len(records) - total_pass
    print(f"Done! {len(records)} grabs processed ({total_bmps} BMPs stitched)")
    print(f"  Pass: {total_pass}  Fail: {total_fail}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
