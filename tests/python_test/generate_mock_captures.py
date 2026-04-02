"""
generate_mock_captures.py
=========================
將 mock BMP 影像轉換為 PICoater AOI 的存檔格式：
  - 每個 CAM 資料夾內的 BMP 逐張獨立處理（不拼接，AniloxRollForm 可自行合成）
  - JPG（縮小 1/5，quality=90）：_raw.jpg, _proc_v.jpg, _proc_h.jpg
  - .bin 曲線檔（MCBF 格式）：_mean_v, _max_v, _mean_h, _max_h
  - CSV 檢測紀錄（每日一個 .csv）

Pipeline（與 C# AniloxCamera.cs 一致）：
  1. 讀取每張 BMP，垂直翻轉
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
    """
    從 BMP 檔名解析 datetime 與 cam_id。
    支援兩種格式：
      - 扁平: yyyyMMdd_HHmmss.fff-{camId}.bmp  → 回傳 (dt, ts_str, cam_id)
      - 純時間戳: yyyyMMdd_HHmmss.fff.bmp       → 回傳 (dt, ts_str, None)
    """
    base = os.path.splitext(filename)[0]  # e.g. "20251117_111952.447-1" or "20251117_111952.447"
    cam_id = None
    ts_part = base
    # 檢查是否有 -{camId} 後綴
    parts = base.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        ts_part = parts[0]
        cam_id = int(parts[1])
    dt = datetime.datetime.strptime(ts_part, "%Y%m%d_%H%M%S.%f")
    return dt, ts_part, cam_id


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
            first_dt, first_ts, _ = parse_timestamp(fn)

    stitched = np.vstack(images)
    return stitched, first_dt, first_ts


def process_cam(bmp_paths, cam_id, output_root, shared_grab_id=None):
    """
    處理單台相機的所有 BMP（逐張獨立處理）→ 每張各自產出 JPG + .bin + CSV record。
    shared_grab_id：若指定，所有影像共用此 GrabId（同一序號多相機多張拼接用）。
    """
    records = []
    for bmp_path in bmp_paths:
        src_img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        if src_img is None:
            raise RuntimeError(f"Failed to read: {bmp_path}")
        src_img = np.flip(src_img, axis=0)

        fn = os.path.basename(bmp_path)
        dt, ts_str, _ = parse_timestamp(fn)
        grab_id = shared_grab_id if shared_grab_id else dt.strftime("%y%m%d-%H%M%S")
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
              f"{fn} -> {base_name}  size={w}x{h}")

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

        records.append({
            "dt": dt,
            "grab_id": grab_id,
            "file_name": base_name,
            "max_exceed": max_exceed,
            "mean_exceed": mean_exceed,
            "mean_peak": mean_peak,
            "max_peak": max_peak,
            "grab_height": h,
            "image_width": w,
        })

    return records


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

    # 偵測 input 格式：扁平（{ts}-{camId}.bmp）或 CAM 子目錄
    cam_dirs = sorted(
        [d for d in os.listdir(input_dir)
         if d.upper().startswith("CAM") and os.path.isdir(os.path.join(input_dir, d))],
        key=lambda x: int(x.upper().replace("CAM", "")),
    )

    flat_bmps = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".bmp")]
    )

    cam_groups = []
    is_cam_mode = False
    earliest_dt = None

    if flat_bmps:
        # 扁平模式：從檔名解析 cam_id，按 cam_id 分組
        by_cam = {}
        for f in flat_bmps:
            _, _, cam_id = parse_timestamp(f)
            if cam_id is None:
                print(f"  Warning: cannot parse cam_id from {f}, skipping")
                continue
            by_cam.setdefault(cam_id, []).append(f)
        for cam_id in sorted(by_cam.keys()):
            cam_groups.append({
                "cam_id": cam_id,
                "bmp_paths": [os.path.join(input_dir, f) for f in by_cam[cam_id]],
                "n_files": len(by_cam[cam_id]),
            })
    else:
        # CAM 子目錄模式：所有相機視為同一序號，共用 GrabId
        earliest_dt = None
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
                # 用第一台相機（CAM1）的最早時間戳作為共用 GrabId
                if earliest_dt is None:
                    earliest_dt, _, _ = parse_timestamp(bmp_files[0])
        if earliest_dt:
            is_cam_mode = True

    total_bmps = sum(g["n_files"] for g in cam_groups)
    print(f"Found {len(cam_groups)} cameras, {total_bmps} BMP files total")
    shared_grab_id = None
    if is_cam_mode and earliest_dt:
        shared_grab_id = earliest_dt.strftime("%y%m%d-%H%M%S")
        print(f"CAM subdirectory mode: shared GrabId = {shared_grab_id}")
    for g in cam_groups:
        print(f"  CAM{g['cam_id']}: {g['n_files']} BMPs (each processed individually)")
    print()

    # 處理每台相機（每張 BMP 獨立產出）
    records = []
    for group in cam_groups:
        cam_records = process_cam(
            group["bmp_paths"],
            group["cam_id"],
            output_dir,
            shared_grab_id=shared_grab_id,
        )
        records.extend(cam_records)

    print()
    write_csv(records, output_dir)

    print()
    total_pass = sum(1 for r in records if r["max_exceed"] == 0 and r["mean_exceed"] == 0)
    total_fail = len(records) - total_pass
    print(f"Done! {len(records)} grabs processed ({total_bmps} BMPs)")
    print(f"  Pass: {total_pass}  Fail: {total_fail}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
