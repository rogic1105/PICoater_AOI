"""picoater_pipeline_reference.py — PICoater 檢測 pipeline 的 Python 參考實作。

native（src/native/modules/GetPICoaterBackground）的前身原型，示範完整 pipeline：
去背（column mean +127）→ Hessian ridge V/H → fixed 正規化 → 曲線 + 顯示圖。

演算法積木全部來自 src/（唯一來源），本檔只負責「把積木串成 pipeline」。
**同 native 的關鍵**：bin 曲線從 float ridge response 算（保峰值，u8 之前）；
顯示圖才用 ridge_to_uint8 clip。讀 repo 外 05_QA_Validation 影像。
"""
import cv2
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.image_processing import remove_column_background, compute_hessian_ridge, ridge_to_uint8
from src.data_processing import plot_and_save_statistics, save_array_to_csv

# --- 參數 ---
IMAGE_FOLDER = "../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
IMAGE_NAME = 'cal_25-11-17_11-19-25-929.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "../artifacts/algtest/pipeline_reference"
OPS = 40.0  # um/pixel
SIGMA = 9.0
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # 1. 載入影像
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print("[Error] Failed to decode image.")
        return
    src_img = np.flip(src_img, axis=0)
    h, w = src_img.shape
    print(f"[Info] Image loaded. Width={w}, Height={h}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_input.png"), src_img)

    # 2. 去背（column mean +127）
    bg_removed = remove_column_background(src_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_bg_removed.png"), bg_removed)

    # 3. Hessian ridge V/H —— 回 float（u8 之前），曲線與顯示都從這個 float 衍生
    t0 = time.time()
    res_v = compute_hessian_ridge(bg_removed, sigma=SIGMA, mode='vertical')
    res_h = compute_hessian_ridge(bg_removed, sigma=SIGMA, mode='horizontal')
    print(f"  [Timing] Hessian ridge V/H: {time.time() - t0:.3f}s")

    # 4. 顯示圖：float → clip u8（同 native scale_clamp）
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_res_v.png"), ridge_to_uint8(res_v))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_res_h.png"), ridge_to_uint8(res_h))

    # 5. 曲線（bin）：直接從 float ridge response 算 —— 同 native，不經 u8 clamp，保峰值
    col_mean = np.mean(res_v, axis=0)   # 切向（沿 X）
    col_max  = np.max(res_v, axis=0)
    col_min  = np.min(res_v, axis=0)
    row_mean = np.mean(res_h, axis=1)   # 法向（沿 Y）
    row_max  = np.max(res_h, axis=1)
    row_min  = np.min(res_h, axis=1)

    plot_and_save_statistics(col_mean, col_max, col_min, OPS,
                             os.path.join(OUTPUT_DIR, "step4_curve_v.jpg"))
    plot_and_save_statistics(row_mean, row_max, row_min, OPS,
                             os.path.join(OUTPUT_DIR, "step4_curve_h.jpg"))

    save_array_to_csv(col_mean, os.path.join(OUTPUT_DIR, "col_mean.csv"))
    save_array_to_csv(col_max,  os.path.join(OUTPUT_DIR, "col_max.csv"))
    save_array_to_csv(row_mean, os.path.join(OUTPUT_DIR, "row_mean.csv"))
    save_array_to_csv(row_max,  os.path.join(OUTPUT_DIR, "row_max.csv"))
    print(f"[Done] artifacts → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
