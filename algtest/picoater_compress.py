"""picoater_compress.py — 壓縮/縮放 在 pipeline 前後 對 mura 曲線的影響研究。

比較 5 種處理順序，找「JPEG 壓縮 / 縮放 到什麼程度，mura 曲線還辨識得出」：
  1 Original             原圖 → 檢測（baseline，曲線從 float，同 native）
  2 Compressed          先 JPEG 壓縮 → 檢測
  3 Resized             先縮放 → 檢測
  4 Original-resized    先檢測 → 再縮放結果（float resize，曲線仍 float）
  5 Original-compressed 先檢測 → 再 JPEG 壓縮結果（JPEG 必經 u8 → 曲線從壓縮後 u8，這正是要看的 degrade）

演算法積木全部來自 src/（唯一來源）。

⚠️ 此研究的 JPEG 用 cv2（libjpeg），與產品 C# 的 GDI+ 是不同 encoder（量化表不同）。
   結論套到產品前，需用 GDI+ 對齊重驗（見對話記錄）。
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.image_processing import (
    remove_column_background, compute_hessian_ridge, ridge_to_uint8, overlay_heatmap,
    img_reduction_compress, img_reduction_resize, img_reduction_resize_average_filter,
)
from src.data_processing import plot_and_save_statistics, save_array_to_csv, plot_comparison

# --- 參數 ---
IMAGE_PATH = "VH.bmp"
OUTPUT_DIR = "../artifacts/algtest/compress"
OPS = 40.0
JPG_QUALITY = 1
RESIZE_MAG = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect(img, sigma):
    """去背 → Hessian ridge V/H（回 float，曲線從 float 算 = 同 native u8 之前）。"""
    bg = remove_column_background(img)
    res_v = compute_hessian_ridge(bg, sigma=sigma, mode='vertical')
    res_h = compute_hessian_ridge(bg, sigma=sigma, mode='horizontal')
    return bg, res_v, res_h


def save_variant(name, input_img, bg, res_v, res_h, ops):
    """存一個變體的所有產物，回傳曲線 dict。

    曲線從傳入的 res_v 算（float 變體→保峰值；壓縮變體→從壓縮後 u8，反映 degrade）；
    顯示圖一律 ridge_to_uint8 clip。input_img 與 res_v 同尺寸時才疊 heatmap。
    """
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    col_mean = np.mean(res_v, axis=0)
    col_max = np.max(res_v, axis=0)
    col_min = np.min(res_v, axis=0)
    plot_and_save_statistics(col_mean, col_max, col_min, ops, os.path.join(out_dir, f"{name}_curve.jpg"))
    save_array_to_csv(col_mean, os.path.join(out_dir, f"{name}_mean.csv"))
    save_array_to_csv(col_max, os.path.join(out_dir, f"{name}_max.csv"))
    save_array_to_csv(col_min, os.path.join(out_dir, f"{name}_min.csv"))

    rv_u8, rh_u8 = ridge_to_uint8(res_v), ridge_to_uint8(res_h)
    cv2.imwrite(os.path.join(out_dir, f"{name}_1_input.jpg"), input_img)
    cv2.imwrite(os.path.join(out_dir, f"{name}_2_bg.jpg"), bg)
    cv2.imwrite(os.path.join(out_dir, f"{name}_3_res_v.jpg"), rv_u8)
    cv2.imwrite(os.path.join(out_dir, f"{name}_4_res_h.jpg"), rh_u8)
    if input_img.shape[:2] == rv_u8.shape[:2]:
        cv2.imwrite(os.path.join(out_dir, f"{name}_5_heatmap_v.jpg"),
                    overlay_heatmap(input_img, rv_u8, lower_limit=20, alpha=0.3))
    return {'mean': col_mean, 'max': col_max, 'min': col_min}


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print("[Error] Failed to decode image.")
        return
    src_img = np.flip(src_img, axis=0)

    cmp = {}

    # 變體 1-3：先變換輸入，再檢測（曲線從 float ridge）
    for name, img, sigma in [
        ('Original',   src_img, 9.0),
        ('Compressed', img_reduction_compress(src_img, JPG_QUALITY), 9.0),
        ('Resized',    img_reduction_resize(src_img, RESIZE_MAG), 6.0),
    ]:
        bg, res_v, res_h = detect(img, sigma)
        cmp[name] = save_variant(name, img, bg, res_v, res_h, OPS)

    # 原圖檢測一次（變體 4、5 共用）
    bg_o, res_v_o, res_h_o = detect(src_img, 9.0)

    # 變體 4：檢測後縮放結果（float resize，input 也縮放以對齊尺寸）
    cmp['Original-resized'] = save_variant(
        'Original-resized', img_reduction_resize_average_filter(src_img, RESIZE_MAG),
        img_reduction_resize_average_filter(bg_o, RESIZE_MAG),
        img_reduction_resize_average_filter(res_v_o, RESIZE_MAG),
        img_reduction_resize_average_filter(res_h_o, RESIZE_MAG), OPS)

    # 變體 5：檢測後壓縮結果（JPEG 必經 u8；曲線從壓縮後 u8，看 degrade）
    cmp['Original-compressed'] = save_variant(
        'Original-compressed', src_img, img_reduction_compress(bg_o, JPG_QUALITY),
        img_reduction_compress(ridge_to_uint8(res_v_o), JPG_QUALITY),
        img_reduction_compress(ridge_to_uint8(res_h_o), JPG_QUALITY), OPS)

    # 5 方案比較圖
    cmp_dir = os.path.join(OUTPUT_DIR, "comparison")
    os.makedirs(cmp_dir, exist_ok=True)
    for stat, ylabel in [('mean', 'Mean Intensity'), ('max', 'Max Intensity')]:
        plot_comparison(
            cmp['Original'][stat], cmp['Compressed'][stat], cmp['Resized'][stat],
            cmp['Original-resized'][stat], cmp['Original-compressed'][stat],
            OPS, RESIZE_MAG, JPG_QUALITY,
            os.path.join(cmp_dir, f"comparison_{stat}.jpg"),
            f"Column {stat.capitalize()} Comparison", ylabel)
    print(f"[Done] artifacts → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
