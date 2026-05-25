import cv2
import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks
from src.image_processing import *
from src.data_processing import *


# --- 參數設定 ---
# IMAGE_FOLDER = "../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
# IMAGE_NAME = 'cal_25-11-17_11-19-25-929.bmp'

IMAGE_FOLDER = ""
IMAGE_NAME = 'VH.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "../artifacts/algtest/compress"

# 新增常數: 解析度 (um/pixel)
OPS = 40.0 
jpg_compression_level = 1
mag_reshape = 5
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 主程式 ---
def main():

    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return
    
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print("[Error] Failed to decode image.")
        return

    src_img = np.flip(src_img, axis=0)
    h, w = src_img.shape



    img_original = src_img.copy()
    img_compressed = img_reduction_compress(src_img, jpg_compression_level)
    img_resized = img_reduction_resize(src_img, mag_reshape)

    

    comparison_data = {}
    # ---------------------------------------------------------
    # 1 原圖->檢測
    # ---------------------------------------------------------
    out_dir_orig = os.path.join(OUTPUT_DIR, 'original-imageProcessing')
    os.makedirs(out_dir_orig, exist_ok=True)

    bg_removed_orig = remove_column_background(img_original)
    res_v_orig = compute_hessian_ridge(bg_removed_orig, sigma=9.0, mode='vertical')
    res_h_orig = compute_hessian_ridge(bg_removed_orig, sigma=9.0, mode='horizontal')
    heatmap_orig_resV = overlay_heatmap(img_original, res_v_orig, lower_limit=20, alpha=0.3)
    heatmap_orig_resH = overlay_heatmap(img_original, res_h_orig, lower_limit=20, alpha=0.3)

    col_mean_orig = np.mean(res_v_orig, axis=0)
    col_max_orig = np.max(res_v_orig, axis=0)
    col_min_orig = np.min(res_v_orig, axis=0)
    
    plot_and_save_statistics(col_mean_orig, col_max_orig, col_min_orig, OPS, os.path.join(out_dir_orig, "Original_4_statistics_plot.bmp"), jpg_quality=95)
    
    save_array_to_csv(col_mean_orig, os.path.join(out_dir_orig, "Original_record_mean.csv"))
    save_array_to_csv(col_max_orig, os.path.join(out_dir_orig, "Original_record_max.csv"))
    save_array_to_csv(col_min_orig, os.path.join(out_dir_orig, "Original_record_min.csv"))
    
    cv2.imwrite(os.path.join(out_dir_orig, "Original_1_input.bmp"), img_original, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original_2_bg_removed.bmp"), bg_removed_orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original_3_res_v.bmp"), res_v_orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original_4_res_h.bmp"), res_h_orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original_5_heatmap_overlay_resV.bmp"), heatmap_orig_resV, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original_6_heatmap_overlay_resH.bmp"), heatmap_orig_resH, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    comparison_data['Original'] = {'mean': col_mean_orig, 'max': col_max_orig, 'min': col_min_orig}


    # ---------------------------------------------------------
    # 2 壓縮圖->檢測
    # ---------------------------------------------------------
    out_dir_comp = os.path.join(OUTPUT_DIR, 'compressed-imageProcessing')
    os.makedirs(out_dir_comp, exist_ok=True)
    
    bg_removed_comp = remove_column_background(img_compressed)
    res_v_comp = compute_hessian_ridge(bg_removed_comp, sigma=9.0, mode='vertical')
    res_h_comp = compute_hessian_ridge(bg_removed_comp, sigma=9.0, mode='horizontal')
    heatmap_comp_resV = overlay_heatmap(img_compressed, res_v_comp, lower_limit=20, alpha=0.3)
    heatmap_comp_resH = overlay_heatmap(img_compressed, res_h_comp, lower_limit=20, alpha=0.3)

    col_mean_comp = np.mean(res_v_comp, axis=0)
    col_max_comp = np.max(res_v_comp, axis=0)
    col_min_comp = np.min(res_v_comp, axis=0)
    
    plot_and_save_statistics(col_mean_comp, col_max_comp, col_min_comp, OPS, os.path.join(out_dir_comp, "Compressed_4_statistics_plot.jpg"), jpg_quality=95)
    
    save_array_to_csv(col_mean_comp, os.path.join(out_dir_comp, "Compressed_record_mean.csv"))
    save_array_to_csv(col_max_comp, os.path.join(out_dir_comp, "Compressed_record_max.csv"))
    save_array_to_csv(col_min_comp, os.path.join(out_dir_comp, "Compressed_record_min.csv"))
    
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_1_input.jpg"), img_compressed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_2_bg_removed.jpg"), bg_removed_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_3_res_v.jpg"), res_v_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_4_res_h.jpg"), res_h_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_5_heatmap_overlay_resV.jpg"), heatmap_comp_resV, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_comp, "Compressed_6_heatmap_overlay_resH.jpg"), heatmap_comp_resH, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    comparison_data['Compressed'] = {'mean': col_mean_comp, 'max': col_max_comp, 'min': col_min_comp}


    # ---------------------------------------------------------
    # 3 縮放->檢測
    # ---------------------------------------------------------
    out_dir_res = os.path.join(OUTPUT_DIR, f'resized-imageProcessing')
    os.makedirs(out_dir_res, exist_ok=True)
    
    
    bg_removed_res = remove_column_background(img_resized)

    res_v_res = compute_hessian_ridge(bg_removed_res, sigma=6.0, mode='vertical')
    res_h_res = compute_hessian_ridge(bg_removed_res, sigma=6.0, mode='horizontal')
    heatmap_res_resV = overlay_heatmap(img_resized, res_v_res, lower_limit=20, alpha=0.3)
    heatmap_res_resH = overlay_heatmap(img_resized, res_h_res, lower_limit=20, alpha=0.3)

    col_mean_res = np.mean(res_v_res, axis=0)
    col_max_res = np.max(res_v_res, axis=0)
    col_min_res = np.min(res_v_res, axis=0)
    
    plot_and_save_statistics(col_mean_res, col_max_res, col_min_res, OPS * mag_reshape, os.path.join(out_dir_res, "Resized_4_statistics_plot.jpg"), jpg_quality=95)
    
    save_array_to_csv(col_mean_res, os.path.join(out_dir_res, "Resized_record_mean.csv"))
    save_array_to_csv(col_max_res, os.path.join(out_dir_res, "Resized_record_max.csv"))
    save_array_to_csv(col_min_res, os.path.join(out_dir_res, "Resized_record_min.csv"))

    cv2.imwrite(os.path.join(out_dir_res, "Resized_1_input.jpg"), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_res, "Resized_2_bg_removed.jpg"), bg_removed_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_res, "Resized_3_res_v.jpg"), res_v_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_res, "Resized_4_res_h.jpg"), res_h_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_res, "Resized_5_heatmap_overlay_resV.jpg"), heatmap_res_resV, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_res, "Resized_6_heatmap_overlay_resH.jpg"), heatmap_res_resH, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    comparison_data['Resized'] = {'mean': col_mean_res, 'max': col_max_res, 'min': col_min_res}

    # ---------------------------------------------------------
    # 4 檢測->縮放
    # ---------------------------------------------------------
    
    out_dir_orig = os.path.join(OUTPUT_DIR, 'imageProcessing-resized')
    os.makedirs(out_dir_orig, exist_ok=True)

    bg_removed_orig_res = img_reduction_resize_average_filter(bg_removed_orig, mag_reshape)
    res_v_orig_res = img_reduction_resize_average_filter(res_v_orig, mag_reshape)
    res_h_orig_res = img_reduction_resize_average_filter(res_h_orig, mag_reshape)
    heatmap_orig_resV = img_reduction_resize_average_filter(heatmap_orig_resV, mag_reshape)
    heatmap_orig_resH = img_reduction_resize_average_filter(heatmap_orig_resH, mag_reshape)

    col_mean_orig_res = np.mean(res_v_orig_res, axis=0)
    col_max_orig_res = np.max(res_v_orig_res, axis=0)
    col_min_orig_res = np.min(res_v_orig_res, axis=0)
    
    plot_and_save_statistics(col_mean_orig_res, col_max_orig_res, col_min_orig_res, OPS, os.path.join(out_dir_orig, "Original-resized_4_statistics_plot.bmp"), jpg_quality=95)
    
    save_array_to_csv(col_mean_orig_res, os.path.join(out_dir_orig, "Original-resized_record_mean.csv"))
    save_array_to_csv(col_max_orig_res, os.path.join(out_dir_orig, "Original-resized_record_max.csv"))
    save_array_to_csv(col_min_orig_res, os.path.join(out_dir_orig, "Original-resized_record_min.csv"))
    
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_1_input.bmp"), img_original)
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_2_bg_removed.bmp"), bg_removed_orig_res)
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_3_res_v.bmp"), res_v_orig_res)
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_4_res_h.bmp"), res_h_orig_res)
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_5_heatmap_overlay_resV.bmp"), heatmap_orig_resV)
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_6_heatmap_overlay_resH.bmp"), heatmap_orig_resH)
    

    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_1_input.jpg"), img_original, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_2_bg_removed.jpg"), bg_removed_orig_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_3_res_v.jpg"), res_v_orig_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_4_res_h.jpg"), res_h_orig_res, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_5_heatmap_overlay_resV.jpg"), heatmap_orig_resV, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-resized_6_heatmap_overlay_resH.jpg"), heatmap_orig_resH, [cv2.IMWRITE_JPEG_QUALITY, 95])
    

    comparison_data['Original-resized'] = {'mean': col_mean_orig_res, 'max': col_max_orig_res, 'min': col_min_orig_res}

    # ---------------------------------------------------------
    # 5 檢測->壓縮圖
    # ---------------------------------------------------------

    out_dir_orig = os.path.join(OUTPUT_DIR, 'imageProcessing-compressed')
    os.makedirs(out_dir_orig, exist_ok=True)

    bg_removed_orig_comp = img_reduction_compress(bg_removed_orig, jpg_compression_level)
    res_v_orig_comp = img_reduction_compress(res_v_orig, jpg_compression_level)
    res_h_orig_comp = img_reduction_compress(res_h_orig, jpg_compression_level)
    heatmap_orig_resV_comp = img_reduction_compress(heatmap_orig_resV, jpg_compression_level)
    heatmap_orig_resH_comp = img_reduction_compress(heatmap_orig_resH, jpg_compression_level)

    col_mean_orig_comp = np.mean(res_v_orig_comp, axis=0)
    col_max_orig_comp = np.max(res_v_orig_comp, axis=0)
    col_min_orig_comp = np.min(res_v_orig_comp, axis=0)
    
    plot_and_save_statistics(col_mean_orig_comp, col_max_orig_comp, col_min_orig_comp, OPS, os.path.join(out_dir_orig, "Original-compressed_4_statistics_plot.bmp"), jpg_quality=95)
    
    save_array_to_csv(col_mean_orig_comp, os.path.join(out_dir_orig, "Original-compressed_record_mean.csv"))
    save_array_to_csv(col_max_orig_comp, os.path.join(out_dir_orig, "Original-compressed_record_max.csv"))
    save_array_to_csv(col_min_orig_comp, os.path.join(out_dir_orig, "Original-compressed_record_min.csv"))
    
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_1_input.bmp"), img_original, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_2_bg_removed.bmp"), bg_removed_orig_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_3_res_v.bmp"), res_v_orig_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_4_res_h.bmp"), res_h_orig_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_5_heatmap_overlay_resV.bmp"), heatmap_orig_resV_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(out_dir_orig, "Original-compressed_6_heatmap_overlay_resH.bmp"), heatmap_orig_resH_comp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    comparison_data['Original-compressed'] = {'mean': col_mean_orig_comp, 'max': col_max_orig_comp, 'min': col_min_orig_comp}

    # --- 生成對比圖表 ---
    
    plot_comparison_dir = os.path.join(OUTPUT_DIR, "comparison")
    os.makedirs(plot_comparison_dir, exist_ok=True)
    
    # 繪製 Column Mean 比較圖
    plot_comparison(
        comparison_data['Original']['mean'], 
        comparison_data['Compressed']['mean'], 
        comparison_data['Resized']['mean'],
        comparison_data['Original-resized']['mean'],
        comparison_data['Original-compressed']['mean'],
        OPS,
        mag_reshape,
        jpg_compression_level,
        os.path.join(plot_comparison_dir, "comparison_column_mean.bmp"),
        "Column Mean Comparison",
        "Mean Intensity"
    )
    
    # 繪製 Column Max 比較圖
    plot_comparison(
        comparison_data['Original']['max'], 
        comparison_data['Compressed']['max'], 
        comparison_data['Resized']['max'],
        comparison_data['Original-resized']['max'],
        comparison_data['Original-compressed']['max'],
        OPS,
        mag_reshape,
        jpg_compression_level,
        os.path.join(plot_comparison_dir, "comparison_column_max.bmp"),
        "Column Max Comparison",
        "Max Intensity"
    )
    



if __name__ == "__main__":
    main()