import cv2
import numpy as np
import os
import csv
import time
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks


# --- 參數設定 ---
IMAGE_FOLDER = "../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
IMAGE_NAME = 'cal_25-11-17_11-19-25-929.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "../artifacts/algtest/method_2"

# 新增常數: 解析度 (um/pixel)
OPS = 40.0 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 原有函數 ---
def remove_column_background(image: np.ndarray) -> np.ndarray:
    """Removes background by subtracting the column-wise mean."""
    print("  [Process] Removing column background...")
    img_float = image.astype(np.float32)
    col_mean = np.mean(img_float, axis=0)
    bg_2d = np.tile(col_mean, (image.shape[0], 1))
    result = img_float - bg_2d + 127

    return np.clip(result, 0, 255).astype(np.uint8)

def compute_hessian_ridge(image: np.ndarray, 
                                sigma: float = 2.0, 
                                mode: str = 'vertical',
                                fixed_max_val: float = 1.0) -> np.ndarray:
    """
    Computes Hessian-based ridge detection with FIXED scaling for production.
    
    Args:
        fixed_max_val: 定義「絕對強度」的上限。
                       Response >= fixed_max_val 的位置會變成 255 (全白)。
                       Response = 0 的位置是 0 (全黑)。
                       這讓不同圖片之間的亮度具有可比性。
    """
    if mode not in ('vertical', 'horizontal', 'both'):
        raise ValueError(f"Invalid mode: {mode}")

    print(f"  [Process] Computing Ridge (Mode={mode}, Sigma={sigma}, FixedMax={fixed_max_val})...")
    ksize = int(6 * sigma + 1) | 1
    
    smooth = cv2.GaussianBlur(image.astype(np.float32), (ksize, ksize), sigma)
    
    # smooth = cv2.GaussianBlur(image, (ksize, ksize), sigma).astype(np.float32)
    

    response = None

    # 2. Compute Derivatives
    if mode == 'vertical':
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        response = np.abs(dxx)
    elif mode == 'horizontal':
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        response = np.abs(dyy)
    elif mode == 'both':
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        response = np.abs(dxx) + np.abs(dyy)

    # 3. Fixed Scaling

    resp_scaled = response / fixed_max_val

    return resp_scaled

# --- 新增函數 ---

def plot_and_save_statistics(mean_arr: np.ndarray, max_arr: np.ndarray, 
                             ops_um: float, output_path: str):
    """
    繪製平均值與標準差圖表，橫軸單位為 mm。
    標準差使用虛線表示 (Mean + Std, Mean - Std)。
    """
    print(f"  [Process] Plotting statistics to {output_path}...")
    
    # 計算橫軸 (Pixels -> mm)
    x_indices = np.arange(len(mean_arr))
    x_mm = x_indices * ops_um / 1000.0  # um to mm

    plt.figure(figsize=(12, 6))
    
    # 繪製平均線
    plt.plot(x_mm, mean_arr, label='Column Mean', color='blue', linewidth=1.5)
    # 繪製極值
    plt.plot(x_mm, max_arr, label='Max', color='orange', linestyle='--', linewidth=1)


    plt.title("Column Intensity Statistics (Ridge Response)")
    plt.xlabel("Position (mm)")
    plt.ylabel("Intensity")
    plt.ylim(0,1)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close() # 釋放記憶體

def save_array_to_csv(data_array: np.ndarray, file_path: str):
    """
    將一維陣列寫入 CSV，第一欄為時間戳記。
    如果檔案不存在，會自動建立。
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 準備要寫入的列: [Timestamp, val1, val2, val3, ...]
    row = [timestamp] + data_array.tolist()
    
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 如果是新檔案，可以選擇寫入 header (這裡省略 header 以簡化，直接寫數據)
            writer.writerow(row)
        print(f"  [Data] Appended data to {file_path}")
    except Exception as e:
        print(f"  [Error] Failed to write CSV: {e}")

def overlay_heatmap(src_image: np.ndarray, overlay_image: np.ndarray, 
                    lower_limit: int = 0, 
                    alpha: float = 0.5) -> np.ndarray:

    if len(src_image.shape) == 2:
        src_bgr = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)
    else:
        src_bgr = src_image.copy()

    heatmap = cv2.applyColorMap(overlay_image, cv2.COLORMAP_JET)
    beta = 1.0 - alpha
    result = cv2.addWeighted(src_bgr, alpha, heatmap, beta, 0)
    mask_indices = (overlay_image <= lower_limit) 
    result[mask_indices] = src_bgr[mask_indices]
    
    return result

# --- 主程式 ---
def main():
    # --- 1. Load Image ---
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return
    
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if src_img is None:
        print("[Error] Failed to decode image.")
        return

    src_img = np.flip(src_img, axis=0)
    h, w = src_img.shape
    print(f"[Info] Image loaded. Width: {w}, Height: {h}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_input.png"), src_img)
    
    # --- 3. Filter Background ---
    time_start = time.time()
    bg_removed = remove_column_background(src_img)
    time_end = time.time()
    print(f"  [Timing] Background removal took {time_end - time_start:.3f} seconds.")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_bg_removed.png"), bg_removed)
    
    # --- 4. Hessian Matrix Ridge Detection ---
    time_start = time.time()
    res_v = compute_hessian_ridge(bg_removed, sigma=9.0, mode='vertical')
    res_h =  compute_hessian_ridge(bg_removed, sigma=9.0, mode='horizontal')
    res_b = compute_hessian_ridge(bg_removed, sigma=9.0, mode='both')
    time_end = time.time()
    print(f"  [Timing] Hessian Ridge Detection took {time_end - time_start:.3f} seconds.")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_v.png"), res_v)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_h.png"), res_h)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_b.png"), res_b)

    
    # --- 5. 統計分析 (Mean & Std) ---
    print("  [Process] Calculating Column Statistics...")
    # 計算每個 Column 的平均值與標準差
    col_mean = np.mean(res_v, axis=0) # Shape: (width,)
    col_max = np.max(res_v, axis=0)   # Shape: (width,)

    row_mean = np.mean(res_h, axis=1) # Shape: (height,)
    row_max = np.max(res_h, axis=1)   # Shape: (height,)

    
    # 5.1 繪製圖表
    plot_path_v = os.path.join(OUTPUT_DIR, "step5_statistics_plot_v.png")
    plot_path_h = os.path.join(OUTPUT_DIR, "step5_statistics_plot_h.png")
    plot_and_save_statistics(col_mean, col_max, OPS, plot_path_v)
    plot_and_save_statistics(row_mean, row_max, OPS, plot_path_h)
    
    # 5.2 寫入 CSV
    csv_mean_path = os.path.join(OUTPUT_DIR, "colMean.csv")
    csv_max_path = os.path.join(OUTPUT_DIR, "Colmax.csv")
    csv_min_path = os.path.join(OUTPUT_DIR, "rowMean.csv")
    csv_max_path = os.path.join(OUTPUT_DIR, "rowmax.csv")
    

    
    save_array_to_csv(col_mean, csv_mean_path)
    save_array_to_csv(col_max, csv_max_path)
    save_array_to_csv(row_mean, csv_min_path)
    save_array_to_csv(row_max, csv_max_path)

    
    # --- 6. 熱力圖疊加 (Overlay) ---
    # 設定顯示範圍 (可以根據 res_v 的 histogram 調整，這裡示範 0-100 讓特徵更明顯)
    # 如果要看全範圍就設 0, 255
    # OVERLAY_LOWER = 20
    # heatmap_result = overlay_heatmap(src_img, res_v, lower_limit=OVERLAY_LOWER, alpha=0.3)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "step6_heatmap_overlay.png"), heatmap_result)
    # print(f"[Done] All artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()