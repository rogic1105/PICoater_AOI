import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks

# --- 參數設定 ---
IMAGE_FOLDER = "../../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
IMAGE_NAME = 'cal_25-11-17_11-19-25-929.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "../../artifacts/python_test/method_2"

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

    # 3. Fixed Scaling (取代原本的 normalize)
    # 公式: Output = (Response / Fixed_Max) * 255
    # 例如: Response=0.5, Max=1.0 -> Output=127.5
    scale_factor = 255.0 / fixed_max_val
    resp_scaled = response * scale_factor
    
    # 4. Clip to 0-255 (超過上限的就切平在 255)
    resp_fixed = np.clip(resp_scaled, 0, 255).astype(np.uint8)
    return resp_fixed

# --- 新增函數 ---

def plot_and_save_statistics(mean_arr: np.ndarray, max_arr: np.ndarray, min_arr: np.ndarray, 
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
    
    # 繪製標準差範圍 (虛線)
    upper_bound = max_arr
    lower_bound = min_arr
    
    plt.plot(x_mm, upper_bound, label='Max', color='orange', linestyle='--', linewidth=1)
    plt.plot(x_mm, lower_bound, label='Min', color='orange', linestyle='--', linewidth=1)
    
    # 填充中間區域 (可選，讓圖更清楚)
    plt.fill_between(x_mm, lower_bound, upper_bound, color='orange', alpha=0.1)

    plt.title("Column Intensity Statistics (Ridge Response)")
    plt.xlabel("Position (mm)")
    plt.ylabel("Intensity / Std Dev")
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
    """
    將 overlay_image 轉換為熱力圖並疊加。
    僅針對 lower_limit < 值 < upper_limit 的區域上色，其餘保留原圖灰階。
    """

    
    # 1. 準備原圖 (轉為 BGR 以便與熱力圖疊加)
    if len(src_image.shape) == 2:
        src_bgr = cv2.cvtColor(src_image, cv2.COLOR_GRAY2BGR)
    else:
        src_bgr = src_image.copy()

    # 2. 應用熱力圖
    # 注意：這裡直接轉換，代表數值 0-255 對應 藍-紅。
    # 如果您希望 50-250 這個區間「展開」成全光譜顏色，需要先做 normalize，
    # 但若只需單純截斷顏色，直接 applyColorMap 即可。
    heatmap = cv2.applyColorMap(overlay_image, cv2.COLORMAP_JET)
    
    # 3. 疊加 (全局疊加)
    beta = 1.0 - alpha
    result = cv2.addWeighted(src_bgr, alpha, heatmap, beta, 0)
    
    # 4. 遮罩還原 (Masking)
    # 邏輯：找出「不需要上色」的區域 mask
    # 條件：數值 <= lower_limit  或  數值 >= upper_limit
    mask_indices = (overlay_image <= lower_limit) 
    
    # 5. 替換像素
    # [修正點]：result 是 3 通道，src_bgr 也是 3 通道，這樣才能正確替換。
    # 原本寫 src_image (單通道) 會報錯。
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
    bg_removed = remove_column_background(src_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_bg_removed.png"), bg_removed)
    
    # --- 4. Hessian Matrix Ridge Detection ---
    res_v = compute_hessian_ridge(bg_removed, sigma=9.0, mode='vertical')
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_v.png"), res_v)
    res_v[res_v<0] = 0
    
    # --- 5. 統計分析 (Mean & Std) ---
    print("  [Process] Calculating Column Statistics...")
    # 計算每個 Column 的平均值與標準差
    col_mean = np.mean(res_v, axis=0) # Shape: (width,)
    col_max = np.max(res_v, axis=0)   # Shape: (width,)
    col_min = np.min(res_v, axis=0)   # Shape: (width,)
    
    # 5.1 繪製圖表
    plot_path = os.path.join(OUTPUT_DIR, "step5_statistics_plot.png")
    plot_and_save_statistics(col_mean, col_max, col_min, OPS, plot_path)
    
    # 5.2 寫入 CSV
    csv_mean_path = os.path.join(OUTPUT_DIR, "record_mean.csv")
    csv_max_path = os.path.join(OUTPUT_DIR, "record_max.csv")
    csv_min_path = os.path.join(OUTPUT_DIR, "record_min.csv")

    
    save_array_to_csv(col_mean, csv_mean_path)
    save_array_to_csv(col_max, csv_max_path)
    save_array_to_csv(col_min, csv_min_path)
    
    # --- 6. 熱力圖疊加 (Overlay) ---
    # 設定顯示範圍 (可以根據 res_v 的 histogram 調整，這裡示範 0-100 讓特徵更明顯)
    # 如果要看全範圍就設 0, 255
    OVERLAY_LOWER = 0
    
    heatmap_result = overlay_heatmap(src_img, res_v, 
                                     lower_limit=OVERLAY_LOWER, 
                                     alpha=0.3) # 原圖佔 60%, 熱力圖佔 40%
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step6_heatmap_overlay.png"), heatmap_result)
    print(f"[Done] All artifacts saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()