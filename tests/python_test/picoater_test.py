import cv2
import numpy as np
import os
from scipy.signal import find_peaks

# Google Style: 模組級常數使用全大寫
IMAGE_FOLDER = "../../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
IMAGE_NAME = 'cal_25-11-17_11-19-38-086.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "out"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def remove_column_background(image: np.ndarray) -> np.ndarray:
    """Removes background by subtracting the column-wise mean.
    """
    print("  [Process] Removing column background...")
    img_float = image.astype(np.float32)
    
    # Calculate column mean (1D array)
    col_mean = np.mean(img_float, axis=0)
    
    # Broadcast to 2D (Expand)
    bg_2d = np.tile(col_mean, (image.shape[0], 1))
    
    # Subtract (Mura = Src - BG + offset)
    result = abs(img_float - bg_2d)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def compute_hessian_ridge(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Detects ridges using Hessian Matrix eigenvalues.
    """
    print(f"  [Process] Computing Hessian Matrix (Sigma={sigma})...")
    img_float = image.astype(np.float32)

    # 1. Gaussian Smoothing
    ksize = int(6 * sigma + 1) | 1
    smooth = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

    k_sobel = 3
    # 2. Second Derivatives
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=k_sobel)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=k_sobel)
    
    dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=k_sobel)
    dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=k_sobel)
    dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=k_sobel)

    # 3. Compute Eigenvalues
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    
    discriminant = tr * tr - 4 * det
    discriminant[discriminant < 0] = 0
    sqrt_disc = np.sqrt(discriminant)
    
    lambda1 = (tr + sqrt_disc) / 2.0
    lambda2 = (tr - sqrt_disc) / 2.0
    
    # 4. Filter Ridges (Maximize curvature response)
    resp = np.maximum(np.abs(lambda1), np.abs(lambda2))
    
    # Normalize result
    resp_norm = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
    return resp_norm.astype(np.uint8)

import cv2
import numpy as np

def compute_hessian_ridge(image: np.ndarray, 
                          sigma: float = 2.0, 
                          mode: str = 'vertical') -> np.ndarray:
    """Computes Hessian-based ridge detection for specific directions.

    This function calculates the 2nd derivative (Hessian) to highlight ridge-like
    structures. It approximates the operation using Gaussian smoothing followed
    by Sobel operators.

    Args:
        image: Input image (grayscale), numpy array.
        sigma: Standard deviation for Gaussian smoothing. Higher values
               detect wider ridges but may lose detail.
        mode: Direction to detect. Options are:
              'vertical'   - Detects vertical lines (Lxx).
              'horizontal' - Detects horizontal lines (Lyy).
              'both'       - Detects ridges in both directions (|Lxx| + |Lyy|).

    Returns:
        A normalized uint8 image (0-255) containing the ridge response.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if mode not in ('vertical', 'horizontal', 'both'):
        raise ValueError(f"Invalid mode: {mode}. Use 'vertical', 'horizontal', or 'both'.")

    print(f"  [Process] Computing Ridge (Mode={mode}, Sigma={sigma})...")
    
    img_float = image.astype(np.float32)

    # 1. Gaussian Smoothing
    # Apply 2D Gaussian blur to suppress noise before differentiation.
    ksize = int(6 * sigma + 1) | 1
    smooth = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

    response = None

    # 2. Compute Derivatives based on mode
    if mode == 'vertical':
        # 2nd derivative in X (Lxx) -> Detects Vertical features
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        response = np.abs(dxx)

    elif mode == 'horizontal':
        # 2nd derivative in Y (Lyy) -> Detects Horizontal features
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        response = np.abs(dyy)

    elif mode == 'both':
        # Compute both and combine
        dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(smooth, cv2.CV_32F, 0, 2, ksize=3)
        # Combine absolute responses to capture features in both directions
        response = np.abs(dxx) + np.abs(dyy)

    # 3. Normalize output to 0-255
    resp_norm = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)

    return resp_norm.astype(np.uint8)





def connect_with_vertical_average(image: np.ndarray,kernel_width: int = 30, kernel_height: int = 180) -> np.ndarray:
    """
    Step 2: 垂直方向卷積 (取平均)
    這相當於一個 (kernel_height, 1) 的 Box Filter
    """
    print(f"  [Process] Applying Vertical Average (Box Filter, Height={kernel_height})...")
    
    # cv2.boxFilter 參數說明:
    # ddepth=-1: 輸出深度與輸入相同
    # ksize=(1, kernel_height): 寬度 1, 高度 30 (只做垂直平均)
    # normalize=True: 做平均 (Sum / Area)。如果設 False 就是單純加總。
    result = cv2.boxFilter(image, -1, (kernel_width, kernel_height), normalize=True)
    
    # 注意：取平均後，原本很亮的線 (255) 加上中間的斷裂黑洞 (0)，
    # 整體亮度會下降 (例如變 150)，這是正常的。
    
    return result

def mark_local_peaks(image: np.ndarray, 
                     n: int, 
                     min_height: float = 20.0,
                     min_distance: int = 50,
                     prominence: float = 10.0,
                     radius: int = 15, 
                     color: tuple = (255, 255, 255)) -> np.ndarray:
    """Detects and marks local maxima in averaged row blocks using Scipy.

    This function averages the image intensity every n rows to create a 1D profile.
    It then uses scipy.signal.find_peaks to identify significant local maxima
    based on height and prominence, allowing for multiple detections per block.

    Args:
        image: Source image (grayscale or color), numpy array.
        n: The height of the row block to average.
        min_height: Required minimum intensity of peaks (0-255).
                    Peaks below this value are ignored (e.g., background noise).
        min_distance: Required minimum horizontal distance between neighboring peaks.
        prominence: Required prominence of peaks. Measures how much a peak stands 
                    out from the surrounding baseline.
        radius: Radius of the circle to draw.
        color: Color of the circle (B, G, R).

    Returns:
        The image with circles drawn at all detected peak locations.
    """
    if n <= 0:
        raise ValueError("Parameter n must be greater than 0.")

    output_img = image.copy()
    h, w = image.shape[:2]

    print(f"  [Process] Scanning for local peaks (Block size={n}, Min Height={min_height})...")

    # Iterate through the image height with step n
    for y in range(0, h, n):
        y_end = min(y + n, h)
        
        # 1. Extract and Average
        strip = image[y:y_end, :]
        # Shape becomes (width, ) - a 1D signal
        avg_row = np.mean(strip, axis=0)

        # 2. Find Peaks using Scipy
        # find_peaks returns indices of peaks satisfying the conditions
        peaks, _ = find_peaks(
            avg_row, 
            height=min_height,      # Absolute height check
            distance=min_distance,  # Avoid multiple detections on thick lines
            prominence=prominence   # Relative height check (peak vs valley)
        )

        # 3. Draw Circles for all found peaks
        center_y = int((y + y_end) / 2)
        
        for x_peak in peaks:
            cv2.circle(output_img, (int(x_peak), center_y), radius, color, -1)

    return output_img

def main():
    # --- 1. Load Image ---
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return
    
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if src_img is None:
        print("[Error] Failed to decode image.")
        return

    h, w = src_img.shape
    print(f"[Info] Image loaded. {IMAGE_PATH}")
    print(f"[Info] Image loaded. Width: {w}, Height: {h}")
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_input.png"), src_img)
    
    # --- 2. Pre-processing ---
    src_img =cv2.GaussianBlur(src_img, (11, 11), 5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_blurred.png"), src_img)
    print("  [Process] Pre-processing completed.")
    
    # --- 3. Filter Background ---
    bg_removed = remove_column_background(src_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_bg_removed.png"), bg_removed)
    print("  [Process] Background removed.")
    
    # --- 4. Hessian Matrix Ridge Detection ---
    res_v = compute_hessian_ridge(bg_removed, sigma=9.0, mode='vertical')
    res_h = compute_hessian_ridge(bg_removed, sigma=9.0, mode='horizontal')
    res_b = compute_hessian_ridge(bg_removed, sigma=9.0, mode='both')
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_v.png"), res_v)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_h.png"), res_h)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_res_b.png"), res_b)

    ridge_map = res_v + res_h
    print("  [Process] Ridge map computed.")
    
    # --- 5.0. Post-processing ---
    ridge_map_peak = mark_local_peaks(
            res_v, 
            n=100, 
            min_height=10,      # Ignore dark noise < 50
            min_distance=50,   # Peaks must be 100px apart
            prominence=10
        )
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step5_peak.png"), ridge_map_peak)
    
    # --- 5. Post-processing ---
    ridge_map_conv = connect_with_vertical_average(res_v, kernel_width=30, kernel_height=180)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step5_conv.png"), ridge_map_conv)
    print("  [Process] Ridge map post-processing completed.")
    
    # --- 6. mask ---
    _, binary = cv2.threshold(ridge_map_conv, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step6_binary.png"), binary)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print(f"  [Info] Detected {num_labels - 1} potential defect segments.")

    mura_label = np.zeros_like(labels)

    mura_num = []
    min_area_threshold = h*10
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area_threshold:
            mura_num.append(i)
            mura_label[labels == i] = i


    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background black
    
    colored_mask = colors[mura_label]
    src_bgr = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(src_bgr, 0.7, colored_mask, 0.3, 0)


    output_path = os.path.join(OUTPUT_DIR, "step6_result_overlay.jpg")
    cv2.imwrite(output_path, overlay)
    
    print(f"[Success] All results saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()