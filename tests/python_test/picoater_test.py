import cv2
import numpy as np
import os

# Google Style: 模組級常數使用全大寫
IMAGE_FOLDER = "../../../../05_QA_Validation/feasibility_test_data/20250117 L5C/Envision/Low_Angle_by_nor_line/mura/"
IMAGE_NAME = 'cal_25-11-17_11-19-38-086.bmp'
IMAGE_PATH = os.path.join(IMAGE_FOLDER, IMAGE_NAME)
OUTPUT_DIR = "out"

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

def compute_hessian_ridge_vertical(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Detects vertical ridges only by computing the 2nd derivative in X (Lxx).
    Faster and more specific to vertical defects.
    """
    print(f"  [Process] Computing Vertical Ridge (Lxx only, Sigma={sigma})...")
    img_float = image.astype(np.float32)

    # 1. Gaussian Smoothing (高斯平滑)
    # 這是為了抗噪，一定要做。雖然只看 X 方向，但建議還是用 2D 高斯，
    # 這樣可以利用垂直方向的像素平均來消除雜訊點。
    ksize = int(6 * sigma + 1) | 1
    smooth = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)

    # 2. Compute 2nd Derivative in X (Lxx)
    # cv2.Sobel 參數說明: 
    # dx=2, dy=0 代表對 X 做二次微分，對 Y 不微分
    dxx = cv2.Sobel(smooth, cv2.CV_32F, 2, 0, ksize=3)

    # 3. Filter Ridges
    # Lxx 的物理意義：
    # - 數值為負大值 (<< 0)：代表亮度凸起 (Bright Line / Peak)
    # - 數值為正大值 (>> 0)：代表亮度凹陷 (Dark Line / Valley)
    # - 數值接近 0：平坦區域或斜坡
    
    # 因為我們要找 "顯著的線條" (不管是亮紋還是暗紋)，取絕對值即可
    resp = np.abs(dxx)

    # 如果你只想找 "亮紋" (Mura 比背景亮)，可以改成：
    # resp = np.maximum(-dxx, 0) 
    
    # Normalize output to 0-255
    resp_norm = cv2.normalize(resp, None, 0, 255, cv2.NORM_MINMAX)
    
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


def main():
    # --- 0. Prepare Output Directory ---
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[Info] Created output directory: {OUTPUT_DIR}")

    # --- 1. Load Image ---
    print(f"[Info] Loading image: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] File not found: {IMAGE_PATH}")
        return

    # Use IMREAD_GRAYSCALE to load as single channel
    src_img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if src_img is None:
        print("[Error] Failed to decode image.")
        return

    h, w = src_img.shape
    print(f"[Info] Image loaded. Width: {w}, Height: {h}")
    src_img =cv2.GaussianBlur(src_img, (11, 11), 5)
    # --- 2. Filter Background ---
    bg_removed = remove_column_background(src_img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step1_bg_removed.png"), bg_removed)

    # --- 3. Hessian Matrix Ridge Detection ---
    ridge_map = compute_hessian_ridge_vertical(bg_removed, sigma=9.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_hessian_response.png"), ridge_map)

    ridge_map = connect_with_vertical_average(ridge_map, kernel_width=30, kernel_height=180)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step3_vertical_max.png"), ridge_map)

    # --- 4. Post-processing & Visualization ---
    print("  [Process] Generating overlay...")
    

    _, binary = cv2.threshold(ridge_map, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_binary.png"), binary)

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


    output_path = os.path.join(OUTPUT_DIR, "step5_result_overlay.jpg")
    cv2.imwrite(output_path, overlay)
    
    print(f"[Success] All results saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()