"""image_processing.py — algtest 影像演算法積木（PICoater mura pipeline）。

對應 C# native（src/native/modules/GetPICoaterBackground）的 Python 參考實作：
- remove_column_background：逐列均值去背（+127 保留方向，同 native）
- compute_hessian_ridge：Hessian ridge V/H + fixed 正規化，回 **float**（×255/正規值，不 clamp）
  —— 同 native：bin 曲線從這個 float 算（保峰值，u8 之前）
- ridge_to_uint8：float ridge → uint8 顯示圖（clip）—— 只給顯示，別拿來算曲線
- overlay_heatmap：ridge 熱圖疊加（視覺化）
- img_reduction_resize / _compress / _resize_average_filter：縮放與 JPEG 壓縮
"""
import cv2
import numpy as np

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

    # 3. Fixed Scaling — 同 native：×255/正規值，但「不 clamp」回傳 float。
    #    曲線（bin）要從這個 float 算（保留 >255 的峰值）；顯示圖才另外用 ridge_to_uint8 clip。
    #    對應 native：曲線 from float hessian response（u8 之前），u8 只是 scale_clamp 顯示圖。
    scale_factor = 255.0 / fixed_max_val
    return response * scale_factor   # float（u8 之前）


def ridge_to_uint8(ridge_response: np.ndarray) -> np.ndarray:
    """float ridge response → uint8 顯示圖（clip 0-255）。對應 native scale_clamp_f32_to_u8。

    ⚠️ bin 曲線不要用這個——會 clamp 掉 >255 的峰值。曲線請直接用 compute_hessian_ridge 的 float。
    """
    return np.clip(ridge_response, 0, 255).astype(np.uint8)

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

def img_reduction_resize(img: np.ndarray, mag: int):
    h, w = img.shape[:2]
    h_reshape, w_reshape = h // mag, w // mag
    img_resized = img[::mag, ::mag]
    return img_resized

def img_reduction_compress(img: np.ndarray, jpg_quality: int = 95):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    img_compressed = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
    return img_compressed

def img_reduction_resize_average_filter(img,avg_resize_scale):
    """
    方式3：使用平均滤波的Resize
    使用cv2.INTER_AREA插值方法实现平均滤波resizing
    缩放到原来的1/10大小
    """
    h, w = img.shape[:2]
    new_h, new_w = h // avg_resize_scale, w // avg_resize_scale
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_img