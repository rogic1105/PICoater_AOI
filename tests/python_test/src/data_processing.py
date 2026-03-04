import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from datetime import datetime

def plot_and_save_statistics(mean_arr: np.ndarray, max_arr: np.ndarray, min_arr: np.ndarray, 
                             ops_um: float, output_path: str, jpg_quality: int = 95):
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
    plt.savefig(output_path, dpi=150, format='jpg', pil_kwargs={'quality': jpg_quality})
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

def plot_comparison(arr_orig: np.ndarray, arr_comp: np.ndarray, arr_resized: np.ndarray,
                    arr_orig_resized: np.ndarray, arr_orig_compressed: np.ndarray,
                    ops_um: float, mag_reshape: int, comp_quality: int, output_path: str, title: str, ylabel: str):
    """繪製五個方案的比較圖"""
    print(f"  [Process] Plotting {title} to {output_path}...")
    
    # 計算橫軸
    x_indices = np.arange(len(arr_orig))
    x_mm = x_indices * ops_um / 1000.0
    
    plt.figure(figsize=(14, 6))
    plt.plot(x_mm, arr_orig, label='Original', color='blue', linewidth=1, linestyle='-')
    plt.plot(x_mm, arr_comp, label=f'Compressed (Quality {comp_quality})', color='red', linewidth=1, linestyle='--')
    
    # Image Processed -> Compressed 也是原圖大小，因此共用 x_mm
    plt.plot(x_mm, arr_orig_compressed, label=f'Orig->Compressed ({comp_quality})', color='orange', linewidth=1, linestyle='dashdot')
    
    # 為了 Resized 的橫軸需要調整（因為寬度不同）
    x_indices_resized = np.arange(len(arr_resized))
    x_mm_resized = x_indices_resized * ops_um * mag_reshape / 1000.0
    
    plt.plot(x_mm_resized, arr_resized, label=f'Resized (1/{mag_reshape} scale)', color='green', linewidth=1, linestyle=':')
    plt.plot(x_mm_resized, arr_orig_resized, label=f'Orig->Resized (1/{mag_reshape})', color='purple', linewidth=1, linestyle=(0, (3, 1, 1, 1, 1, 1)))

    plt.title(title)
    plt.xlabel("Position (mm)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, format='jpg', pil_kwargs={'quality': 95})
    plt.close()
