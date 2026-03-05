import cv2
import numpy as np
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python pseudo_merge.py <image_path>")
        return
        
    # 影像路徑
    img_path = sys.argv[1]
    base, ext = os.path.splitext(img_path)
    out_path = f"{base}_merged{ext}"
    
    # 讀取原圖
    img1 = cv2.imread(img_path)
    if img1 is None:
        print(f"Error: 無法讀取檔案 {img_path}")
        return
        
    # 取得原圖尺寸 (高度, 寬度)
    h, w = img1.shape[:2]
    
    # 90度旋轉 (順時針)
    rotated_img = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
    
    # resize 到跟原圖一樣大
    resized_rotated_img = cv2.resize(rotated_img, (w, h))
    
    # 兩張圖片做各自 50% 疊加
    blended_img = cv2.addWeighted(img1, 0.5, resized_rotated_img, 0.5, 0)

    # 轉成灰階
    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2GRAY)
    
    # 存檔
    cv2.imwrite(out_path, blended_img)
    print(f"成功儲存疊加後的圖片至: {out_path}")

if __name__ == "__main__":
    main()
