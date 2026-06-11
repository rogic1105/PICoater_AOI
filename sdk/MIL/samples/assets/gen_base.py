"""MilGrabber icon 底圖生成：裁切設計參考圖（四色相機格子截圖）的白邊 → 置中正方形 256px。

icon 實際做法（忠於設計參考圖，非 PIL 重畫）：
  1. python gen_base.py <參考圖.png>   → 產 MilGrabber_base.png（已 .gitignore，中間產物）
  2. cd ../../../tools/icon-gen && python make_icon.py \
       --photo ../../MIL/samples/assets/MilGrabber_base.png --no-band --border F9A825 \
       --out ../../MIL/samples/assets/MilGrabber.ico
  （黃框 F9A825 = 影像範例程式語意；見 sdk/tools/icon-gen/README.md）
"""
import sys
from PIL import Image, ImageChops

ref = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Desktop\螢幕擷取畫面 2026-06-11 110707.png"
src = Image.open(ref).convert("RGB")

# 自動裁掉近白邊：與全白底的差異 bounding box
bbox = ImageChops.difference(src, Image.new("RGB", src.size, (255, 255, 255))).getbbox()
crop = src.crop(bbox)

# 置中貼到正方形畫布（取較長邊 + 4% 留白讓黃框不貼死色塊），縮到 256
side = max(crop.size)
pad = int(side * 0.04)
canvas = Image.new("RGB", (side + 2 * pad, side + 2 * pad), (255, 255, 255))
canvas.paste(crop, ((canvas.size[0] - crop.size[0]) // 2, (canvas.size[1] - crop.size[1]) // 2))
canvas.resize((256, 256), Image.LANCZOS).save("MilGrabber_base.png")
print("saved MilGrabber_base.png 256x256 (from", ref, ")")
