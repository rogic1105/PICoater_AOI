#!/usr/bin/env python
"""make_icon.py — 把圖片做成 app icon（圓角 + 邊框 + 可選底部文字）

統一本專案 icon 風格：圓角方形、彩色邊框貼合外緣、可選底部 1/5 橫條文字。
配色語意慣例：綠 = 產品主程式、藍 = 測試 tool。

用法：
  # 照片 + 藍框 + 底部 AUTO 字（io 工具）
  python make_icon.py --photo ET7044.png --text AUTO --out io-auto.ico

  # 純藍底 + LIGHT 字（無照片）
  python make_icon.py --text LIGHT --out light.ico

  # 照片 + 綠框、不加字（產品主程式風格）
  python make_icon.py --photo roll.png --out app.ico --border 6FBF44 --no-band

依賴：pip install pillow
"""
import argparse
import os
from PIL import Image, ImageDraw, ImageFont

S = 256                                  # icon 基準尺寸
SIZES = [(256, 256), (64, 64), (48, 48), (32, 32), (16, 16)]
DEFAULT_FONT = r"C:\Windows\Fonts\arialbd.ttf"


def _hex(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def make_icon(out, photo=None, text=None,
              border="2563EB", band="1E3A8A", radius=40,
              font_path=DEFAULT_FONT):
    border_c, band_c = _hex(border), _hex(band)

    # 底圖：有照片用照片，否則純邊框色背景
    if photo:
        img = Image.open(photo).convert("RGBA").resize((S, S), Image.LANCZOS)
    else:
        img = Image.new("RGBA", (S, S), border_c + (255,))
    d = ImageDraw.Draw(img)

    # 底部 1/5 橫條 + 白字（option）
    if text:
        band_h = S // 5
        band_top = S - band_h
        d.rectangle([0, band_top, S, S], fill=band_c + (255,))
        size = int(band_h * 0.60)
        font = ImageFont.truetype(font_path, size)
        while size > 8:
            bb = d.textbbox((0, 0), text, font=font)
            if bb[2] - bb[0] <= S * 0.84:
                break
            size -= 2
            font = ImageFont.truetype(font_path, size)
        d.text((S // 2, band_top + band_h // 2), text,
               font=font, fill="white", anchor="mm")

    # 圓角削四角 + 邊框貼最外緣
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=radius, fill=255)
    out_img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out_img.paste(img, (0, 0), mask)
    ImageDraw.Draw(out_img).rounded_rectangle(
        [0, 0, S - 1, S - 1], radius=radius, outline=border_c, width=5)

    out_img.save(out, format="ICO", sizes=SIZES)
    print("saved", os.path.abspath(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="圖片 → app icon（圓角+邊框+可選文字）")
    ap.add_argument("--out", required=True, help="輸出 .ico 路徑")
    ap.add_argument("--photo", help="底圖（省略=純邊框色背景）")
    ap.add_argument("--text", help="底部文字（省略 或 --no-band=不加字）")
    ap.add_argument("--no-band", action="store_true", help="不加底部文字橫條")
    ap.add_argument("--border", default="2563EB", help="邊框 hex 色（預設藍 2563EB；產品用綠 6FBF44）")
    ap.add_argument("--band", default="1E3A8A", help="底字橫條 hex 色（預設深藍）")
    ap.add_argument("--radius", type=int, default=40, help="圓角半徑（預設 40）")
    ap.add_argument("--font", default=DEFAULT_FONT, help="字體 ttf 路徑")
    a = ap.parse_args()
    make_icon(a.out, a.photo, None if a.no_band else a.text,
              a.border, a.band, a.radius, a.font)
