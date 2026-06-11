# icon-gen — app icon 生成工具

把圖片做成統一風格的 app icon（圓角方形 + 邊框貼合外緣 + 可選底部文字橫條），
多尺寸 `.ico`（256/64/48/32/16）。

## 配色語意慣例

| 顏色 | 用途 | 範例 |
|---|---|---|
| 綠 `6FBF44` | 產品主程式 | `app.ico`（Anilox 滾筒）|
| 藍 `2563EB` | Bridge 工具 | `io-auto.ico` / `io-manual.ico` / `light.ico` / `storage.ico` |
| 黃 `F9A825` | 影像範例程式 | `MilGrabber.ico`（四色相機格子）|

> 邊框色＝類別語意（綠主程式 / 藍 Bridge / 黃 影像範例程式）；都可選加底部文字橫條（`--text`）。

## 用法

```bash
pip install pillow

# 照片 + 藍框 + 底部 AUTO 字（io 工具）
python make_icon.py --photo ET7044.png --text AUTO --out ../../../assets/io-auto.ico

# 純藍底 + LIGHT 字（無照片）
python make_icon.py --text LIGHT --out ../../../assets/light.ico

# 照片 + 綠框、不加字（產品主程式風格）
python make_icon.py --photo roll.png --out ../../../assets/app.ico --border 6FBF44 --no-band
```

## 參數

| 參數 | 說明 | 預設 |
|---|---|---|
| `--out` | 輸出 .ico（必要）| — |
| `--photo` | 底圖（省略=純邊框色背景）| 無 |
| `--text` | 底部文字（option）| 無 |
| `--no-band` | 強制不加底部文字橫條 | false |
| `--border` | 邊框 hex 色 | 2563EB（藍）|
| `--band` | 底字橫條 hex 色 | 1E3A8A（深藍）|
| `--radius` | 圓角半徑 | 40 |
| `--font` | 字體 ttf | arialbd.ttf |

## 設計

1. 底圖（照片 resize 256，或純色）
2. 底部 1/5 橫條 + 白字（anchor='mm' 垂直置中，自動 fit 寬度）
3. 圓角遮罩削四角
4. 邊框沿圓角貼最外緣（width 5，不內縮）

產出的 icon 套用到 exe 用 csproj 的 `<ApplicationIcon>..\assets\xxx.ico</ApplicationIcon>`。
