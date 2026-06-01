# algtest — 演算法原型 / 測試資料工具

PICoater mura 演算法的 Python 原型與測試資料準備工具。
**探索想法用 Python**（快、好畫圖、LLM 寫得準）；要**驗證產品行為一致性**（如 JPEG）才改用產品技術棧。

## 目錄

| 檔 | 做什麼 | 輸入 | 輸出 |
|---|---|---|---|
| `src/image_processing.py` | 影像演算法積木：去背 / Hessian ridge（回 float）/ `ridge_to_uint8` / 縮放壓縮 / heatmap | — | — |
| `src/data_processing.py` | 資料積木：統計圖 / CSV / 多方案比較圖 | — | — |
| `picoater_pipeline_reference.py` | **native pipeline 的 Python 參考**（去背→ridge→曲線→存）| QA 影像 | `artifacts/algtest/pipeline_reference` |
| `generate_mock_captures.py` | 少量 BMP → 模擬多相機 mock 存檔（測 review/data，2→7 相機）| BMP 資料夾 | 存檔格式（jpg+bin+csv）|
| `pseudo_merge.py` | 單向圖 → 旋轉疊加成「偽雙向」mock | 單張圖 | `<name>_merged` |
| `picoater_compress.py` | 壓縮/縮放 對 mura 曲線影響研究（5 種順序找極限）| `VH.bmp` | `artifacts/algtest/compress` |

## 關鍵原則（與 native 一致）

- **bin 曲線從 float ridge response 算**：`compute_hessian_ridge` 回 **float**（×255/正規值，不 clamp，保峰值）— 同 native「u8 之前算曲線」
- **顯示圖才 `ridge_to_uint8` clip u8** — 同 native 的 `scale_clamp_f32_to_u8`
- 演算法**唯一來源在 `src/`**，各腳本只組合積木、不自抄

## ⚠️ 與產品的一致性注意

- **JPEG**：`picoater_compress` 用 cv2（libjpeg），產品 C# 存檔用 GDI+ — **不同 encoder、量化表不同**。壓縮極限結論套到產品前，需用 GDI+ 對齊重驗（寫 C# 小工具或 pythonnet）。
- **resize**：產品用 CUDA `CoreCV_Resize_GPU`；研究若要精確一致，可用 ctypes 呼 `core_cv_api.dll`。

## 輸入 / 輸出資料

- **輸入**（raw 影像）：repo 外 `05_QA_Validation`（不可重生 → 要備份）。路徑目前硬編在各腳本，未來建議改環境變數。
- **輸出**（產物）：`artifacts/`（已 gitignore、可重生、不備份）
