# 系統資源用量說明

本文件說明 PICoater AOI 各功能頁面的 GPU、CPU、記憶體需求，供硬體規格評估參考。

---

## 即時監控（Live tab）

| 資源 | 項目 | 估計用量 |
|------|------|---------|
| CUDA Pinned Memory | NativeBufferPool（input + ridge + mura + thumbnail + curves） | ~634 MB（啟動配置，全程共用） |
| GPU | 每幀 ProcessPipeline（16384×10000 max）+ Resize_GPU | 持續使用，依相機 FPS |
| RAM — MIL 顯示 | 7 台主/副顯示 buffer | 由 MIL 管理 |
| RAM — 曲線快取 | _liveCurveMean/Max[7]，float[]，len ≤ 16384 | ~1 MB |
| RAM — Overview Chart | MaxOverviewPoints = 2000，merge 7 台 | ~200 KB |
| RAM — Telemetry | listViewCameras 16 欄 × 7 行 | < 100 KB |
| CPU | MIL callback marshal、chart 更新、PLC polling | 多執行緒，中等負載 |

**即時監控是最主要的 GPU 消耗來源**，取決於相機 FPS 和解析度。

---

## 歷史查詢（Review tab）

| 資源 | 項目 | 估計用量 |
|------|------|---------|
| CUDA Pinned Memory | 與 Live 共用同一個 NativeBufferPool | 0（已配置） |
| GPU | JPEG 模式：不使用；BMP 模式：ProcessPipeline + Resize_GPU | 0 或 ~160M pixels/次 |
| RAM — Gallery 縮圖 | 7 台 × 縮圖（~1000×600 JPEG） | 25–35 MB |
| RAM — 拼接圖 | _stitchedImages[7]，每台垂直拼接所有 grab，寬 ~3276 px | 150–250 MB（依 grab 數） |
| RAM — 曲線 | _stitchedCurveMean/Max × 7 × 2 方向，float[] | ~1 MB |
| RAM — Overview Chart | MaxOverviewPoints = 2000 | ~200 KB |
| CPU | JPEG 解碼、Bitmap 拼接、MergeCurves | 單核密集，短暫 |

### JPEG vs BMP 模式差異

| 模式 | 條件 | GPU 使用 | 說明 |
|------|------|---------|------|
| JPEG（預設） | SaveOriginalBmp = false | 不使用 | 讀取 _raw.jpg + _proc_v.jpg + .bin，純 CPU 解碼 |
| BMP | SaveOriginalBmp = true | 使用 | 讀 BMP → GPU pipeline → 存 .bin 快取，後續不再重算 |

**拼接圖是 Review tab 記憶體用量的主要來源**，grab 數量越多拼接圖越大。

---

## 檢測報表（Data tab）

| 資源 | 項目 | 估計用量 |
|------|------|---------|
| GPU | 完全不使用 | 0 |
| CUDA Pinned Memory | 不使用 | 0 |
| RAM — GrabIdInfo | 每筆 ~60 bytes | ~120 KB / 月 |
| RAM — GrabDetail | 每筆 ~70 bytes | ~700 KB @ 10K 筆 |
| RAM — listViewGrabDetail | 每行 ~150 bytes | ~1.5 MB @ 10K 行 |
| RAM — Period Charts | 3 張 StackedColumn（12/31/24 bucket） | ~300 KB |
| CPU | CSV 遞迴掃描 + 統計計算 | 單核，I/O bound |

**Data tab 資源消耗極低**，瓶頸在 CSV 檔案數量影響的掃描時間。

---

## 取像資料流

```
Camera Grabber ──DMA──→ MIL Grab Buffer（VRAM/Host）
        │
        ├──MbufGet2d──→ managed byte[]（_hostInputBuffer）
        │       │
        │    Marshal.Copy
        │       ▼
        │   CUDA pinned buffer（NativeBufferPool._inputBuffer）
        │       │
        │   GPU ProcessPipeline
        │       ▼
        │   ridge / mura / curves（CUDA pinned）
        │       │
        │    Marshal.Copy
        │       ▼
        ├──MbufPut2d──→ MIL Display Buffer → 即時顯示
        │
        └── 存檔路徑（非同步，不影響 pipeline）：
            GPU Resize → Marshal.Copy → JPEG encode → 寫磁碟
```

**Grabber → CUDA 全程在記憶體中**，不經過磁碟。磁碟 I/O 只發生在存檔時。

## 取像吞吐量

### 實際生產規格（觀測值）

解析度：**16384 × 3000 px**，7 台相機，每次檢測 10 張，每日 2000 次檢測

| 項目 | 計算 | 數值 |
|------|------|------|
| 單張存檔（JPG 原圖 + V/H mura + V/H bin） | 實測 | **~0.7 MB** |
| 單次檢測（7 台 × 10 張） | 0.7 MB × 7 × 10 | **~50 MB** |
| 每日寫入 | 2000 次 × 50 MB | **~100 GB/日** |
| 每月寫入 | 100 GB × 30 | **~3 TB/月** |
| 單幀 raw（GPU pipeline 輸入） | 16384 × 3000 × 8bit | 46.9 MB |
| GPU pipeline（單幀） | 49 Mpixel ProcessPipeline | **34–42 ms**（RTX 5080 實測） |

### 最大設計規格（上限）

最大規格：**16384 × 10000 px × 7 台 / 秒**（線掃相機，GrabHeight=10000, LineRate=10000Hz）

| 項目 | 計算 | 數值 |
|------|------|------|
| 單幀 raw data | 16384 × 10000 × 8bit | 156 MB |
| 7 台 raw 吞吐 | 156 MB × 7 | **1.07 GB/秒** |
| GPU pipeline（單幀） | 164 Mpixel ProcessPipeline | **~0.5 秒**（RTX 5080 實測） |
| .bin 曲線 | (16384+10000) × 4bytes × 4檔 × 7 台 | **~3 MB/秒** |
| CSV 日誌 | 7 行/秒 × ~200 bytes | 極小 |

> 實際 FPS 由 GrabHeight 決定：Height=3000（實際）→ ~3.3 FPS/台；Height=10000（最大）→ ~1 FPS/台

---

## GPU Pipeline 演算法分析

基於 `picoater_api.dll` 原始碼（`Module_GetPICoaterBackground.cu`）分析。

### Pipeline 步驟

| Step | 演算法 | Kernel | 記憶體存取模式 |
|------|--------|--------|--------------|
| 1 | Column Mean（去離群值，2-pass） | `k_calcColumnMeans_RemoveOutliers` | 1 thread/col，垂直掃描 H 列 ×2 pass |
| 2 | Background Removal（逐像素減背景） | `k_calcColumnBackground` | 2D grid，read input + col_mean → write output |
| 3 | Gaussian Blur（可分離，Row+Col） | `k_gaussianBlurRow` + `k_gaussianBlurCol` | 2D grid，float 精度，ksize ×2 pass |
| 4 | Hessian Response（3×3 stencil） | `k_hessianResponse` | 1D grid，9 鄰域 float 讀取 |
| 5 | Scale + Clamp（float→uint8） | `k_scale_clamp_f32_to_u8` | 1D grid，逐像素 |
| 6 | Column Mean + Max（ridge 統計） | `k_calcColumnMeans` + `k_calcColumnMax` | 1 thread/col，垂直掃描 |
| 7 | cudaMemcpy H2D + D2H | Host↔Device 傳輸 | input + bg + mura + ridge + curves |

> Step 1 的 column-wise 掃描是最大瓶頸：每個 thread 垂直掃描 H 列，stride = W，**記憶體不連續**（coalescing 極差），實際頻寬效率僅 ~10–20%。

### 單幀頻寬消耗（實際 vs 最大）

| | 實際（16384×3000 = 49 Mpixel） | 最大（16384×10000 = 164 Mpixel） |
|---|---|---|
| 單幀總頻寬 | **~2.1 GB** | **~6.9 GB** |
| cudaMemcpy H2D+D2H | ~188 MB | ~625 MB |
| RTX 5080 處理時間 | **~0.15 秒**（估算） | **~0.5 秒**（實測） |

### GPU 效能估算

| GPU | 記憶體頻寬 | 實際 49MP 估計 | 最大 164MP 估計 |
|-----|----------|--------------|----------------|
| **RTX 5080** | 960 GB/s | **34–42 ms（實測）** | **~0.5 秒**（實測） |
| RTX 4080 | 717 GB/s | ~55 ms | ~0.7 秒 |
| RTX 4070 Ti | 504 GB/s | ~80 ms | ~1.0 秒 |
| RTX 4060 Ti | 288 GB/s | ~140 ms | ~1.7 秒 |
| RTX A2000 | 288 GB/s | ~140 ms | ~1.9 秒 |

> 以 RTX 5080 實測（49 Mpixel = 34–42 ms, 164 Mpixel = 0.5 秒）為基準，按記憶體頻寬比例估算。Pipeline 為 memory-bandwidth bound，估算誤差 ±30%。

### 效能瓶頸分析

| 因素 | 影響程度 | 說明 |
|------|---------|------|
| **記憶體頻寬** | 最高 | Pipeline 每步都是逐像素讀寫，總頻寬 ~6.9 GB/幀 |
| **Column-wise coalescing** | 高 | Step 1/6 每 thread 沿 Y 方向掃描，stride=W，L2 cache 有限 |
| **Gaussian ksize** | 中 | ridgeSigma=2 → ksize=13，ridgeSigma=5 → ksize=31，影響 Step 3 |
| **CUDA cores** | 低 | 非計算密集型，core 數量不是瓶頸 |
| **H2D/D2H 傳輸** | 低 | ~625 MB/幀，PCIe Gen4 x16 = 32 GB/s → ~20 ms |

---

## 各 Tab 資源用量對照

| | 即時監控 | 歷史查詢 | 檢測報表 |
|---|---|---|---|
| GPU VRAM | 持續使用（7×49MP/幀，實際） | 0（JPEG）或偶爾（BMP） | 0 |
| CUDA Pinned | ~634 MB（共用） | 共用 | 不使用 |
| RAM 峰值 | ~50 MB + MIL | 200–300 MB | 3–5 MB |
| CPU 瓶頸 | GPU callback + chart | JPEG 解碼 / 拼接 | CSV I/O |
| 主要瓶頸因素 | 相機 FPS × 解析度 | grab 數量 → 拼接圖大小 | CSV 檔案數量 |

---

## 共用資源

- **NativeBufferPool**：啟動時一次性配置 ~634 MB CUDA pinned memory，Live 和 Review 共享
  - input buffer：16384 × 10000 = 156 MB
  - mura buffer：156 MB
  - ridge buffer：156 MB
  - thumbnail buffer：2000 × 2000 = 3.8 MB
  - curve buffers：~170 KB
- **picoater_api.dll 內部 GPU VRAM**：
  - `AoiPipelineContext` 固定 buffer（`export_api.cpp::EnsureBuffers`）：
    - d_input / d_background / d_mura / d_ridge：各 W×H uint8
    - d_curve_mean/max + d_row_curve_mean/max: ~0.2 MB
  - `PICoaterDetector` 固定 buffer（`Module_GetPICoaterBackground.cu::Initialize`）：
    - d_col_mean (W×float) + d_col_bg_ (W×H) + d_blur_tmp_ (W×H)
  - `PICoaterDetector` 共用 workspace（取 Gaussian / Hessian 中較大者）：
    - Gaussian 需求：3 × W×H×float + mask（較大，決定 workspace 大小）
  - CUDA runtime 額外開銷：~200–300 MB

  | | 實際（16384×3000） | 最大（16384×10000） |
  |---|---|---|
  | 固定 buffer（×6） | 281 MB | 936 MB |
  | Workspace（Gaussian） | 563 MB | 1,877 MB |
  | CUDA runtime | ~200 MB | ~300 MB |
  | **總 VRAM** | **~1.0 GB** | **~3.1 GB** |
- **InspectionEngine**：單一共用實例，不會同時處理多張影像

---

## 建議硬體規格（生產機）

### 檢測電腦（低配 / 標配）

| 項目 | 低配 | 標配 | 實測依據 |
|------|------|------|---------|
| 工業電腦 | **Advantech IPC-7130** (4U) | 同左 | 雙 PCIe x16（GPU + Grabber） |
| 主機板 | **W790** | 同左 | PCIe 5.0 多通道（GPU + Grabber + U.2） |
| CPU | **Xeon W5-3423**（12C/24T） | **Xeon W5-3525**（16C/32T） | CPU 實測 2-5%（2 cam），7 cam 推估 10-15% |
| RAM | **32 GB DDR5 ECC RDIMM** | 同左 | 7 cam Grab 峰值 ~3.4 GB + OS ~2 GB + MIL ~1 GB ≈ 7 GB；32 GB 餘裕充足 |
| GPU | **RTX 5080 16GB** | 同左（+備品 1 張） | VRAM 實測 3.0 GB（2 cam），7 cam 推估 **~7.3 GB**（每台相機獨立 pipeline）；8 GB 卡不可行，16 GB 為最低需求 |
| SSD（系統） | **M.2 500GB**（Kioxia XG8） | **M.2 1TB**（Kioxia XG8） | 系統碟寫入極低（< 5 GB/日） |
| SSD（存圖） | **U.2 3.84TB × 2** | **U.2 3.84TB × 3** | 低配 7.7 TB ≈ 2.5 個月；標配 11.5 TB ≈ 3.8 個月 |
| PSU | **700W 工業級** | 同左 | — |
| OS | **Win 10 IoT LTSC 2021** | 同左 | 10 年安全更新 |
| UPS | **APC Smart-UPS 1500VA** | 同左 | 防斷電保護 SSD |

> 低配與標配差異：CPU（12C→16C，影響極小）、系統碟（500G→1T）、存圖碟（×2→×3，多 1 顆多撐 1 個月）、GPU 備品。

### 儲存電腦（低配 / 標配）

| 項目 | 低配 | 標配 | 實測依據 |
|------|------|------|---------|
| 主機板 | **Q670E** | 同左 | 嵌入式長供貨，vPro 遠端管理 |
| CPU | **i5-13500TE**（6P+8E, 35W） | 同左 | Review CPU 實測 1-6%（2 cam） |
| RAM | **16 GB DDR5** | **32 GB DDR5** | Review 峰值 ~1.6 GB（2 cam），7 cam 推估 ~3.5 GB；16 GB 足夠 |
| GPU | **不需要（內顯）** | 同左 | 已驗證無 GPU 可正常啟動，JPG Review 不用 CUDA |
| SSD（系統） | **M.2 500GB** | 同左 | — |
| HDD（存圖） | **企業 HDD 18TB × 1** | **企業 HDD 18TB × 2（RAID1）** | 低配無冗餘，硬碟故障 = 資料全失 |
| PSU | **300W** | 同左 | 無獨顯，TDP 35W |
| OS | **Win 10/11 Pro** | 同左 | 遠端桌面 |

> 低配與標配差異：RAM（16→32 GB）、HDD（×1→×2 RAID1）。**RAID1 強烈建議**：省一顆 HDD 約 $400 USD，但硬碟故障即失去 6 個月資料。

### 低配 / 標配成本差異摘要

| 差異項目 | 省下金額（約） | 風險 |
|---------|-------------|------|
| 檢測機 U.2 ×2→×3 | ~$300 USD | 存圖空間少 1 個月，需更頻繁清理或仰賴傳輸 |
| 檢測機 GPU 備品 | ~$1,000 USD | GPU 故障需等採購，停機時間拉長 |
| 檢測機系統碟 500G→1T | ~$30 USD | 影響極小 |
| 儲存機 RAM 16→32 GB | ~$30 USD | 7 cam 全域合圖仍可運行 |
| 儲存機 HDD ×1→×2 | ~$400 USD | **無 RAID1，硬碟故障 = 資料全失** |

#### SSD 詳細規格（工業級）

**存圖碟 — U.2**

| 項目 | 規格 |
|------|------|
| Type | 2.5" U.2 SSD (SFF-8639), 15mm |
| Model | Solidigm D7-P5520 |
| Capacity | 3.84 TB |
| Interface | PCIe Gen4 x4, NVMe 1.4 |
| Flash | 3D TLC (Enterprise Grade) |
| Temp. Range | 0°C ~ 70°C (Commercial) |
| PLP | Yes — Hardware capacitor-based |
| Endurance | 1 DWPD (~7,008 TBW) |
| MTBF | > 2,000,000 hours |
| Fixed BOM | Yes |

> U.2 15mm 金屬外殼自帶散熱，適合 ~100 GB/日持續寫入。需主板 U.2 接口或 M.2→U.2 轉接卡（SFF-8639 to M.2 adapter）。

**系統碟 — M.2**

| 項目 | 規格 |
|------|------|
| Type | M.2 2280 M-Key |
| Model | Kioxia XG8 (KXG80ZNV1T02) |
| Capacity | 1 TB |
| Interface | PCIe Gen4 x4, NVMe 1.4 |
| Flash | 3D TLC (Enterprise Grade, BiCS FLASH) |
| Temp. Range | 0°C ~ 70°C (Commercial) |
| PLP | No（系統碟寫入量低，搭配 UPS 即可） |
| Endurance | 1 DWPD (~600 TBW) |
| MTBF | > 1,500,000 hours |
| Fixed BOM | Yes（OEM 供貨） |

> M.2 2280 直插主板插槽，不佔額外空間。系統碟每日寫入極低（< 5 GB），不需 PLP。Kioxia XG8 為日本 Kioxia（原東芝）企業 OEM 系列，Fixed BOM 確保主控與顆粒不替換。

### 消費等級（開發/測試用）

| 項目 | 推薦型號 | 說明 |
|------|---------|------|
| CPU | Intel Core i5-14500 | LGA1700，開發用足夠 |
| 主機板 | ASUS Prime B760M-A DDR5 | 需 PCIe x16（GPU）+ x4/x8（grabber） |
| RAM | Kingston Fury Beast DDR5-5600 32GB（2×16GB） | 非 ECC，開發夠用 |
| GPU | NVIDIA GeForce RTX 5080 16GB | 同生產機（算力需求） |
| SSD（系統） | WD Black SN770 1TB NVMe（M.2 2280） | OS + 應用程式 |
| SSD（存圖） | WD Black SN850X 4TB NVMe（M.2 2280） | 開發用不需 3 個月容量，4 TB ≈ 40 天 |
| PSU | Seasonic Focus GX-650（650W 80+ Gold） | — |
| OS | Windows 11 Pro x64 | .NET 4.8；Pro 版遠端桌面 |

### 工業 vs 消費等級關鍵差異

| 項目 | 消費級（開發用） | 工業級（產線用） | 影響 |
|------|--------|--------|------|
| SSD 耐寫（TBW） | ~2,400 TBW (M.2 4TB) | ~7,008 TBW (U.2 3.84TB) | 100 GB/日 → 消費級 ~66 年 vs 工業級 ~192 年（理論值） |
| RAM ECC | 無 | 有（檢測機） | 24/7 長時間運轉防止記憶體 bit flip |
| 主機板供貨 | 1–2 年 | 7 年以上 | 產線備品更換不斷料 |
| 工作溫度 | 0–35°C | 0–50°C（寬溫） | 工廠環境溫度較高 |
| OS 更新策略 | 頻繁功能更新 | LTSC 10 年只有安全更新 | 產線穩定性優先 |
| GPU | 消費級（RTX 5080） | 消費級（算力硬需求，A 系列不足） | 備品策略彌補壽命差距 |
| 儲存機 GPU | 不需要 | 不需要（內顯） | JPG Review 不用 CUDA（實測驗證） |
| UPS | 通常沒有 | 必要（檢測機） | 防異常斷電損壞 SSD |

### 實測資源用量（2 cam, RTX 5080, 16384×3000）

> 測試日期：2026-04-07，2 台相機實測，數據來自 resource-monitor CSV。

#### Grab 模式（即時取像存檔）

| 項目 | 實測值 | 說明 |
|------|--------|------|
| **RAM** | **2.85–2.91 GB** | 程序 WorkingSet；含 MIL + CUDA pinned + WinForms |
| **VRAM** | **2,983–2,996 MB（~3.0 GB）** | nvidia-smi 實測；含 MIL DMA + CUDA pipeline + 驅動 |
| **CPU%** | **2–5%** | 程序 CPU 佔用（全核平均） |
| **GPU Time** | **34–42 ms**（穩態） | 首幀 ~240 ms（CUDA JIT），之後穩定 |
| **SaveKB** | **~600 KB/幀** | JPG×3 + bin×4（排除 BMP） |
| **VRAM 基線** | **~1,280 MB（idle/Review）** | 顯示驅動 ~500–800 MB + CUDA context 殘留 ~400 MB |

> VRAM 分解：idle 基線 ~1.3 GB + MIL grabber DMA ~0.7 GB + CUDA pipeline ~1.0 GB ≈ 3.0 GB。
> VRAM 基線 ~1.3 GB 是常態，任何有 CUDA context 的程式都會佔用，不可避免。

#### Review 模式（讀圖）

| 模式 | RAM | VRAM | CPU% | Load Time | 說明 |
|------|-----|------|------|-----------|------|
| **Stitch**（合圖） | 915–1,011 MB | 1,279–1,284 MB | 2–6% | 469–827 ms | 垂直拼接多張 |
| **Global**（全域合圖） | 1,020–1,592 MB | 1,285–1,287 MB | 1–4% | 419–1,245 ms | 水平合圖＋垂直拼接 |

> Review 時 MIL grabber 已釋放，VRAM 降至 ~1.3 GB（純 CUDA context + 驅動）。
> RAM 峰值取決於拼接影像張數（60 張合圖 → 1.6 GB）。

#### 7 cam 推估

每台相機各自擁有獨立的 AoiService（CUDA pipeline）+ NativeBufferPool（pinned memory）+ MIL grab buffers，**VRAM 和 RAM 隨相機數線性增長**。

**Per-camera 資源佔用（16384×3000 = 47 MB/frame）：**

| 類型 | 項目 | 大小 |
|------|------|------|
| RAM (managed) | hostInputBuffer + hostOutputBuffer | 94 MB |
| RAM (CUDA pinned) | NativeBufferPool: input + mura + ridge + thumb + curves | ~142 MB |
| VRAM (MIL) | grab buffer ×2 + display + source | 188 MB |
| VRAM (CUDA) | AoiPipelineContext (4×W×H) + PICoaterDetector + workspace | ~660 MB |
| **合計 per-cam** | | **RAM ~236 MB, VRAM ~848 MB** |

> CUDA pipeline 的 VRAM 是 lazy allocate（首次 ProcessImage 才分配），反推：(3.0 GB - 1.3 GB baseline) / 2 cam ≈ **0.85 GB/cam**，與理論值 0.85 GB 吻合。

| 項目 | 2 cam 實測 | 7 cam 推估 | 推估依據 |
|------|-----------|-----------|---------|
| **RAM（Grab）** | 2.9 GB | **~4.1 GB** | 2.9 + 5 cam × 236 MB |
| **RAM（Review 峰值）** | 1.6 GB | **~3.5 GB** | 7 台拼接圖同時載入記憶體 |
| **VRAM（Grab）** | 3.0 GB | **~7.3 GB** | 1.3 baseline + 7 × 0.85 GB |
| **VRAM（Review）** | 1.3 GB | **~1.3 GB** | 不隨 cam 數變化（MIL 已釋放，CUDA 未初始化） |
| **CPU%** | 2–5% | **~10–15%** | MIL callback + JPEG encode + chart 更新 |
| **GPU Time** | 34–42 ms/幀 | **34–42 ms/幀** | 逐張處理，不隨 cam 數變化 |

> **重要**：7 cam Grab VRAM 推估 ~7.3 GB，RTX 5080 的 16 GB 仍有餘裕（~8.7 GB 剩餘），但 8 GB 顯卡**不可行**。

### 資源用量與規格對照

| 項目 | 峰值用量（7 cam 推估） | 最低規格餘裕 | 建議規格餘裕 |
|------|---------|-------------|-------------|
| RAM | ~4.1 GB（Grab）/ ~3.5 GB（Review） | 16 GB → 11 GB 可用 | 32 GB → 27 GB 可用 |
| GPU VRAM | **~7.3 GB（Grab）** / ~1.3 GB（Review） | ~~8 GB → 不可行~~ | 16 GB → 8.7 GB 可用（RTX 5080） |
| GPU 處理 | 34–42 ms/幀（實測 49MP） | RTX 5080 硬需求 | 記憶體頻寬為瓶頸，無法降級 |
| CPU | 10–15%（7 cam 推估） | 6C → 充裕 | 20T → 非常充裕 |
| 磁碟寫入 | ~100 GB/日（實測 JPG） | SATA SSD 夠用 | NVMe 更穩定 |
| 磁碟容量 | 3 TB/月 | 4 TB ≈ 40 天 | 10 TB ≈ 3 個月（檢測機） |

### 磁碟 I/O 效能分析

#### 檢測電腦寫入（U.2 NVMe SSD）

| 情境 | 寫入量 | 速度需求 | U.2 NVMe 能力 | 餘裕 |
|------|--------|---------|--------------|------|
| 日均寫入 | 100 GB / 86400 秒 | **1.2 MB/s** | 3,000+ MB/s | >2000× |
| 單次取像峰值（JPG） | 7 cam × 600 KB = 4.2 MB | **4.2 MB/s** | 3,000+ MB/s | >700× |
| 單次取像峰值（BMP） | 7 cam × 47 MB = 329 MB | **329 MB/s** | 3,000+ MB/s | ~9× |

> 磁碟寫入完全不是瓶頸。即使 SATA SSD（~500 MB/s）也足夠，NVMe 選用是為了耐久度（企業級 U.2）和穩定延遲，非速度需求。

#### 檢測電腦讀取（U.2 NVMe SSD — Review）

| 情境 | 讀取量 | 實測時間 | 說明 |
|------|--------|---------|------|
| Stitch（2 cam） | 20–60 張 JPG | 469–827 ms | 瓶頸在 JPEG decode（CPU bound），非磁碟 |
| Global（2 cam） | 同上 + 水平合圖 | 419–1,245 ms | 合圖用 GDI+ DrawImage，CPU bound |

> NVMe 隨機讀取 ~500K IOPS，70 個小檔案 < 1 ms，完全不是瓶頸。

#### 儲存電腦讀取（HDD — Review）⚠️

| 情境 | 讀取量 | HDD 效能 | 預估時間 |
|------|--------|---------|---------|
| 單次 Review（7 cam） | 70 個檔案 × 0.7 MB = 49 MB | 順序讀 ~200 MB/s | ~0.25 秒（最佳） |
| 小檔案隨機讀取 | 70 個檔案散佈不同目錄 | 隨機讀 ~1–2 MB/s（seek ~8 ms） | **~560 ms 尋道** |
| **合計（JPEG decode + HDD seek）** | | | **~1.5–2 秒** |

> 檢測電腦同樣操作 ~700 ms（NVMe），儲存電腦 HDD 約慢 2–3 倍。**可接受但不算快**。

#### 儲存電腦 HDD 讀取加速方案（選配）

| 方案 | 成本 | 效果 | 說明 |
|------|------|------|------|
| **SSD 讀取快取** | +$100 USD（M.2 1TB） | Review 回到 ~700 ms | 最近 1 個月資料放 SSD，舊資料在 HDD |
| **全 SSD** | +$300 USD（M.2 4TB） | 最快 | 但消費級 SSD 壽命短，不適合長期 |
| **維持 HDD** | $0 | ~1.5–2 秒/次 | 查詢頻率低的話可接受 |

> 建議先用 HDD 實際測試，若使用者反映查詢太慢再加 SSD 快取。

#### 儲存電腦寫入（HDD — 接收檢測資料）

| 情境 | 寫入量 | 速度需求 | HDD 能力 | 餘裕 |
|------|--------|---------|---------|------|
| 單次接收 | 50 MB | ~5 MB/s（10 秒間隔） | 順序寫 ~200 MB/s | ~40× |
| 日均接收 | 100 GB / 86400 秒 | 1.2 MB/s | 200 MB/s | >150× |

> HDD 順序寫入完全足夠。資料傳輸是批次順序寫入（非隨機），HDD 強項。

### 儲存容量估算

**生產情境（實測）**：每日 2000 次檢測 × 10 張/次 × 7 台，GrabHeight=3000，全年無休

| 項目 | 數值 |
|------|------|
| 單張存檔（JPG 原圖 + V/H mura + V/H bin） | ~0.7 MB |
| 單次檢測 | 0.7 MB × 7 cam × 10 張 = **~50 MB** |
| 每日 | 2000 次 × 50 MB = **~100 GB** |
| 每月 | 100 GB × 30 = **~3 TB** |
| 每年 | 3 TB × 12 = **~36 TB** |

#### 儲存架構：檢測機 + 儲存機

| | 檢測電腦（現場） | 儲存電腦（機房） |
|---|---|---|
| 用途 | 即時取像 + 檢測 + 存檔 | 歷史查詢 + 檢測報表 + 長期儲存 |
| 保留期間 | **3 個月** | **6 個月** |
| 所需容量 | 3 TB × 3 = **9 TB** | 3 TB × 6 = **18 TB** |
| 建議磁碟 | U.2 NVMe 3.84 TB × 3 | 企業 HDD 18TB × 2（RAID1 鏡像） |
| 傳輸方式 | 每次檢測完成即傳（~50 MB/次） | 被動接收 |
| 網路 | **1 Gbps**（Cat6A 佈線，預留 10G） | **1 Gbps** |

> **檢測電腦**需高速 NVMe（即時寫入），3.84 TB × 3 顆提供 ~11.5 TB 可用空間，覆蓋 3 個月 + 緩衝。
>
> **儲存電腦**不需高速 I/O，HDD RAID1 即可。每次檢測 50 MB，1 Gbps 網路傳輸 ~0.5 秒，檢測間隔 ≥10 秒，餘裕充足。
>
> **BMP 模式**（`SaveOriginalBmp = true`）：單張 ~47 MB，每日 ~6.6 TB，僅短期調參數用，不建議常開。

#### 儲存電腦建議規格

> 詳見上方「儲存電腦（低配 / 標配）」表格。
>
> 儲存電腦放機房有空調，不需工業等級 IPC。一般桌機或入門工作站即可。RAID1 比 UPS 更重要（資料冗餘）。程式啟動不需 NVIDIA GPU（已實作 graceful fallback，無 GPU 時跳過 CUDA 初始化，JPG Review 正常運作）。

### 元件壽命預估（基於實際產線負載）

**工作負載**：每日 2000 次檢測 × 10 張 × 7 台 = 140,000 幀/日。GPU 每幀 **34–42 ms**（實測 49 Mpixel），每日運算 ~1.4 小時（負載率 ~6%），24/7 通電不關機。磁碟寫入 ~100 GB/日。

#### GPU 壽命（GeForce RTX 5080）

RTX A 系列（A2000 等）算力不足（處理時間遠超 0.5 秒），因此生產機必須使用消費級 GeForce RTX 5080。以下為消費級 GPU 在 24/7 產線環境的壽命預估。

| 失效因素 | GeForce RTX 5080 | 說明 |
|---------|-------------------|------|
| 風扇軸承 | ~30,000 hr → **~3.4 年** | 24/7 通電風扇持續轉；消費級軸承壽命較短 |
| 電容老化 | 105°C 規格，~5 年 | 消費級用料，ESR 隨時間上升 |
| VRAM | 非 ECC，可能偶發 bit flip | 不會壞但可能造成檢測誤判 |
| 散熱膏 | ~2–3 年開始劣化 | 劣化後溫度升高加速其他老化 |
| Driver 支援 | ~5 年 | NVIDIA 消費級 driver 支援週期 |

| 情境 | 預期壽命 |
|------|---------|
| 樂觀（良好環境 + UPS） | 5 年 |
| **本工況（6% GPU + 24/7 通電）** | **3–4 年** |
| 悲觀（高溫 + 無 UPS） | 2 年 |
| 首個預期失效點 | 風扇軸承（~3 年） |

#### GPU 維護策略

| 項目 | 建議 |
|------|------|
| 備品 | **現場備一張相同型號 RTX 5080**，風扇異音時立即更換 |
| 監控 | 定期檢查 GPU 溫度，異常升溫代表散熱膏/風扇劣化 |
| 更換週期 | 建議 3 年預防性更換，舊卡轉為備品 |
| 年均成本 | RTX 5080 ~NT$35,000 ÷ 3 年 ≈ **~NT$11,700/年**（含備品攤提 ~NT$15,600/年） |

> 消費級 GPU 用於產線的代價是較短壽命和較高年均成本，但 RTX 5080 的算力是功能硬需求（處理時間 ~0.5 秒），無工業級替代方案。透過**備品 + 預防性更換**策略降低停機風險。

#### SSD 壽命（存圖碟，依 Flash 類型比較，實測 ~100 GB/日）

基準：3.84 TB 容量存圖碟，DWPD 為業界該 Flash 類型典型規格。

| Flash 類型 | 典型 DWPD | 額定 TBW | 代表型號（非韓系） |
|-----------|----------|---------|------------------|
| **QLC** | 0.3 | ~2,102 TB | Solidigm D5-P5336（U.2） |
| **TLC** | 1 | ~7,008 TB | Solidigm D7-P5520（U.2） |
| **pSLC** | 3 | ~21,024 TB | Kioxia FL6（U.2） |

> TBW 計算：3.84 TB × DWPD × 365 天 × 5 年保固

##### JPG 模式（~100 GB/日，日常生產，實測）

| | QLC (0.3 DWPD) | TLC (1 DWPD) | pSLC (3 DWPD) |
|---|---|---|---|
| 額定 TBW | 2,102 TB | 7,008 TB | 21,024 TB |
| 每年寫入 | 36.5 TB | 36.5 TB | 36.5 TB |
| **理論寫入壽命** | **~58 年** | **~192 年** | **~576 年** |
| 實際壽命瓶頸 | 控制器/電容 5–7 年 | 控制器/電容 7–10 年 | 控制器/電容 7–10 年 |
| PLP | 視型號 | 有（企業級標配） | 有 |

> JPG 模式下三種 Flash 的寫入壽命均遠超硬體老化壽命，**QLC 即足夠**。差異在 PLP 和穩定延遲。

##### BMP 模式（~6.6 TB/日，短期調參數用）

| | QLC (0.3 DWPD) | TLC (1 DWPD) | pSLC (3 DWPD) |
|---|---|---|---|
| 額定 TBW | 2,102 TB | 7,008 TB | 21,024 TB |
| 每年寫入 | 2,409 TB | 2,409 TB | 2,409 TB |
| **理論寫入壽命** | **~319 天** | **~2.9 年** | **~8.7 年** |

> BMP 模式每日 ~6.6 TB（1.7 DWPD），**QLC 不到一年即耗盡寫入壽命**。TLC 可撐 ~3 年但不建議長期。若需長期開啟 BMP 模式，建議使用 pSLC。正常使用下 BMP 僅短期開啟（數天），對 TLC 影響可忽略。

##### 結論

| 使用情境 | 建議 Flash 類型 | 原因 |
|---------|----------------|------|
| JPEG 日常生產（推薦） | **TLC**（1 DWPD） | 壽命遠超硬體老化，PLP + 穩定延遲；QLC 亦可但 PLP 較少見 |
| BMP 短期調參數（偶爾） | TLC 即可 | 短期數天使用，耗損可忽略 |
| BMP 長期開啟（特殊） | **pSLC**（3 DWPD） | 唯一能承受 5.7 DWPD 持續寫入的選擇 |

#### 其他元件

| 元件 | 預期壽命 | 維護建議 |
|------|---------|---------|
| CPU | 10 年+ | 無需維護 |
| RAM（ECC） | 10 年+ | 無需維護 |
| 主機板（工業級） | 7–10 年 | 注意電容膨脹、BIOS 電池（5 年更換） |
| PSU（工業級） | 5–7 年 | 風扇軸承為主要失效點 |
| UPS 電池 | 2–3 年 | **定期更換電池** |
| Grabber card | 10 年+（無主動散熱） | 被動元件，壽命長 |

> **注意**：以上為生產機建議。開發機因需同時跑 Visual Studio、測試、Claude Code 等，建議 64 GB RAM 以上。
