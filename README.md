# PICoater AOI Solution

PI Coater 線上自動光學檢測系統（Automated Optical Inspection），用於 Anilox Roll 塗佈品質即時監控與離線分析。

## 1. 專案架構概覽 (Repository Structure)

```text
PICoater_AOI/
├── bin/                 # 編譯輸出目錄 (所有的 .exe, .dll, .lib 都會產生於此)
├── build/               # 編譯中間檔案 (Intermediate files, obj)
├── sdk/                 # [核心] AOI_SDK 影像處理引擎
│   ├── core_cv          # CUDA 底層運算庫
│   ├── framework        # 測試框架
│   ├── cpp_utils        # C++ 工具庫
│   └── src/dotnet/
│       └── AOI.SDK/     # .NET SDK（SmartCanvas 等）
├── src/native/          # [演算法] 專案特定的 C++ 模組
│   ├── modules/         # 各式檢測功能模組 (如 GetPICoaterBackground)
│   └── c_api/           # 導出給 C# 使用的 DLL 介面層
├── src/dotnet/          # [介面] C# 使用者介面
│   ├── AniloxRoll.Monitor/ # 主程式 (WinForms)
│   └── IoBridge/       # IO Modbus TCP 通訊 (Core / ManualControl / Automation)
├── tests/               # [測試] C++ / C# 單元測試與壓力測試
│   ├── cpp_test         # (picoater_tests)
│   └── dotnet_test/AniloxRoll.Monitor.Tests/ # NUnit 3.x + Moq 4.x
├── docs/                # 架構與模式文件
│   ├── sample/          # 示範程式（AOI.SDK.TestApp — 不參與主 build，未來會分離回 AOI_SDK repo）
│   ├── config/          # config / dcf 範例
│   ├── dev/             # 廠商規格書、CLProtocol 範例、Grabber / 光源控制器手冊
│   └── user-manual/     # 操作員說明書 + IO 圖 + 硬體規格
├── TestRunner/          # 測試啟動器（雙擊 TestRunner.bat 執行）
└── Directory.Build.props # 全域 MSBuild 設定檔
```

## 2. 功能概述

- **即時監控**（Live View）：7 台線掃相機同步取像，MIL 顯示 + GPU Mura 曲線即時更新
- **影像回顧**（Review）：離線瀏覽歷史影像，支援原圖/V強化圖/H強化圖切換，多張拼接模式
- **檢測數據**（Data）：良率統計（序號/時間模式）、逐序號明細、年月日趨勢圖

## 3. 技術棧

| 層 | 技術 |
|------|------|
| UI | C# WinForms (.NET Framework 4.8) |
| 相機 | Matrox MIL 10.x（Camera Link + CLProtocol） |
| GPU | CUDA（Hessian Ridge Detection、Background Removal、StandardBgSub） |
| 存檔 | JPEG + .bin 曲線（GPU resize 1/5x），可選 BMP 原圖 |

## 4. 系統需求 (Prerequisites)

* **IDE**: Visual Studio 2022 (Community/Pro/Enterprise)
* **Workloads**:
  * 使用 C++ 的桌面開發 (Desktop development with C++)
  * .NET 桌面開發 (.NET desktop development)
* **Framework**: .NET Framework 4.8
* **GPU Toolkit**: NVIDIA CUDA Toolkit 12.8 (必須啟用 Visual Studio 整合)
* **Platform**: Windows x64

## 5. 建置指南 (Build Instructions)

本專案使用 `Directory.Build.props` 統一管理路徑，不需要手動設定 Include/Library 路徑。

1. 使用 Visual Studio 2022 開啟根目錄下的 **`PICoater_AOI.sln`**。
2. 組態設定：**Configuration**: `Release` / **Platform**: `x64`
3. 在方案總管中的「方案」上點擊右鍵 -> **建置方案**。
4. 建置成功後，執行檔位於 `bin/x64/Release/`。

## 6. 執行與測試 (Running & Testing)

### C++ 模組測試 (底層驗證)
* **專案**: `picoater_tests`
* 驗證 CUDA 演算法是否正確，不涉及 GUI。

### C# 單元 + 壓力測試
* **專案**: `AniloxRoll.Monitor.Tests`（NUnit 3.x + Moq 4.x）
* 40 個單元測試 + 6 個壓力測試（IO FSM、CSV、Settings 讀寫）

#### 使用 TestRunner（推薦）

雙擊 **`TestRunner/TestRunner.bat`** 即可選擇測試模式：

```
 1. Unit tests only    (~2 sec)     ← 快速單元測試
 2. Stress tests only               ← 僅壓力測試
 3. All tests                        ← 全部（先 unit 再 stress，不交錯）
 4. Exit
```

選擇壓力測試後會詢問要跑幾分鐘（預設 60 分鐘），例如：
- `1` = 快速驗證（~1 分鐘）
- `60` = 標準壓力（~1 小時）
- `1440` = 整天跑（~24 小時）

| 檔案 | 用途 |
|------|------|
| `TestRunner/TestRunner.bat` | 測試啟動器（雙擊執行） |
| `TestRunner/TestRunner.ps1` | PowerShell 輔助腳本（bat 內部呼叫，處理 UTF-8 log） |
| `TestRunner/TestRunner.log` | 測試結果 log（自動產生，已加入 .gitignore） |

壓力測試執行時會即時顯示每 10% 的進度：
```
[11:07:13] ▶ SettingsRW  cycles=2,500  est=10s
  [11:07:14]   SettingsRW  10%  (250/2,500)
  [11:07:14]   SettingsRW  20%  (500/2,500)
  ...
[11:07:20] ✔ SettingsRW  elapsed=6.6s
```

#### 使用 CLI

```bash
# 快速單元測試（~2 秒）
dotnet test tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj -p:Configuration=Release --filter "TestCategory!=Stress"

# 壓力測試（自訂分鐘數，透過環境變數 STRESS_MINUTES）
set STRESS_MINUTES=60
dotnet test tests/dotnet_test/AniloxRoll.Monitor.Tests/AniloxRoll.Monitor.Tests.csproj -p:Configuration=Release --filter "TestCategory=Stress"
```

### Mock Data 產生器

`tests/python_test/generate_mock_captures.py` — 從原始 BMP 影像產生 AniloxRoll.Monitor 可讀取的存檔格式（JPG + .bin 曲線 + CSV）。

```bash
cd tests/python_test
python generate_mock_captures.py <input_dir> <output_dir>
```

#### 支援兩種 Input 格式

**1. 扁平模式**（推薦）— 檔名含 camId，每張獨立 GrabId：
```
input_dir/
├── 20251117_111952.447-1.bmp    ← CAM1
├── 20251117_111952.447-2.bmp    ← CAM2
├── 20251117_111952.447-3.bmp    ← CAM3
├── 20251117_111958.759-1.bmp
├── ...
```
```bash
python generate_mock_captures.py "C:/Users/User/Downloads/mura" "D:/AniloxCaptures"
```

**2. CAM 子目錄模式** — 所有相機視為同一序號（共用 GrabId），可垂直合圖：
```
input_dir/
├── CAM1/
│   ├── 20251117_111919.181.bmp
│   ├── 20251117_111925.929.bmp
│   └── 20251117_111932.600.bmp
├── CAM2/
│   └── ...
└── CAM3/
    └── ...
```
```bash
python generate_mock_captures.py "D:/MockData" "D:/AniloxCaptures"
```
GrabId 取自 CAM1 的最早時間戳（模擬單次 DO_PC_BUSY 觸發）。

#### 每張 BMP 的產出

| 檔案 | 說明 |
|------|------|
| `{ts}-{camId}_raw.jpg` | 原圖縮圖（1/5，JPEG Q90） |
| `{ts}-{camId}_proc_v.jpg` | V 方向 Hessian Ridge 處理圖 |
| `{ts}-{camId}_proc_h.jpg` | H 方向 Hessian Ridge 處理圖 |
| `{ts}-{camId}_mean_v.bin` | V 方向 Column Mean 曲線（MCBF） |
| `{ts}-{camId}_max_v.bin` | V 方向 Column Max 曲線 |
| `{ts}-{camId}_mean_h.bin` | H 方向 Row Mean 曲線 |
| `{ts}-{camId}_max_h.bin` | H 方向 Row Max 曲線 |
| `{yyyyMMdd}.csv` | 每日檢測紀錄（含 #CFG 設定行） |

#### 相依套件

```bash
pip install numpy opencv-python
```

### C# GUI (主程式)
* **專案**: `AniloxRoll.Monitor`
* 程式啟動時自動載入 `picoater_api.dll` 及 `core_cv_api.dll`。

## 7. 開發文件

詳見 `CLAUDE.md`（Claude Code 規則 + 文件路由）和 `docs/` 目錄：

| 文件 | 內容 |
|------|------|
| `docs/architecture-ui.md` | UI 架構、控制項觸發關係 |
| `docs/architecture-image-pipeline.md` | GPU pipeline、存檔格式 |
| `docs/architecture-acquisition.md` | MIL 取像模組、IO FSM（IoState enum、IO 快照、Watchdog） |
| `docs/user-manual/io_diagrams.html` | IO 狀態機視覺化（State Machine / SFC / Ladder / Timing） |
| `docs/dev/MIL_API_Reference.md` | MIL API 參考 |
| `.claude/skills/*.md` | 開發模式（依修改範圍觸發對應 skill — 詳見 CLAUDE.md §Skills 路由）|
| `CLAUDE.md` | Claude Code 規則 + 架構原則 + 控制項速查 + Skills 路由 |

## 8. 開發規範 (Development Guide)

### 加入新的 C++ 演算法
1. 在 `src/native/modules` 下建立新的專案 (Static Library)。
2. 實作 `.hpp` 與 `.cu`（參考 `Module_GetPICoaterBackground`）。
3. 在 `Directory.Build.props` 定義新模組的路徑變數。
4. **重要**: 透過**專案參考**加入模組，切勿手動加入 `.lib`。

### 解決連結錯誤 (Troubleshooting)
* **LNK2001 / LNK1181**: 檢查 References 是否勾選相依專案。若依賴 CUDA Code，將「使用程式庫相依性輸入」設為 True。

### Coding Style
* **C++**: 遵循 Google C++ Style Guide。
* **路徑**: 所有專案均繼承 `Directory.Build.props`，請勿寫死絕對路徑。

---

### 聯絡資訊

* **Maintainer**: Chunkuan
* **Department**: AUO / PICoater Project Team
