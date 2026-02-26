# AniloxRoll.Monitor Core Modules

為了讓功能更容易定位，Core 依責任拆成以下模組：

## 1) Acquisition（擷取卡取圖）
路徑：`Core/Modules/Acquisition`

- `AniloxCamera.cs`：單一相機生命週期與取圖流程。
- `CameraSystemManager.cs`：MIL Application/System 配置與釋放。
- `Inspection/InspectionData.cs`：影像與運算結果資料模型。

## 2) ImageProcessing（影像處理）
路徑：`Core/Modules/ImageProcessing`

- `InspectionEngine.cs`：Native 引擎資源管理（建立、buffer、釋放）。
- `InspectionEngine.ImageProcessing.cs`：縮圖、全尺寸檢測、曲線輸出。
- `BatchInspectionService.cs`：多相機批次流程與參數更新。
- `InspectionEngineConfig.cs`：演算法預設參數。
- `MIL/CameraImageProcessor.cs`：MIL 專用影像前處理與轉換（與 Interop DLL 模組分離）。

## 3) ImageCatalog（圖片編號管理 / 索引）
路徑：`Core/Modules/ImageCatalog`

- `ImageRepository.cs`：掃描檔案、依時間/相機編號查圖。
- `ImageMetadata.cs`：影像檔名解析後的欄位。

## 4) Settings（圖片參數管理）
路徑：`Core/Modules/Settings`

- `InspectionSettings.cs`：UI/流程使用的設定模型。
- `InspectionSettingsStore.cs`：設定序列化與持久化。
- `UserSettingsService.cs`：存取使用者設定。

## 5) Interop（DLL 讀取）
路徑：`Core/Modules/Interop`

- `NativeMethods.cs`：`picoater_api.dll` P/Invoke 入口與簽名。

---

> UI 顯示相關（即時畫面、縮圖牆、表單互動）仍位於 `Forms/Helpers`，可作為下一步再拆分成 `Forms/Display` 與 `Forms/Workflow`。
