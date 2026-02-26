# Settings Module

此資料夾採用「集中化配置管理」，依領域拆成 Models / Services / Stores / Providers / Utilities / State / System。

## 檔案職責

- `Models/MachineLayoutConfig.cs`
  - 機台物理佈局：OPS 與相機起始位置。

- `Models/AcquisitionSettings.cs`
  - 取像設定：取像高度與曝光時間。

- `Models/InspectionRecipe.cs`
  - 檢測配方：Hessian 與誤差參數。

- `Models/StorageSettings.cs`
  - 儲存設定：截圖開關與路徑。

- `InspectionSettings.cs`
  - 聚合根模型（供 PropertyGrid 與流程使用），整合上述子模型並提供相容屬性。

- `System/SystemSettings.cs`、`System/CameraHardwareConfig.cs`
  - 系統層（硬體）設定：MIL System Descriptor、System Number、Device Number、DCF 路徑。
  - 提供集中管理的相機硬體配置來源。

- `Providers/InspectionSettingsDefaultsProvider.cs`、`Utilities/JsonConfigLoader.cs`
  - JSON 設定載入工具與預設參數提供者。
  - 將相機硬體參數與 PropertyGrid 預設值改為由 JSON 檔集中管理。

- `Stores/InspectionSettingsStore.cs`
  - 底層 JSON 序列化儲存層（統一 JSON，不再混用 XML）。
  - `Validate()` 校驗移至 Model，Store 專注讀寫。

- `Services/ConfigManager.cs`
  - 設定整合入口（載入/儲存 InspectionSettings 與 SystemSettings）。

- `State/UserSettingsService.cs`
  - Core 設定儲存（例如 `InspectionConfigJson`），不承載 UI Session 狀態。

- `UI/State/UserSessionState.cs`
  - UI 使用者操作狀態（最後資料夾、時間篩選、影像處理勾選狀態）。

## 設計重點

- **模型與儲存分離**：`InspectionSettings` 不直接碰 IO；由 Store/Service 處理。
- **容錯優先**：設定檔異常時回退到安全預設值，避免 UI/流程中斷。
- **單一入口**：外部透過 `InspectionSettingsStore` 與 `UserSettingsService` 操作設定，降低耦合。


## 三層配置建議

1. **SystemSettings（硬體/系統）**：相機與擷取卡拓樸。
2. **InspectionSettings（檢測/流程）**：演算法與操作參數。
3. **UserSessionState（UI AppState）**：UI 行為狀態與最近使用資訊。

## JSON 設定檔

- `Config/system-settings.json`：相機硬體參數（SystemSettings）。
- `Config/inspection-settings.defaults.json`：PropertyGrid 預設值（InspectionSettings defaults）。
