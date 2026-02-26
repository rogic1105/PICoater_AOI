# Settings Module

此資料夾負責 **使用者設定與檢測參數** 的定義、序列化與持久化。

## 檔案職責

- `InspectionSettings.cs`
  - 定義可由 UI (`PropertyGrid`) 編輯的檢測/相機參數模型。
  - 包含相機 OPS、相機位置、演算法參數、取像參數與截圖設定。

- `InspectionSettingsStore.cs`
  - 負責 `InspectionSettings` 的 XML 序列化/反序列化。
  - 對外提供 `Load()` / `Save()`，並在讀取時補齊安全預設值。

- `UserSettingsService.cs`
  - 封裝 `Properties.Settings` 的存取。
  - 集中處理設定檔損毀復原（刪除壞檔 + `Reset()`）與例外保護。

## 設計重點

- **模型與儲存分離**：`InspectionSettings` 不直接碰 IO；由 Store/Service 處理。
- **容錯優先**：設定檔異常時回退到安全預設值，避免 UI/流程中斷。
- **單一入口**：外部透過 `InspectionSettingsStore` 與 `UserSettingsService` 操作設定，降低耦合。
