# AniloxRoll.Monitor 模組目錄（扁平化）

目前已改為以功能域扁平化整理，不再集中於 `Core/Modules` 或 `Forms`：

## 1) Acquisition（擷取）
路徑：`Acquisition`

- `AniloxCamera.cs`
- `CameraSystemManager.cs`
- `Inspection/InspectionData.cs`

## 2) ImageProcessing（影像處理）
路徑：`ImageProcessing`

- `InspectionEngine.cs`
- `InspectionEngine.ImageProcessing.cs`
- `BatchInspectionService.cs`
- `InspectionEngineConfig.cs`
- `MIL/CameraImageProcessor.cs`

## 3) ImageCatalog（影像索引）
路徑：`ImageCatalog`

- `ImageRepository.cs`
- `ImageMetadata.cs`

## 4) Settings（設定）
路徑：`Settings`

- `InspectionSettings.cs`
- `Models/*`
- `Providers/*`
- `Services/ConfigManager.cs`
- `Stores/InspectionSettingsStore.cs`
- `State/UserSettingsService.cs`
- `Utilities/*`
- `System/SystemSettings.cs`
- `System/CameraHardwareConfig.cs`
- `README.md`

> UI Session 狀態（LastDataPath / 時間篩選記憶）已移至 `UI/State/UserSessionState.cs`。監控強化已改為 InspectionSettings.EnableMuraEnhance。

## 5) UI（介面）
路徑：`UI`

- `Form/AniloxRollForm.cs`
- `Presenters/*`
- `Navigators/*`
- `Managers/*`
- `Widgets/*`

## 6) Interop（原生 DLL）
路徑：`Interop`

- `NativeMethods.cs`
