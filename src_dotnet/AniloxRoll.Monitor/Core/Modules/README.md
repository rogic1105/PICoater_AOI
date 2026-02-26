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
- `Services/ConfigManager.cs`
- `InspectionSettingsStore.cs`
- `UserSettingsService.cs`
- `System/SystemSettings.cs`
- `System/CameraHardwareConfig.cs`
- `README.md`

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
