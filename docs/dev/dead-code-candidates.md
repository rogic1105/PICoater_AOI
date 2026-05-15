# Dead Code 偵測紀錄

生成日期：2026-05-15
用 `tests/python_test/find_dead_code.py` 掃出引用次數 ≤ 1（即只剩定義本身、沒有任何 caller）。

**清理結果**：
- 第一輪（commit 9dbac3e）：35 → 4 框架（保留）+ 31 刪 + 1 衍生（ReinitializeForAcquisitionSettings）
- 第二輪（commit code-review fixes）：又刪 2 個（SetExposureForAll / SetGrabHeightForAll，L1）

最後 find_dead_code 重掃只剩 4 個框架介面：
- BoolYesNoConverter.GetStandardValues（TypeConverter override）
- DcfFileEditor.GetEditStyle / EditValue（UITypeEditor override）
- LiveCameraManager.WheelZoomFilter.PreFilterMessage（IMessageFilter）

清理方式：功能導向、一次性。若有功能缺失再補回。

---

## A. 框架/介面實作（**不是 dead，保留**）

這些方法是 .NET / WinForms 介面要求的覆寫，由框架呼叫，grep 找不到 caller 屬正常。

| 檔案 / 方法 | 屬於介面 |
|---|---|
| `LiveCameraManager.cs:1283` `WheelZoomFilter.PreFilterMessage` | `IMessageFilter`（由 `Application.AddMessageFilter` 註冊）|
| `BoolYesNoConverter.cs:31` `GetStandardValues` | `BooleanConverter` override |
| `DcfFileEditor.cs:10` `GetEditStyle` | `UITypeEditor` override |
| `DcfFileEditor.cs:13` `EditValue` | `UITypeEditor` override |

---

## B. 高機率真 dead code（**建議刪除或評估**）

### B1. MIL 影像處理輔助（CUDA 遷移後殘留）

`ImageProcessing/MIL/CameraImageProcessor.cs` 整個檔案是 MIL（CPU）時代的處理 helper，現在 pipeline 已遷移到 CUDA（`Module_GetPICoaterBackground`）。

| 方法 | 用途 |
|---|---|
| L12 `ApplyColMeanSubtraction` | MIL 欄平均減背景 — CUDA 取代 |
| L90 `ApplyHessianVerticalFixed` | MIL Hessian — CUDA 取代 |
| L170 `ApplyBinarize` | MIL 二值化 — 未使用 |
| L176 `CopyImage` | MIL 拷貝 — 未使用 |

**建議**：整個檔案刪除（**外加** `Acquisition/AniloxCamera.cs` 內的 `EnableHessian` / `BinarizeThreshold` / `HessianSigma` / `HessianFixedMax` / `RidgeMode` 等 MIL pipeline 殘留設定）。

### B2. PlcService（過時的 PLC 介面）

`Services/PlcService.cs` 整個 class — 現在用的是 `Services/PlcGrabController` + `PlcBridge.Core.IModbusTcpClient`。

| 方法 | 狀態 |
|---|---|
| L10 `InitializeMock` | 未使用 |
| L34 `ReadBit` | 未使用 |
| L47 `WriteBit` | 未使用 |

**建議**：整個 `PlcService.cs` 刪除（也要刪除對應 `NativeMethods.PICoaterAPI_CreateMockPlc` 等 P/Invoke 宣告 — **但要先確認 native side 沒人用**）。

### B3. ImageRepository 6 層日期 cascade（已被 cbDate + cbTime 取代）

`ImageCatalog/ImageRepository.cs`:

| 方法 | 狀態 |
|---|---|
| L67-72 `GetYears` / `GetMonths` / `GetDays` / `GetHours` / `GetMinutes` / `GetSeconds` | 舊 6 層 cascade UI，已被 `GetDates` / `GetTimesForDate` 取代 |
| L94 `GetImagesByDateTime` | 同上，舊介面 |

**建議**：刪 6 個 cascade 方法 + `GetImagesByDateTime`。注意 `GetDates` / `GetTimesForDate` 仍要保留（新介面在用）。

### B4. 其他

| 檔案 / 方法 | 我的判斷 |
|---|---|
| `Services/InspectionStatisticsService.cs:495` `TryGetDateRange` | 未使用 helper，可刪 |
| `Settings/Providers/InspectionSettingsDefaultsProvider.cs:6` `LoadDefaults` | 整個 provider class 沒人用？整檔評估 |
| `Settings/Services/ConfigManager.cs:23` `LoadSystemSettings` | 整檔評估 — 可能整個 ConfigManager class 都廢了 |
| `UI/Widgets/GrabImageStitcher.cs:218` `LoadGdiBmpResized` | GDI fallback，CUDA 取代 |
| `UI/Widgets/GrabImageStitcher.cs:242` `ReencodeAsJpeg` | 未使用 |
| `UI/Widgets/RowCurveChartHelper.cs:35` `SetRowPitch` | 已被 `SetRowPitchFromSpeed` 取代？check |
| `UI/Widgets/ProportionalScaler.cs:55` `RegisterControl` | 公開但無 caller |

---

## C. 需要手動確認（**可能是 public API 或近期新增還沒接線**）

| 檔案 / 方法 | 風險 |
|---|---|
| `Acquisition/AniloxCamera.cs:765` `GetMemorySizeMB` | 可能是 debug log 用 — 需要 check Trace.WriteLine 等模糊呼叫 |
| `Acquisition/AniloxCamera.cs:844` `ApplyAcquisitionSettings` | 看起來像 public lifecycle method — 看是否從測試或外部呼叫 |
| `UI/Managers/LiveCameraManager.cs:332` `SetLiveDisplayDirection` | 公開 API，看是否從 Form 直接呼叫（grep 可能漏 lambda）|
| `UI/Managers/LiveCameraManager.cs:404` `SetLineRateForAll` | 公開 API，疑似 batch 操作但無 caller |
| `UI/Managers/LiveCameraManager.cs:481` `SwitchToCamera` | 公開 API，需要 check |
| `UI/Presenters/DataStatisticsPresenter.cs:424` `SyncReviewGrabIdFromData` | Public method，可能跨 tab sync 用但未接線 |
| `UI/Presenters/ThumbnailGridPresenter.cs:92` `GetCurrentSelectionData` | Public 但無 caller — 整個 presenter 是否還在用？|
| `UI/Widgets/FormInteractionHelper.cs:345` `CleanupSystem` | Public lifecycle，看 Form 是否該呼叫但漏了 |

---

## 建議刪除策略（**分批執行，每批 commit 一次方便回退**）

1. **第一批：低風險（B1 + B3）**
   - 刪 `ImageProcessing/MIL/CameraImageProcessor.cs` 整檔
   - 刪 `ImageRepository` 6 層 cascade
   - 刪 `AniloxCamera` 殘留 MIL 設定欄位（EnableHessian、BinarizeThreshold 等）
2. **第二批：中風險（B2 + B4）**
   - 評估 `PlcService.cs` 整檔（含 native side P/Invoke）
   - 刪其他 B4 個別方法
3. **第三批：C 類**
   - 逐項對 ui-flow.html 比對；ui-flow 沒提到 = 對應實作沒人用
   - 確認後刪或補上 caller / 文件

---

## 工具

- `tests/python_test/find_dead_code.py` — 自動掃描
- 跑法：`python tests/python_test/find_dead_code.py`
- 預設掃 `src_dotnet/AniloxRoll.Monitor/`，可傳路徑換目標
