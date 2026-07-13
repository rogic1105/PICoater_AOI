# UI 目錄分類

`UI` 目錄已扁平化，避免再多一層 `Helpers`：

- `Form/`
  - `AniloxRollForm.cs`
  - `AniloxRollForm.Designer.cs`
  - `AniloxRollForm.resx`
  - 主畫面與 WinForms 資源。

- `Presenters/`
  - `AniloxRollPresenter.cs`

- `Navigators/`
  - `DateTimeNavigator.cs`

- `Managers/`
  - `LiveCameraManager.cs`

- `Coordinators/`
  - `ReviewFolderCoordinator.cs`：資料夾選擇、repository refresh、navigator 初始化。
  - `InspectionSettingsCoordinator.cs`：檢測設定套用到 pipeline。

- `Binders/`
  - `BusyUiBinder.cs`：回顧載入期間的等待游標與命令按鈕鎖定。
  - `GrabDetailListBinder.cs`：報表明細虛擬清單、繪圖、捲動與選取視覺；報表選取規則留在 presenter。

- `Services/`
  - `ImageCacheService.cs`：回顧處理影像的生命週期與釋放。

- `Widgets/`
  - `ColumnCurveChartHelper.cs`
  - `RowCurveChartHelper.cs`
  - `CurveMergeHelper.cs`

- `State/`
  - `UserSessionState.cs`
  - `ReviewRuntimeState.cs`
  - UI 使用者操作狀態（最後路徑、上次選項、時間篩選記憶）。
