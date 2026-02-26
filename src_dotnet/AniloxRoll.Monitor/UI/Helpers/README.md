# Forms Helpers 分類

依角色將 `Forms/Helpers` 重新分為以下子資料夾：

- `Presenters/`
  - `AniloxRollPresenter.cs`
  - `ThumbnailGridPresenter.cs`
  - 負責 UI 與流程協調、縮圖選取狀態管理。

- `Navigators/`
  - `DateTimeNavigator.cs`
  - 負責年/月/日/時/分/秒連動與時間篩選導覽。

- `Managers/`
  - `LiveCameraManager.cs`
  - 負責即時相機硬體生命週期與抓圖顯示。

- `Helpers/`
  - `FormInteractionHelper.cs`
  - `MuraChartHelper.cs`
  - 負責表單互動流程與圖表顯示輔助。
