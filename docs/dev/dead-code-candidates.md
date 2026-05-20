# Dead Code 偵測工具

## 工具

`tests/python_test/find_dead_code.py` — 自動掃描 `private` / `internal` / `public` / `static` 方法引用次數。

```
python tests/python_test/find_dead_code.py [src_root]
```

- 預設 `src_root = src/dotnet/AniloxRoll.Monitor`
- 引用次數 ≤ 1（即只剩定義本身）= dead code 候選
- 排除：`InitializeComponent` / `Dispose` / `Main` / 各 WinForms framework hook / `Properties` 目錄

## 框架介面（**永遠是 refs=1，不是 dead，掃描時要忽略**）

| 檔案 / 方法 | 屬於介面 |
|---|---|
| `LiveCameraManager.WheelZoomFilter.PreFilterMessage` | `IMessageFilter`（由 `Application.AddMessageFilter` 註冊）|
| `BoolYesNoConverter.GetStandardValues` | `BooleanConverter` override |
| `DcfFileEditor.GetEditStyle` / `EditValue` | `UITypeEditor` override |

## 清理歷史

- **2026-05-15 第一輪**（commit `9dbac3e`）：35 個候選 → 4 框架保留 + 31 刪 + 1 衍生（`ReinitializeForAcquisitionSettings`）
- **2026-05-15 第二輪**（commit `2294621`）：補刪 `SetExposureForAll` / `SetGrabHeightForAll`

清理方式：功能導向、一次性。若有功能缺失再補回。
