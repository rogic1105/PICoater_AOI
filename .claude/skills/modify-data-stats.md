# modify-data-stats

修改 Data tab 統計、CSV、Period Charts、跨 Tab 同步相關程式碼。

## 使用時機

修改 DataStatisticsPresenter、InspectionStatisticsService、InspectionLogService、Period Charts 或跨 Tab 同步邏輯時。

## 關鍵檔案

- `UI/Presenters/DataStatisticsPresenter.cs` — Data tab 統計、combo 串聯、Period Charts、跨 Tab 同步
- `Services/InspectionStatisticsService.cs` — CSV 統計服務、LoadConfigForDate
- `Services/InspectionLogService.cs` — 每日 CSV 寫入、GrabId 格式
- `Services/CsvConfigSnapshot.cs` — 不可變設定快照

## 注意事項

### 統計模式（`_activeStatMode` 追蹤）
- **序號模式**：`ComputeByGrabIdRange` — 分母=唯一序號數，同序號同相機一票否決
- **時間模式**：找時間範圍內 GrabIds → 同樣用 `ComputeByGrabIdRange`
- Period Charts（`ScanCsvByDateRange`）同樣用 (GrabId, CamId) 一票否決

### CSV 格式
- 路徑：`{Root}\{yyyy}\{yyyyMM}\{yyyyMMdd}.csv`
- 欄位：`Id,FileName,MaxExceed,MeanExceed,MeanPeak,MaxPeak,...`
- GrabId：`yyMMdd-HHmmss`（時間戳，字典序=時間序）
- CamId 從 FileName 提取：`fileName.LastIndexOf('-')` 後的數字
- `#CFG` 行格式：`#CFG,ISO-timestamp,key=value,...`

### Period Charts
- StackedColumn 綁 `YAxisType.Secondary`（AxisY2 右側顯示 label）
- AxisY（Primary）驅動 MajorGrid（只顯示 0 和 niceMax）
- 無資料時 `InitOneChart` 預填 zero-value 資料點（月1-12/日1-31/時0-23），防止整個 chart 空白
- `FillPeriodChart` 先 clear 再填真實資料
- `niceMax = max(5, ceil(maxTotal/5)*5)`，AxisY 和 AxisY2 兩軸 Maximum/Interval 必須同步
- chart.Tag = `"auto"` 代表 AutoScale 模式，null = FixedScale

### 跨 Tab 同步
| 方向 | Guard |
|------|-------|
| Review → Data | `_grabIdCrossGuard` |
| Data → Review | `_grabIdCrossGuard` |
| 時間 → GrabId | `_grabIdNavGuard` |
| `_chartNavGuard` | chart 年月日 cascade 填充時 |

### PropertyGrid 持久化
- 新增 Category/屬性 → 必須同步更新 `InspectionSettingsStore` 的 `SerializeJson`+`ParseJson`
- 不使用 `JavaScriptSerializer`（`user.config` 損毀時會拋例外）

## 步驟

1. 讀取 DataStatisticsPresenter 中相關方法
2. 確認統計模式是否受影響
3. 修改 + build 驗證
