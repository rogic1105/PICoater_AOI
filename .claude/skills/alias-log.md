# alias-log

記錄使用者對 UI 控制項的稱呼，建立別名對照表。

## 使用時機

每次完成 UI 相關修改後，主動檢查對話中是否出現新的控制項稱呼。若有，執行此 skill 記錄。

## 執行步驟

1. **回顧對話** — 找出使用者用過的控制項描述詞（中文、口語、位置描述皆算）

2. **對應控制項** — 查 Designer.cs 或 code 確認對應的 Name

3. **建議標準名稱** — 告訴使用者這個控制項適合的簡短稱呼（中文），讓雙方有共同語言。格式：
   ```
   你說的「XXX」→ 控制項 `Name` → 建議稱呼「YYY」
   ```

4. **更新對照表** — 將新別名寫入 `CLAUDE.md` 的「控制項速查」區段：
   - 「你可能說的」欄位加入新的別名（用 / 分隔）
   - 若是全新控制項，新增一行

## 範例

對話中使用者說：「下面那個紅綠柱狀圖的年份切不過去」

```
你說的「紅綠柱狀圖」→ `chartDataYieldYearly/chartDataYieldMonthly/chartDataYieldDaily` → 建議稱呼「良率柱狀圖」
你說的「年份切不過去」→ `btnChartYearPrev/Next` + `cbDataYieldYear` → 建議稱呼「年份導航」
```

更新 CLAUDE.md：
```
| 良率柱狀圖 / 紅綠柱狀圖 / 年月日圖表 | chartDataYieldYearly/Monthly/Daily | Data tab |
```

## 原則

- 別名不限數量，越多越好找
- 位置描述也記（「Review 下面第二張圖」）
- 建議稱呼要簡短、明確、中文
- 不要刪舊別名，只增不減
