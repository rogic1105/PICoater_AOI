# 遷移 Plan：回顧主畫面接 LiveDisplayView（與監控同源，絞殺榕收官）

> 狀態：**規劃完成、待執行**（分支 `refactor/review-same-source`）。
> 原則沿用 tanuki 遷移：**分階段 + 每階段 build 驗證 + checkpoint commit + 平行建新不拆舊**。

## 1. 目標
`camReviewMain`（回顧主畫面）改接 sdk `TanukiCv.Controls.LiveDisplayView`（監控已用），刪回顧自己的顯示方法
（`CanvasInteractionHelper` 顯示路徑 + `GrabImageStitcher.MergeHorizontal` 顯示用合圖）→ **兩畫面徹底同源**。
回顧直接繼承：動態 LOD（解掉「回顧 LOD 待辦」）、縮圖↔主畫面雙向連動（已在 sdk）、mm overlay、雙三擊、游標剖面。

## 2. 功能盤點（2026-06-12 完成；終局前提）
| 功能 | 結論 |
|---|---|
| zoom/pan/overlay/雙三擊/曲線連動/座標 | SmartCanvas+LiveDisplayView 內建 ✓ |
| 縮圖↔主畫面雙向連動 | ✅ 已移植 sdk（ThumbView 高亮+CenterOnCamera+SelectedCamChanged；框色 ThumbSelectedColor 各 UI 自選） |
| 動態 LOD | 監控有 → 回顧接入後白賺 |
| 視野保留（換 ID 保 zoom/pan） | LiveDisplayView 換幀天然不動視野；驗「影像尺寸變化」edge case；顯示順序需調 |
| NavigateCamera（上下台） | app 一行接 `SetSelected`+`CenterOnCamera` |
| CFG 座標基準（回顧用當時 ops/pos） | `SetLayout` 吃陣列 → app 餵 ReviewConfig 值，sdk 不改 |
| 時段合併（多 grab 垂直拼高圖） | app 先垂直拼好 → PushFrame 推高圖；**或刪除**（使用者測回顧 LOD 後決定） |
| 回顧強化（EnableReviewEnhance） | 內容層（載哪張圖），與畫布無關 |

## 3. 接點地圖（camReviewMain 引用 8 檔）
`AniloxRollForm.{cs,Review,Background,Data,DirectionStitch,Helpers}.cs` + `ReviewStitchCoordinator` + `DataStatisticsPresenter`。
顯示寫入主要走 `CanvasInteractionHelper.UpdateCanvas(Bitmap)`（FormInteractionHelper.OnGallerySelectionChanged 唯一直呼點）
與 ReviewStitchCoordinator 的合圖貼圖。

## 4. 關鍵設計決策（執行前確認）
1. **餵圖路徑**：LiveDisplayView 吃 8bpp 灰階 bytes；回顧載入是 Bitmap（JPEG 解碼）。
   選項 A=Bitmap→gray bytes（LockBits 一次性轉，換 ID 才發生、可接受）；
   選項 B=載入管線直接保留 gray bytes（較快但動載入層）。建議 A 起步。
2. **容器**：camReviewMain 現為 Designer 上的 SmartCanvas 控制項；LiveDisplayView 要 Panel 容器
   （建構時自建 SmartCanvas）。需 Designer 改 Panel（或用現有 parent panel）+ camReview1~7 縮圖 panel 傳入。
3. **單張模式語意差**：回顧單張=「載該相機full-res圖」（高解析）vs LiveDisplayView 單張=顯示該台最新幀。
   接入後：7 台圖都 PushFrame，單張=SetSelected 切換（圖已在記憶體）→ 換台變即時、不用重載。記憶體：7×full-res
   灰階 bytes（~16384×H×7）需評估（LOD 顯示便宜但快照仍駐留）。
4. **chart 取代案（使用者刪除清單）**：`chartLiveVertical`→`chartLivePatch`、`chartReviewVertical`→待確認
   （使用者寫 chartDataPatch，疑為 chartReviewPatch）。接入時視野連動接線一起改，省二次工。
5. **時段合併去留**：待使用者測回顧 LOD 後決定（刪→簡化；留→app 垂直拼好推高圖）。

## 5. 分階段
| 階段 | 內容 | 驗證 |
|---|---|---|
| 0 | 決策確認（§4）+ Designer 容器調整 | build |
| 1 | 平行建新：回顧側建 LiveDisplayView（旗標切換新舊路徑），PushFrame 餵載入圖 + SetLayout 餵 CFG | 新路徑顯示正常 |
| 2 | 功能接線：StitchMode/MergeAll/Flip/LOD/雙向連動/NavigateCamera/chart 視野連動（含 chart 取代案） | 上機逐項 |
| 3 | 時段合併處理（按 §4.5 決定） | 上機 |
| 4 | 刪舊：CanvasInteractionHelper 顯示路徑、GrabImageStitcher.MergeHorizontal 顯示用部分、舊接線 | 全方案 build + 上機回歸 |
| 5 | docs/skills 同步（CLAUDE.md 速查表、ui-flow.html） | /update-docs |

## 6. 風險
- 回顧的「強化重載」「背景預覽」也畫在 camReviewMain → 接入後改推幀（Background.cs 接點）。
- ReviewStitchCoordinator 的平行解碼計時 log（CSV/Stitch/Merge(bg)/UIapply）要保留量測語意。
- DataStatisticsPresenter 跨 tab 同步（chartDataPatch 對齊 chartReviewPatch）的座標假設要重驗。
