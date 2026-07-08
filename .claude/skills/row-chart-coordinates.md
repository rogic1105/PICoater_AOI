# 列圖表（chartLiveRow / chartReviewRow）座標設計說明書

> 2026-07-08 定版。改任何「上下方向 / 列 chart / 主畫面垂直座標」相關 code 前**必讀**。
> 歷史：此區多重座標轉換疊加曾造成「列 chart 上下顛倒」「拖曳反向」「排版毀損」多次事故。

## 鐵則（違反任一條＝直接退回）

1. **排版嚴禁變動**：XY 軸排版（InnerPlotPosition、軸位置、旋轉）當年調整很久才定案。
   任何需求都不准動排版屬性。
2. **`AxisY.IsReversed` 永久禁用**：它不只翻值方向，會連動重排軸標籤/InnerPlot 對齊 → 毀排版
   （2026-07-08 第三次驗證；前兩次見記憶 feedback_axis_direction）。方向一律用「資料映射＋視窗鏡射」。
3. **拖曳中 chart 連動不可節流/抑制**（踩過 3 次）：優化只能降單次重畫成本（降採樣），不能降頻。

## 座標定版規格（唯一真相）

**垂直物理座標：0 錨定「方向原點」**

| 上下方向設定 | 畫面 0 的位置 | 遞增方向 | 圖表 0 的位置 |
|---|---|---|---|
| 由上而下（TopToBottom） | 畫面**最上面** | 往下 | 圖表**頂端** |
| 由下而上（BottomToTop） | 畫面**最下面** | 往上 | 圖表**底端** |

主畫面 overlay 四邊值、**游標 Y**、chart 軸數字、chart 視窗——**全部同一套物理數字**（所見即所得，
框選範圍可直接比對）。

**座標約定與影像翻轉解耦**：live 影像不翻轉（設計如此）、回顧影像翻轉——但兩者的「座標約定」
都跟「上下方向」設定走。故 sdk `ImageDisplayView.VerticalZeroAtBottom`（座標）與 `FlipVertical`
（影像）是**兩個獨立旗標**，app 各自設定。

## 轉換架構（各只有一個轉換點，嚴禁再疊層）

```
ImageDisplayView.OnCanvasStatus / RefireViewRange   ← 顯示幾何 → 物理 的唯一轉換點
  由上而下：直通                                       （overlay/游標/ViewRangeMmChanged 同源）
  由下而上：phys = 總高(ContentH×sf×pitch) − v
        ↓ 物理座標
RowCurveDisplayAdapter（app policy）                 ← 方向旗標推送 + 視窗排序歸一（lo<hi）
  zeroAtTop = 瀑布(顯示順序資料) || 由上而下
        ↓
RowCurveChartHelper.PhysToChart（sdk）               ← 物理 → 圖表軸值 的唯一映射點
  chartVal = zeroAtTop ? total − phys : phys           （資料點與視窗共用同一函式，方向不可能分岔）
```

**反模式（全是踩過的雷）**：adapter 內 Array.Reverse 資料、adapter 內視窗鏡射、helper 內寫死
`(n-1-i)` 反向映射、IsReversed——四層各自轉一次疊出顛倒。現制：轉換只在上表兩點。

## 驗證（每次改完跑一輪，log 判準）

- `set:[顯示基線] 上下方向=… 主畫面=…`（開機一行）
- `IC/WF viewEdges X a~b｜Y c~d`（拖曳放開＝畫面四邊實際值）：由下而上時 **Y 下緣≈0/小、上緣大**
- `LC/RV row rowView view a~b → chart lo~hi`（chart 收到的視野與套用值；每秒一樣本）
- 眼睛：影像特徵與曲線峰對位、拖曳同向、瀑布照舊、切換方向後鏡像成立、**排版不動**

相關記憶：project_row_coordinate_design_spec / feedback_axis_direction / feedback_review_chart_live_during_drag
