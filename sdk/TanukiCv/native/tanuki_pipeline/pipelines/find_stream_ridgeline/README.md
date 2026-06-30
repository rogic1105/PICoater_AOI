# find_stream_ridgeline

找「**流水圖**」的**脊線**（mura 檢測）。anilox 滾筒的 mura 缺陷在影像上呈現流水/條紋狀，缺陷處是脊線（ridge）。
本 pipeline 去掉條紋背景後，用脊線偵測把 mura 凸顯出來，並輸出欄/列曲線供判定。

## 流程
```
輸入灰階影像
   │
   ▼  background_sub        ── 去背（robust column 背景估計 + column 相減）
去背影像
   │
   ▼  ridge_hessian         ── 脊線（gaussian blur → hessian 響應 → 曲線 → scale）
脊線圖 + 欄/列曲線（mean/max）
```

## module（可抽換）
| 步驟 | module | 組合的 tanuki_core primitive |
|---|---|---|
| 去背 | `background_sub` | calcColumnMeans_RemoveOutliers + calcColumnBackground |
| 脊線 | `ridge_hessian` | gaussianBlur + computeHessianResponse + calcColumn/Row Means/Max + scale_clamp |

**換脊線方法**：`CreateFindStreamRidgeline("hessian")` → 之後可加 `"gabor"` 等（實作同角色 module，pipeline 結構不變）。

## 參數（Params）
| 欄位 | 意義 |
|---|---|
| `bg_sigma_factor` | 去背 robust 估計的離群門檻 |
| `ridge_sigma` | 脊線前的高斯模糊 sigma |
| `hessian_max_factor` | hessian 響應正規化（scale = 255 / factor） |
| `ridge_mode` | `vertical` / `horizontal` / `vertical+horizontal` |
| `precomputed_col_mean` | 非 null 時跳過每幀 column mean，直接用此背景 |

## 範例圖
> 待補：放 `docs/images/` —— 原圖 → 去背 → 脊線 三張對照（PNG）。

## 用法
```cpp
auto pipe = tanuki::pipeline::CreateFindStreamRidgeline("hessian");
pipe->Process(input, params, &output);   // output.ridge_data + 曲線
```
