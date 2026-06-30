# 改 grab 高度／相機參數造成相機 stall — 根因、修法、硬體上限

> 適用：Matrox **Radient eV-CL**（板載 1GB DDR3、4 acquisition path）+ Teledyne DALSA **Linea** 線掃描相機（寬 16384、8-bit mono），MIL .NET 連續取像（`MdigProcess`）。
> 本文是 2026-06 整段調查的**統整定稿**（取代舊的 `grabheight-max-buffer-stage2.md` 分層歷史筆記）。

---

## 0. TL;DR（先記這幾條）

- **改 grab 高度/線掃/曝光時相機 stall 有「兩個並列主因」，要同時處理**：
  - **① 軟體競態**：多餘 realloc / 熱路徑 inquire 撞 CLProtocol → CAM1 stall（即使高度正常也會中）。
  - **② per-camera 硬體高度上限（~12062，每台浮動）**：高度逼近上限那台會 stall（即使競態修好也會中）。
  - 缺一不可：只修競態、高度還是會在上限附近 stall；只 cap 高度、競態還是會在正常高度 stall。
- **三條鐵則**（違反就會 stall）：
  1. `SetGrabHeight` 開頭要有**同值守門**：高度沒變 + buffer 已配 → 直接 return，不 realloc。
  2. 改尺寸/grab 熱路徑、相機 MIL init 序列裡**絕不可呼 `MsysInquire`/`MdigInquire`**（會讓 CAM1 stall）。
  3. 改尺寸前要 `M_STOP+M_WAIT` 再 `MdigControl(M_GRAB_ABORT)` drain；**絕不可寫相機 GenICam `Height` feature**。
- **高度硬上限**：`AcquisitionDefaults.MaxGrabHeightPx`（固定值、**不分台數**）。實測單台「grab 中往上拉」約 **12062** 會 stall（每台略有浮動），cap 取其下並**留足餘裕**。
- **stall 偵測用「幀數有沒有前進」**（`M_PROCESS_FRAME_COUNT`），**不是 FPS 門檻**（低線掃合法 FPS 極低，固定門檻會誤判）。
- 難重現的 stall：**先看 `D:\Anilox\Logs` 的 `trace-*.log` + `dropdiag-*.csv` 再下結論**，別憑現象推理論。

---

## 1. 症狀

- 某台（常是 CAM1）`M_PROCESS_FRAME_COUNT` 凍在固定值、FPS=0、畫面不動；**同板其他台正常**。
- 停/開（含「停止抓取→開始抓取」）救不回 → 硬體層 CL 失鎖，**只有重開程式**（無 `MdigReset` API）。
- 穩態不動參數＝0 stall；改參數（尤其線掃、其次高度）才觸發。

---

## 2. 真正的根因（**兩個並列主因**，缺一不可）

### 2-1. 主因一：啟動／套設定競態
套用設定時會對每台呼一次 `SetGrabHeight(同值)`。若不擋，會做**多餘的 free+realloc（UI 執行緒）**，
正好撞上該台 **CLProtocol enable（背景執行緒，2~5s）** → MIL 內部並發 → **CAM1 stall（即使高度正常也會中）**。

> 證據（trace log）：`[CAM1] CLProtocol: using device ID` 與 `[CAM1] CLProtocol enabled successfully` 之間，
> 插進了 `[CAM1][HtRealloc] 改高度 12000->12000`。

**修法**：`MilCamera.SetGrabHeight` 開頭 —— 高度未變且 `_milGrabBuffers[0]!=M_NULL` → 直接 return（log `跳過 realloc`）。

**同類陷阱（會加重競態）**：為查板載而在改高度/init 熱路徑呼 `MsysInquire`（`GetMemoryFreeMB`）＝把 inquire 插進相機 MIL 序列 → **CAM1 stall**（既有鐵則，這次診斷碼自己踩到）。→ 熱路徑**禁所有 MsysInquire/MdigInquire**；板載記憶體只背景 telemetry 查（寫 `resource-monitor-*.csv`）。

### 2-2. 主因二：per-camera 硬體高度上限（~12062，每台浮動）
「grab 中把單台高度往上拉」的上限約 **12062**（每次 12062~12065 浮動）；且**每台相機之間也有差異**
（實測 CAM2 在 12000 正常、CAM1 在 12000 stall）。**這不是競態，是板載/相機硬體上限的個體差異 —— 即使競態修好，高度逼近上限那台還是會 stall。**

**修法**：`MaxGrabHeightPx` cap 訂在最弱那台的浮動天花板**之下、留足餘裕**（例：需到高處用 11000；日常用不到就訂更低如 8000~10000，極限根本碰不到）。

> ⚠ 兩個主因**缺一不可**：只修 2-1，高度逼近上限還是 stall（2-2）；只 cap 2-2，正常高度套設定還是 stall（2-1）。

---

## 3. 硬體記憶體模型（Matrox 官方文件佐證）

| 事實 | 官方來源 |
|---|---|
| grab buffer **預設配在 Host 非分頁記憶體**，不是板載 | `UserGuide/data-buffers/Attribute.htm`、`BoardSpecificNotes/radient_ev-cl/Minimum_latency_and_grabbing_all_frames.htm` |
| 板載 1GB 當 **PCIe latency 緩衝** + 每 path 的 **temporary buffer（依 frame size 在 `MdigAlloc` 時預留）** | `Grabbing_large_images.htm`、`Minimum_latency…` |
| → 板載占用＝**所有「已設定」digitizer 的 frame size 總和**，與實際插幾台無關（拔相機不變） | 同上（temporary buffer 在 MdigAlloc 預留） |
| 「grab 中拉大」上限（~12062）< 「初配」上限（~14303）：連續取像時 on-board 還要兼 latency 緩衝，剩給放大單幀的空間較少 | `Minimum_latency…`（文件支持的推論，非單句明文） |
| **單 path 上限無法用旋鈕提高**（無關 FIFO / M_DEGRADED 之類開關） | 全文件未找到此開關 |
| 要更大影像的官方正路＝**Host 大 buffer + child-buffer 分段擷取**（每段 ≤ DCF frame size），非把單幀撐大 | `Grabbing_large_images.htm` |
| 連續取像中**不建議改 digitizer 設定**；改只影響「下一次 grab」 | `Reference/dig/MdigControl.htm`、`MdigProcess.htm` |

> 板載總量 `MsysInquire(M_MEMORY_SIZE)=1024`（硬體固定，不可軟體調）。

---

## 4. SetGrabHeight 正確步驟（`MilCamera.Params.cs`）

```
同值守門（高度未變+buffer已配 → return）
 → M_STOP + M_WAIT（drain：等佇列 grab 全跑完）
 → MdigControl(M_GRAB_ABORT)（立即中止 in-flight+佇列；guard try/catch）
 → FreeGrabBuffers
 → MdigControl(M_SOURCE_SIZE_Y, h)   ← 純 digitizer 端切幀；**絕不可寫相機 Height feature**（會搞壞兩台 + FPS 算錯）
 → MbufAlloc2d × N（檢查 M_NULL＝配置失敗）
 → MdispSelectWindow → settle → M_START
失敗 rollback：FreeGrabBuffers → AllocateAndBind(oldHeight)
```

---

## 5. stall 偵測（`LiveCameraManager.Telemetry.cs`）

- 判據＝**`M_PROCESS_FRAME_COUNT`（`GetFrameCount`）有沒有前進**，**不是 FPS 門檻**。
  - 真 stall：幀數凍住不動。慢速 grab：幀數仍慢慢加（不誤判）。
  - 幀數變化（含重啟歸零＝減少）一律視為「活著」→ 重置計數。
- **偵測窗自動拉長**：`needed = 基準(2s) + ⌈預期幀週期 × 1.5⌉`，預期幀週期 = `FrameHeight / AppliedLineRateHz`。
  - 例：100Hz/12000＝0.0083fps（一幀 120s）→ 窗自動 ~182s，**合法慢速不誤判**；真卡死仍會偵到（只是慢）。
  - 例：10000Hz/12000＝0.83fps → 窗 ~4s，快速偵到。
- 偵到只用縮圖紅「STALL」標示（停/開救不回，不做無效 thrash）。

---

## 6. 診斷工具（`D:\Anilox\Logs\`）

| 檔案 | 內容 | 設定點 |
|---|---|---|
| `trace-*.log` | 所有 `Trace.WriteLine`（`[HtRealloc]`/`[LineRate]`/`[Exposure]`/CLProtocol…），AutoFlush（stall/hang 也已落地） | `Program.cs`（`TextWriterTraceListener`） |
| `dropdiag-*.csv` | 每 500ms 每台 `fps,lineRateHz,frameH,frames,procMissed,grabMissed` | `LiveTelemetryPresenter.DropDiagLogPath` |
| `paramchange-*.csv` | 每次改參數 `time,scope,cam,param,value` | `AniloxRollForm.ParamChangeLogPath` |
| `phaselog-*.csv` | 每幀硬體 frame-start tick（多相機相位/掉幀位置） | Data Latch |
| `resource-monitor-*.csv` | CPU/RAM/VRAM/**板載記憶體**（背景執行緒安全查，非熱路徑） | ResourceLog |

**判讀 SOP**：`dropdiag` 看「哪台 frames 凍住、當下 lineRate/height」→ 對 `trace`/`paramchange` 看「凍住前改了什麼」。

---

## 7. 試過但「棄用」的路（別再走）

| 方案 | 為何棄用 |
|---|---|
| **max-buffer**（一次配 max 高度 buffer、改高度只改 `M_SOURCE_SIZE_Y` 不 realloc） | 7 台 host ~3.6GB 逼爆非分頁池；且**兩個主因都沒解**（competition 與 per-camera 上限都不是 realloc 造成）。`MilCamera.UseMaxHeightBuffers` flag/scaffold 留著當紀錄、預設 false |
| **auto-allocate**（`MdigProcess` bufarray=M_NULL） | 官方確認對 on-board 占用沒幫助；**兩個主因都沒解** |
| **per-board 高度公式**（`板載 ÷ 同板台數 ÷ 每行成本`，曾算板0 6963/板1 9284） | 被競態現象誤導 + 診斷污染導出的錯誤模型；實機證明同板可 4×12062，**不該按台數減**。已移除 |

---

## 8. 實測數據（原始記錄，如實保留）

> 即使部分當時的「解讀/係數」後來被修正，**底下的量測數字都原樣保留**（數字是真的，有重測對照價值）。
> 機型：Radient eV-CL QB（板載 1GB、4 path）+ Linea LA-CM-16K05A（寬 16384）。

### 8-1. stall 高度邊界
| 情境 | 結果 |
|---|---|
| 開機初配，單台高度 | **14303 正常；14304+ 一開機就 stall**（`M_PROCESS_FRAME_COUNT` 凍 0） |
| grab 中 realloc 拉高（多台在線） | **~12063 開始 stall**（每次 12062~12065 浮動） |
| grab 中 max-buffer no-realloc（buffer 固定 14000、板0 4 digitizer） | source 8650 正常、10800 正常、**14000 stall** |
| 高度 12000（板0 4 digitizer 配置、2 台實體在線） | **CAM2 正常、CAM1 stall**（per-camera 差異） |

### 8-2. 板載記憶體 vs 高度
| 日期/條件 | 量測 |
|---|---|
| 6/22 兩台在線 | 高度 1893 → 板載剩 779MB；高度 10000 → 剩 395MB |
| 早期推算 | 10000 × 4 台 → 板載用 944MB |
| 6/24 兩台受控（buffer==source realloc） | 3000 → 用 192MB；9220 → 用 581MB |
| 6/24 改高度到 12000（4 digitizer 配置） | `allocFail=False`、板載剩 **647MB**（buffer 配得出來＝**非記憶體不足**） |

**每台每行成本（反推，未強行統一）**：早期 ~**0.0236** MB/行/台；6/24 受控 ~**0.03125** MB/行/台（=2 grab buffer×寬16384÷1MB）。
兩者不一致（量測模式/條件不同），**如實並列**。板載總量 `M_MEMORY_SIZE=1024`MB（硬體固定）。

### 8-3. dropdiag 證據（每 500ms，`time,cam,fps,lineRateHz,frameH,frames,procMissed,grabMissed`）
高度 12000 時：
```
14:35:51.434,1,0.00,10000.0,12000,0,0,0    ← CAM1 frames 凍 0
14:35:51.434,2,0.83,10000.0,12000,6,0,0    ← CAM2 frames 遞增、FPS 0.83(=10000/12000 正確)
...（CAM1 持續 0；CAM2 6→7→8→9→10→11）
```

### 8-4. 相機 Height feature（唯讀查，CLProtocol 就緒後）
`MdigInquireFeature("Height")`：Min=0、Max=4294967295、Increment=1 → line-scan **無格點限制**（高度合法性非此因）。

> 解讀演進（保留當推理過程）：6/18 以為純「M_STOP 沒 drain」；6/22~6/23 以為「板載溢位/realloc 累積」並推 per-board 係數公式（板0 6963/板1 9284）；6/24 dropdiag 揪出 CAM2 在 12000 正常 → 推翻「板載是唯一因」，定為**競態 + per-camera 上限兩個主因**。數字未動，只動解讀。

---

## 9. 換相機／換 grabber 時要重測什麼

- **`MaxGrabHeightPx`**：grab 中把單台高度從小往大拉，找「剛好開始 stall」的值，保守取整 + 留餘裕。
- 板載記憶體模型（若換板）：看 `resource-monitor-*.csv` 的板載占用隨高度/台數變化。
- 相機 source size 合法格點：`MdigInquireFeature(M_FEATURE_MIN/MAX/INCREMENT, "Height")`（本機 line-scan 回 0/4294967295/1＝無格點限制）。

---

## 相關程式碼

| 路徑 | 角色 |
|---|---|
| `sdk/MIL/MilGrabber.Core/MilCamera.Params.cs` | `SetGrabHeight`（同值守門 + drain + realloc）、`SetLineRateHz`/`SetExposureUs`（診斷 log） |
| `sdk/MIL/MilGrabber.Core/MilCamera.CLProtocol.cs` | CLProtocol 背景啟用（競態的另一方） |
| `sdk/MIL/MilGrabber.Core/MilCamera.Telemetry.cs` | `GetFrameCount`（stall 判據）、`GetMemoryFreeMB`（**只背景用**） |
| `src/dotnet/AniloxRoll.Monitor/UI/Managers/LiveCameraManager.Telemetry.cs` | stall 偵測（幀數前進 + 自動窗） |
| `src/dotnet/AniloxRoll.Monitor/Settings/Models/Defaults/AcquisitionDefaults.cs` | `MaxGrabHeightPx` 唯一來源 |
| `src/dotnet/AniloxRoll.Monitor/Program.cs` | 檔案 trace listener |
