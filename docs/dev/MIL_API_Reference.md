# MIL .NET API Quick Reference

> **DLL**: `Matrox.MatroxImagingLibrary.dll` v10.70.963.1085
> **Namespace**: `Matrox.MatroxImagingLibrary`
> **路徑**: `C:\Program Files\Matrox Imaging\MIL\MIL.NET\`
> **Runtime**: .NET Framework 4.0 (AMD64)

---

## 目錄

- [1. 常用常數速查](#1-常用常數速查)
- [2. Application (Mapp)](#2-application-mapp)
- [3. System (Msys)](#3-system-msys)
- [4. Digitizer (Mdig)](#4-digitizer-mdig)
- [5. Display (Mdisp)](#5-display-mdisp)
- [6. Buffer (Mbuf)](#6-buffer-mbuf)
- [7. Image Processing (Mim)](#7-image-processing-mim)
- [8. Edge Detection (Medge)](#8-edge-detection-medge)
- [9. 資源生命週期順序](#9-資源生命週期順序)
- [10. CLProtocol 啟動流程](#10-clprotocol-啟動流程)
- [11. 已知 .NET Wrapper 限制](#11-已知-net-wrapper-限制)
- [12. 專案使用對照](#12-專案使用對照)
- [13. 完整方法清單](#13-完整方法清單)

---

## 1. 常用常數速查

### 通用

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_NULL` | 0x0 | 空值 |
| `M_DEFAULT` | 0x10000000 | 預設值 |
| `M_ENABLE` | 0xFFFFD8F3 | 啟用 |
| `M_DISABLE` | 0xFFFFD8F1 | 停用 |
| `M_ONCE` | 0x3 | 執行一次 |
| `M_START` | 0x2 | 開始 |
| `M_STOP` | 0x4 | 停止 |
| `M_UNHOOK` | 0x4000000 | 解除 Hook |

### Buffer 屬性

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_IMAGE` | 0x4 | 影像 buffer |
| `M_GRAB` | 0x8 | 可被 Digitizer 抓圖 |
| `M_PROC` | 0x10 | 可被處理函式使用 |
| `M_DISP` | 0x20 | 可被 Display 顯示 |
| `M_UNSIGNED` | 0x0 | 無號整數 |
| `M_SIGNED` | 0x8000000 | 有號整數 |
| `M_FLOAT` | 0x48000000 | 浮點數 |
| `M_SIZE_X` | 0x600 | 寬度 |
| `M_SIZE_Y` | 0x601 | 高度 |
| `M_OWNER_SYSTEM` | 0x44D | 所屬 System ID |

### Digitizer 查詢

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_SOURCE_SIZE_Y` | 0xFB7 | 設定/查詢 Grab 高度 |
| `M_SCAN_MODE` | 0xFB5 | Line / Progressive |
| `M_EXPOSURE_TIME` | 0x19FC | 曝光時間（ns，legacy） |
| `M_PROCESS_FRAME_RATE` | 0x19CF | 實測 FPS |
| `M_SELECTED_FRAME_RATE` | 0x19CE | DCF 目標 FPS |
| `M_PROCESS_FRAME_COUNT` | 0x14D7 | 累計處理幀數 |
| `M_PROCESS_FRAME_MISSED` | 0x14D6 | Callback 遺漏幀數 |
| `M_GRAB_FRAME_MISSED` | 0x14E3 | 硬體遺漏幀數 |
| `M_CAMERA_PRESENT` | 0x14C3 | 相機是否在線 |

### CLProtocol

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_GC_CLPROTOCOL` | 0x1B3E | 啟用/停用 CLProtocol |
| `M_GC_CLPROTOCOL_DEVICE_ID` | 0x1E32 | CLProtocol 裝置 ID |
| `M_FEATURE_VALUE` | 0xFA0000 | Feature 值存取 |
| `M_TYPE_DOUBLE` | 0x5000000000 | double 型別 |

### Display

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_SCALE_DISPLAY` | 0x293D | 縮放顯示 |
| `M_CENTER_DISPLAY` | 0x2723 | 置中顯示 |
| `M_MOUSE_USE` | 0xC93 | 啟用滑鼠互動 |
| `M_ZOOM_FACTOR_X` | 0x19CD | X 方向 zoom 倍率 |
| `M_ZOOM_FACTOR_Y` | 0x19CE | Y 方向 zoom 倍率 |
| `M_PAN_OFFSET_X` | 0x19CB | X 方向平移偏移 |
| `M_PAN_OFFSET_Y` | 0x19CC | Y 方向平移偏移 |

### Mouse Hook

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_MOUSE_MOVE` | 0x40 | 滑鼠移動事件 |
| `M_MOUSE_LEFT_BUTTON_DOWN` | 0x38 | 滑鼠左鍵按下 |
| `M_MOUSE_POSITION_BUFFER_X` | 0x4 | buffer 座標 X |
| `M_MOUSE_POSITION_BUFFER_Y` | 0x5 | buffer 座標 Y |

### Processing Hook

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_MODIFIED_BUFFER` | 0x40000000 | 被修改的 buffer 旗標 |
| `M_BUFFER_ID` | 0x160000 | Buffer ID 查詢 |

### System 查詢

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_TEMPERATURE_FPGA` | 0x1C85 | 擷取卡 FPGA 溫度 (°C) |
| `M_MEMORY_FREE` | 0x914 | 可用記憶體 (bytes) |
| `M_MEMORY_SIZE` | 0x913 | 總記憶體 (bytes) |
| `M_PCIE_NUMBER_OF_LANES` | 0xA1C | PCIe 通道數 |
| `M_PCIE_SPEED` | 0xA26 | PCIe 速度 |

### Error

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_ERROR` | 0x40000000 | 錯誤控制 |
| `M_PRINT_DISABLE` | 0x0 | 停用錯誤輸出 |

### Image Processing

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_SUB` | 0x1 | 減法 |
| `M_ADD_CONST` | 0x8000 | 加常數 |
| `M_MULT_CONST` | 0x8100 | 乘常數 |
| `M_ABS` | 0xC | 取絕對值 |
| `M_FILL_DESTINATION` | 0xFFFFFFFF | 填滿目標 |
| `M_NEAREST_NEIGHBOR` | 0x40 | 最近鄰插值 |
| `M_0_DEGREE` | 0x0 | 0° 投影方向 |
| `M_MEAN` | 0x8000000 | 均值統計 |
| `M_FIXED` | 0x50 | 固定閾值 |
| `M_GREATER` | 0x5 | 大於 |

### Edge Detection

| 常數 | 值 (Hex) | 說明 |
|------|----------|------|
| `M_CREST` | 0x801 | 脊線偵測模式 |
| `M_SAVE_DERIVATIVES` | 0xB | 保存微分影像 |
| `M_FLOAT_MODE` | 0x46 | 浮點模式 |
| `M_FILTER_SMOOTHNESS` | 0x6C | 平滑濾波 (sigma) |
| `M_DRAW_SECOND_DERIVATIVE_X` | 0x1200000 | 繪製 X 方向二階微分 |

---

## 2. Application (Mapp)

### MappAlloc — 建立 MIL Application

```csharp
MIL.MappAlloc(MIL.M_NULL, MIL.M_DEFAULT, ref milApplication);
```

### MappControl — 全域控制

```csharp
// 停用錯誤訊息（避免 MessageBox 彈出中斷程式）
MIL.MappControl(MIL.M_DEFAULT, MIL.M_ERROR, MIL.M_PRINT_DISABLE);
```

### MappFreeDefault — 釋放 Application

```csharp
MIL.MappFreeDefault(milApplication, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL);
```

---

## 3. System (Msys)

### MsysAlloc — 建立 MIL System（對應一張擷取卡）

```csharp
MIL_ID sysId = MIL.M_NULL;
MIL.MsysAlloc(milApplication, systemDescriptor, systemNum, MIL.M_DEFAULT, ref sysId);
// systemDescriptor 例: MIL.M_SYSTEM_RADIENTEVCL
// systemNum: 0-based 擷取卡編號
```

### MsysInquire — 查詢 System 資訊

```csharp
double fpgaTemp = 0;
MIL.MsysInquire(sysId, MIL.M_TEMPERATURE_FPGA, ref fpgaTemp);  // FPGA 溫度 (°C)

MIL_INT memFree = 0;
MIL.MsysInquire(sysId, MIL.M_MEMORY_FREE, ref memFree);        // 可用記憶體 (bytes)

MIL_INT memSize = 0;
MIL.MsysInquire(sysId, MIL.M_MEMORY_SIZE, ref memSize);        // 總記憶體 (bytes)

MIL_INT lanes = 0;
MIL.MsysInquire(sysId, MIL.M_PCIE_NUMBER_OF_LANES, ref lanes); // PCIe 通道數

MIL_INT speed = 0;
MIL.MsysInquire(sysId, MIL.M_PCIE_SPEED, ref speed);           // PCIe 速度 (1=Gen1, 2=Gen2, 3=Gen3)
```

### MsysFree — 釋放 System

```csharp
MIL.MsysFree(sysId);
```

---

## 4. Digitizer (Mdig)

### MdigAlloc — 開啟 Digitizer

```csharp
MIL_ID digId = MIL.M_NULL;
MIL.MdigAlloc(sysId, devNum, dcfPath, MIL.M_DEFAULT, ref digId);
// devNum: Digitizer 編號 (0-based)
// dcfPath: DCF 檔案路徑 (Camera Link 設定)
```

### MdigControl — 設定 Digitizer 參數

```csharp
// 設定 Grab 高度（必須在 MdigInquire SIZE 之前）
MIL.MdigControl(digId, MIL.M_SOURCE_SIZE_Y, (MIL_INT)height);

// CLProtocol 啟用
MIL.MdigControl(digId, MIL.M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT");
MIL.MdigControl(digId, MIL.M_GC_CLPROTOCOL, MIL.M_ENABLE);

// Legacy 曝光設定（CLProtocol 未啟用時，單位 ns）
MIL.MdigControl(digId, MIL.M_EXPOSURE_TIME, exposureUs * 1000.0);
```

### MdigControlFeature — 透過 CLProtocol Feature API 設定

```csharp
// 曝光時間（μs，CLProtocol 啟用後）
double expUs = 500.0;
MIL.MdigControlFeature(digId, MIL.M_FEATURE_VALUE,
    "ExposureTime", MIL.M_TYPE_DOUBLE, ref expUs);

// 線掃速率（Hz，CLProtocol 啟用後）
double hz = 5000.0;
MIL.MdigControlFeature(digId, MIL.M_FEATURE_VALUE,
    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
```

### MdigInquire — 查詢 Digitizer 資訊

```csharp
// 影像尺寸（回傳值 + ref 皆可用）
MIL_INT sizeX = MIL.MdigInquire(digId, MIL.M_SIZE_X, MIL.M_NULL);
MIL_INT sizeY = MIL.MdigInquire(digId, MIL.M_SIZE_Y, MIL.M_NULL);

// 效能資訊
double fps = 0;
MIL.MdigInquire(digId, MIL.M_PROCESS_FRAME_RATE, ref fps);     // 實測 FPS

double targetFps = 0;
MIL.MdigInquire(digId, MIL.M_SELECTED_FRAME_RATE, ref targetFps); // DCF 設定 FPS

MIL_INT frameCount = 0;
MIL.MdigInquire(digId, MIL.M_PROCESS_FRAME_COUNT, ref frameCount); // 累計幀數

MIL_INT missed = 0;
MIL.MdigInquire(digId, MIL.M_PROCESS_FRAME_MISSED, ref missed);   // Callback 遺漏

MIL_INT grabMiss = 0;
MIL.MdigInquire(digId, MIL.M_GRAB_FRAME_MISSED, ref grabMiss);    // 硬體遺漏

MIL_INT scanMode = 0;
MIL.MdigInquire(digId, MIL.M_SCAN_MODE, ref scanMode);            // Line / Progressive

// 相機存在偵測
MIL_INT presence = 0;
MIL.MdigInquire(digId, MIL.M_CAMERA_PRESENT, ref presence);       // 0=離線, 1=在線

// Legacy 曝光查詢（CLProtocol 未啟用，回傳 ns）
double valNs = 0;
MIL.MdigInquire(digId, MIL.M_EXPOSURE_TIME, ref valNs);
double expUs = valNs / 1000.0;
```

### MdigInquireFeature — 透過 CLProtocol Feature API 查詢

```csharp
// 曝光時間（μs）
double expUs = 0;
MIL.MdigInquireFeature(digId, MIL.M_FEATURE_VALUE,
    "ExposureTime", MIL.M_TYPE_DOUBLE, ref expUs);

// 線掃速率（Hz）
double lineRate = 0;
MIL.MdigInquireFeature(digId, MIL.M_FEATURE_VALUE,
    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref lineRate);

// 相機溫度（°C）
double camTemp = 0;
MIL.MdigInquireFeature(digId, MIL.M_FEATURE_VALUE,
    "DeviceTemperature", MIL.M_TYPE_DOUBLE, ref camTemp);
```

### MdigProcess — 連續抓圖控制

```csharp
// 雙緩衝抓圖
MIL_ID[] grabBufs = new MIL_ID[2];
MIL_INT bufCount = 2;
GCHandle hUserData = GCHandle.Alloc(this);
MIL_DIG_HOOK_FUNCTION_PTR callback = new MIL_DIG_HOOK_FUNCTION_PTR(ProcessingFunction);

// 開始
MIL.MdigProcess(digId, grabBufs, bufCount,
    MIL.M_START, MIL.M_DEFAULT, callback, GCHandle.ToIntPtr(hUserData));

// 停止
MIL.MdigProcess(digId, grabBufs, bufCount,
    MIL.M_STOP, MIL.M_DEFAULT, callback, GCHandle.ToIntPtr(hUserData));
```

### MdigGetHookInfo — Callback 內取得修改的 Buffer

```csharp
private static MIL_INT ProcessingFunction(MIL_INT hookType, MIL_ID eventId, IntPtr userData)
{
    MIL_ID modifiedBuffer = MIL.M_NULL;
    MIL.MdigGetHookInfo(eventId, MIL.M_MODIFIED_BUFFER + MIL.M_BUFFER_ID, ref modifiedBuffer);
    // modifiedBuffer 就是本次抓到的影像
    return MIL.M_NULL;
}
```

### MdigFree — 釋放 Digitizer

```csharp
MIL.MdigFree(digId);
```

---

## 5. Display (Mdisp)

### MdispAlloc — 建立 Display

```csharp
MIL_ID dispId = MIL.M_NULL;
MIL.MdispAlloc(sysId, MIL.M_DEFAULT, "M_DEFAULT", MIL.M_DEFAULT, ref dispId);
```

### MdispSelectWindow — 綁定 Display 到 WinForms Panel

```csharp
// 綁定
MIL.MdispSelectWindow(dispId, displayBuffer, panel.Handle);

// 解除綁定
MIL.MdispSelectWindow(dispId, MIL.M_NULL, IntPtr.Zero);
```

### MdispControl — Display 顯示控制

```csharp
MIL.MdispControl(dispId, MIL.M_SCALE_DISPLAY, MIL.M_ONCE);    // 自適配一次
MIL.MdispControl(dispId, MIL.M_CENTER_DISPLAY, MIL.M_ENABLE);  // 置中
MIL.MdispControl(dispId, MIL.M_MOUSE_USE, MIL.M_ENABLE);       // 滑鼠 zoom/pan
```

### MdispInquire — 查詢 Display 狀態（AniloxRoll.Monitor 獨有）

```csharp
// 查詢使用者 zoom/pan 狀態（用於 chart 對齊）
double zoomX = 0, zoomY = 0, panX = 0, panY = 0;
MIL.MdispInquire(dispId, MIL.M_ZOOM_FACTOR_X, ref zoomX);
MIL.MdispInquire(dispId, MIL.M_ZOOM_FACTOR_Y, ref zoomY);
MIL.MdispInquire(dispId, MIL.M_PAN_OFFSET_X, ref panX);
MIL.MdispInquire(dispId, MIL.M_PAN_OFFSET_Y, ref panY);
```

### MdispHookFunction — 掛載 / 解除 Mouse Hook

```csharp
MIL_DISP_HOOK_FUNCTION_PTR mouseDelegate = new MIL_DISP_HOOK_FUNCTION_PTR(OnMouseStatus);

// 掛載
MIL.MdispHookFunction(dispId, MIL.M_MOUSE_MOVE, mouseDelegate, (IntPtr)cameraId);
MIL.MdispHookFunction(dispId, MIL.M_MOUSE_LEFT_BUTTON_DOWN, clickDelegate, (IntPtr)cameraId);

// 解除
MIL.MdispHookFunction(dispId, MIL.M_MOUSE_MOVE + MIL.M_UNHOOK, mouseDelegate, IntPtr.Zero);
MIL.MdispHookFunction(dispId, MIL.M_MOUSE_LEFT_BUTTON_DOWN + MIL.M_UNHOOK, clickDelegate, IntPtr.Zero);
```

### MdispGetHookInfo — Mouse Hook 內取得座標

```csharp
private static MIL_INT OnMouseStatus(MIL_INT hookType, MIL_ID EventId, IntPtr userData)
{
    MIL_INT posX = 0, posY = 0;
    MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_X, ref posX);
    MIL.MdispGetHookInfo(EventId, MIL.M_MOUSE_POSITION_BUFFER_Y, ref posY);
    return MIL.M_NULL;
}
```

### MdispFree — 釋放 Display

```csharp
MIL.MdispFree(dispId);
```

---

## 6. Buffer (Mbuf)

### MbufAlloc2d — 建立 2D Buffer

```csharp
MIL_ID bufId = MIL.M_NULL;

// Grab Buffer（可被 Digitizer 寫入 + 處理）
MIL.MbufAlloc2d(sysId, width, height, 8 + MIL.M_UNSIGNED,
    MIL.M_IMAGE + MIL.M_GRAB + MIL.M_PROC, ref bufId);

// Display Buffer（可顯示 + 處理）
MIL.MbufAlloc2d(sysId, width, height, 8 + MIL.M_UNSIGNED,
    MIL.M_IMAGE + MIL.M_DISP + MIL.M_PROC, ref bufId);

// Processing Buffer（僅處理用）
MIL.MbufAlloc2d(sysId, width, height, 8 + MIL.M_UNSIGNED,
    MIL.M_IMAGE + MIL.M_PROC, ref bufId);

// 16-bit 有號（影像運算中間結果）
MIL.MbufAlloc2d(sysId, width, height, 16 + MIL.M_SIGNED,
    MIL.M_IMAGE + MIL.M_PROC, ref bufId);

// 32-bit 浮點（Hessian 等精密計算）
MIL.MbufAlloc2d(sysId, width, height, 32 + MIL.M_FLOAT,
    MIL.M_IMAGE + MIL.M_PROC, ref bufId);
```

### MbufClear — 清空 Buffer

```csharp
MIL.MbufClear(bufId, 0);  // 填充 0
```

### MbufCopy — Buffer 間複製

```csharp
MIL.MbufCopy(srcBuffer, dstBuffer);
```

### MbufGet2d / MbufPut2d — Host ↔ Device 資料傳輸

```csharp
// Device → Host（讀取影像到 CPU byte 陣列）
byte[] hostBuffer = new byte[width * height];
MIL.MbufGet2d(srcBuffer, 0, 0, width, height, hostBuffer);

// Host → Device（寫入處理後的資料回 MIL Buffer）
MIL.MbufPut2d(dstBuffer, 0, 0, width, height, hostBuffer);

// 讀取單一像素
byte[] pixel = new byte[1];
MIL.MbufGet2d(dispBuffer, x, y, 1, 1, pixel);
```

### MbufInquire — 查詢 Buffer 資訊

```csharp
MIL_INT w = MIL.MbufInquire(bufId, MIL.M_SIZE_X, MIL.M_NULL);
MIL_INT h = MIL.MbufInquire(bufId, MIL.M_SIZE_Y, MIL.M_NULL);
MIL_ID owner = MIL.MbufInquire(bufId, MIL.M_OWNER_SYSTEM, MIL.M_NULL);
```

### MbufFree — 釋放 Buffer

```csharp
MIL.MbufFree(bufId);
```

---

## 7. Image Processing (Mim)

> **注意**：MilGrabSample 使用 MIL 影像處理，AniloxRoll.Monitor 改用 CUDA pipeline。

### MimProjection — 列/行投影

```csharp
// 沿 0° 方向（垂直）投影，計算每列均值 → 1×W 的 1D buffer
MIL.MimProjection(srcBuffer, meanLine1D, MIL.M_0_DEGREE, MIL.M_MEAN, MIL.M_NULL);
```

### MimResize — 影像縮放

```csharp
// 最近鄰插值，填滿目標 buffer
MIL.MimResize(srcBuffer, dstBuffer, MIL.M_FILL_DESTINATION, MIL.M_DEFAULT, MIL.M_NEAREST_NEIGHBOR);
```

### MimArith — 影像算術運算

```csharp
MIL.MimArith(src1, src2, dst, MIL.M_SUB);           // dst = src1 - src2
MIL.MimArith(src, 127.0, dst, MIL.M_ADD_CONST);     // dst = src + 127
MIL.MimArith(src, factor, dst, MIL.M_MULT_CONST);   // dst = src * factor
MIL.MimArith(src, MIL.M_NULL, dst, MIL.M_ABS);      // dst = |src|
```

### MimBinarize — 二值化

```csharp
MIL.MimBinarize(srcBuffer, dstBuffer, MIL.M_FIXED + MIL.M_GREATER, threshold, MIL.M_NULL);
```

---

## 8. Edge Detection (Medge)

> **注意**：僅 MilGrabSample 使用。AniloxRoll.Monitor 改用 CUDA Hessian。

### 完整流程

```csharp
MIL_ID edgeCtx = MIL.M_NULL, edgeRes = MIL.M_NULL;

// 1. 建立 Context（脊線偵測模式）
MIL.MedgeAlloc(sysId, MIL.M_CREST, MIL.M_DEFAULT, ref edgeCtx);
MIL.MedgeAllocResult(sysId, MIL.M_DEFAULT, ref edgeRes);

// 2. 設定參數
MIL.MedgeControl(edgeCtx, MIL.M_SAVE_DERIVATIVES, MIL.M_ENABLE);
MIL.MedgeControl(edgeCtx, MIL.M_FLOAT_MODE, MIL.M_ENABLE);
MIL.MedgeControl(edgeCtx, MIL.M_FILTER_SMOOTHNESS, sigma);  // sigma 值

// 3. 執行計算
MIL.MedgeCalculate(edgeCtx, srcBuffer, MIL.M_NULL, MIL.M_NULL, MIL.M_NULL, edgeRes, MIL.M_DEFAULT);

// 4. 繪製結果（二階微分 X）
MIL_ID bufFloat = MIL.M_NULL;
MIL.MbufAlloc2d(sysId, width, height, 32 + MIL.M_FLOAT, MIL.M_IMAGE + MIL.M_PROC, ref bufFloat);
MIL.MedgeDraw(MIL.M_DEFAULT, edgeRes, bufFloat, MIL.M_DRAW_SECOND_DERIVATIVE_X, MIL.M_DEFAULT, MIL.M_DEFAULT);

// 5. 後處理
MIL.MimArith(bufFloat, MIL.M_NULL, bufFloat, MIL.M_ABS);           // 取絕對值
MIL.MimArith(bufFloat, scaleFactor, bufFloat, MIL.M_MULT_CONST);   // 縮放
MIL.MbufCopy(bufFloat, destBuffer);                                  // 轉回 8-bit

// 6. 釋放
MIL.MedgeFree(edgeCtx);
MIL.MedgeFree(edgeRes);
MIL.MbufFree(bufFloat);
```

---

## 9. 資源生命週期順序

### 初始化（必須依序）

```
MappAlloc          → MIL Application
  ↓
MappControl        → M_PRINT_DISABLE
  ↓
MsysAlloc × N      → 每張擷取卡一個 System
  ↓
MdigAlloc          → 開 Digitizer
  ↓
MdigControl(M_SOURCE_SIZE_Y)  → 設 Grab 高度 ⚠ 必須在 MdigInquire SIZE 之前
  ↓
MdispAlloc         → 開 Display（可開多個：primary + secondary）
  ↓
MdigInquire(SIZE)  → 查實際尺寸
  ↓
MbufAlloc2d × N    → Grab/Display/Proc Buffer
  ↓
MbufClear          → 清空（避免殘影）
  ↓
MdispSelectWindow  → 綁定 Panel
  ↓
MdispControl       → SCALE_DISPLAY / CENTER / MOUSE_USE
  ↓
MdispHookFunction  → Mouse Hook
  ↓
MdigProcess(START) → 開始抓圖
  ↓
CLProtocol 啟動    → 背景 Task.Run（見下節）
```

### 釋放（逆序）

```
MdigProcess(M_STOP)                    → 停止抓圖
  ↓
MdispHookFunction(M_UNHOOK)            → 解除 Mouse Hook
MdispSelectWindow(M_NULL, IntPtr.Zero) → 解除 Display 綁定
  ↓
MbufFree × N                           → 釋放所有 Buffer
  ↓
MdispFree                              → 釋放 Display
  ↓
MdigFree                               → 釋放 Digitizer（最後）
  ↓
GCHandle.Free()                        → 釋放 Callback GCHandle
  ↓
MsysFree × N                           → 釋放 System
  ↓
MappFreeDefault                        → 釋放 Application
```

### SetGrabHeight Buffer 重配流程

```
MdigProcess(M_STOP)           → 1. 停止抓圖
  ↓
MbufFree(Grab × 2)           → 2. 釋放舊 Buffer
MbufFree(Display)
MbufFree(Proc)
NativeBufferPool.Dispose()    → 3. 釋放 CUDA Pinned Memory
  ↓
MdigControl(M_SOURCE_SIZE_Y)  → 4. 設新高度
MdigInquire(SIZE)             → 5. 重查實際尺寸
  ↓
new NativeBufferPool(W,H)    → 6. 重新分配 CUDA Pinned
MbufAlloc2d × N              → 7. 重新分配 Buffer
  ↓
MdispSelectWindow(new buf)   → 8. 重新綁定
MdigProcess(M_START)          → 9. 恢復抓圖
```

> ⚠ 舊尺寸 Buffer + 新尺寸 = MIL 崩潰。**不可省略步驟 2–3**。

---

## 10. CLProtocol 啟動流程

```
MdigProcess(M_START) 成功後：
  ↓
Task.Run(async) {
    MdigControl(M_GC_CLPROTOCOL_DEVICE_ID, "M_DEFAULT")
    MdigControl(M_GC_CLPROTOCOL, M_ENABLE)     // 耗時 1–2 秒
    _clProtocolEnabled = true
    SetExposureUs(...)    // 重套曝光（改走 Feature API）
    SetLineRateHz(...)    // 重套線掃速率
}
```

### 曝光 Set/Get 分支

| CLProtocol | Set | Get | 單位 |
|------------|-----|-----|------|
| ✅ 啟用 | `MdigControlFeature("ExposureTime", M_TYPE_DOUBLE, ref μs)` | `MdigInquireFeature("ExposureTime")` | μs |
| ❌ 未啟用 | `MdigControl(M_EXPOSURE_TIME, μs×1000)` | `MdigInquire(M_EXPOSURE_TIME)` ÷ 1000 | ns→μs |

### 線掃速率（僅 CLProtocol 啟用時可設）

```csharp
// Set
MIL.MdigControlFeature(digId, MIL.M_FEATURE_VALUE,
    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);

// Get
MIL.MdigInquireFeature(digId, MIL.M_FEATURE_VALUE,
    "AcquisitionLineRate", MIL.M_TYPE_DOUBLE, ref hz);
```

---

## 11. 已知 .NET Wrapper 限制

| 常數 | 狀態 | 替代方案 |
|------|------|----------|
| `M_LINE_RATE` | ❌ 不存在 | `MdigInquireFeature("AcquisitionLineRate")` (需 CLProtocol) |
| `M_LINE_RATE_CURRENT` | ❌ 不存在 | 同上 |
| `M_GRAB_SIZE_Y` | ❌ 不存在 | `MdigControl(M_SOURCE_SIZE_Y, height)` |
| `MdigHookFunction(M_CAMERA_PRESENT)` | ❌ 已移除 | Timer 每 500ms 輪詢 `MdigInquire(M_CAMERA_PRESENT)` |

CLProtocol 初始化期間（約 1–2 秒）以下查詢無法取值，屬正常：
- `"AcquisitionLineRate"`
- `"ExposureTime"`（Feature 版）
- `"DeviceTemperature"`

---

## 12. 專案使用對照

| API | MilGrabSample | AniloxRoll.Monitor | 差異說明 |
|-----|:---:|:---:|------|
| **MappAlloc/Control/Free** | ✅ | ✅ | 用法一致 |
| **MsysAlloc/Free/Inquire** | ✅ | ✅ | 用法一致 |
| **MdigAlloc/Free** | ✅ | ✅ | 用法一致 |
| **MdigControl(M_SOURCE_SIZE_Y)** | ✅ | ✅ | 用法一致 |
| **MdigControl(CLProtocol)** | ✅ | ✅ | 用法一致 |
| **MdigControl(M_EXPOSURE_TIME)** | ✅ | ✅ | 用法一致（legacy ns 路徑） |
| **MdigControlFeature(ExposureTime)** | ✅ | ✅ | 用法一致（CLProtocol μs） |
| **MdigControlFeature(LineRate)** | ✅ | ✅ | 用法一致 |
| **MdigInquire(SIZE/FPS/MISSED...)** | ✅ | ✅ | 用法一致 |
| **MdigInquireFeature(各 Feature)** | ✅ | ✅ | 用法一致 |
| **MdigProcess(START/STOP)** | ✅ | ✅ | 用法一致 |
| **MdigGetHookInfo** | ✅ | ✅ | 用法一致 |
| **MdispAlloc** | ✅(×1) | ✅(×2) | Monitor 多一個副顯示器 |
| **MdispControl** | ✅ | ✅ | 用法一致 |
| **MdispSelectWindow** | ✅ | ✅ | 用法一致 |
| **MdispInquire(zoom/pan)** | ❌ | ✅ | Monitor 獨有：chart 對齊用 |
| **MdispHookFunction(MOUSE_MOVE)** | ✅ | ✅ | 用法一致 |
| **MdispHookFunction(BUTTON_DOWN)** | ❌ | ✅ | Monitor 獨有：點選切換相機 |
| **MdispGetHookInfo** | ✅ | ✅ | 用法一致 |
| **MbufAlloc2d** | ✅ | ✅ | 用法一致 |
| **MbufClear/Copy** | ✅ | ✅ | 用法一致 |
| **MbufGet2d/Put2d** | ✅ | ✅ | 用法一致 |
| **MbufInquire** | ✅ | ✅ | 用法一致 |
| **MimProjection/Resize/Arith** | ✅ | ❌ | Monitor 改用 CUDA |
| **MimBinarize** | ✅ | ❌ | Monitor 改用 CUDA |
| **MedgeAlloc/Calculate/Draw** | ✅ | ❌ | Monitor 改用 CUDA Hessian |

**結論**：兩個專案的 MIL API 用法**完全一致**。AniloxRoll.Monitor 是 MilGrabSample 的功能超集，額外增加了副顯示器、MdispInquire（zoom/pan 查詢）和滑鼠點擊 Hook。影像處理部分 Monitor 已遷移至 CUDA pipeline，不再使用 MIL 的 Mim/Medge。

---

## 13. 完整方法清單

MIL .NET wrapper 共提供以下模組的靜態方法（僅列出前綴分類）：

| 模組前綴 | 用途 | 本專案使用 |
|----------|------|:---:|
| `Mapp` | Application 管理 | ✅ |
| `Msys` | System（擷取卡）管理 | ✅ |
| `Mdig` | Digitizer（相機）控制 | ✅ |
| `Mdisp` | Display 顯示管理 | ✅ |
| `Mbuf` | Buffer 記憶體管理 | ✅ |
| `Mim` | 影像處理（算術/濾波/形態學） | MilGrabSample only |
| `Medge` | 邊緣/脊線偵測 | MilGrabSample only |
| `Mgra` | 圖形繪製（標註/overlay） | ❌ |
| `Mmod` | 模型匹配（Geometric Model Finder） | ❌ |
| `Mpat` | 樣板匹配（Pattern Matching） | ❌ |
| `Mcode` | 條碼/QR Code 讀取 | ❌ |
| `Mocr` | OCR 字元辨識 | ❌ |
| `Mcol` | 色彩分析 | ❌ |
| `Mcal` | 校正（Calibration） | ❌ |
| `Mblob` | Blob 分析 | ❌ |
| `Mmeas` | 量測（Measurement） | ❌ |
| `Mreg` | 影像對齊（Registration） | ❌ |
| `Mclass` | 深度學習分類 | ❌ |
| `Mseq` | 影像序列/編碼 | ❌ |
| `Mthr` | 多執行緒管理 | ❌ |
| `Mfpga` | FPGA 程式設計 | ❌ |
| `Mfunc` | 自定義函式 | ❌ |
| `Mobj` | 物件管理/訊息傳遞 | ❌ |
| `Mmet` | 計量學（Metrology） | ❌ |
| `Magm` | 自適配幾何匹配 | ❌ |
| `Mbead` | 膠條檢測 | ❌ |
| `Mcom` | 通訊（Serial I/O） | ❌ |
| `Mstr` | 字串讀取（String Reader） | ❌ |
| `Mdmr` | 資料矩陣讀取 | ❌ |
| `Mdlocr` | 深度學習 OCR | ❌ |
