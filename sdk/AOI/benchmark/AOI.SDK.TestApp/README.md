# AOI.SDK.TestApp — AOI.SDK benchmark UI

WinForms exe，AOI.SDK 的 benchmark 工具：載入 BMP → GPU 濾鏡 + 計時 + 顯示到
`SmartCanvas`，量端到端速度。示範 `core_cv_api.dll` 的 P/Invoke 用法
（`CoreCV_AllocPinned` / `CoreCV_FastReadBMP` / `CoreCV_Resize_GPU` 等）。

**不參與主 build**：不在 `PICoater_AOI.sln` 內，build 主程式不會 build 這裡。
可獨立用 `AOI.SDK.TestApp.csproj` 開啟 / build（依賴 `..\..\dotnet\AOI.SDK\AOI.SDK.csproj`）。

**位置**：跟 C++ 的 `core_cv_benchmark` 同層（`sdk/AOI/benchmark/`），因為它是
SDK 的 benchmark 工具而非文件範例。原本住獨立的 `AOI_SDK` repo
（<https://github.com/rogic1105/AOI_SDK>），LLM 工具還無法跨 repo 時暫搬進本 repo。

**參考價值**：
- `Forms/SdkForm.cs` — 如何快速搭一個 GPU pipeline 測試介面
- `Program.cs` — 單一 Form 啟動 pattern
- `Properties/Settings.settings` — 設定持久化基底
