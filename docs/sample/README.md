# docs/sample/ — 可分離的示範程式

此目錄存放「**目前 PICoater_AOI 不使用、但保留作為新專案參考**」的程式碼。
不在 `PICoater_AOI.sln` 內，build 主程式不會 build 這裡。

## 內容

### `AOI.SDK.TestApp/`

AOI.SDK 的 WinForms 測試工具，示範 `core_cv_api.dll` 的 P/Invoke 用法
（`CoreCV_AllocPinned` / `CoreCV_FastReadBMP` / `CoreCV_Resize_GPU` 等）。

**歷史背景**：原本住在獨立的 `AOI_SDK` repo（<https://github.com/rogic1105/AOI_SDK>），
但當時 LLM 工具還沒辦法跨 repo 操作，所以暫時搬過來。現在 Claude Code 可跨目錄、
未來會把 TestApp 同步回 AOI_SDK repo，本 repo 就刪除這份。

**參考價值**：
- 看 `Forms/SdkForm.cs` 了解如何快速搭一個 GPU pipeline 測試介面
- 看 `Program.cs` 的單一 Form 啟動 pattern
- 看 `Properties/Settings.settings` 的設定持久化基底

## 為什麼放在 docs/sample/ 而非 sdk/

- `sdk/AOI/` 是當作 git submodule 的延伸（雖然目前實際是 monorepo），核心 SDK 程式不應該包示範
- `docs/sample/` 明確表達「給後續開發者看的範例、不是 build target」
