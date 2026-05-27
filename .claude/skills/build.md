# build

手動觸發完整 Build 驗證。

## 使用時機

使用者輸入 `/build` 時。

## 原則

**一律 Release|x64**，不 build Debug。本專案依賴 AMD64 MIL SDK，必須 `Platform=x64`
（AnyCPU/MSIL 會 MSB3270 警告且執行期崩潰）。開發用 agent + `Trace.WriteLine` /
`Console.WriteLine` 檢查（`Debug.WriteLine` 在 Release 是 no-op）。csproj 殘留的 Debug
配置請忽略，不要選用。

## 執行步驟

1. **Build 產品主程式**（Release|x64）：
   ```
   "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "D:\Chunkuan\AUO\02_Projects_Active\PICoater\07_Source_Code_Repo\PICoater_AOI\src\dotnet\AniloxRoll.Monitor\AniloxRoll.Monitor.csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal
   ```
   （PowerShell 用 `& "...MSBuild.exe" "...csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal`）

2. **Build sdk 工具**（Release|x64，所有 samples → bin/x64/Release/tools/）：
   ```
   "...MSBuild.exe" "D:\...\PICoater_AOI\sdk\Tools.sln" /p:Configuration=Release /p:Platform=x64 /v:minimal
   ```

3. **回報結果** — 列出 error/warning 數量，若有 error 顯示前 10 條。
