# build

手動觸發完整 Build 驗證。

## 使用時機

使用者輸入 `/build` 時。

## 執行步驟

1. **Build Release|x64**：
   ```
   cat > /tmp/build.bat << 'EOFBAT'
   @echo off
   "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "D:\Chunkuan\AUO\02_Projects_Active\PICoater\07_Source_Code_Repo\PICoater_AOI\src_dotnet\AniloxRoll.Monitor\AniloxRoll.Monitor.csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal
   EOFBAT
   cmd //c "$(cygpath -w /tmp/build.bat)"
   ```

2. **Build Debug|x64**（同上換 Configuration=Debug）

3. **回報結果** — 列出 error/warning 數量，若有 error 顯示前 10 條

**重要**：永遠使用 `Platform=x64`，不可省略。本專案依賴 AMD64 的 MIL SDK，AnyCPU/MSIL 會產生 MSB3270 警告且執行期會崩潰。
