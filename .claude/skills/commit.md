# commit

提交並推送變更，遵循專案規則。

## 使用時機

使用者輸入 `/commit` 時。

## 執行步驟

1. **確認分支** — `git branch --show-current`，若在 main 且改動較大，詢問是否建 feature branch

2. **Build 驗證** — 用 MSBuild 確認零錯誤（**必須帶 Platform=x64**）：
   ```
   cat > /tmp/build.bat << 'EOFBAT'
   @echo off
   "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe" "D:\Chunkuan\AUO\02_Projects_Active\PICoater\07_Source_Code_Repo\PICoater_AOI\src_dotnet\AniloxRoll.Monitor\AniloxRoll.Monitor.csproj" /p:Configuration=Release /p:Platform=x64 /v:minimal
   EOFBAT
   cmd //c "$(cygpath -w /tmp/build.bat)"
   ```

3. **更新文件** — 根據改動內容更新：
   - `CLAUDE.md` — 關鍵檔案速查、路由索引
   - `docs/*.md` — 對應的架構或模式文件
   - `README.md` — 對外說明（若有重大功能變更）

4. **檢視變更** — `git status` + `git diff --stat`，確認無敏感檔案

5. **Commit** — 使用 conventional commit 格式，附加 Co-Authored-By

6. **Push** — 僅在使用者明確要求時推送
