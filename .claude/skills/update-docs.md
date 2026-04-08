# update-docs

Commit 前批次更新專案文件，確保三方同步 + 控制項存在性。

## 使用時機

使用者輸入 `/update-docs` 時，或 `/commit` 流程中的文件更新步驟。

## 執行步驟

### 1. 控制項存在性驗證

掃描 `CLAUDE.md` 控制項速查表中所有 `Name`（反引號內的名稱），逐一比對 `AniloxRollForm.Designer.cs`：

- **不存在** → 標記 ⚠️，詢問使用者是否從速查表移除
- **Designer.cs 有但速查表沒有** → 提示是否需要新增（僅列出有 `.Text` 或 `.Click` 的控制項）

### 2. 三方名稱同步檢查

比對三層的控制項名稱是否一致：

| 層 | 檔案 | 檢查內容 |
|----|------|---------|
| Form | `Designer.cs` | `.Text = "..."` 畫面文字 |
| CLAUDE.md | 控制項速查 | 標準名稱 + 畫面文字欄 |
| HTML | `docs/user-manual/ui-flow.html` | 【】內的控制項名稱 |

不一致的列出差異，詢問以哪一方為準。

### 3. 掃描變更範圍

`git diff --name-only HEAD` 找出所有修改的 .cs 檔。

### 4. 判斷需更新的文件

| 改動範圍 | 更新目標 |
|---------|---------|
| `UI/Form/AniloxRollForm*.cs` | CLAUDE.md 控制項速查 + `ui-flow.html` |
| `UI/Presenters/*.cs` | 對應 `.claude/skills/modify-*.md` |
| `UI/Widgets/*.cs` | `.claude/skills/modify-ui.md` |
| `ImageProcessing/*.cs` | `.claude/skills/modify-pipeline.md` |
| `Acquisition/*.cs` | `.claude/skills/modify-acquisition.md` |
| `Services/Plc*.cs` | `.claude/skills/modify-acquisition.md` |
| `Interop/NativeMethods.cs` | `CLAUDE.md` Native API 表 |
| 新增/移除 .cs 檔 | `CLAUDE.md` 關鍵檔案速查 + `.csproj` |

### 5. 更新文件

- **CLAUDE.md** — 關鍵檔案速查表、控制項速查表、Skills 路由
- **`.claude/skills/*.md`** — 對應 skill 的注意事項
- **`docs/user-manual/`** — 操作說明（若影響使用者操作流程）

### 6. 回報摘要

列出：
- 控制項驗證結果（新增/移除/不一致）
- 更新了哪些文件
- 主要變更內容
