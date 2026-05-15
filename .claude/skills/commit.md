# commit

提交並推送變更。Git 規則見 `CLAUDE.md §Git Workflow 規則`。

## 使用時機

使用者輸入 `/commit` 時。

## 執行步驟

1. **Build 驗證**（`/build` 也可獨立呼叫；Release|x64 必須）
2. **跑 `/update-docs`** — 對齊 CLAUDE.md / skills / docs/ 與本次改動
3. **檢視變更** — `git status` + `git diff --cached --stat`
4. **Conventional commit message** — 附 `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>`
5. **Push** — 僅在使用者明確要求時
