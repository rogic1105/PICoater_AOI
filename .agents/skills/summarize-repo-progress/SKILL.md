---
name: summarize-repo-progress
description: Extract and summarize repository progress from Git history, branches, diffs, and current worktree changes. Use when the user asks what changed today or on a specified date, requests a daily report, work log, progress summary, or wants repo activity condensed into short management-friendly highlights.
---

# Summarize Repo Progress

## Collect evidence

1. Run `scripts/collect_repo_progress.ps1 -Date YYYY-MM-DD` from the repository root.
2. Add `-IncludeWorkingTree` only when reporting the current day.
3. Treat Git commits and diffs as the source of truth. Do not use file modification times to reconstruct past work.
4. Inspect relevant diffs when a commit title does not explain the user-visible action and result.
5. Separate work merged to `main`, committed feature work, experiments, and uncommitted work. Describe experiments as tests, not released behavior.

## Condense the work

- Merge related commits into outcome-based themes.
- Identify the day's top-level task or goal before listing implementation details. Use branch changes, commit intent, diffs, and user context to recognize priority shifts during the day.
- When work pivots, present the earlier task and the new priority in their actual importance; do not flatten every technical edit into an equal-sized item.
- Assign every theme to the subject that best owns the result. Use this vocabulary:
  - Product flows: `監控`, `回顧`, `報表`.
  - Product capabilities: `取像／硬體`, `影像處理`, `設定`, `Bridge`, `資料格式`, `儲存／傳輸`, `UI／操作`.
  - Engineering support: `SDK／共用元件`, `部署／網路`, `測試／驗證`, `架構／效能`.
- Prefer a product flow over an internal implementation label when the outcome clearly belongs to monitoring, review, or report. Use capability or engineering subjects for cross-cutting work or work with no single product-flow owner.
- Use a combined subject when the same action and result genuinely span multiple areas, such as `回顧／報表`, `監控／回顧`, or `Bridge／資料格式`.
- Prefer one combined bullet over duplicating the same outcome under multiple subjects. Keep subjects separate when their actions or benefits differ.
- Cover every subject with a material result that day. Do not impose a fixed subject or bullet count, and do not force subjects with no meaningful work.
- Prefer highlights meaningful in a daily report: faster reading, smoother operation, higher accuracy, shorter waits, safer persistence, or easier maintenance.
- Omit commit hashes, file names, line counts, implementation class names, and detailed chronology unless requested.
- Preserve an important limitation when omitting it would make an experiment sound complete.

## Default output standard

- Return one bullet per meaningful subject. Use a fixed count only when the user explicitly requests one.
- Write one short sentence per bullet in Task + Result (TR) form.
- Start with the product subject, then state the task and result: `主題：任務，成果。`
- Describe the task as the goal being handled, not the implementation action. Prefer `加速大量序號切換` over `新增排程器`.
- State only an evidenced result. For unfinished or experimental work, use `完成初步驗證`, `降低預期等待`, or similarly qualified wording instead of claiming a released improvement.
- Keep each bullet compact, ideally 18-30 Chinese characters.
- Use plain Traditional Chinese suitable for a daily report.
- Do not add headings, preamble, statistics, or a concluding paragraph.

Example:

- 監控：縮短相機啟動等待，完成同步流程優化。
- 回顧／報表：加速大量序號切換，降低畫面卡頓。
- 資料格式：提升封裝資料讀取速度，完成 ACAP 預覽設計。
- 測試／驗證：確保效能修改穩定，補齊流程檢查。

Return only evidence-backed outcomes; never invent work to reach a target count.
