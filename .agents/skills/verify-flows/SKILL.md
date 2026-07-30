---
name: verify-flows
description: Verify PICoater AOI monitoring, review, report, storage, and hardware behavior against DVT log-flow and code-flow contracts. Use after changing event wiring, async sequencing, display modes, coordinates, persistence, navigation, or UI architecture, and when diagnosing a full-day trace log.
---

# verify-flows

Use the DVT contract to check both sides of every affected behavior:

- `log-flow`: runtime evidence, ordering, tokens, counts, snapshots, and forbidden lines.
- `code-flow`: current responsibility chain, state owner, conversion formulas, and structural invariants.

## Workflow

1. Read the effectiveness and interpretation rules at the start of
   [`references/dvt-contract.md`](references/dvt-contract.md).
2. Identify every affected flow family from the change or observed operation.
3. Read the corresponding sections in the contract; do not assume a previously verified contract is immutable.
4. Audit current code before judging it. Code is implementation fact; the contract is design intent. Use git history when they conflict.
5. Run the smallest applicable checker under `tools/python/`. Use `check_all_flows.py` for a broad or full-day trace and domain checkers for focused work.
6. Compare checker output with the raw trace around each failure. A checker narrows evidence; it does not replace causal analysis.
7. When behavior or architecture intentionally changes, update the contract in the same change and explain which clause changed and why.
8. After build, DVT, and required on-machine smoke checks pass, commit the verified state immediately as a recoverable baseline.

For coordinate or mirror changes, also use the `row-chart-coordinates` skill.

## Log recording modes

PropertyGrid `5. Log 設定（記錄／除錯） > 記錄範圍` is the runtime SSoT:

- `日常運行`: operations, connections, errors, persistence, capture lifecycle,
  and anomaly-triggered performance evidence.
- `流程驗證`: adds coordinate/direction snapshots, prefit, main/chart ranges, and
  other evidence required for full DVT checking.
- `完整診斷`: adds per-second paint/stat evidence and raw `UiActionLogger`
  JSONL under `D:\Anilox\Logs\fsm\`.

`check_all_flows.py` reads the `log mode=...` session marker. Rules that require DVT-only evidence
must return `NOT COVERED`, not `FAIL`, when a session used the operational mode. Legacy traces have
no marker and are treated as fully instrumented because all probes were unconditional then.

## Automated UI driver

`tools/dvt/AniloxRoll.DvtRunner` operates the real WinForms controls through Windows UI
Automation and Win32 control messages. Its scenario steps reference contract IDs and wait for
the smallest required Flow evidence; cross-step order, forbidden lines, counts, and completeness
remain owned by `tools/python/check_all_flows.py`.

- Use the runner to automate repeatable smoke/DVT actions while an operator observes the screen.
- A scenario must not duplicate the full contract or invent a second PASS standard.
- Every UI action or setting change step must identify its owning contract.
- The runner restores changed PropertyGrid values and attempts to stop an active Grab on abort.
- Automated cleanup gives the product up to 60 seconds to close normally. If failure recovery or
  disconnected hardware prevents shutdown, the runner force-terminates its test process so a
  failed campaign cannot leave an orphaned product instance; that run does not satisfy the normal
  shutdown contract.
- `NOT COVERED` still means the scenario did not exercise that behavior. Physical disconnects,
  visual correctness, stress, and soak evidence remain separate test layers.
