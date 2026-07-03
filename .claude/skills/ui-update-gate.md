# ui-update-gate

Use this skill when changing WinForms UI refresh timing, chart startup rendering, timer-driven view updates, or canvas/chart synchronization.

## Single Authoritative Gate

- One UI update flow must have one authoritative gate only. Do not add a second gate to "make it safer".
- For live overview charts, the authoritative gate is the main-display fit range: `_liveViewLeftMm/_liveViewRightMm`.
- Delay `chartLiveColumn` drawing until that fit range is ready. Do not also gate the same refresh on chart `PostPaint`, `Resize`, `ClientSize`, or chart-layout-ready state.
- Chart helpers own deterministic rendering only: axis style, tick interval, font, plot position, and valid axis assignment. They must not decide whether startup data is ready.
- Do not hide startup range bugs with axis/tick lower-bound clamps. If a range is wrong, fix the range source or the single gate that admits it.
- If chart startup is unstable, first inspect the existing fit-range gate path: `ApplyLiveViewRange` -> `_liveViewLeftMm/_liveViewRightMm` -> `LiveOverviewTimer_Tick`.

## Practical Check

Before adding a new guard flag, timer gate, paint/layout gate, or ready event:

1. Identify the current authoritative source of readiness.
2. Check whether the proposed guard duplicates that source.
3. Prefer delaying at the data/view-range boundary over delaying at the chart paint/layout boundary.
4. If two gates appear necessary, stop and refactor ownership first.
