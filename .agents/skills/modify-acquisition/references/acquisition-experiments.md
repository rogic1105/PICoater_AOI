# Acquisition Experiment History

This file records closed acquisition hypotheses. Branches are temporary work queues; use the
commit or annotated tag below when an old implementation must be reconstructed.

| Experiment | Reference | Evidence | Decision |
|---|---|---|---|
| V1 warm camera phases | `a7d1657` | No independent final hardware result was recorded in the commit. | Closed; not merged. |
| V2 sequenced parameter timing | `3c45759` | No independent final hardware result was recorded in the commit. | Closed; not merged. |
| V3 verified standby | `a4b501d` | Release x64, 115 unit, 98 integration, and 87 Python checks passed. Bench testing still found an opening gap before the first complete post-edge frame. | Closed; superseded by the accepted boundary design. |
| No hot standby plus 100 ms IO polling | tag `experiment/no-hot-standby-20260724` (`e8f5ce3`) | Cold-start testing produced 2/7 complete ten-frame cycles; 5/7 cycles had only nine frames. Release x64 later built successfully. | Rejected; do not merge. |
| Queued timing parameters after Grab | tag `experiment/queued-camera-parameters-20260724` (`bc4e467`) | Release x64, 110 unit, and 80 Python checks passed. Hardware smoke still produced dropped-frame and timing regressions. | Deferred; do not merge without a new hardware design and test plan. |
| Accepted synchronized IO boundaries | `f02e668` | Release x64, 119 C# and 90 Python checks passed; hardware smoke completed 26/26 cycles and DVT reported 21 PASS / 0 FAIL. | Merged to `main`. |

When revisiting a closed experiment, start a new short-lived branch from current `main`. Do not
continue directly from these historical commits because later storage, review, display, and DVT
changes are absent from them.
