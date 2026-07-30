# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260730-021810`
- Commit: `9307c6c`
- Working tree: dirty
- Mode: `All`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-07-30 02:20:10 +08:00
- Raw artifacts: `artifacts/test-reports/20260730-021810-9307c6c/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Unit | Resource trend guard tests | **PASS** | One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail. | exit code 0 | 0.23 |
| Functional | Python flow checker tests | **PASS** | All discovered checker self-tests pass; 0 failures. | Ran 119 tests in 0.015s; OK | 0.12 |
| Unit | .NET unit tests | **PASS** | All discovered unit tests pass; 0 failures. | total 147, passed 147, failed 0, not-executed 0 | 3.43 |
| Integration | .NET integration tests | **PASS** | All discovered integration tests pass; 0 failures. | total 114, passed 114, failed 0, not-executed 0 | 6.81 |
| DVT functional | DVT Runner self-check | **PASS** | Launch the exact app, restore changed settings, close cleanly, and finish the checker with exit code 0. | Result: PASS; Status: PASS：Runner 自我檢查（不 Grab） | 7.15 |
| Large-data UI DVT | Review and report 30,000-record DVT | **PASS** | Load exactly 30,000 grab IDs; reload jumps to newest; Review rapid/period navigation, enhancement, direction, heatmap, and display crop preserve data contracts; Report single/range curves, Y-axis toggle, fail filter, cross-tab curve reuse, clean shutdown, and the full checker pass. | Result: PASS; Status: PASS：回顧與報表 30,000 筆; grabIds=30000; maxUiStall=1000ms; checker=44 PASS / 0 FAIL | 39.7 |
| Stress | Offline stress tests | **PASS** | All 9 high-frequency and mock Bridge cases pass for the configured wall-clock budget. | total 9, passed 9, failed 0, not-executed 0 | 54.12 |
| Soak | Offline endurance tests | **PASS** | The mixed IO, CSV/CFG, statistics, remote-copy, and cleanup workload runs continuously; queue drains; temp files clean up; Private Bytes <=512 MB, handles <=50, and threads <=15 after warm-up. | total 1, passed 1, failed 0, not-executed 0 | 8.3 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered without wiring

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run the on-machine DVT and soak campaign when wiring is available.
