# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260813-115046`
- Commit: `7d2f389`
- Working tree: dirty
- Mode: `Functional`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-08-13 11:51:11 +08:00
- Raw artifacts: `artifacts/test-reports/20260813-115046-7d2f389/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Build | Release x64 solution build | **PASS** | Release\|x64 build; 0 compiler errors; 0 warnings. | exit code 0 | 5.25 |
| Unit | Resource trend guard tests | **PASS** | One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail. | exit code 0 | 0.81 |
| Functional | Python flow checker tests | **PASS** | All discovered checker self-tests pass; 0 failures. | Ran 205 tests in 0.042s; OK | 0.25 |
| Unit | .NET unit tests | **PASS** | All discovered unit tests pass; 0 failures. | total 266, passed 266, failed 0, not-executed 0 | 4.77 |
| Integration | .NET integration tests | **PASS** | All discovered integration tests pass; 0 failures. | total 137, passed 137, failed 0, not-executed 0 | 6.53 |
| DVT functional | DVT Runner self-check | **PASS** | Launch the exact app, restore changed settings, close cleanly, and finish the checker with exit code 0. | Result: PASS; Status: PASS：Runner 自我檢查（不 Grab） | 7.64 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
