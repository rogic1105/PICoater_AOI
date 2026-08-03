# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260803-083546`
- Commit: `1459410`
- Working tree: dirty
- Mode: `Soak`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-08-03 09:35:52 +08:00
- Raw artifacts: `artifacts/test-reports/20260803-083546-1459410/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Build | Release x64 solution build | **PASS** | Release\|x64 build; 0 compiler errors; 0 warnings. | exit code 0 | 3.48 |
| Soak | Offline endurance tests | **PASS** | The mixed IO, CSV/CFG, statistics, remote-copy, and cleanup workload runs continuously; queue drains; temp files clean up; Private Bytes <=512 MB, handles <=50, and threads <=15 after warm-up. | total 1, passed 1, failed 0, not-executed 0 | 3602.34 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
