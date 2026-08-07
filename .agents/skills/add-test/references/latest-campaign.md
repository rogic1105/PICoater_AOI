# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260807-084007`
- Commit: `2e447d5`
- Working tree: dirty
- Mode: `Functional`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-08-07 08:40:26 +08:00
- Raw artifacts: `artifacts/test-reports/20260807-084007-2e447d5/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Unit | Resource trend guard tests | **PASS** | One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail. | exit code 0 | 0.25 |
| Functional | Python flow checker tests | **PASS** | All discovered checker self-tests pass; 0 failures. | Ran 171 tests in 0.027s; OK | 0.14 |
| Unit | .NET unit tests | **PASS** | All discovered unit tests pass; 0 failures. | total 204, passed 204, failed 0, not-executed 0 | 3.99 |
| Integration | .NET integration tests | **PASS** | All discovered integration tests pass; 0 failures. | total 127, passed 127, failed 0, not-executed 0 | 6.6 |
| DVT functional | DVT Runner self-check | **PASS** | Launch the exact app, restore changed settings, close cleanly, and finish the checker with exit code 0. | Result: PASS; Status: PASS：Runner 自我檢查（不 Grab） | 7.76 |

## Improvement record

- Live and review normalization now update retained Hessian standard maps and column/row curves with latest-only coalescing; DVT probes compare same-frame display payload and curve peaks, while report summaries retain their capture-normalized semantics.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
