# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260731-082930`
- Commit: `eec7b84`
- Working tree: dirty
- Mode: `PhysicalRetention`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-07-31 08:29:41 +08:00
- Raw artifacts: `artifacts/test-reports/20260731-082930-eec7b84/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Physical retention DVT | Physical low-disk retention recovery | **PASS** | A marker-protected TEMP root holds two complete historical days; the threshold is derived from current free space; only the oldest day and its CSV are deleted; the newer day remains; low-space and cleanup incidents complete raise, resolve, and individual acknowledgement; settings and fixture are cleaned up. | threshold=1554GiB free=1668911194112 fixture=450621440B oldest=deleted newer=preserved; freed=429MB; outputHealth=6 events/5 states/0 invalid; checker=16 PASS/0 FAIL | 10.75 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption and backlog transfer are covered by their separate recovery campaign, not this run.
- Retention on the storage PC's own local disk remains separate; this run exercised the shared cleanup core and inspection-PC UI state with a marker-protected isolated fixture.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
