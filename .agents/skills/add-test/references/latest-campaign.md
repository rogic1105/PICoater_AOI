# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260731-133218`
- Commit: `96035a3`
- Working tree: dirty
- Mode: `PhysicalCaptureSoak`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-07-31 15:33:07 +08:00
- Raw artifacts: `artifacts/test-reports/20260731-133218-96035a3/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Physical capture soak | Physical repeated capture soak | **PASS** | High 10 seconds / Low 4 seconds for the configured duration; every High produces one request, gate, aligned first set, image-before-curve result, clean gate close, archive, and remote enqueue; storage and light remain green; UI and resource guards pass; clean shutdown. | cycles=514/514; flowCountGuards=6/6; checker=31 PASS/0 FAIL; resources samples=239 privateMB=17059.1->16274.4 max=42366.9 handles=1485->1633 max=1633 gdi=136->137 user=283->286 threads=149->150 ratesPerHour=private:-408.3MB medianPrivate:125082.1MB postExpansionPrivate:0MB postExpansionSeconds=0 largestPrivateStepMB=26890.4 cyclic=True troughDeltaMB=-801.7 troughRateMBPerHour=-834.3 handles:77 observer=external | 7248.6 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Seven-camera full-load acquisition remains untested; this run covered only the connected cameras.
- Background capture and preview are covered by the separate PhysicalCamera scenario, not this run.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
