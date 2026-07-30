# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260730-171929`
- Commit: `458942f`
- Working tree: dirty
- Mode: `PhysicalBridgeRecovery`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-07-30 17:20:25 +08:00
- Raw artifacts: `artifacts/test-reports/20260730-171929-458942f/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Physical bridge recovery DVT | Physical IO and light software recovery | **PASS** | The physical IO TCP endpoint and light serial device are each isolated and restored three times in software; every cycle raises one disconnect edge and health incident, then reconnects and resolves before clean shutdown. | IO disconnect/reconnect/raise/resolve each 3; light disconnect/reconnect/raise/resolve each 3; checker 17 PASS / 0 FAIL | 55.84 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO/light cable removal and power-cycle recovery remain untested; this run covered repeatable software endpoint and device isolation.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
