# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260731-083512`
- Commit: `8f184b4`
- Working tree: dirty
- Mode: `PhysicalSoak`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-07-31 10:35:23 +08:00
- Raw artifacts: `artifacts/test-reports/20260731-083512-8f184b4/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Unit | Resource trend guard tests | **PASS** | One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail. | exit code 0 | 0.23 |
| Physical soak | Physical IO and storage soak | **PASS** | Fixed hardware topology; IO and storage stay green; UI always responds; Private Bytes sustained growth <=256 MB/hour and total delta <=4 GB; handles/GDI/USER/threads stay within guards; clean shutdown. | Result: PASS; Status: PASS：IO＋儲存電腦待機耐久測試（不 Grab）; resources samples=237 privateMB=2751.6->3213.1 max=3213.1 handles=1256->1365 max=1367 gdi=134->135 user=281->283 threads=106->129 ratesPerHour=private:241.1MB medianPrivate:154.5MB postExpansionPrivate:19.9MB postExpansionSeconds=1245.4 largestPrivateStepMB=203.6 handles:57 observer=external | 7209.9 |

## Improvement record

- Inspect the tested commit and worktree diff for product changes; the campaign runner does not infer them.
- The commit STAR body records the exact implementation change and verified result.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
