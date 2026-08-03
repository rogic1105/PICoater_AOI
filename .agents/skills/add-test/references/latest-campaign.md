# Latest automated test campaign

> This file is overwritten by the next recorded campaign. Git history is the durable record.

- Result: **PASS**
- Run: `20260803-100950`
- Commit: `9036db0`
- Working tree: clean
- Mode: `PhysicalSoak`
- Machine: `DESKTOP-C1MN5KD`
- Finished: 2026-08-03 14:10:10 +08:00
- Raw artifacts: `artifacts/test-reports/20260803-100950-9036db0/` (local, ignored by Git)

## Results

| Layer | Check | Result | Theory / acceptance | Experimental value / evidence | Seconds |
|---|---|---:|---|---|---:|
| Build | Release x64 solution build | **PASS** | Release\|x64 build; 0 compiler errors; 0 warnings. | exit code 0 | 4.11 |
| Unit | Resource trend guard tests | **PASS** | One-time heap expansion passes; sustained 330 MB/hour growth and post-expansion growth fail. | exit code 0 | 0.3 |
| Physical soak | Physical IO, storage, and light soak | **PASS** | Fixed hardware topology; IO, storage, and light stay healthy; UI always responds; Private Bytes sustained growth <=256 MB/hour and total delta <=4 GB; handles/GDI/USER/threads stay within guards; clean shutdown. | 4.00 hours; 475 samples; Private 2761.3->3241.8 MB; median 71.5 MB/hour; post-expansion 22.2 MB/hour; handles +109; GDI +2; USER +4; threads +22; UI non-responsive=0; checker 17 PASS / 0 FAIL. | 14415.2 |

## Conclusion

- This is a valid four-hour connected qualification run, not the final eight-hour release soak.
- IO polling, storage heartbeat, light connectivity, UI responsiveness, resource guards, and clean shutdown passed for the full run.
- The uninterrupted eight-hour connected soak and seven-camera full-load test remain pending.

## Not covered by this campaign

- Physical camera/grabber acquisition, seven-camera frame load, background capture, and live Grab.
- Physical IO and light disconnect/reconnect timing.
- Storage-PC SMB interruption, remote backlog transfer, and real-disk/UI low-space status and recovery.
- Shift/24-hour product soak with the IO simulator, cameras, storage transfer, and operator interactions.

These cases remain **NOT COVERED**, not PASS. Run their dedicated DVT or soak campaign before release.
