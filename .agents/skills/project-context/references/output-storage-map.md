# PICoater AOI output and storage map

> Sole owner of produced-file locations and copy/retention classification. Runtime ordering,
> recovery, and pass/fail evidence belong to `$verify-flows`. Paths are defaults; runtime values
> come from `StorageSettings`, `AppModeConfig`, and the executable directory.

## Monitoring PC

| Output | Default path | Source of truth | Remote copy | Retention |
|---|---|---|---|---|
| Daily inspection records | `D:\Anilox\Captures\yyyy\yyyyMM\yyyyMMdd.csv` | Yes | Yes | Keep |
| Raw image | `...\yyyyMMdd\{yyyyMMdd_HHmmss.fff}-{cam}_raw.jpg` | Yes | Yes | Delete oldest when low on space |
| Column processed image | `...\yyyyMMdd\{base}_proc_c.jpg` | Yes | Yes | Delete oldest when low on space |
| Row processed image | `...\yyyyMMdd\{base}_proc_r.jpg` | Yes | Yes | Delete oldest when low on space |
| Column curves | `...\yyyyMMdd\{base}_{mean|max}_c.bin` | Yes | Yes | Delete oldest when low on space |
| Row curves | `...\yyyyMMdd\{base}_{mean|max}_r.bin` | Yes | Yes | Delete oldest when low on space |
| Frame-start tick index | `...\yyyyMMdd\_ticks.csv` | Alignment source while images exist | Yes | Delete with the day output |
| Rebuildable curve summary | `...\yyyyMMdd\_curve_summary\{grabId}.mcsf` | No; bins are authoritative | No | Delete with the day output |
| Background calibration | `D:\Anilox\Bg\bg_{width}_{cam}.bin` | Yes for local acquisition | No | Not part of capture retention |
| Runtime trace and diagnostics | `D:\Anilox\Logs\` | Diagnostic evidence | No | Managed separately |
| Runtime settings | `{ExeDir}\Config\*.json`, `Radient_Config.dcf` | Yes for that machine | No | Not part of capture retention |
| Review/session state | `{ExeDir}\Config\session-state.json` | UI convenience only | No | Replaceable |
| Durable remote-copy ledger | `D:\Anilox\Captures\.remote-copy-pending\*.pending` | Delivery state | No | Remove only after confirmed publish |
| Stress dataset | `D:\Anilox\StressCaptures_30000` | Test-only | No | Remove manually after testing |

Readers accept legacy `_proc_v.jpg` / `_proc_h.jpg` and legacy curve-bin names. New writers must
emit only the c/r names above.

`_ticks.csv` is a shared index inside each date image folder. Each row maps one image base name to
its frame-start monotonic tick, allowing review to align cameras even when filenames jitter. It is
appended by `CameraFrameSaver` and recopied whenever it changes.

`_curve_summary\{grabId}.mcsf` is a per-grab UI acceleration cache created by
`SingleGrabCurveSummaryStore`; the C/R curve bins remain authoritative. Retention therefore deletes
the summary directory together with the images and bins for that date.

## Storage PC

| Output | Default path | Meaning |
|---|---|---|
| Mirrored production data | `D:\Anilox\Captures\...` | Same relative layout as monitoring PC for remotely copied outputs |
| Storage-app heartbeat | `D:\Anilox\Config\storage-app-heartbeat.json` | Atomic liveness/status snapshot from the Storage-role app |
| Cleanup request | `D:\Anilox\Config\cleanup-request.flag` | Transient fixed command; watcher consumes and deletes it |
| Publish temporary file | `{destination}.part-{guid}` | Incomplete remote copy; length-verified then atomically renamed, normally absent |
| Storage runtime settings | `D:\AniloxMonitor\Config\*.json` | Configuration for the Storage-role app only |
| Storage runtime logs | `D:\Anilox\Logs\` | Local evidence from the Storage-role app |

The SMB defaults are `\\192.168.10.20\Anilox\Captures` and
`\\192.168.10.20\Anilox\Config`. A green share probe proves SMB write/delete access; a fresh
heartbeat proves the Storage-role application is running. These are separate states.

A `.part-*` file is a remote staging file, not a capture result. The copy worker publishes it under
the final filename only after the source stayed stable and both lengths match. A crash can leave an
old part file behind; readers ignore it and a later retry publishes a fresh snapshot.

## Verification ownership

- File naming and path derivation: unit/integration tests.
- Copy, restart recovery, pending protection, and retention: StorageBridge integration tests.
- Large backlog and repeated disconnect/reconnect: stress tests.
- Shift/24-hour operation with IO simulator and hardware failure injection: soak procedure.
- Runtime sequence and operator-visible state: `$verify-flows` C/H contracts and validators.

Low-disk tests must use an isolated root or test volume. Set the effective threshold above the test
drive's current free space but below its total capacity to trigger cleanup without physically filling
the disk; thresholds at or above total capacity are invalid and must delete nothing. Never aim the test
at production captures. Storage role uses `app-mode.json` `StorageMinFreeGB`; Inspection role uses
`LocalMinFreeGB`. The storage-role deployment default is 100 GB; smaller test volumes therefore exercise
the invalid-threshold guard until their deployment setting is lowered deliberately.
