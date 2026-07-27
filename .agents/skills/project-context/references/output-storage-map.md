# PICoater AOI output and storage map

> Sole owner of produced-file locations and copy/retention classification. Runtime ordering,
> recovery, and pass/fail evidence belong to `$verify-flows`. Paths are defaults; runtime values
> come from `StorageSettings`, `AppModeConfig`, and the executable directory.

## Monitoring PC

| Output | Default path | Source of truth | Remote copy | Retention |
|---|---|---|---|---|
| Daily inspection records | `D:\Anilox\Captures\yyyy\yyyyMM\yyyyMMdd.csv` | Yes | Yes | Delete with the oldest complete day when low on space |
| Per-grab capture archive | `...\yyyyMMdd\{grabId}.acap` | Yes; contains authoritative raw/processed JPEG, C/R curves, camera id, frame ticks, plus three rebuildable 1920x1080 preview atlases | Yes, after Grab finalization | Delete with the oldest complete day when low on space |
| Legacy loose JPG/BIN and `_ticks.csv` | Manually selected old roots only | Read compatibility, never written by the current product | No new delivery | Existing old data follows its original root policy |
| Rebuildable curve summary | `...\yyyyMMdd\_curve_summary\{grabId}.mcsf` | No; legacy bins or `.acap` curve records are authoritative | No | Delete with the day output |
| Background calibration | `D:\Anilox\Bg\bg_{width}_{cam}_{yyyyMMdd-HHmmssfff}.bin`; active manifest `Bg\active-background.json` | Yes for local acquisition | No | Active set protected; a successful low-space cleanup also removes inactive timestamped sets |
| Runtime trace and diagnostics | `D:\Anilox\Logs\` | Diagnostic evidence | No | Cataloged logs expire after `LogRetentionHours` (default 168 h) |
| Runtime settings | `{ExeDir}\Config\*.json`, `Radient_Config.dcf` | Yes for that machine | No | Not part of capture retention |
| Review/session state | `{ExeDir}\Config\session-state.json` | UI convenience only | No | Replaceable |
| Durable remote-copy ledger | `D:\Anilox\Captures\.remote-copy-pending\*.pending` | Delivery state | No | Remove after confirmed publish or explicit retention cancellation; corrupt markers move to `quarantine\` |
| Stress dataset | `D:\Anilox\StressCaptures_30000` | Test-only | No | Remove manually after testing |

Readers accept legacy loose JPG/BIN names when an engineer explicitly selects an old root. New
production writes never create loose capture assets.

Each new Grab writes one `{grabId}.acap`. Stop waits for every accepted camera frame to finish
appending, then adds one raw, column-processed, and row-processed atlas. Each atlas packs the
tick-aligned camera strips into one JPEG bounded by 1920x1080 and stores camera rectangles for
in-memory splitting. Only after this finalization are the archive and daily CSV enqueued for remote
copy. Full JPEG and curve records inside the archive remain authoritative.

The repository source of truth for the MIL binary configuration is
`sdk/MIL/Config/Radient_Config.dcf`. The product and MIL monitor sample link that one file into
their projects and copy it to `{ExeDir}\Config\Radient_Config.dcf` at build time.

Daily inspection CSV keeps data rows and versioned `#CFG` rows together. Each data row belongs to
the nearest preceding `#CFG`; this intentionally avoids a second settings file that could become
out of sync after a crash. A new snapshot contains the complete `OPS + START + CROP` layout,
column/row normalization, `RidgeSigma` (細線濾除), thresholds, and capture parameters.

Frame-start ticks now live in each archive record. `_ticks.csv` is legacy read compatibility only.

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
| Storage runtime settings | `C:\AniloxMonitor\Config\*.json` | Configuration for the Storage-role app only |
| Storage runtime logs | `D:\Anilox\Logs\` | Local evidence from the Storage-role app |

The SMB defaults are `\\192.168.10.20\Anilox\Captures` and
`\\192.168.10.20\Anilox\Config`. A green share probe proves SMB write/delete access; a fresh
heartbeat proves the Storage-role application is running. These are separate states.

A `.part-*` file is a remote staging file, not a capture result. The copy worker publishes it under
the final filename only after the source stayed stable and both lengths match. A crash can leave an
old part file behind; readers ignore it, and the worker deletes parts older than 24 hours when that
destination directory is next used. Full-day retention removes the rest with the date folder.

The managed log catalog is `trace-*.log`, `resource-monitor-*.csv`, `dropdiag-*.csv`,
`phaselog-*.csv`, `paramchange-*.csv`, `ui-actions-*.jsonl`, `io-*.log`, and
`AniloxRoll-crash.log`. Product startup injects the selected writable runtime-log directory into
`IoLogger`, so trace, IO, and crash evidence are retained together. `paramchange` is created lazily
only after an actual operator parameter change; control initialization is not a parameter change.
`LogRetentionService` deletes only cataloged files older than the configured hours; unknown operator
files and logs created by the current process are not cleanup candidates.

## Verification ownership

- File naming and path derivation: unit/integration tests.
- Copy, restart recovery, pending cancellation, and full-day retention: StorageBridge integration tests.
- Large backlog and repeated disconnect/reconnect: stress tests.
- Shift/24-hour operation with IO simulator and hardware failure injection: soak procedure.
- Runtime sequence and operator-visible state: `$verify-flows` C/H contracts and validators.

Low-disk tests must use an isolated root or test volume. Set the effective threshold above the test
drive's current free space but below its total capacity to trigger cleanup without physically filling
the disk; thresholds at or above total capacity are invalid and must delete nothing. Never aim the test
at production captures. Storage role uses `app-mode.json` `StorageMinFreeGB`; Inspection role uses
`LocalMinFreeGB`. In Storage role the deployment value is copied into the same PropertyGrid field at
startup, and edits are synchronized back to `app-mode.json`; the operator therefore sees the effective
threshold. The storage-role deployment default is 100 GB; smaller test volumes exercise the input clamp
until their setting is lowered deliberately.
