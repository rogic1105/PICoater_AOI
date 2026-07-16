# Stress and soak verification

Use this reference for long-loop, load, endurance, or failure-recovery work. Keep exact limits tied to a dated baseline log; do not copy historical hardware estimates into assertions.

## Automated stress tests

- Project: `tests/AniloxRoll.Monitor.Stress.Tests/`
- Runner: `tests/TestRunner.bat` / `tests/TestRunner.ps1`
- Duration input: `STRESS_MINUTES`
- Build and run only `Release|x64`.

Inspect the current test names before documenting a case; do not rely on old counts.

## On-machine phases

1. **Smoke, 10-30 minutes**: grab/start/stop, display modes, review navigation, report ranges, persistence, and hardware status.
2. **Load**: production camera count, frame dimensions, line rate, save mode, and accumulated CSV volume.
3. **Interaction under load**: parameter changes, tab switches, rapid review IDs, report range scrolling, zoom/pan, and background preview.
4. **Soak, shift or 24 hours**: no crash; frame, save, remote-copy, RAM, handle, VRAM, and queue trends remain bounded.
5. **Failure injection**: disconnect/reconnect PLC, light, camera, storage network, and low-disk conditions one at a time. Verify recovery and retained data.

For the shift/24-hour phase, run `IoBridge.IoSimulator` as the repeatable IO source and schedule
start/stop/Mura transitions throughout the run. Include at least one simulator restart and one
inspection-app restart; the soak still needs separate short runs against the physical IO before
release because the simulator cannot reproduce wiring, switch, or power-supply faults.

Low-disk retention does not require filling a disk. On an isolated test volume, set
`LocalMinFreeGB` above that volume's current free space, run cleanup once, and verify oldest-first
deletion, pending-copy protection, `_ticks.csv`/`_curve_summary` deletion, and daily CSV retention.
Never point this test at production captures.

## Evidence

- `D:\Anilox\Logs\trace-*.log`: flow, stalls, slow handlers, hardware transitions.
- `D:\Anilox\Logs\resource-monitor-*.csv`: process/GPU/save/frame measurements.
- `D:\Anilox\Logs\dropdiag-*.csv`, `phaselog-*.csv`, `paramchange-*.csv`: acquisition timing and parameter-change diagnosis.
- `tools/python/check_all_flows.py`: broad DVT check for a long session.
- Domain checkers under `tools/python/flow_checks/`: focused diagnosis.

## Pass criteria

- Define numeric limits before the run from a recent clean baseline on the same machine and recipe.
- Judge trends, not one sample: no monotonic RAM/handle/VRAM/queue growth after warm-up.
- Every user intent must retain its DVT owner; no forbidden flow, orphan, incomplete begin/done, or stale-token winner.
- A failure-injection case passes only if the state transition, operator indication, data integrity, and recovery path all agree.
- Save the tested commit, runtime settings, duration, hardware layout, and relevant logs with the result.

Do not delete captures or fill a production disk as part of an automated test. Destructive low-disk and retention scenarios require an isolated test volume.
