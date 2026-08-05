# Stress and soak verification

Use this reference for long-loop, load, endurance, or failure-recovery work. Keep exact limits tied to a dated baseline log; do not copy historical hardware estimates into assertions.

## Automated stress tests

- Project: `tests/AniloxRoll.Monitor.Stress.Tests/`
- Runner: `tests/TestRunner.bat` / `tests/TestRunner.ps1`
- Duration input: `STRESS_MINUTES`
- Build and run only `Release|x64`.

Inspect the current test names before documenting a case; do not rely on old counts.

## On-machine phases

Follow this verification ladder in order:

1. **Refactor slice**: move one responsibility at a time. After every slice, require the affected
   Release x64 build, automated checks, and a short smoke of the touched behavior. Do not stack
   several unverified slices and postpone all functional evidence until the end.
2. **Complete functional DVT**: after the planned refactor slices are integrated, run monitoring,
   background capture/preview, review, report, settings, bridges, persistence, and shutdown once
   against the golden flow contract.
3. **Stress/load**: run repeated IO cycles, production-sized files, rapid navigation, parameter
   changes, tab switches, zoom/pan, remote-copy recovery, and low-disk retention. This phase proves
   throughput and race boundaries, not multi-hour stability.
4. **Soak, shift or 24 hours**: no crash; frame, save, remote-copy, RAM, handle, VRAM, and queue
   trends remain bounded.
5. **Failure injection**: disconnect/reconnect PLC, light, camera, storage network, and low-disk
   conditions one at a time. Verify recovery and retained data.

The first-frame phase gate needs a short Grab/Curve/save smoke before its branch can be integrated.
Its rare-event evidence is deferred to stress: repeat at least the previous comparable baseline
(158 IO cycles / about 30 minutes) and require every rejected head probe to stop before firstFrame,
CSV, or persistence. The shift/24-hour run remains a separate final phase.

For the shift/24-hour phase, run `IoBridge.IoSimulator` as the repeatable IO source and schedule
start/stop/Mura transitions throughout the run. Include at least one simulator restart and one
inspection-app restart; the soak still needs separate short runs against the physical IO before
release because the simulator cannot reproduce wiring, switch, or power-supply faults.

Low-disk retention does not require filling a disk. On an isolated test volume, set
`LocalMinFreeGB` above that volume's current free space, run cleanup once, and verify oldest-first
deletion, pending-copy protection, complete-day `.acap` deletion, and matching daily CSV deletion.
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
