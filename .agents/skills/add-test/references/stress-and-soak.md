# Stress and soak verification

Use this reference for long-loop, load, endurance, or failure-recovery work. Keep exact limits tied to a dated baseline log; do not copy historical hardware estimates into assertions.

## Automated stress tests

- Project: `tests/AniloxRoll.Monitor.Stress.Tests/`
- Runner: `tests/TestRunner.bat` / `tests/TestRunner.ps1`
- Duration input: `STRESS_MINUTES`
- Build and run only `Release|x64`.

`tests/TestRunner.bat` is the operator entry point. It keeps functional, stress, and soak
classification separate while producing one campaign report. Raw output is written under ignored
`artifacts/test-reports/<run>/`. A recorded complete campaign overwrites
`references/latest-campaign.md`; Git history retains previous reports without accumulating report
files in the working tree.

Long campaigns use **120 minutes per cycle** by default. Each cycle must start from a known
hardware topology, complete its own shutdown, and produce its own report. Four passing two-hour
cycles provide repeatability evidence but do not replace a later uninterrupted eight-hour run when
that release criterion is required. Functional DVT scenarios remain action-bounded and finish when
their contract steps complete.

If an operator intentionally powers off, disconnects, or changes the required hardware topology,
the raw guard still fails immediately, but the durable verification record classifies the run as
`INTERRUPTED`, not as product `FAIL`. Preserve the raw failing artifact and cite the trace transition
that proves the external interruption. An interrupted run never counts toward the requested duration.

The offline stress option runs nine high-frequency logic, persistence, and mock Bridge cases.
Each of the six adjustable workloads receives one sixth of `STRESS_MINUTES`; the three bounded
Bridge/resource cases run their complete cycle counts.

The offline soak option runs one persistent mixed workload for `SOAK_MINUTES`. It interleaves
lightweight fake IO start/stop transitions, CSV/CFG persistence, statistics recomputation, remote-copy queue
drain, and temporary-file cleanup in one testhost process. After warm-up, it requires Private
Bytes growth <=512 MB, handle growth <=50, and thread growth <=15. All files stay under an
isolated `%TEMP%` directory. This proves bounded logic and file behavior only; it does not replace
the on-machine shift/24-hour phase below.

Do not use an invocation-recording mock such as Moq in a long-running polling or soak loop. Moq
retains every invocation until explicitly cleared, so the test harness itself can create linear
memory growth. Use a state-only handwritten fake when call-history verification is not part of the
acceptance criteria. The 2026-08-03 baseline replaced the Moq PLC with `SoakModbusTcpClient` and
held Private Bytes at 43.5-43.6 MB for 60 minutes and 108,486 cycles.

The physical IO + storage soak runs the real product without Grab or light output. It samples
Working Set, Private Bytes, total handles, GDI objects, USER objects, threads,
CPU time, and UI responsiveness every 30 seconds,
while the DVT runner continuously requires IO Idle and the storage two-layer green state.
Resource sampling is owned by the external runner. Do not add periodic
`Process.GetCurrentProcess().Threads` enumeration or resource probes inside the product process:
the observer must not create handles or alter the resource trend being judged. Target-process GC
generation counts are diagnostic-only and are not required by the soak acceptance gate. After
the first five minutes, the default resource guards are: handle growth <= 200, GDI/USER growth
<= 100 each, thread growth <= 25, and zero non-responsive samples. Private Bytes is judged by the
median 30-second interval rate and, after a large heap-expansion step, by the rate from that new
baseline. This distinguishes sustained growth from Server GC reserving a larger heap once while
managed memory drops. Sustained growth above 256 MB/hour fails once movement reaches 64 MB; an
absolute 4 GB increase is always an emergency failure. The report keeps the total first-to-last
rate as context but does not use that single slope as the verdict. Handle growth above 200/hour
fails once movement reaches 50. These are leak guardrails, not hardware lifetime estimates.

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
- IO Bridge polling has a dedicated kernel-resource regression:
  `IoBridgeReconnectStressTests.SustainedPolling_ReusesKernelResources` performs 1,000
  serialized Modbus polls after warm-up and permits at most 10 additional process handles.
  This guards against reintroducing per-poll `NetworkStream` overlapped Event allocation.
- Every user intent must retain its DVT owner; no forbidden flow, orphan, incomplete begin/done, or stale-token winner.
- A failure-injection case passes only if the state transition, operator indication, data integrity, and recovery path all agree.
- Save the tested commit, runtime settings, duration, hardware layout, and relevant logs with the result.

Do not delete captures or fill a production disk as part of an automated test. Destructive low-disk and retention scenarios require an isolated test volume.
