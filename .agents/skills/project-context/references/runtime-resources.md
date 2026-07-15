# Runtime resource measurement

Use measured runtime evidence instead of static memory or hardware estimates. Frame size, camera count, save mode, LOD state, accumulated history, and GPU implementation all change the result.

## Current instruments

| Evidence | Source | What it measures |
|---|---|---|
| `resource-monitor-*.csv` | `CameraFrameSaver.InitResourceLog` / `WriteResourceLine` | mode, camera/frame size, processing time, save bytes, session frames, RAM, CPU, VRAM |
| Settings hardware/resource list | `AniloxRollForm.SettingsTabs.cs` + `Telemetry.cs` | current frame size, GPU time, save size, session volume, RAM, estimated VRAM |
| `[UiStall]`, `[UiPing]`, `[UiSlow]` | `FlowTrace` and instrumented handlers | UI starvation, blocking, GC correlation, slow owners |
| review resource rows | `CameraFrameSaver.AppendReviewResourceLog` | review mode, camera/image count, load time, RAM |
| drop/phase/parameter logs | acquisition telemetry | frame progress, physical timing, parameter-change gaps |

## Measurement method

1. Record commit, runtime JSON, machine role, camera count, frame dimensions, line rate, save mode, display mode, and dataset size.
2. Capture an idle baseline after warm-up.
3. Run one workload at a time: live grab, review ID/range navigation, report range query, or remote copy.
4. Compare steady windows and worst cases. Separate one-time allocation from monotonic growth.
5. Correlate resource samples with trace timestamps and DVT intents before assigning a cause.

## Interpretation

- Live GPU/RAM scales mainly with active frame dimensions, camera count, pipeline buffers, and LOD/display caches.
- Review load scales with decoded images, stitched frames, curve arrays, and selected range; latest-only cancellation must release stale results.
- Report memory should remain bounded by virtualized visible rows and cached parsed data, not ListView item count.
- Save throughput and disk lifetime must be calculated from `SaveKB`/session bytes measured in the actual recipe. Do not reuse old BMP/JPEG daily-volume estimates.
- A simulated extra pipeline is not equivalent to another physical camera: MIL buffers, grabber memory, transport, and scheduling differ. Label simulated and physical counts separately.
- Low CPU usage during GPU saturation or IO backpressure does not prove CPU headroom; always read it together with per-camera fps, `ProcessMs`, dropped frames, and queue growth.
- Save-mode comparisons must include per-camera frame counts. Similar aggregate throughput can hide one camera chain falling behind another.
- Hardware purchasing or lifetime recommendations require a dated measurement report outside the agent rules; they are not durable architecture knowledge.

## Dated baseline

[`docs/user-manual/hardware-specs.html`](../../../../docs/user-manual/hardware-specs.html) is the
2026-04-10 Phantom seven-camera measurement report. It supersedes the earlier 2026-04-09 drafts.
Treat its numbers as historical evidence for that commit, branch, machine, and recipe only;
remeasure before applying them to current production sizing.
