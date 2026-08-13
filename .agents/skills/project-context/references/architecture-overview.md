# PICoater AOI architecture overview

> Sole owner of current logical topology, feature owners, and refactor pressure points. The app and
> SDK `AGENTS.md` files own normative rules. Detailed paths and controls belong to
> `repository-reference.md`. Confirm owner names against code before a refactor.

## Layer model

```text
User intent
    |
    v
View / Form partial
    |
    v
ControlAdapter / Binder
    |
    v
Feature Coordinator / Presenter -----> State (SSoT / feature runtime state)
    |
    v
Application Service / Repository
    |
    v
SDK / Bridge
    |
    v
Native library / vendor SDK / hardware / persisted files

Results and events travel upward; dependencies and commands travel downward.
```

## Layer ownership matrix

| Layer | Responsibility | Monitoring | Review | Report |
|---|---|---|---|---|
| **View / Form** | Read controls, render results, emit intent | `AniloxRollForm.Live.cs`, `Background.cs` | `AniloxRollForm.Review.cs` | `AniloxRollForm.Data.cs` |
| **Adapter / Binder** | One control group's wiring, guards, formatting, busy/selection visuals | Control event adapters; display gestures are owned by SDK controls | `DateTimeNavigator`, `BusyUiBinder` | `DataDateGrabIdNavigator`, `GrabDetailListBinder`, and chart selection adapters |
| **Coordinator / Presenter** | One feature workflow, async token, debounce/latest-only, state/service composition | `LiveCameraManager` acquisition facade; `LiveDisplayCoordinator`; `GlobalMergeCoordinator`; `BackgroundCaptureCoordinator`; `CaptureStopCoordinator`; `LightConnectionCoordinator`; `IoConnectionCoordinator`; `MainWorkspaceLayoutController`; `OutputHealthPresenter` | `ReviewStitchCoordinator` (grab load orchestration), `ReviewPeriodImagePresenter` (period image lookup/decode/publish), `ReviewChartPresenter` (chart apply), `LatestGrabLoadCoordinator`, `ReviewPeriodLoadCoordinator`, `ReviewFolderCoordinator`, `InspectionSettingsCoordinator` | `DataStatisticsPresenter`, `ReportCurveVerdictPresenter` (O/X apply and DVT audit), `DataRangePreviewCoordinator` (range timing/cancellation), `ReportSingleGrabSelectionCoordinator` (33ms latest-only selection), `ReportCurveVerdictIndexCoordinator` (summary/bin fallback generation), `YieldPeriodChartPresenter`, `MuraProfileChartPresenter` |
| **State** | One truth per setting/session/runtime fact; no file IO or control mutation | `SettingsHub` plus explicit live coordinator state | `SettingsHub`, `ReviewRuntimeState`, `ReviewDisplayContent` (current images/curves and ownership) | `SettingsHub`, `ReportCurveVerdictIndex` (detail/column/row peaks plus threshold identity), plus presenter-local navigation/chart state |
| **Service / Repository** | Product rules, inspection, statistics, persistence, hardware FSM; no WinForms | `AniloxCamera`, `CameraFrameSaver`, `InspectionEngine`, `InspectionLogService`, `BackgroundProfileRepository`, `CaptureArchiveStore` (runtime container IO), `CapturePreviewAtlasCodec` (preview cache), `CaptureArchiveLegacyConverter`, `CaptureArchiveThumbnailMaintenance`, and `CaptureArchiveMigration` (offline maintenance only) | `ImageRepository`, `FrameTickIndex`, `ReviewImageDataLoader`, `ReviewPeriodDataLoader`, `SingleGrabCurveDataLoader`, `ImageCacheService`, `InspectionConfigRepository`, `InspectionImagePathRepository`, `InspectionMuraProfileRepository`, batch loaders | `InspectionCsvReader`, `InspectionStatisticsService`, `InspectionConfigRepository`, `InspectionImagePathRepository`, `InspectionMuraProfileRepository`, `SingleGrabCurveSummaryStore`, `SingleGrabCurveCache` |
| **SDK / Bridge** | Reusable mechanism and protocol/transport | `MilGrabber.Core`, `TanukiCv.Core`, `TanukiCv.Controls`, hardware bridges | `TanukiCv.Controls.ImageDisplayView`, `TanukiCv.Core` | Shared chart helpers and reusable parsers only |
| **Native / External** | ABI, vendor runtime, hardware, filesystem | `tanuki_pipeline_api`, MIL, cameras, PLC, light, storage PC | Pipeline/native image decoding and capture files | CSV and curve bin files |

Normative dependency rules and layer prohibitions are defined only in the app and SDK `AGENTS.md`.

## Main feature flows

### Monitoring

```text
Camera / MIL
  -> per-camera acquisition
  -> inspection + persistence
  -> acquisition/display coordination
  -> ImageDisplayView | WaterfallView + charts
```

Behavior contract: `$verify-flows` F1-F8.

### Review

```text
Capture files + CSV/bin
  -> repository + alignment + curve loading
  -> review workflow coordination
  -> review runtime state
  -> ImageDisplayView + review charts
```

Behavior contract: `$verify-flows` R series.

### Report

```text
CSV + curve bins
  -> statistics/query services
  -> report + chart presenters
  -> virtual detail list + yield/profile charts
```

Behavior contract: `$verify-flows` D series.

### Settings and side effects

```text
PropertyGrid / button / chart intent
  -> SettingsHub (SSoT)
  -> feature owner reacts
  -> view refresh or service/hardware side effect
```

Normative SSoT and transition rules are defined only in the app `AGENTS.md`.

## Current architecture pressure points

| Area | Current status | Refactor direction |
|---|---|---|
| `FormInteractionHelper` | Removed; responsibilities split into binder/coordinators/service/state | Do not recreate a Form-wide helper or service locator |
| `InspectionStatisticsService` | Report statistics owner after CSV parsing, CFG, image-path, and Mura-profile queries were extracted | Keep report aggregation here; do not move persistence queries back in |
| Review loading/display boundary | `ReviewStitchCoordinator` owns grab-id loading and tokens; `ReviewPeriodImagePresenter` owns period image lookup/decode/publish; `ReviewDisplayContent` owns current Bitmap/Curve lifetime; `ReviewChartPresenter` owns column/row chart application | Preserve this boundary; loaders and presenters must not reacquire each other's state or IO responsibilities |
| Background capture | `BackgroundCaptureCoordinator` owns first-set wait, timed sampling, averaging, version publication, and rollback; `BackgroundProfileRepository` owns all background file IO. `AniloxRollForm.Background` retains the UI/light/grab/health/preview envelope. | Keep native buffer ownership in `AniloxCamera` and preview behavior in `LiveDisplayCoordinator`; do not move those policies into the capture coordinator |
| Capture stop conditions | `CaptureStopCoordinator` owns the per-grab IO/time/height snapshot, timer generation, common-row threshold, and single terminal transition. The Form executes the shared stop/cleanup command. | Keep physical stop, light, persistence finalization, and IO notification in their existing owners; no duplicate stop-condition fields in Form partials |
| Hardware/output status | `LightConnectionCoordinator` owns light discovery/probe/reconnect/controller lifetime. `StorageHealthCoordinator` owns local/remote capacity observation, SMB reachability, heartbeat and remote re-probe cadence. `IoConnectionCoordinator` owns the single active IO controller generation and serializes replacement/shutdown. `AniloxRollForm.HardwareStatus` retains IO capture policy and renders hardware/output snapshots; `OutputHealthPresenter` exclusively renders independent output incidents | Keep IO Grab/Stop, request-generation validation, stop-condition behavior and BUSY/ALIVE product policy in the Form plus `IoGrabController`; keep retention and remote-copy policy in their existing services |
| `LiveCameraManager` | Acquisition facade after display extraction | Keep display state in `LiveDisplayCoordinator`; only split further from evidence |
| `OnSettingChanged` | Deliberate single serialized sequencer; route classification is centralized while owner handlers live in their feature partials | Keep one `SettingsHub.Changed` side-effect subscriber; add behavior to the owning handler and reserve the sequencer for ordering and cross-feature impacts |

## Where to read next

- Physical directory placement: [`repository-layout.md`](repository-layout.md)
- File, API, setting, and control lookup: [`repository-reference.md`](repository-reference.md)
- UI ownership rules and SSoT details: app `src/dotnet/AniloxRoll.Monitor/AGENTS.md`
- Runtime behavior and evidence: `$verify-flows`
