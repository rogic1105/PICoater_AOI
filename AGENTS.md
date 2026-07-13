# PICoater AOI - Codex Project Rules

This file is the repository instruction SSoT. Nested `AGENTS.md` files add rules for their subtree.
`CLAUDE.md` files are compatibility entry points only and must not duplicate rules.

## Repository Boundaries

- `src/dotnet/AniloxRoll.Monitor/`: product WinForms application. Product policy stays here.
- `sdk/`: reusable libraries only. Read `sdk/AGENTS.md` before editing; SDK code must not depend on the app.
- `sdk/Bridges/`: hardware and external-system protocol/transport libraries. Product policy stays in the app.
- `tools/`: repository-wide tools. Component-specific executable samples stay under that component's `samples/`.
- `tests/`: automated unit, integration, and stress tests.
- `algtest/`: algorithm prototypes and feasibility work, not automated product tests.
- `docs/`: developer references and operator documentation.
- `.agents/skills/`: repo-scoped Codex workflows. Use `$project-context` for detailed file, setting, control, and repository maps.

Dependency direction is `src -> sdk`; SDK and Bridge projects must never reference
`AniloxRoll.Monitor`, `InspectionSettings`, GrabId, Mura, or other product policy.

## Instruction And Skill Routing

- App UI, monitoring, review, or report architecture: read nested
  `src/dotnet/AniloxRoll.Monitor/AGENTS.md` and use `$modify-ui`.
- SDK, algorithms, reusable controls, or shared mechanisms: read `sdk/AGENTS.md`.
- IoBridge changes: also read `sdk/Bridges/IoBridge/AGENTS.md` and use `$add-hardware-bridge` when appropriate.
- Native C/C++ API or C# interop: use `$add-native-api`.
- Tests: use `$add-test` before choosing unit, integration, stress, UI, or DVT.
- Acquisition, MIL, cameras, CLProtocol, PLC, telemetry, background capture: use `$modify-acquisition`.
- Data tab, CSV, statistics, report list, period/range charts: use `$modify-data-stats`.
- Image pipeline, buffers, curve generation, persistence: use `$modify-pipeline`.
- C# 7.3, .NET Framework 4.8, WinForms, settings persistence: use `$csharp-patterns`.
- Row chart, vertical coordinates, mirroring, top/bottom direction: use `$row-chart-coordinates`.
- Event wiring, async flow, display mode, persistence, navigation, or trace diagnosis: use `$verify-flows`.
- Network, SMB, inspection/storage PC deployment: use `$deploy-network`.

Codex may select skills implicitly from their descriptions. When a task is high-risk or crosses
several domains, name the required `$skill-name` explicitly and use all applicable skills.

## Single Source Of Truth

Any formula, algorithm, constant, or order-sensitive sequence that represents the same truth must
have one implementation. When a second copy appears, extract it immediately unless the two copies
are intentionally allowed to evolve independently.

- Code is implementation fact; documents express design intent. When they conflict, inspect code
  and git history before deciding which side changes.
- A verified contract is the current approved behavior, not immutable truth. Intentional evolution
  updates the contract consciously before or with code.
- Defaults live in `Settings/Models/Defaults/*Defaults.cs`; models and stores reference them rather
  than copying literals.
- Settings changes flow through `SettingsHub.Set`, `SetBatch`, or `NotifyExternalChange` except for
  clearly marked bootstrap code.
- Installing a flow log instrument and updating the `$verify-flows` DVT contract are one action.

## Product Terminology

Use one axis vocabulary without exceptions:

| Physical axis | Code term | Chinese UI | Meaning |
|---|---|---|---|
| X | `col` / Column | 欄 | Along roller width; camera stitching direction |
| Y | `row` / Row | 列 | Along material travel direction |

Do not reintroduce ambiguous Vertical/Horizontal business names. Framework properties such as
`Orientation`, `DockStyle`, or `ScrollBars.Vertical` are exempt. Existing WinForms control names are
not mechanically renamed unless the task explicitly includes that migration.

## Test Layers

| Layer | Location | Rule |
|---|---|---|
| Unit | `tests/AniloxRoll.Monitor.Tests/` | Pure logic, no file or hardware IO, mock boundaries, normally under 5 ms/case |
| Integration | `tests/AniloxRoll.Monitor.Integration.Tests/` | File/JSON/CSV IO or mock hardware, normally under 1 s/case |
| Stress | `tests/AniloxRoll.Monitor.Stress.Tests/` | Long loops, soak, load, or timing-sensitive endurance work |
| DVT log checker | `tools/python/check_*_flows.py` | Validates runtime behavior contracts from trace logs; not a unit test |

Benchmarks live with the measured component, not under `tests/`. Add `InternalsVisibleTo` when a
new test assembly needs internal app APIs.

## Native And Interop Boundaries

- Product P/Invoke declarations live in
  `src/dotnet/AniloxRoll.Monitor/Interop/NativeMethods.cs`.
- Reusable wrapper APIs live in `sdk/TanukiCv/dotnet/TanukiCv.Core` only when they are independent
  of product policy.
- Keep ABI details explicit: calling convention, bool/string marshaling, ownership, dimensions,
  and buffer lifetime.
- Persistent config models use `int`, `double`, and `string`; do not serialize `MIL_INT` structs.

## Configuration

- Runtime JSON lives under `{ExeDir}/Config/`, not AppData or ProgramData.
- Missing config files regenerate from `*Defaults.cs` and are written back.
- Runtime config JSON is not committed or copied from source. DCF binary files are the explicit
  exception because defaults do not generate them.
- To restore defaults, delete the executable's generated `Config/*.json` and restart.

## Implementation Rules

- Before implementing a state machine, write the complete `State + Event -> Next State + Action`
  table. Prefer explicit transitions over compound boolean guards.
- Keep edits scoped to one responsibility. Do not combine architecture migration, behavior changes,
  and unrelated cleanup in one commit.
- Retired modes, controls, and APIs leave no compatibility branches or stale terminology unless an
  explicit external compatibility contract requires them.
- Comments explain non-obvious constraints or ordering, not line-by-line mechanics.
- New `.cs` files must be added to the .NET Framework project file; this repository uses C# 7.3.

## Documentation Ownership

- Root `AGENTS.md`: durable repository rules and skill routing.
- Nested `AGENTS.md`: subtree architecture and ownership.
- `.agents/skills/*/SKILL.md`: repeatable task workflow.
- `$verify-flows` reference: behavior/DVT contracts.
- `docs/dev/`: large engineering context and rationale.
- `docs/user-manual/`: operator-visible behavior.

Do not duplicate full rules across these surfaces. When behavior changes, update the owning skill or
DVT contract and operator docs in the same change. Use `$project-context` instead of expanding root
instructions with large lookup tables.

## Build Verification

- Build only `Release|x64`; do not use Debug or AnyCPU.
- After changing `.cs`, `.csproj`, or `.sln`, build immediately and require zero compiler errors.
- Product entry: `PICoater_AOI.sln`. SDK tools: `sdk/Tools.sln`. Bridge solutions live under
  `sdk/Bridges/*/`. A single csproj may be built directly when narrower verification is appropriate.
- Always pass `/p:Configuration=Release /p:Platform=x64`; the product depends on the AMD64 MIL SDK.
- Do not place custom imports in Visual Studio reserved `ImportGroup` sections.
- Do not change shared `.Core` output paths inside this monorepo; project references depend on them.
- Release logging uses `Trace.WriteLine` or `Console.WriteLine`; `Debug.WriteLine` is a no-op.

## Git Workflow

Do not commit or push unless the user explicitly requests it.

Before commit:

1. Build the affected Release|x64 target and run proportional unit/integration/DVT checks.
2. Verify affected `AGENTS.md`, skills, DVT contract, lookup references, and operator docs.
3. Grep retired terminology and require zero unintended remnants.
4. Keep unrelated user changes and untracked files out of the commit.
5. Split unrelated themes into separate commits; commit messages explain why, not only what.
6. Commit an on-machine verified green state immediately when the user has authorized commit/push.
