# PICoater AOI repository layout

> Sole owner of physical directory responsibilities and placement decisions. Logical layers and
> current feature owners belong to [`architecture-overview.md`](architecture-overview.md); runtime
> behavior belongs to `$verify-flows`. Confirm paths against the current tree before moving code.

## Top-level map

```text
PICoater_AOI/
|-- .agents/                  Codex rules, skills, and engineering references
|-- src/
|   `-- dotnet/AniloxRoll.Monitor/   WinForms product application
|-- sdk/
|   |-- TanukiCv/             image processing, native pipeline, and shared controls
|   |-- Bridges/              IO, light, and storage adapters
|   |-- MIL/                  MIL acquisition wrapper, samples, and vendor material
|   `-- docs/                 shared SDK HTML guidance
|-- tools/                    cross-component maintenance and analysis tools
|-- tests/                    unit, integration, and stress test projects
|-- algtest/                  Python algorithm prototypes and feasibility work
|-- deploy/                   inspection/storage PC deployment scripts
`-- docs/                     operator HTML and non-agent vendor artifacts
```

## Directory responsibilities

| Directory | Owns | Must not own |
|---|---|---|
| `.agents/skills/` | Repeatable engineering workflow, architecture/API references, DVT contracts | Operator manuals or generated output |
| `src/` | Product policy, composition, WinForms behavior | Reusable SDK algorithms or vendor source |
| `sdk/TanukiCv/` | Native CV pipeline, .NET core APIs, reusable WinForms controls | PICoater-specific settings or tab workflow |
| `sdk/Bridges/` | External hardware/service interfaces and adapters | Main-form UI policy |
| `sdk/MIL/` | MIL camera acquisition, samples, and MIL vendor references | Product display ownership |
| `tools/` | Cross-component scripts and maintenance executables | A sample useful only with one SDK component |
| `tests/` | Automated correctness/load tests | Manual DVT trace evidence |
| `docs/` | Operator-facing HTML and non-agent vendor artifacts | Agent engineering Markdown |

## SDK component map

| Component | Important contents | Boundary |
|---|---|---|
| `sdk/TanukiCv/native/` | `tanuki_core`, `tanuki_utils`, `tanuki_cv_api`, `tanuki_pipeline` | Native processing and exported ABI |
| `sdk/TanukiCv/dotnet/` | `TanukiCv.Core`, `TanukiCv.Controls` | Reusable managed APIs and display controls |
| `sdk/Bridges/IoBridge/` | Modbus interface, ET-7044 implementation, samples | IO protocol and state access |
| `sdk/Bridges/LightBridge/` | LTS-3DPA24 serial control | Light communication |
| `sdk/Bridges/StorageBridge/` | SMB/copy/retention primitives | Storage transport |
| `sdk/MIL/MilGrabber.Core/` | `MilCamera` and acquisition wrapper | MIL access without product UI policy |

Component-specific structure and invariants belong to the closest nested `AGENTS.md` and owning
skill. Do not copy their implementation history into this map.

## Placement rules

### Samples vs tools

- Put an executable in `sdk/<component>/samples/` when it only demonstrates that component and
  should move with the component if the SDK is split.
- Put it in `tools/` when it remains useful across components or operates the product repository.

### Benchmarks vs tests

- Correctness and regression tests live under `tests/`.
- Benchmarks live beside the measured component, such as `sdk/TanukiCv/benchmark/` or a pipeline's
  own `benchmark/` directory.
- Prototype analysis that is not an automated regression test belongs under `algtest/`.

### Engineering references vs docs

- Agent-readable Markdown belongs in the owning `.agents/skills/<skill>/references/` directory.
- `docs/` retains operator material and vendor artifacts that are not agent instructions.
- Completed migration plans and historical reviews are Git history, not active architecture rules.
