---
name: project-context
description: Locate PICoater AOI projects, files, controls, settings, services, native APIs, harnesses, and documentation. Use when mapping an unfamiliar feature, finding the owner of a control or setting, or checking repository placement before a refactor.
---

# project-context

Start with [`references/architecture-overview.md`](references/architecture-overview.md) when the
task crosses layers or features. Read
[`references/repository-layout.md`](references/repository-layout.md) for physical directory and
placement decisions. Read
[`references/repository-reference.md`](references/repository-reference.md) only for the lookup
tables needed by the current task. Treat root and nested `AGENTS.md` files as authoritative rules.

For performance or hardware sizing work, read
[`references/runtime-resources.md`](references/runtime-resources.md) and use current measurements.
For produced files, local/remote placement, and retention classification, read
[`references/output-storage-map.md`](references/output-storage-map.md).

## Document ownership

Each fact has one documentation owner. Do not copy the same table or rule into another file.

| Source | Sole ownership | Do not put here |
|---|---|---|
| Root or nested `AGENTS.md` | Normative architecture rules and prohibitions | Current owner inventories or file catalogs |
| `architecture-overview.md` | Current logical topology, feature owners, and refactor pressure points | Detailed paths, control names, or normative rules |
| `repository-layout.md` | Physical directory responsibilities and placement decisions | Feature workflow or current class ownership |
| `repository-reference.md` | File/API/setting/control lookup facts | Architecture decisions or refactor plans |
| `output-storage-map.md` | Produced-file paths and copy/retention classification | Runtime sequencing or operator instructions |
| `$verify-flows` | Runtime behavior, sequencing, log evidence, and DVT contracts | Static directory or owner catalogs |

Update only the owner of the changed fact:

- Role or feature ownership changed -> `architecture-overview.md`.
- Directory placement policy changed -> `repository-layout.md`.
- Symbol, path, setting, or control changed -> `repository-reference.md`.
- Architecture rule changed -> closest `AGENTS.md`.
- Runtime behavior or wiring changed -> `$verify-flows`.

When one code change affects multiple fact types, update all affected owners in the same commit.

## Parallel-work rule

- Parallel workers do not independently edit shared files under `project-context/references/`.
- They report owner/path/behavior changes with their code result.
- The primary integrating worker updates the shared references once after code integration and
  verifies them against the final tree with `rg`.
- A worker may edit a shared reference directly only when that reference is its explicitly assigned
  deliverable and no other worker owns it.

Before editing:

1. Locate the feature, control, setting, or service in the reference.
2. Confirm the entry against current code with `rg`; the reference can become stale.
3. Read the closest nested `AGENTS.md` and the task-specific skill.
4. Use the ownership table above to update only the reference that owns the changed fact.
