# Contributor workflow

Shared project development uses **`project-work`**. The user explicitly authorized
**`phase1-scene-preparation`** in a separate checkout for Phase 1. Continue that
work on the Phase 1 branch rather than creating a branch for every substep.
Use descriptive project names without assistant or vendor prefixes.

Find both existing checkouts with:

```bash
git worktree list
```

The branch rename does not move any checkout. The open working paper remains at
`docs/paper/working_draft.tex` in the existing shared-development checkout.
Raw recordings and runtime assets remain under
`/Users/nilakarthikesan/grasp-failure-prediction/runs`.

For a clean partner clone, fetch and switch to the shared branch:

```bash
git fetch origin --prune
git switch project-work
```

Existing collaborators should update their local branch name and tracking branch
to match the renamed remote. Keep local changes intact when doing so.

The original local project checkout remains on `main` at snapshot `c95dd8d`.
Remote main includes Enoch's provider PRs 21 through 24 at `b503c2f`.
Those changes are integrated into `project-work` with later diagnostics and
curated documentation. Phase 1 starts from the shared branch at `e1a90be`.

Preserve dirty files, ignored runtime assets, experiment outputs and the open
paper path. Keep source consolidation separate from scientific protocol versions.
Append evidence-backed changes to [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md).

[Branch cleanup history](archive/WORKING_BRANCH_HISTORY_2026-10-12.md) and
[recovery tags](BRANCH_ARCHIVE.json) now use normalized historical labels.
The [naming migration receipt](PROJECT_NAMING_MIGRATION.json) records original
Git blobs, unchanged evidence byte hashes, and separate display-label aliases.
Historical validation receipts still refer to the unchanged original evidence.
Real storage paths and machine tool identifiers remain literal so existing data
stays loadable. No historical experiment was rerun by the naming update.
