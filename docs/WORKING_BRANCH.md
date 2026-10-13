# Contributor workflow

Continue work on **`codex/project-work`**, the project's single ongoing branch.
Use the existing checkout that holds it:

```bash
git worktree list
```

On Nila's machine, that checkout is:

```text
/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction
```

The working paper remains at `docs/paper/working_draft.tex` in this checkout.
Raw recordings and runtime assets remain under
`/Users/nilakarthikesan/grasp-failure-prediction/runs`.

The original project checkout is on `main`. PR 19 integrated snapshot
`c95dd8d`; later work is on `codex/project-work` until separately merged.
From a clean partner clone, fetch and switch to the existing working branch.

Preserve dirty files, ignored runtime assets, experiment outputs and the open
paper path. Keep source consolidation separate from scientific protocol versions.
Append evidence-backed changes to [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md).

[Branch cleanup history](archive/WORKING_BRANCH_HISTORY_2026-10-12.md) and
[recovery tags](BRANCH_ARCHIVE.json) preserve earlier branch references.
