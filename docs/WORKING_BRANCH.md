# One working branch for this project

Continue project work on **`codex/project-work`**. It contains the simulation,
complete noise demo website, literature review, recorded datasets, trained
baseline, fresh object evaluation, mug correction, paper and decision history.
The earlier design proposals and optional Docker runtime are preserved too.

`main` remains the published repository base. The consolidated working branch
is available for review and has not automatically been merged into `main`.
Ordinary code, experiment and paper updates now share this working branch and
one consolidated PR. A separate persistent branch is created only at the user's
explicit request.

The only open project PR is [19](https://github.com/nilakarthikesan/grasp-failure-recovery/pull/19).
The 13 superseded PRs are closed, and the 17 old remote branch names and
14 redundant local aliases are retired. Only `main` and `codex/project-work`
remain as branch names; original tips are recoverable from the archive tags.

## Why the older branches existed

The earlier workflow created a branch and PR for each milestone so execution,
collection, dataset splitting, model training and paper changes could be reviewed
in small pieces. Most later branches included the earlier ones, forming a stack.
Separate demo, literature, design and runtime branches also had unique files.
Those needed explicit integration before cleanup.

Branches are pointers to commit history. Removing redundant pointers does not
require removing the commits, authorship, files or experiment evidence. The
consolidation preserves the original commit histories through merge commits.
The rewritten preflight checkpoint contains the same four patches already
present; its history is preserved without replacing newer code.

## Where to work

The active checkout remains:

```text
/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction
```

The existing paper stays at `docs/paper/working_draft.tex` in that checkout.
Raw run data and the local dependency runtime remain under
`/Users/nilakarthikesan/grasp-failure-prediction/runs`; cleanup does not remove them.
The original project checkout is on published `main`. Older auxiliary checkouts
are detached at saved commits, preserving their ignored files; they are historical
copies rather than additional working branches. The active paper and source
checkout remains on `codex/project-work`.

## Recovery and historical references

The [branch archive](BRANCH_ARCHIVE.json) records each old branch tip and its
recovery tag. Original tips are retained as `archive/2026-10-11/...` tags, with
a full Git bundle and local notes saved under:

```text
/Users/nilakarthikesan/grasp-failure-prediction/runs/branch_cleanup_2026-10-11
```

The earlier PR pages remain historical records. Their redundant open PRs are
closed after the consolidated PR exists. Removing their active chat attachments
does not delete those GitHub records.

Scientific versions remain separate: the original v2 model evaluation and
corrected v3 controller checks keep their existing hashes, labels and metrics.
Putting source on one branch does not combine those experiments or retrain a model.

## Validation

The integrated suite passed 373 tests with 32 optional skips. Frozen source and
published evidence hashes must continue to match their manifests. The Docker
implementation is preserved with one OpenCV provider and research data excluded
from the build context. Its actual image build requires a Docker host; it was
not tested on this Mac, where Docker is unavailable.
