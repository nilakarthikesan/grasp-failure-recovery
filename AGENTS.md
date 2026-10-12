# Project workflow

Continue work on `codex/project-work`, the user's single ongoing working branch.
Do not create a new branch or PR for each implementation or documentation step.
Update the existing consolidated PR; create another persistent branch only at
the user's explicit request.

Use `git worktree list` to locate the checkout holding that branch. Work there
rather than creating another checkout or resetting another branch. Preserve
dirty files, ignored runtime assets, experiment outputs and the open paper path.

Follow `.cursor/rules/project-workflow.mdc` for research documentation. Append
dated, evidence-backed decisions to `docs/RESEARCH_DECISIONS.md` and distinguish
implemented results from hypotheses and proposed work. Source consolidation
does not merge scientific protocol versions or rewrite historical results.
