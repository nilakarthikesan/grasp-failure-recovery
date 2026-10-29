# Project workflow

Use `project-work` for shared project development. The user explicitly authorized
`phase1-scene-preparation` in an isolated checkout for Phase 1. Continue Phase 1
work on that branch; do not create another branch or PR for each substep.
Create other persistent branches only at the user's explicit request.
Use descriptive project names without assistant or vendor prefixes.

Use `git worktree list` to locate the checkout holding the intended branch.
Preserve dirty files, ignored runtime assets, experiment outputs and the open
paper path. Do not reset another checkout or move its manuscript.

Follow `.cursor/rules/project-workflow.mdc` for research documentation. Append
dated, evidence-backed decisions to `docs/RESEARCH_DECISIONS.md` and distinguish
implemented results from hypotheses and proposed work. Source consolidation
does not merge scientific protocol versions or rewrite historical results.
