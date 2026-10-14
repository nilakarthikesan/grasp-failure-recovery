# Preserved HUG runtime and performance history

14 October 2026. This staging preserves distinct source/protocol versions and
their recorded statuses. It does not merge scientific protocols or historical
results. The current canonical V2.1 candidate fits and their independent audit
are packaged separately and are not copied into this history selection.

- `runtime_v1_interrupted`: two interrupted cohorts, zero completed cases and
  no selected-candidate report. Logged objective/geometry probes are incomplete
  optimizer evaluations. Final full-run integration counters were not serialized;
  a complete-run zero-step guard report is unavailable.
- `runtime_v2_import_abort`: six startup-only failures before `main`, case
  loading or any solver callback. Final whole-run guard counters are unavailable.
- `runtime_v3_unexecuted`: frozen faster-helper preparation with import smoke
  evidence only. V3 fitting never ran and produced no candidate records.
- `performance_one_two_threads`: measured static helper/reference parity on six
  originals plus three fixed FK perturbations. Canonical V2.1 uses the verified
  two-thread mask with its unchanged original helper. Timing scope is the normal
  helper only; it is not complete-fit speed or mapping acceptance.
- `performance_aabb_unused`: measured static AABB pilot, never activated in a
  constrained fit or in canonical V2.1.
- `performance_exact_buried_tested_unused`: measured static and manufactured
  exact-buried helper parity, prepared for V3 but unused in canonical fitting.
- `validation_v1_unexecuted` and `validation_v3_unexecuted`: separately bound
  independent wrapper sources/protocols that were not run on real candidates.

Each group retains its original protocol. Original metadata statements remain
historical statements with their original scope. In particular, the preserved
`runtime_v1_interrupted/read_only_code_review.json` describes the canonical code
review while real-grasp results were pending at that review; it is not a V1 fit
verdict. `reviews/case8_read_only_review.json` separately records the completed
canonical case-8 trace review, bound to its report hash, without copying that
canonical fit report into history. Exact fallback does not prove global
infeasibility or scientific mapping PASS.

Only literal allowlisted metadata, text source, notes, diffs and logs are copied.
No licensed assets, scene binaries, query NPZ/NPY arrays or raw point arrays are
included. Duplicate partial benchmark reports and V2 draft protocols/manifests
remain preserved at their original paths and are excluded here. The canonical
direction helper and its original fixtures are packaged separately by root.

All logs are gzip-compressed with an empty filename and `mtime=0`; even the
small logs use the same deterministic rule. `manifest.json` binds both each
original uncompressed byte SHA-256 and each stored byte SHA-256, with sizes and
source paths. The archiver verifies decompression and rechecks source bytes after
copying. The archiver reads metadata only, imports no experiment code, reruns no
tests, and writes only beneath its own temporary staging directory.
