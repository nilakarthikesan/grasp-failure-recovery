# Native MANO static comparison — 2026-10-15 v1

Read [the report](../../docs/HUG_NATIVE_MANO_COMPARISON.md) for methods, measured claims and limits. All 32 existing HUG grasps were retained; no inference, optimization, physics integration, evaluation episodes, labels or training.

- `measurement/protocol.json`: pre-measurement input/source bindings and numerical scope.
- `measurement/cases/`: 32 original scalar reports; no licensed MANO asset or original proposal geometry is redistributed.
- `measurement/audit_at_measurement.py`: exact measured driver. `source/audit_hug_native_mano.py` adds only the integer conversion that fixes final-summary JSON serialization.
- `run.log`: complete original run and final summary export error. `summarize_saved_comparison.py` and `summary.log`: scalar-only aggregate recovery, with every other frozen input binding verified. `measurement/results.json` reports the recovery and reconstructed counter provenance explicitly.
- `source/`: native model and test/driver snapshots. Model/test receipt hashes identify these pre-portability-test-fix snapshots; repository tests later add only optional SciPy skips, recorded in `verification.json`.
- `model_verification/`: 21 manufactured model controls receipt.
- `independent_review/`: separate scalar FK, compiled readback and metric controls, saved-measurement aggregation, original manifest and preserved setup-failure logs. Those source scripts retain original absolute runtime paths; their archived evidence is the historical result, not a portable rerun command.
- `comparison.png`, `make_figure.py`: visualization from saved scalars.
- `hug_section.tex`, `paper_update.json`: owned manuscript subsection and guarded edit receipt.
- `manifest.json`: archive-relative file hashes excluding the manifest itself.

The original protocol references external existing assets and exact code bytes. For historical hash verification, resolve its `scripts/audit_hug_native_mano.py` binding to the archived `measurement/audit_at_measurement.py`; do not rewrite the original protocol to point to the later serialization fix. Every other binding was unchanged during measurement and recovery. Fresh reproduction requires those licensed/private inputs and pinned dependencies, using a separate diagnostic directory. No HUG checkpoint is needed.

Native MANO here is an articulated static collision approximation, not an outcome-producing simulator. Its skeleton import passes and average geometry improves; contact surfaces, physical parameters, actuation and execution remain unvalidated. The 1/2/5 mm pointwise probe counts do not replace historical whole-cell acceptance results. See the report's limitations before interpreting the numbers.

The archived `hug_section.tex` deliberately preserves the exact section bytes, including its final blank line. Its SHA-256 matches the guarded paper-section receipt. This evidence-only trailing blank is excluded from the source whitespace check.
