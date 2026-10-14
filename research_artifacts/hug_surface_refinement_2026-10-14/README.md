# HUG surface-refinement development evidence — 14 October 2026

This bundle records a concrete native-relative correction on six existing HUG
grasps and an independent unchanged-rule audit on six coarse/three finer domains.
All nine strict relationship statuses remain FAIL. Numerical/engineering checks
pass; improvement is not scientific mapping acceptance. No inference, physics
integration, eval episodes/labels or prediction training occurred.

- [Project explanation](../../docs/HUG_SURFACE_REFINEMENT.md).
- [Frozen correction report](prototype/REPORT.md),
  [settings/guard decisions](prototype/summary.json) and
  [tradeoff figure](prototype/refinement_tradeoff.png).
- [Independent report](validation/REPORT.md),
  [all cases/bands/parts/resolutions](validation/summary.json),
  [original witness recheck](validation/existing_witness_recheck.json) and
  [comparison figure](validation/guarded_candidate_tradeoffs.png).
- [Experimental API derivative](prototype/production_candidate/README.md)
  preserves immutable experiment modules separately; 70 manufactured tests and
  11 exact valid-input parity comparisons pass. Stored-assessment replay agrees
  on all six guarded targets, without recomputing geometry.

Submanifests preserve each selected file exactly. Raw and guarded full reports
are exact-byte gzip archives; binding files give decompressed SHA-256 values.
Unexecuted drafts and pre-case abort versions remain separate. No raw MANO
model, checkpoint, native proposal, source object asset, scene binary or scratch
query array is included. Original inputs and runtime assets remain local.
Derived surface measurements are not generator/eval collection records.

The static scripts intentionally bind original absolute runtime and input paths.
Re-running a solver requires those recorded assets/caches and a copied temporary
workspace; do not write over committed historical reports. Some scripts expect
the JSON report archives to be decompressed in that temporary copy first.
The default mapper, evaluator/controller and existing predictor are unchanged.

Read-only bundle verification, with the original runtime still present:

```sh
PYTHONDONTWRITEBYTECODE=1 python verify_bundle.py
PYTHONDONTWRITEBYTECODE=1 python prototype/verify_static_outputs.py
```

`verify_bundle.py --offline` checks published bytes/protocols/API-copy bindings
without claiming unavailable original runtime input verification.
The prototype verifier also supports `--skip-runtime-sources` for its explicit
offline scope. Checks do not load a generator, integrate physics or produce labels.
