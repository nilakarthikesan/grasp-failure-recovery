# HUG bounded surface-refinement evidence

See `REPORT.md` for method, measured tradeoffs and version history. This is
static development evidence on six existing saved proposals. The separate
independent validation subbundle applies the unchanged strict mapping rule;
the geometric objective and existing fingertip engineering guard cannot grant
scientific mapping acceptance.

Only the files listed in `manifest.json` belong to the publication bundle.
Runtime caches, partial reports, original proposals, scene binaries and
licensed MANO model arrays are excluded. The two complete raw/guarded reports
are exact-byte gzip copies; their binding files state the original byte hashes.
Selected native closest points are derived measurements, not copied model
arrays.

The reusable bounded solver is `surface_refinement.py`; the fixed engineering
guard is `guarded_refinement.py`. `run_refinement.py` is the frozen six-case raw
driver and `apply_guard.py` its guarded derivative. They use read-only existing
data and time-zero FK, with integration APIs explicitly blocked. The operational
protocols bind their exact implementations and original source inputs.

For a read-only artifact/source check in the preserved runtime:

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python verify_static_outputs.py
```

Use `--skip-runtime-sources` only for an offline file/hash check. Its output
explicitly marks the actual runtime source and source-URDF bound checks absent.
This verifier never loads a model or advances physics. `verification.json`
records the performed source/guard checks; the manifest separately binds all
selected artifact bytes.

The original frozen drivers expect the preserved project/runtime absolute
paths and cached query files recorded in the protocols. Do not rerun them in
this immutable evidence directory: their outputs would overwrite historical
reports. Any replay belongs in a scratch directory with source path bindings
checked and fresh report provenance. Fixed scientific/objective parameters must
stay unchanged. Historical source protocols, reports and version folders remain
the record of this run.
