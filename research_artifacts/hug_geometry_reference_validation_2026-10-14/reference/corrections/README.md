# Correction: source physical limits versus captured scene metadata

The original reference comparison mislabeled two limit tables:
`methods[...].limits.physical_lower_rad` and `physical_upper_rad` were populated
from the captured tabletop scene's rounded ranges. Their legacy
`filter_output_physical_violation_rad` and `max_physical_violation_rad` values
likewise describe captured-scene ranges. These should be interpreted as **scene
range metadata**, not authoritative full-precision physical source limits.
All original reports, scripts, protocols and manifests remain unchanged.

The retargeting optimizers and final safety clamp used the full-precision URDF
limits correctly. `verify_bounds_metadata.py` independently parses that original
URDF and the captured scene XML, then verifies existing arrays without optimizer
calls, geometry recalculation, inference, physics or evaluation generation.

Across all 32 cases and all four recorded methods:

- Every final target satisfies the original URDF bounds with zero exceedance.
- Every final vector exactly equals the filtered endpoint clipped to source bounds.
- Raw/filtered source-limit exceedance is at most 0.001000006773 rad, within the
  predeclared 1.01e-3 rad numerical safety margin. Original clip deltas remain valid.
- Current, regenerated-current and as-shipped reference targets satisfy the
  strict captured-scene ranges. The metric-static sensitivity exceeds them in
  21 cases: 13 LFJ3 and eight FFJ3 cases. All excesses are 3.87799e-7 rad.

The source lower bound is −0.261799387799 rad; captured XML records −0.261799 rad.
The scene builder uses `mj_saveLastXML` before tabletop recompilation; MuJoCo
explicitly documents small numeric differences from text serialization.
[MuJoCo 3.3.7 saving models](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#saving-models)

Production provider preflight checks the directly compiled full URDF with 1e-9
rad tolerance, so this captured-scene discrepancy does **not** demonstrate a
production preflight rejection. No preflight, production mapping, controller,
scene asset or evaluation protocol was changed. Execution implications remain
untested. The supplemental table is the authority for interpreting source limits;
use the untouched original reports for unchanged solver traces and geometric results.

`bounds_metadata_correction.json` records the source/scene bounds, preserved-file
hashes, per-case verifications and summary. `manifest.json` binds this supplement.
The original report's uncompressed SHA remains
`0b7a2df411b9763f29c9cda6a5a252417716869dc5728de90369ef5f632738f6`.

To reproduce, retain this directory beside the exact original reference bundle
(or copy the complete original bundle plus corrections into a temporary directory)
and run `python verify_bounds_metadata.py`. It uses only Python's standard library,
reads either the original full JSON or its exact compressed copy, and rewrites only
this supplemental JSON in its own directory. Original saved local source XML/URDF
paths must remain accessible. It checks all original manifest files before and after.
