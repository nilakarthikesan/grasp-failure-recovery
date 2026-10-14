# Static grasp mapping audit artifacts

14 October 2026, source branch `codex/project-work`, audited source `7b14f26`.
See [the interpretation and source references](../../docs/GRASP_MAPPING_FIDELITY.md).

These files are diagnostics, not a training or evaluation dataset. No model
inference, physics integration, new task labels or predictor fitting occurred.
The HUG files reuse one saved real cube/seed-0 output. ContactGen checks
transform algebra only; UniDex tests a labeled reference proxy; DGA tests
synthetic static joints against its official URDF. None is a four-source
native-output benchmark. The manifest records hashes of the diagnostic files.

## Evidence

- `hug_static_fidelity.json`: full frame composition, each nominal tip error,
  joint-bound positions, declared force-close displacements and joint changes.
- `hug_native_vs_shadow_object_frame.png`: native MANO skeleton versus nominal
  Shadow task landmarks around the same 50 mm cube. Dashed task links are not
  full robot collision geometry.
- `hug_shadow_force_close_static_displacement.png`: separate controller command
  change; it does not show achieved actuator motion.
- `hug_saved_final_static_contacts.json`: reconstruction of an existing final
  Shadow state. Forces are not reconstructed because control/mocap/actuator
  state was not saved. No corresponding native MANO contact rollout exists.
- `hug_nominal_initial_contact_regions.json`: guarded static nominal placement
  at the unchanged initial object state, coarse native vertex/Shadow contact
  cube-face coverage, and penetration limits. This is before controller motion.
- `contactgen_transform_review.json`: two static composition checks, exact
  runtime/hash provenance, and explicit missing real-ContactGen evidence.
- `dga_static_fk.json`: 65 synthetic URDF FK configurations, native 24-joint/root
  facts and candidate bridge discrepancies, without a registered adapter.
- `unidex_reference_static_fidelity.json`: 112 synthetic FK configurations,
  reference-proxy scope, certificate blind spots and upstream parser comparison.

## Reproduction

The scripts are snapshots using the exact local paths in the reports. They
write new diagnostics only into `/private/tmp`; they do not overwrite recorded
runs. All MuJoCo scripts forbid `mj_step`, `mj_step1` and `mj_step2` before
constructing models. DGA uses XML and NumPy FK with no simulator.

The existing HUG runtime command is:

```sh
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR='/private/tmp/grasp-mapping-audit-2026-10-14/matplotlib-cache' \
PYTHONPATH='/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/src:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps:/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/cmeel.prefix/lib/python3.12/site-packages' \
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python \
research_artifacts/grasp_mapping_audit_2026-10-14/audit_hug_static.py
```

The saved-contact snapshot uses the same import path. ContactGen's exact
runtime/import path is recorded in its JSON. UniDex needs the project core and
existing numerical/kinematics dependencies, the pinned upstream source at
`/private/tmp/gfp-unidexgrasp-source`, and the hash-pinned Dex reference XML at
`/private/tmp/gfp-dex-shadow-source.xml`; that reference is not the actual UniDex
generation asset. DGA expects the pinned official source extracted at
`/private/tmp/DexGrasp-Anything-86353d710ca71d0f547f0d8b15cc8ca959c78163`.
Runtime assets/source are external and are not bundled in these diagnostics.

Future audits should store actual native outputs and contact-bearing surface
comparisons, with the declared generation stage, before making a fidelity claim.
