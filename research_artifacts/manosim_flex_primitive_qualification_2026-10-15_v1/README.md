# MuJoCo 3.3.7 primitive flex qualification

Status: SOURCE PREPARED; no actual model compilation, FK, contact queries or integration has been performed by this implementation step. The first declaration `protocol.json` at b0e9db979853222c3ce35b8e24fb3ba210794dd3 was NEVER EXECUTED. Its bytes and original `controls.json` are preserved and bound as provenance. Active preparation uses `protocol_v2.json` and `controls_v2.json`, adding correct handling for excluded exact-touch contacts without an active constraint row. Root must run manufactured controls, prepare, commit the declaration and source, then execute exactly once.

This study uses synthetic vertices only. No HUG, MANO, MANOSim asset, project mapper, neural model or task outcome is loaded. It cannot admit collection or validate articulated MANO dynamics.

The fixed plan has 12 models: zero or 1 micrometre radius, identity or nonzero rigid frame, and three configurations: one triangle on one free body; one triangle whose three vertices belong to independent free bodies; and two separated triangles on one free body. All owners retain active free joints. The separated triangles retain six vertices and two faces; sphere probes over their convex-hull gap must find no contact while probes over each actual triangle must find contact. This detects accidental whole-flex convexification.

Each model has one initial and six probe `mj_forward` calls, at most 84 explicit forwards, 12 compilation attempts and 12 data constructions in 120 seconds (130-second outer process timeout). These are static forwards, never integration. Compilation may perform internal compiler work: counters describe explicit Python API attempts. A compilation rejection is retained as a measurement, not silently skipped; unexpected guard/numerical errors hard-stop and preserve attempted/remaining case accounting.

Single triangles use sphere centres on both sides at analytical penetration 2 micrometres, touching, and separation 2 micrometres. Touching-contact presence is descriptive because it is a strict numerical boundary. Disconnected triangles use both sides of each triangle and both sides over the clear gap. Contact dimension is one, with zero margin/gap/gravity and dense Jacobians. Original ordered face indices and body ownership must read back exactly; vertex arithmetic tolerance is 1e-14 metres. Contact distance/point and rigid-transform comparisons use 1e-10 metres, normals and normal-row translation Jacobians 1e-10. Single-owner rotational normal-row derivatives use 1e-10. These are predeclared manufactured-control limits, unrelated to the unchanged native surface rule.

Three-owner rotational Jacobians are recorded against barycentrically weighted vertex velocities but remain diagnostic. MuJoCo uses common contact-position owner Jacobians; passing translation/contact controls is not a claim that its rotational owner semantics equal MANO skinning.

The official 3.3.7 manual requires positive radius for 2D flex. Zero radius is therefore explicitly undocumented even if this bounded compiler/contact test passes. Positive radius geometrically inflates triangles; its value must remain visible in any later native field measurement. Neither radius may receive a blanket fidelity allow flag.

`prepare` only reads distribution metadata and hashes package/source files. `run` requires unchanged exact source/protocol Git blobs at the supplied HEAD, identical runtime bytes and once-only output directory. The worker imports MuJoCo only after validation, allows only exact planned synthetic XMLs, and blocks physics, extra FK/collision APIs and native/project imports. Warnings, nonfinite state and time changes refuse the run. Every completed XML/NPZ/case result is hash-indexed. The pure `qualified_radius_policy` helper rechecks measured case summaries and artifact hashes; it returns primitive scope only and leaves native-surface/admission gates closed.

Commands (original venv; no package installation):

```
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python scripts/check_hug_build.py research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/test_run_qualification.py -q --report research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/controls_v2.json
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/run_qualification.py prepare --protocol research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/protocol_v2.json
# Root commits protocol and all source/runtime-control receipts before the next command.
/Users/nilakarthikesan/grasp-failure-prediction/.venv/bin/python research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/run_qualification.py run --protocol research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1/protocol_v2.json --freeze-commit COMMITTED_SHA
```

The protocol is generated only after controls exist. This README is the protocol draft; no empirical result is predeclared. Runtime scientific operations remain NOT_RUN until that separate frozen execution.
