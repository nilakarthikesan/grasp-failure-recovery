# Minimal engineering contract for the declared HUG→Shadow pipeline

This is a separately named implementation contract. It does not replace, weaken, or relabel the failed strict native-surface criterion. The chosen prediction target is failure of the executed, retargeted Shadow proposal.

**Acceptable static claim:** For a specified immutable native HUG proposal, mapper profile, hand/object assets and scene, the simulator's nominal Shadow configuration realizes the declared transformed root and named-joint command to the documented numerical precision. The original native proposal is retained and its measured geometric distortion is reported.

This does not assert that Shadow exactly reproduces MANO anatomy, that native and mapped contacts or failure labels coincide, that a command is physically reached during a rollout, or that a failure predictor generalizes. 'Static realization' and 'dynamic execution' must remain separate statuses while no new physics is authorized.

## Mandatory evidence

| Requirement | Acceptance evidence | Current saved-record status |
|---|---|---|
| Immutable proposal identity | Native file/shape/landmark/frame hashes; exact native decoding; original source retained even if poor |32 saved native reconstructions were reported exact in the frozen native-contract audit. |
| Declared mapper and command | Versioned profile/config/URDF/solver settings, explicit operator axes and wrist/palm policy, complete24 named joints and root pose, deterministic state/reset or recorded history; errors/status preserved |32 legacy commands are saved. Replaying them does not prove deterministic rerunning of the old stateful retargeter or validate a new optional profile. New mapper profiles require their own evidence. |
| Current implementation linkage | Hash-bound current loader/mapper and runtime; demonstrate that the exact intended current path is the path checked; no silent fallback or obsolete frozen import |Historical static evidence is tied to its old loader and source hashes. The separate current-execution bridge is being prepared; it must pass before transferring that evidence to current source. |
| Command realization | Independent FK from authoritative URDF versus compiled model, named-joint/order/bounds checks, root/palm/wrist composition, object-frame and world-frame agreement; explicit numerical precision basis |Historical32-pose audit passed its scoped checks, with maximum fixed-URDF/stored-scene task-point discrepancy7.8449e−8m and zero violations of the stored scene’s reported limits. Those limits came from the rounded stored scene, not an independent read of the authoritative source XML; the current bridge must check source-precision limits separately. These are measured residuals, not a universal anatomical tolerance. Current-code bridge remains separately reviewable. |
| Geometry intended for simulation | Compatible collision masks/exclusions, body/geom transforms, mesh hull/primitive dimensions, visual/collision distinction, margins/gaps and exact asset hashes |Historical frozen compiled-geometry inspection covered six development cases, but did not establish source-OBJ collider identity or current-loader equivalence. The current loader separately repairs a mesh-basename alias collision issue (commit18c5a0d); its repair cannot be credited to that older inspection. All32/current-source collider linkage remains a bridge requirement. |
| Explicit scope and distortion | Preserve original source/native targets; log mapped pose, point errors and available proximity/orientation diagnostics; retain failed/undefined regions and all modified candidates |Strict native relationship failures remain reported. Two latest SE3 segment candidates improve retained violation severity without gaining a new passing row; they are experimental candidates, not a new globally validated mapper. |
| Execution/label semantics | If later authorized, freeze controller, approach/force-close/lift, noise, friction/contact parameters and failure definition; associate labels with the actual recorded Shadow trajectory and all upstream versions |Not established by static FK. No new physics, labels or eval data were produced here. A later execution check cannot be silently inferred from source code or time-zero pose agreement. |
| Generalization and method comparison | Separate development objects/observations from independent validation; account for mapper/geometry changes and generator identity |32 saved proposals cover four objects and one MANO shape. They are inspected development data, not independent unseen-object validation. Predictor adequacy is untested. |

## Finite signoff procedure now

1. Freeze one **named command/mapper profile** and explicitly state whether the claim is replay of the existing legacy commands or new retargeting. Do not pool their provenance or promote experimental per-case corrections as a general algorithm.
2. Complete the current-source static bridge on the existing32 commands: independent FK, named joints/root, object state and collision geometry. Use justified numerical allowances tied to represented arithmetic/serialization, not fitted morphological success thresholds. Keep every discrepancy visible.
3. Publish a scoped matrix: native decoding, declared command realization, geometry linkage, distortion reporting, strict native fidelity, dynamic execution, and independent predictor evidence. Each status can pass, fail, or remain untested separately.
4. If mandatory static checks pass, sign off **static implementation fidelity of that declared Shadow pipeline on those32 saved commands**. Keep strict native proxy-preservation FAIL/UNRESOLVED and dynamic/predictor evidence untested. This is a precise achievable claim, not 'HUG completely accurate' without qualification.

No numerical anatomical allowance is invented. A future claim of bounded approximate native fidelity needs separately justified distortion limits. A claim that the simulator faithfully realizes its own declared Shadow commands instead needs an implementation/geometry contract and explicit distortion disclosure. These are different claims, and both the paper and data records must say which one is being made.

HUG itself evaluates MANO in simulation and retargets to Ability/WUJI for robot deployment; its fixed embodiment-specific alignment is not a Shadow tolerance or a proof of exact native-contact preservation. [Official HUG paper,§§5.2,F.2](https://arxiv.org/html/2606.17054v1).

## Evidence bindings

Historical32-pose FK: `research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json`.
Native/strict split: `research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_application.json` and its immutable rule.
Current certificate-capability diagnosis: `/private/tmp/hug_acceptance_contract_review_2026-10-15_v1/contract_evidence.json`.
Current bridge preparation: `/private/tmp/hug_current_execution_bridge_2026-10-15_v1`; its outcome was not yet available when this contract was written.
