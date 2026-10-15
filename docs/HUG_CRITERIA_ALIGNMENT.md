# HUG evaluation alignment and the mapping claim

15 October 2026. The historical strict criteria and failures remain unchanged. This note clarifies what they establish and proposes a separate path to practical retargeting acceptance. It does not authorize new inference, physics, evaluation episodes or training, and it does not assign an approximate acceptance threshold.

## Correction to the interpretation

Nine failed surface comparisons mean that the tested HUG-to-Shadow poses do not preserve every relationship required by our strict geometric contract. They do not establish failure of HUG's published generator benchmark, nor prove that every approximate Shadow mapping is unusable. Earlier shorthand such as “HUG cannot be signed off” must name the claim: strict native surface preservation is rejected; practical adequacy for the executed Shadow task remains untested.

The user selected the executed retargeted Shadow grasp as the predictor's target. Native MANO failure-label equivalence is therefore a separate, stronger claim and is not required merely to define a Shadow outcome. Retargeting can still distort source diversity or create systematic failures, so the adapter's correctness, distortion and downstream effect need evidence.

## What comes from HUG

HUG reports lift success and fingertip-to-object error, with penetration/intersection and miss diagnostics. Its simulated hand approximates MANO; Ability and WUJI deployment uses robot-specific retargeting and alignment. These evaluations provide no universal Shadow fidelity tolerance or requirement of zero changed native surface relationships. Its reported errors are observations, not mapping pass limits. [HUG §§5.2, E, F.2](https://arxiv.org/html/2606.17054v1)

The inspected official source snapshot `8d1c52d4c24bfae5a369e32e3f134f5601a02630` supports native representation checks: preserve the supplied shape, metric camera-frame pose, wrist convention and local articulation. It does not contain a verified released benchmark evaluator. Our Shadow study must not be described as reproducing HUG-Bench. [Pinned MANO exporter](https://github.com/KevinyWu/hug/blob/8d1c52d4c24bfae5a369e32e3f134f5601a02630/src/models/mano.py), [official release](https://github.com/KevinyWu/hug)

## Origins and roles of our checks

| Check | Origin and purpose | What passing would establish |
| --- | --- | --- |
| Native reconstruction, units, transforms, handedness, named joints and FK | Source representation and robot implementation invariants | Correct implementation within the checked scope |
| Mean fingertip-to-native-target error ≤10 mm | Project engineering admission guard; not a HUG-paper tolerance | Meets that configured guard, not validated grasp fidelity |
| Each fingertip no worse than its original mapping error | Project safeguard added after average-loss fitting worsened individual fingers | No regression against that particular baseline; baseline error can remain substantial |
| Surface bands at 1/2/5 mm | Inherited project measurement scales, originally diagnostic | Relationships at those declared scales; not simulator contact thresholds |
| Zero lost/added relationships and no reliable direction reversal | Project definition of strict native-relative preservation | The strong equivalence claim, provided coverage is complete |
| ±0.35 rad joint correction and source-reach-derived translation boxes | Bounded search design | Candidate stays within the tested search domain, not a universal accuracy margin |

The frozen rule is SHA-256 `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`. [Original acceptance explanation](HUG_MAPPING_ACCEPTANCE.md), [original static protocol](HUG_STATIC_VALIDATION_PROTOCOL.md). The numerical 10 mm gate and exact baseline caps must not be conflated with HUG's fingertip-to-object quality measure: our error compares two hands' coordinates, whereas the latter compares fingertips with the object.

## What a surface error means

For a fixed location on the object, compare its distance and direction relative to the corresponding native and robot hand part. A lost-near region lies close to the native part but far from the robot part; an added-near region does the reverse. A direction reversal changes whether a reliable hand surface normal opposes the object's outward normal. These are changes to a declared geometric relationship, not proof of actual force-bearing contact or a failed lift.

The historic distance audit includes the literal surfaces of individual robot colliders, including buried interfaces. Its normal checks exclude unreliable/buried samples, but these domains are still not equivalent to the exposed robot boundary or the simulator's active contacts. An exposed-boundary contact audit must be separately named and implemented; it cannot silently replace the frozen measurements.

One new mug pose repaired 10 retained proximity examples yet increased definite thumb mismatches and introduced reliable direction reversals. That is a measured tradeoff, not a rounding error. Separately, the current independent-interval certificate can remain ambiguous for identical fields; that limits proof of a pass, but does not invalidate positive mismatch counterexamples.

## What remains before practical acceptance

1. **Keep the hard implementation checks.** Preserve source and mapped states, physical joint limits, world/object transforms and independent forward kinematics. Confirm that the simulator realizes the declared pose rather than only trusting command values.
2. **Improve and measure retargeting.** Compare a fixed baseline and each new mapper on the same saved inputs. Report per-finger placement, contact-region displacement, object-relative distances, directions, penetrations/misses and sampling uncertainty. Keep native poor proposals and refusals; report any intentional repair separately.
3. **Define a new claim prospectively.** Practical Shadow adequacy needs its own protocol, distinct from exact MANO equivalence. Any geometric allowance must have an external rationale or separate calibration evidence. No number is selected because it makes these six inspected grasps pass. The allowance, calibration dataset and approval decision are currently unset.
4. **Validate outside these development cases.** A later authorized calibration/sensitivity study should determine whether measured mapping changes materially alter outcomes under a fixed controller, with object-size/feature strata. Freeze the resulting policy before held-out testing. Do not turn the six repeatedly inspected grasps into an “independent” validation split.
5. **Then evaluate the prediction question.** Compare one-to-four-generator training under the same Shadow/task/controller and declared mapper versions. The labels describe executed robot attempts. Successful lifting alone is not a mapping gate, and physical failures are required examples for a failure predictor.

A useful approximate mapping is a plausible goal; complete native-to-robot surface equality is a substantially stronger claim and is not guaranteed for different hand geometries. Existing evidence does not yet establish an acceptable practical error budget or generalization. We should improve the mapping and validate its intended use without presenting either unknown as solved.

## Current improvement experiment

A separately frozen feasibility-restoration experiment reuses twelve V2 terminal poses from the same six existing grasps. It selects the nearest observed exact-feasible checkpoint to each endpoint within a small fixed trust box; this is not a globally optimal projection. No surface objective guides checkpoint selection.

All twelve restored checkpoints pass the original individual fingertip caps and mean guard. None improves the frozen surface diagnostic rank. Ten invalidate previously eligible orientation rows; six also lose previously passing records. These are counts of checkpoints failing retained-row checks, not physical contacts or complete coverage. A still-eligible milk-box B sample reverses its native-relative facing sign in each of its two restored endpoints. Missing/invalid direction evidence is not an actual measured sign reversal, and no claim is made that the tiny projection caused these issues: the previously infeasible endpoints were not given the same exact surface check.

Mug A has no retained direction rows. One restored endpoint passes eleven of its fifty-seven proximity examples instead of the selected baseline's ten, yet worsens the worst normalized violation from 5.000 to 5.445. More passing examples alone therefore do not establish a better mapping. All six previously selected poses are retained; no new coarse or fine fields are needed. The prior nine full-grid FAIL results remain prior evidence, not newly rerun tests.

This result removes one uncertainty: small fingertip violations were repairable, but repairing them did not reveal an acceptable surface correction. The experiment used 1,101 tip-FK callbacks and 161 analytic Jacobian callbacks (2,524 actual projection forward calls), plus 24 retained-surface checks. No inference, physics, evaluation episodes, outcome labels or training ran. [Restoration evidence](../research_artifacts/hug_feasibility_restoration_2026-10-15_v1/REPORT.md).
