# DGA practical retargeting: static admission protocol v1

15 October 2026. This is a new scientific contract for **DGA plus a named retargeter on the unchanged Shadow hand**. Historical strict native-preservation protocols and their failures remain intact. The intended prediction target is the eventual executed Shadow attempt. This protocol can admit a static implementation in a stated domain; it cannot establish physical adequacy or native failure-label equivalence.

The criteria were frozen before the new correction run (SHA-256 `6eb351dc097daffb4d08f04117615714de2120d3395367f900e4cefb444d4d34`). Separate method and validation supplements bind code, inputs, assets, numerical libraries and search/validation settings before their respective tests. Repairs produce additive versions; prior results remain available. No physics, evaluation episodes, outcome labels or training are permitted.

## Why these criteria are justified

Literal joint legality, correct named transforms, preserved units and independent compiled forward kinematics are implementation invariants. The 1 nm / 1 nrad agreement limits concern the numerical realization of the same prescribed pose; they are not acceptable robot-to-native error budgets.

There is no currently established empirical millimeter threshold connecting DGA-to-Shadow distortion to failure-label stability. A morphology difference or finger length alone cannot supply that threshold. Therefore this protocol does not manufacture one. Instead, it uses a transparent, always-reproducible **legal reference** and proves that an adopted correction does not worsen its declared per-finger geometric errors. Absolute baseline and final errors remain reported, so a poor baseline is visible rather than described as faithful.

For each source-valid input, the legal reference projects each finger angle into its literal target interval, preserves both wrists and the native palm transform, and derives the target root by counter-transforming the target wrist FK. Projection is an explicit retargeting step and is recorded in the output. It minimizes unweighted joint-space change; it does **not** minimize geometric distortion, demonstrate an unavoidable morphology error, or repair the native output. If a wrist is incompatible with the target, the input has an explicit unsupported status.

Only finger branches containing a target-limit violation may change. Other branches, both wrists and the palm remain exact. This restriction protects already-representable native articulation and prevents a fitting loss from moving the whole hand to hide a thumb problem. It also means that pre-existing native/target palm mismatch remains and must be reported; fixed-palm field identity proves no additional change relative to the legal reference, not removal of native palm overlap.

## Correction and admission

A correction must retain all literal joint limits and must not increase the maximum landmark displacement or sum of squared landmark errors of **any finger** relative to the legal reference. Per-landmark and aggregate errors are retained as diagnostics. An aggregate fitting improvement cannot override a failing finger gate.

The geometric objective compares native-relative distances symmetrically. It seeks to preserve both close and far relationships, including source penetrations and misses. It must not pull all fingers into contact, maximize a success score, or remove poor grasps from the cohort. Such actions would modify generator quality and bias a failure-prediction dataset.

For each corrected finger, independently compare the maximum absolute native-to-target **uncapped** unsigned surface-distance residual across the complete object surface. Distance to a closed triangle set is 1-Lipschitz, so their residual difference is at most 2-Lipschitz. If an independent grid covers the surface with radius rho and its largest residual is m, the supremum lies in `[max(0,m-a), m+2*rho+a]`, conditional on the declared residual arithmetic allowance a = 2 nm (two distance calculations, each assumed within 1 nm). A correction passes nonregression when its upper bound is at most the baseline's lower bound; exact geometry identity proves equality separately. The independently shifted grids use cell fractions `(9/23, 7/19)` and fixed 1, 0.5 and 0.25 mm steps. Their covering radius is `h * sqrt((14/23)^2 + (12/19)^2)`. If the intervals overlap, refine through those frozen grids. Remaining ambiguity leads to the legal-reference fallback, not an optimistic pass. These numerical bounds do not establish signed occupancy, normal-direction or contact-force equivalence.

Every input receives a result. Source-invalid inputs and implementation errors retain explicit reasons and stay in the denominator. An actual solver exception or an unproved correction returns the recorded legal reference. Ordinary iteration/evaluation budget exhaustion ends a bounded proposal search; its saved valid trials may still be considered by the independent gates. An input/provenance/implementation validation error remains an explicit error, never a hidden pass. Such fallback is a successful deterministic static mapping operation, but is not a successful geometric correction. Any claimed improvement must be supported by the independent metrics; lower optimizer loss alone is insufficient.

## Coverage and claims

The 16 recent outputs and four older outputs have already informed development. They are regression material. The five source-invalid members of the recent 16 remain identified; no native values are changed to admit them. Older DexGraspNet, native MultiDex and each retargeting version retain separate provenance and reports.

Independent validation means both an independent implementation of the saved-state checks and, for a later generalization claim, fresh native outputs collected only after the mapper is frozen. Existing outputs cannot become an untouched holdout by changing their label. If a new native smoke cohort is run, it contains native proposals for mapping validation only; it produces no physics episodes or outcome labels. Object and checkpoint coverage must be stated exactly.

Passing the static contract permits the status **STATIC_PILOT_ADMITTED_TESTED_SCOPE**. It means that the explicit DGA-plus-retargeter implementation produces auditable legal Shadow poses, with measured distortion and conservative no-worse-than-reference correction decisions, over its tested domain. It does not mean that native surfaces are preserved, that the reported distortion is physically harmless, or that the method generalizes to untested objects. Shared provider execution remains disabled until later execution gates are addressed under appropriate authorization.

## Interpretation limits and method references

- A legal-reference error cap is a prospective engineering comparator, not a universal scientific accuracy tolerance. Report large errors prominently.
- The 2-rho bound can be conservative: with fine-grid overlap, fallback is expected and should not be recast as a failed native generator.
- The arithmetic allowance is conditional until supported by certified numerics. Independent-backend agreement is useful evidence, but not a universal floating-point proof.
- A future morphology-based envelope can add a global guarantee. For corresponding link-local surfaces S and T and transforms A and B, `d_H(AS,BT) <= d_H(S,T) + max_{v in vertices(T)} ||(A-B)[v,1]||`. The Hausdorff term requires a complete-surface upper bound; vertex nearest-neighbor samples alone are insufficient. Such an envelope still does not define task adequacy.
- External retargeting libraries are method references, not sources of our acceptance numbers. Dex Retargeting uses position/vector fitting with named joints; its DexPilot mode deliberately encourages pinch behavior, and its default joint-limit padding must not replace this protocol's exact literal bounds. [Official implementation](https://raw.githubusercontent.com/dexsuite/dex-retargeting/main/src/dex_retargeting/optimizer.py).

## Evidence

The [versioned implementation and validation evidence](../research_artifacts/dga_practical_retargeting_2026-10-15_v1/README.md) records actual outcomes. The [original exact-preservation audit](DGA_ACCEPTANCE_FEASIBILITY_AUDIT.md) remains unchanged. Static admission is a readiness decision for the tested pipeline; the baseline comparator does not establish that its absolute distortion is physically acceptable.


## Completed static pilot, 15 October 2026

All 20 saved MultiDex outputs were retained: 13 source-valid mappings pass the static contract and seven native-invalid outputs remain refusals. One of 23 proposed branch corrections is admitted (recent row 4 thumb); 22 remain unresolved and fall back. The independently bounded worst thumb residual falls from [2.508736, 2.947318] mm to [1.678521, 2.117103] mm, with lower maximum landmark error and SSE. This is a worst-residual improvement over the whole cube surface, not pointwise nonregression at every location or preservation of all near/contact regions.

The final maximum landmark error across the cohort remains 6.075600 mm and existing palm overlap remains. The scoped status is `STATIC_PILOT_ADMITTED_TESTED_SCOPE`, with no native-fidelity or physically harmless-distortion claim. All outputs are previously inspected cube regression data. Provider execution remains disabled.

The final validator uses face-referenced vertex bounds, conservative triangle sphere/AABB pruning and an exact triangulation of the target thumb's 1 mm visual box. Adaptive refinement retains every leaf's bound in a verified full-surface covering ledger. It changes runtime, not the frozen grid levels or acceptance inequalities. Earlier capped proposer runs, incomplete checker runs and all code versions remain separate in the evidence bundle.
