# HUG acceptance contract: what can actually be certified

15 October2026. Read-only code, frozen-report and primary-source review. No new geometry, FK, fitting, proposals, physics or eval data.

**The current strict evaluator is a falsifier and uncertainty reporter, not an implemented positive certificate. Continuing to optimize until that implementation says PASS cannot terminate successfully. This does not make its existing definite failures false.**

## Three different claims

1. **Declared Shadow pipeline correctness:** the saved native data are decoded in the stated frames, the named retargeter produces documented Shadow commands, and the compiled simulator realizes those declared commands with the expected geometry, coordinates, limits and state. This is an implementation claim. It can be checked without claiming that Shadow has MANO's anatomy or that native and mapped outcomes coincide.
2. **Strict native proxy-relationship preservation (the frozen rule):** each of six proxy parts preserves every declared1/2/5mm object-side proximity relationship, with continuous valid orientation coverage. This is much stronger than claim1. Existing definite counterexamples refute it in the tested scope. A better certificate must preserve those failures, not reclassify them because its uncertainty accounting improves.
3. **Usefulness/generalization of a failure predictor on the declared Shadow pipeline:** requires independent predictive evidence and explicit mapper/geometry confounding controls. Neither claim1 nor static claim2 proves this. Outcome labels would describe the executed retargeted Shadow pipeline, not the unexecuted native MANO proposal. No such new outcomes are authorized or generated here.

The frozen rule explicitly distinguishes numerical implementation, strict preservation, approximate adequacy and untested outcome equivalence. Engineering signoff for claim1 is therefore a separately named supported claim, not a relaxed version of claim2. Distortion reporting remains necessary; a numerical distortion budget is needed if claiming bounded approximate native fidelity, but no universal morphology tolerance is needed merely to verify the realization of declared Shadow commands.

## Exact proximity certificate condition

For a positive-area query cell, let `ND,NP` mean native definitely/possibly near; `MD,MP` the same for mapped geometry. The frozen function computes:

- lost lower/upper masks: `ND & ~MP`, `NP & ~MD`;
- added lower/upper masks: `MD & ~NP`, `MP & ~ND`.

Each field has only three consistent classes: far=(0,0), ambiguous=(0,1), near=(1,1). **Both mismatch upper bounds are zero if and only if every positive-area cell is certified near in both fields or certified far in both fields.** This condition is necessary and sufficient for this particular estimator to certify zero proximity mismatch. It is sufficient, not necessary, for actual relationship equality.

If native is ambiguous, zero lost upper requires `MD=1`; this forces `MP=1`, so added upper is positive because `ND=0`. Thus, for that fixed native cell, **no possible mapped pose can make both upper masks vanish**. This is impossibility of the fixed certificate, not of geometric equivalence or Shadow reachability.

For identity geometry (`dN=dM`) on a native-ambiguous cell, actual lost/added area is zero, both lower bounds are zero, and both independent upper bounds equal the cell's area. The prior manufactured identity control already demonstrates this without any real-case measurements. Refinement can shrink ambiguous regions, but a fixed finite subdivision can retain cells crossing a proximity boundary; independent intervals do not use the exact correlation between the two distance fields. A zero-area target cannot simply be reached by deciding that a small remaining bound is zero.

Saved-report readback shows fixed native ambiguity in every current grid:

| Case index | Density(mm) | Blocked part/band checks(out of18) |
|---|---:|---:|
|8|2|13|
|8|1|11|
|12|2|14|
|16|2|13|
|16|1|13|
|20|2|12|
|24|2|18|
|24|1|18|
|28|2|18|

These are source-fixed blockers independent of the mapped pose. Part/band cell tuples and areas overlap; they are not independent samples or disjoint physical area. Full per-check cell counts/areas and source hashes are in `contract_evidence.json`.

## Orientation and the unreachable overall PASS

`acceptance/evaluate_acceptance.py` explicitly assigns continuous orientation coverage `UNRESOLVED`. Its overall strict expression has only `FAIL` and `UNRESOLVED`, even if all proximity checks pass. The literal source has no full strict PASS branch. This is consistent with its documented limitation: sampled signs can supply a counterexample, but do not certify continuous nearest-feature normals.

A valid sampled sign reversal can refute the property. The absence of reversals cannot prove it. Normals and nearest labels can change discontinuously across feature/Voronoi boundaries; distance Lipschitz bounds do not become normal-angle bounds. Query/closest-point exterior tests, ties, buried component surfaces and object-side normal definition need coverage too. The rule's phrase 'required normals' needs a precise declared domain before implementing a positive certifier. Undefined/tangent/tied native directions must not silently become passing records or be filtered away to improve a result.

This is a missing positive-proof capability, not evidence that the old FAIL outputs are a software error. Adding an otherwise-PASS branch alone would be unsound.

## Sound certificate development that retains the zero-changed criterion

A separate version can extend the proof method while keeping the semantic target unchanged and preserving all historical outputs:

1. **Structural identity route.** Canonical, exactly identical labelled geometric sets in the same frame provide a direct identity proof for distance fields. This resolves the correlation problem even in band-crossing cells. It must verify the represented sets, not merely matching sampled distances, and still establish the required orientation-domain coverage. This supplies a necessary positive control for a certifier; it does not apply automatically to MANO versus Shadow.
2. **Certified cell route.** Use conservative, outward-rounded distance bounds to prove both fields near or both far, or certify a genuine mismatch. Subdivide unresolved cells. This retains valid existing lower-bound witnesses. Finite termination is not guaranteed at threshold crossings; report unresolved at a frozen resource limit.
3. **Paired/feature-aware route for remaining boundaries.** Where the independent route cannot close, prove equality of the two threshold predicates jointly. One possible implementation partitions nearest-feature regions and proves signs of squared-distance-minus-band-squared expressions agree, including their zero sets. Identity/canonical symmetry certificates can shortcut this. An approximate bound on distance difference alone does not prove identical threshold sets near the threshold.
4. **Continuous orientation route.** Certify object-facet normals, stable closest feature/part and all eligibility conditions over each required cell, then bound both facing scores away from zero with matching signs. Split feature-boundary regions or prove their treatment explicitly under the original required-domain interpretation. Never infer this from point sampling. A proper positive control has known identity/clearance and stable normals over its entire declared domain; negative controls include sign reversal, lost/added regions, feature switches and unavailable normals.

The first deliverable should be a **finite certifier capability check on mathematical positive/negative controls**, not another six-case optimization. Prove that at least one valid complete-domain control can obtain PASS, identity under threshold ambiguity is handled correctly, and known mismatches remain FAIL. This is proposed work; it has not been implemented or executed by this review. For a zero-changed claim on different anatomies, boundary coincidence may still be demanding; no global impossibility has been proved.

## Morphology and contact assumptions

Native part labels are MANO skinning-weight/triangle-majority proxies; Shadow labels are body/component groupings. They define an auditable semantic correspondence, not established homologous material patches. The actual saved cases showed no demonstrated accidental fallback-label bug in the independent source audit. Changing the correspondence to improve scores would be a new policy, not fixing a proven indexing error.

The query domain is the compiled visual-object surface. Native open anatomical surface patches and target per-component collider boundaries are the compared sets; target sets can include buried/internal surfaces. They are not physical-contact masks, nor automatically the boundary of the solid collider union. MuJoCo's pairwise component contacts and geometric exterior accessibility are also distinct. The prior exposed-boundary diagnostic found surviving counterexamples, so the entire historical failure cannot be dismissed as a buried-surface artifact.

## Original HUG evidence

HUG's simulation evaluates a position-actuated MANO hand using lift success and fingertip-contact error. Its robot deployment retargets to Ability and WUJI, using a fixed embodiment-specific alignment for Ability; it acknowledges that retargeting can fail when a robot cannot realize a feasible analog. The paper does not establish our zero-changed, six-part,1/2/5mm continuum rule as a universal retargeting requirement or supply a Shadow-specific tolerance. These observations support evaluating a declared robot pipeline while reporting morphology distortion, not claiming native-contact identity from simulator correctness. [HUG paper, §§5.2,7,F.2](https://arxiv.org/html/2606.17054v1), [official implementation](https://github.com/KevinyWu/hug).

## Recommended decision now

Stop using 'make the frozen evaluator output full PASS' as the optimizer's stopping condition. Preserve strict native-preservation FAIL/UNRESOLVED evidence. Separate the finite engineering contract for the chosen common Shadow embodiment from the optional stronger native-fidelity claim. Continue only bounded, explicitly versioned mapper improvements against the declared engineering objectives, and build positive-proof certificate capability before claiming full strict equivalence. Neither a changed label nor this diagnosis authorizes new eval data, proves physical fidelity, or validates the predictor.
