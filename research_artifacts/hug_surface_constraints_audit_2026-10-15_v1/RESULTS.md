# Direct HUG surface-constraint audit results

This is a static development-case mapping audit of the same six existing HUG grasps. It contains no new inference, physics rollout, evaluation dataset, outcome labels or model training. The Shadow Hand and original acceptance criteria were retained.

All six independent source/FK/root/domain and original fingertip-cap checks passed. The 9 fresh grid comparisons returned proximity {'FAIL': 9}, pointwise orientation {'FAIL': 9}, and strict relationship {'FAIL': 9}. These counts are case-resolution comparisons, not independent grasps.

The predeclared cross-version selection uses the identical exact retained-witness rank with stable V1 ties. It retained V1 for all six cases; five were exact ties, and Mug A had a better V1 rank. This is a fitting selection, not scientific acceptance. Original fallback and refused diagnostics remain explicitly labeled.

| Case index | Role | Palm shift (mm) | Original → selected mean tip error (mm) | Resolution: proximity / orientation / strict |
|---|---|---:|---:|---|
| 8 | tip_feasible_constraint_refused_diagnostic | 0.063565 | 5.844763 → 5.838904 | 2 mm: FAIL / FAIL / FAIL; 1 mm: FAIL / FAIL / FAIL |
| 12 | tip_feasible_constraint_refused_diagnostic | 1.994976 | 5.866399 → 5.708878 | 2 mm: FAIL / FAIL / FAIL |
| 16 | original_fallback | 0.000000 | 6.646875 → 6.646875 | 2 mm: FAIL / FAIL / FAIL; 1 mm: FAIL / FAIL / FAIL |
| 20 | tip_feasible_constraint_refused_diagnostic | 4.891691 | 8.189192 → 7.240657 | 2 mm: FAIL / FAIL / FAIL |
| 24 | tip_feasible_constraint_refused_diagnostic | 22.359525 | 6.609030 → 4.492666 | 2 mm: FAIL / FAIL / FAIL; 1 mm: FAIL / FAIL / FAIL |
| 28 | tip_feasible_constraint_refused_diagnostic | 3.770633 | 6.608610 → 6.181258 | 2 mm: FAIL / FAIL / FAIL |

## Saved fitting witnesses are separate evidence

The following counts refer only to the fixed capped retained rows, including repeated nested-band/history records. They are not full-grid coverage or independent sites. Orientation with no saved row is unavailable, not passed.

| Case | Proximity retained rows passed: original → selected | Orientation retained rows passed: original → selected | Orientation coverage |
|---|---:|---:|---|
| 8 | 0/33 → 0/33 | 0/54 → 0/54 | PRESENT_CAPPED_EXAMPLES |
| 12 | 0/15 → 0/15 | 2/11 → 0/11 | PRESENT_CAPPED_EXAMPLES |
| 16 | 0/37 → 0/37 | 0/52 → 0/52 | PRESENT_CAPPED_EXAMPLES |
| 20 | 0/14 → 2/14 | 0/15 → 2/15 | PRESENT_CAPPED_EXAMPLES |
| 24 | 0/57 → 10/57 | 0/0 → 0/0 | UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS |
| 28 | 0/22 → 0/22 | 0/2 → 1/2 | PRESENT_CAPPED_EXAMPLES |

## Palm bounds on the unchanged coarse grid

Values below use the same 2 mm band, 2 mm grid and primary 10 nm arithmetic allowance. They are conditional lower/upper bounds on anatomical proximity mismatch. Anatomical parts overlap and must not be summed as physical contact area. A smaller lower bound alone does not prove smaller total mismatch.

| Case | Lost lower: original → selected (mm²) | Lost upper (mm²) | Added lower (mm²) | Added upper (mm²) |
|---|---:|---:|---:|---:|
| 8 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 |
| 12 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 |
| 16 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 |
| 20 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 | 0.000000 → 0.000000 |
| 24 | 105.161503 → 86.996801 | 1155.734449 → 1137.312617 | 0.000000 → 0.000000 | 0.000000 → 304.058360 |
| 28 | 53.090089 → 53.090089 | 827.107025 → 827.107025 | 0.000000 → 0.000000 | 0.000000 → 0.000000 |

## Interpretation and reproducibility

These numerical checks do not establish faithful surface mapping or failure-label preservation. Definite proximity mismatches and reliable sign reversals remain failures. Absence of a sampled reversal remains unresolved for continuous orientation. The independent-interval identity control also shows why ambiguous cells cannot certify zero possible mismatch even when surfaces agree; that certificate limitation does not erase the observed definite failures.

Replaying the identical fitter import sequence showed four effective Numba threads per process, overriding the requested two. The unchanged launch drivers contain no later reset, so the primary archives disclose the inferred four-thread launch setting rather than claiming compliance with the requested limit. The final independent grid audit resets threads after imports and verifies two active threads per process, with at most two audit processes.

- Original acceptance-rule SHA: `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`.
- Final audit protocol SHA: `3b9e46fb8ff69fb54f62ac46778498bd53c984580eb3e499dfc2f8755d8e95de`.
- Pooled selected candidate SHA: `acb4870253fc81b42a4972eb22d28e8970ff751d60ec12ad41b63a822ee67ee3`.
- Independent arithmetic/source read-back: 578 checks passed; 56 saved orientation witness records, 50 distinct by case/density/query/direction.
- Preflight time: 11.743 s. Cohort times: 481.689 s, 966.152 s.
- See `summary.json` for per-part lower/upper comparisons and orientation denominators; full reports retain all bands and allowances.
- Separate saved-row and document-review supplements preserve their own scopes. Document review covers method and retained-row interpretation, not pending-at-review full-grid outcome prose.
