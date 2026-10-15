# Static mapping results — 2026-10-15

HUG→Shadow mapping is not accepted under the original frozen criteria.

| Separate audit | Existing native inputs | Fresh grid comparisons | Proximity | Strict relationship | Pointwise orientation |
|---|---:|---:|---|---|---|
| Initial objective-ranked translated diagnostics | 6 | 9 | 9 FAIL | 9 FAIL | 7 FAIL, 2 UNRESOLVED |
| Explicit translation-proposal supplement | Existing indices 8/12/16 | 5 | 5 FAIL | 5 FAIL | 5 FAIL |

There were 14 actual fresh grid comparisons in two separately frozen protocols, not 14 grasps. The six-case intentional-palm-placement coverage view uses the five supplemental comparisons for indices 8/12/16 and the four initial comparisons for indices 20/24/28: it contains nine comparisons, all proximity/strict FAIL, with seven pointwise-orientation FAIL and two UNRESOLVED. This view does not rewrite or combine the protocols' historical results.

All audited poses passed independent source/FK and exact original fingertip caps, with the unchanged strict 10 mm mean limit. All were diagnostic alternatives refused by the extra direction-selection guard. Both fit versions' guarded returned outputs remain exact original fallbacks. No diagnostic was promoted by the audit.

The first objective-ranked diagnostics for indices 8/12/16 moved the palm by only approximately 1–3 nm, arising from restoration of joint-dominated proposals; they did not provide a meaningful intentional translation test. The separate supplement requires ancestry from an explicitly nonzero proposed palm translation before restoration, without an arbitrary final displacement threshold. The intentional audited palm movements are 0.063565, 1.994976, 0.031372, 4.891691, 3.716480, and 3.770633 mm for indices 8, 12, 16, 20, 24, and 28 respectively.

At the same coarse grid and frozen 2 mm proximity band, the two mug palm lost-mismatch certified lower bounds remain 105.1615 and 53.0901 mm² after the 3.716/3.771 mm movements. Finger relationships change in both directions. Mug A's ring lost-mismatch lower bound decreases from 106.5790 to 63.2204 mm², while its thumb gains an added-mismatch lower bound of 4.2279 mm² (formerly zero); mug B gains 5.7687 mm² of certified added thumb mismatch (formerly zero). Apple B's intentional movement reduces the ring lost-mismatch lower bound from 58.4935 to 50.4771 mm² but increases its added-mismatch lower bound from 88.9289 to 104.7713 mm². Its 2 mm-band native-facing/mapped-away witness count changes from zero to one, with eligible query count changing from 38 to 49.

These are per-part bounds; anatomical regions may overlap and are not summed into contact area. A lower-bound decrease alone does not certify that the total mismatch area decreased. Full summaries retain upper bounds, ambiguous area, and orientation denominators.

Independent arithmetic read-back completed 1,586 initial and 1,478 supplemental consistency checks. It recomputed the aggregate criteria and witness normal dot products. The initial reports retain 44 witness records (39 distinct case/density/query/direction combinations); the supplement retains 42 records (36 distinct). The maximum dot-product arithmetic discrepancy is 1.11e-16 in both. This is a read-back verification, not a second full geometry implementation.

The initial cohorts ran for 567.61 and 970.45 seconds in their recorded reports; the supplemental cohort ran for 385.69 seconds. Initial six-case preflight took 10.09 seconds, supplemental three-case preflight 8.63 seconds. At most two full audit processes ran concurrently, each with two active Numba threads. Every process recorded zero actual physics integration attempts; only the three manufactured integration-guard self-tests were invoked per process.

The immutable criterion SHA remains `87da8d6c03c0f5692f08cbad99f3af42f72f16d818b5b30df581791361d400d4`. No new HUG inference, eval episode, failure label, or training data was generated. Continuous contact equivalence and unseen-object validity remain unconfirmed. These bounded diagnostics do not prove that every possible palm/joint placement is impossible.

The optional `selection_source` display omission in frozen reports is documented in a separate read-only `selection_source_metadata_addendum.json`; the full candidate hash, native hash, case index, and actual pose were always bound. Scientific reports and code were not rewritten.
