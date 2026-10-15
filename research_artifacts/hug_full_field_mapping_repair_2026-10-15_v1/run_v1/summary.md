# Full-field current-geometry diagnostic: completed, no mapping PASS

15 October 2026. Frozen commit `354452ef64f695426b2dd12d5622a7bf4d536f77`,
protocol SHA-256 `6b4b30041bead914a0790df7fc4b95767c54a34b6c9870c352d87efe9c5e5109`.
Both fixed case subprocesses completed within their original 600-second limits.
There were10static forwards,4,914,008collider-point distance computations,
zero retarget solves, zero inference calls and zero physics steps. No historical
MJB or historical Case constructor was loaded. No new task data or fitting ran.

The audit covers every cell of the two original2mmgrids:19,174milk-box cells and
45,484mug cells. For each of the original-current and saved experimental poses,
it evaluates all341,721and804,343native-definite anatomical-part/band
relationships respectively. Native-ambiguous tuples remain separately visible
(3,411milk and14,369mug). Query points, bounds, classifications, complete
constraint indices and current distances are retained in each pose's NPZ.

Milk-box B/index20:

- Original current pose:530definite violations (215lost-near,315added-near),
  matching all530historical definite mismatch tuples. Its literal historical
  fingertip comparison remains refused; the10mmmean gate passes.
- Saved experimental pose:684definite violations (247lost,437added), despite
  passing all original literal fingertip caps and the mean gate.
- Of the530old definite tuples,17are now proved preserved,150are unresolved,
  and363remain definite violations. It introduces321new definite violations
  and loses909relationships that were proved passing in the original-current
  pose. The latter includes relationships becoming unresolved.
- Worker measurement time27.208s; parent subprocess elapsed28.424s;5forwards.

Mug B/index28:

- Original current pose:3,183definite violations (2,359lost,824added), matching
  all3,183historical definite mismatch tuples. Its literal historical fingertip
  comparison remains refused; the10mmmean gate passes.
- Saved experimental pose:2,874definite violations (1,810lost,1,064added),
  passing all original literal fingertip caps and the mean gate.
- Of the3,183old definite tuples,53are now proved preserved,771are unresolved,
  and2,359remain definite violations. It introduces515new definite violations
  and loses2,112previously proved-passing original-current relationships,
  including transitions into uncertainty.
- Worker measurement time64.481s; parent subprocess elapsed65.426s;5forwards.

All four poses therefore have a definite full-field proximity `FAIL` within the
declared geometry/arithmetic assumptions. The previously improved capped-row
candidates do not preserve the complete measured field. Reducing one count or
turning an old definite mismatch into an unresolved interval is not an accepted
repair. All original caps and strict/inclusive boundary predicates are unchanged.

These are overlapping **cell/part/band tuples**, not independent contacts,
unique failure sites, disjoint areas, task outcomes or success rates. Many
native-definite tuples are far relationships. Anatomical parts and1/2/5mm bands
overlap. There are two repeatedly inspected native development grasps and four
robot poses; this is not independent validation or a global infeasibility proof.

Numerical limitation: the frozen distance kernel emitted NumPy `matmul`
divide-by-zero, overflow and invalid-value RuntimeWarnings in hull/halfspace
operations. All returned unsigned distances passed the finite-value guard;
the warnings and call sites remain in `case_20.log` and `case_28.log`.
Finiteness alone does not establish the cause or absence of a correctness
effect. Independent warning diagnosis is separate and pending at this receipt.
No source was patched and no measurement retried during this run. These are
conditional numerical diagnostics, not formally outward-rounded certificates.

Full orientation and continuous coverage remain `NOT_RUN`/`NOT_PROVED`.
The historical feasible solver was not invoked: its current-scene30Dadapter
was not yet qualified. No fidelity PASS, production promotion or permission for
new task collection/training is issued.

A read-only saved-array replay independently reconstructed the lost/added
Boolean predicates from the serialized native/current distance arrays, cell
radii and unchanged10nmarithmetic allowance. All four full masks and counts
matched. This replay performed no geometry query, model construction or FK.

Final report SHA-256:
`519ea6c52556496ee785bb57f6176f83378ff63823c39e5d2950d22ef1dd0445`.
Case20result SHA-256:
`17c1361a1193af6383e3f7f424d394ac3016983488ab303e572254a2c32e0358`.
Case28result SHA-256:
`fd0cb96c3b270223819d6cd7404430889ce33e1a1c0109cc269cad14f1275e99`.
All raw pose reports, NPZ arrays and worker logs remain unchanged.
