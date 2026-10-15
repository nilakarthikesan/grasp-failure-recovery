from pathlib import Path
import hashlib,json,shutil,datetime
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
B=W/'research_artifacts/dga_practical_retargeting_2026-10-15_v1'
C=Path('/private/tmp/dga_practical_retargeting_20261015_v1/control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
src=W/'docs/paper/working_draft.tex'
expected='8d55b452eb31e1f978a27933d7deb2dc8075eb4127e777d287bcbe2d32da3c54'
assert sha(src)==expected,'Paper changed since successful compilation; preserve latest and recheck'
paper=B/'paper';paper.mkdir(exist_ok=True)
shutil.copyfile(src,paper/'compiled_working_draft.tex')
write(paper/'compilation_receipt.json',{'tool':'mcp__codex_app__compile_latex_document','path':str(src),'source_sha256_before':expected,'source_sha256_after':sha(src),'kind':'success','message':"The current source compiled successfully with the desktop editor's compiler.",'compiler_installation':'built-in desktop editor','scope':'Current working paper; may include preserved concurrent uncommitted provider sections. Not a claim that the entire working file was committed in this change.','outside_dga_preserved':True})
progress='\nDGA practical-study compilation confirmed with the desktop compiler at whole-source SHA-256 `'+expected+'`. The exact working-source snapshot and tool receipt are archived; unrelated sections and the open paper path were preserved.\n'
pp=W/'docs/PAPER_PROGRESS.md';old=pp.read_text();assert progress not in old;pp.write_text(old+progress)
u=C/'owned_doc_updates.json';d=json.loads(u.read_text());d['append_updates']['docs/PAPER_PROGRESS.md']+=progress;write(u,d)

s=json.loads((B/'validation/authoritative_all20_summary_v1.json').read_text())
summary={'schema_version':'dga_practical_static_admission_decision_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'STATIC_PILOT_ADMITTED_TESTED_SCOPE','pipeline':'Pinned DGA MultiDex plus explicit legal-reference projection and independently gated practical correction on unchanged Shadow','accounting':s['cohort_accounting'],'branch_accounting':s['counts'],'max_final_landmark_mm':s['max_final_landmark_m']*1000,'pooled_final_rms_mm':s['pooled_final_rms_m']*1000,'source_invalid_handling':'Refuse before projection; preserve unmodified source and full denominator','native_holdout':False,'objects_tested':'One unchanged 50 mm cube; 20 previously inspected native outputs','native_fidelity_accepted':False,'physically_calibrated_distortion_tolerance':None,'geometry_property':'Conditional nonincrease of each finger worst uncapped native-relative distance residual over full cube surface versus legal reference; not pointwise nonregression','arithmetic_assumption_m_per_distance':1e-9,'absolute_native_palm_overlap_repaired':False,'execution_eligible':False,'physics_steps':0,'native_inference_calls':0,'evaluation_data_generated':False,'training_calls':0,'authoritative_summary':{'path':'validation/authoritative_all20_summary_v1.json','sha256':sha(B/'validation/authoritative_all20_summary_v1.json')},'final_pose_archive':{'path':'validation/authoritative_all20_final_static_poses.npz','sha256':sha(B/'validation/authoritative_all20_final_static_poses.npz')}}
write(B/'final_summary_v1.json',summary)

readme='''# DGA practical retargeting: completed static pilot

15 October 2026. **STATIC_PILOT_ADMITTED_TESTED_SCOPE** for the pinned DGA MultiDex plus explicit retargeting pipeline on the unchanged Shadow hand, limited to these saved cube outputs and the conditional numerical assumptions below. Provider execution remains disabled. Native-hand fidelity and physical adequacy are not accepted by this result.

## Actual result

| Saved native cohort | Inputs | Source-valid static mappings | Native-invalid refusals |
| --- | ---: | ---: | ---: |
| Recent MultiDex regression | 16 | 11 | 5 |
| Original four MultiDex regression | 4 | 2 | 2 |
| Total | 20 | 13 | 7 |

All outputs are previously inspected development/regression material on the same 50 mm cube. None is an untouched native holdout. Original DexGraspNet outputs remain a separate lineage and were not substituted into this cohort; the inventory records 18 unique original-checkpoint outputs with native-domain failures.

Of 65 finger branches across the thirteen valid poses, 42 preserve reference target geometry exactly. Twenty-three branches propose a correction. One passes independent geometry admission; 22 remain unresolved at the frozen finest resolution and fall back to the explicit legal reference. “Unresolved” does not prove regression or impossibility. Every refusal and fallback remains in the accounting.

The admitted correction is recent row 4's thumb. Under the declared 1 nm per-distance arithmetic assumption, its continuous worst native-relative distance residual over the entire cube surface lies in **[1.678521, 2.117103] mm**, compared with reference **[2.508736, 2.947318] mm**. The candidate upper bound is 0.391633 mm below the reference lower bound. Thumb maximum landmark error falls from 0.852864 to 0.584096 mm; thumb SSE decreases from 1.348861 to 1.013983 mm². The little-finger proposal does not pass and is reverted, so the final whole-pose maximum is **0.605692 mm**, not the provisional candidate's 0.584096 mm. All other branches satisfy their own unchanged/nonregression gates.

Across all thirteen final poses, maximum native-to-target landmark error remains **6.075600 mm**. Maximum per-pose RMS is 1.778242 mm; pooled RMS over 13 × 28 sites decreases only from 1.179927 to 1.179537 mm. These are distortion measurements, not experimentally calibrated acceptable error thresholds. Prior palm overlap remains: final root/wrist identities carry forward six recent-row strict palm failures. The whole-surface test controls each finger's worst distance residual; it does not establish pointwise nonregression, preserved near/contact regions, signed occupancy, normals, grasp success or failure-label equivalence.

Independent literal XML/SciPy FK and compiled MuJoCo kinematics agree across final poses to 2.483e−16 m and 7.094e−16 rad; target limits match exactly. These numbers concern numerical realization of the prescribed target pose, not native-to-target accuracy. Only static kinematics is used; simulation/forward/inverse/integration entry points are blocked in the checker.

## Prospective criteria and repairs

See the [protocol](../../docs/DGA_PRACTICAL_RETARGETING_PROTOCOL.md). The criteria SHA is `6eb351dc097daffb4d08f04117615714de2120d3395367f900e4cefb444d4d34`; proposal method v3 SHA is `44c2c1e12fc51d912090c7b2987af8ec0631832127c1d764215972c4826ecf6e`; validator policy v5 SHA is `a37227a4ee6702ae2005f4be1159b148b9a98fa5dfe9b4fbfecfbc79a5fa18df`. Each was frozen before its associated run. Earlier versions remain preserved, including the incomplete v2 validation.

Source-invalid data are rejected before retargeting. The reference is explicit Euclidean finger-joint projection into literal target intervals, with native palm and wrists fixed. Already legal branches cannot change. Projection minimizes angular displacement only; it neither repairs the native output nor proves geometric optimality. Optional proposals need per-finger maximum landmark and SSE nonregression plus independently bounded uncapped worst-distance nonregression and a strict improvement. A lower fitting loss alone cannot pass.

Repairs address solver exceptions after an improved trial, the capped-vs-uncapped objective mismatch, unused vertices in mesh search bounds, the missing 1 mm target thumb visual box and conservative validation runtime. The final triangle search uses face-referenced vertex bounds, radius buckets and sphere/AABB pruning. Adaptive refinement preserves all leaf upper bounds and verifies full cube coverage on the original shifted 1, 0.5 and 0.25 mm grids. None of these changes relaxes the frozen admission inequalities.

Static repository tests passed (126 provider tests, 10 practical tests, 18 independent adversarial tests). Separate solver-exception, proposer and independent geometry controls are retained with their receipts. The fitting stage's numerical warnings are recorded rather than erased; final acceptance is independently recomputed with warnings treated as errors. Legacy proposer dispatch counts omit uncapped mesh work and are not total computation counts.

## Authoritative evidence and replay

- [Decision summary](final_summary_v1.json).
- [All-20 independent summary](validation/authoritative_all20_summary_v1.json), including per-pose and per-finger errors, counts and final compiled checks.
- [Final selected poses](validation/authoritative_all20_final_static_poses.npz), with original/recent cohort keys; these are static mapping artifacts, not evaluation episodes or outcome data.
- `validation/recent16_authoritative_v3_part_a/`, `validation/recent16_authoritative_v3_part_b/`, `validation/original4_authoritative_v3/`: full queries, leaf-cover ledgers, row reports and compiled snapshots.
- `protocol/*complete_geometry_recount_v1.json` and the combined independent recount: independent saved-evidence recomputation of coverage, bounds, adoption and final mixed poses.
- `mapper/implementation_v3/` and `mapper/results_v3/`; `mapper/results_original_four_regression_v3/` is kept separate. These contain **provisional** results, superseded for admission by the independent selected-pose archive.
- `validation/final_palm_diagnostic_identity_v2.json`: unchanged palm evidence, including inherited failures.
- `dependency_index.json`: original-path/hash-to-repository-file resolution, including dependencies already preserved by the prior committed study. Historical reports keep their original paths and hashes; resolve those paths through this index and the `mapper/`, `protocol/`, `validation/` directory mapping when replaying elsewhere. Do not edit a frozen method in place or mistake historical candidate files for final admission.
- `paper/`: exact compiled working manuscript and desktop compiler receipt. Only the DGA section was edited; concurrent unrelated manuscript work was preserved.
- `evidence_manifest.json`: final file hashes and preservation receipt.

The source APIs deliberately report geometry/physics as not assessed when called without the independent study gates. No provider registry promotion occurs. The new static protocol treats DGA plus the named mapper as a distinct generator pipeline; future outcome labels must describe the actual Shadow attempt and retain mapper identity, distortion, native-invalid refusals and the full input denominator. Additional native object coverage and physical/predictor adequacy remain separate future checks. No physics, new native inference, evaluation episodes/labels or training ran here. Historical strict-preservation studies and the unchanged Shadow target remain intact.
'''
(B/'README.md').write_text(readme)
print(json.dumps({'summary':str(B/'final_summary_v1.json'),'compiled_source_sha256':expected,'counts':s['counts']},indent=2))
