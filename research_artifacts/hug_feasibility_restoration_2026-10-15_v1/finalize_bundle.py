"""Format already saved results and bind the finalized local artifact files."""
from pathlib import Path
import hashlib,json
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((OUT/'restoration_summary.json').read_text());d=json.loads((OUT/'endpoint_outcome_diagnostics.json').read_text());names={8:'Apple A',12:'Apple B',16:'Milk box A',20:'Milk box B',24:'Mug A',28:'Mug B'}
lines=[]
for c in r['cases']:
 endpoints=c['endpoints'];gates=[e['gate'] for e in endpoints];n=[(len(g['lost_orientation_valid_rows']),len(g['lost_orientation_pass_rows'])) for g in gates]
 prox=[f"{e['exact_surface_summary']['proximity_pass']}/{e['exact_surface_summary']['proximity_rows']}" for e in endpoints]
 lines.append(f"| {names[c['index']]} | {n[0][0]} / {n[0][1]} | {n[1][0]} / {n[1][1]} | {prox[0]}, {prox[1]} |")
fk=sum(e['counts']['FK_executions'] for c in r['cases'] for e in c['endpoints']);jac=sum(e['counts']['analytic_jacobian_executions'] for c in r['cases'] for e in c['endpoints']);fwd=sum(e['actual_projection_mj_forward_calls'] for c in r['cases'] for e in c['endpoints']);checks=sum(c['exact_surface_checks'] for c in r['cases'])
assert (fk,jac,fwd,checks)==(1101,161,2524,24) and not r['eligible_coarse_audit_indices']
assert all(e['restored_tip_feasible'] for c in r['cases'] for e in c['endpoints'])
report='''# Saved-endpoint fingertip feasibility restoration

15 October 2026. **12 restored endpoints; 0 accepted improvements; all six prior pooled poses retained unchanged.** No fresh coarse or fine grids were warranted or run by this method.

## Frozen question and method

The preceding V2 surface search rejected all twelve terminal states for small individual fingertip-cap excesses. This study asks whether a bounded, FK-only projection can remove those excesses while exposing a useful retained-surface improvement. All twelve source endpoints, their complete 27-coordinate states and exact trace provenance were frozen before any new geometry or projection.

One projection per endpoint minimizes normalized squared distance from that endpoint within a fixed per-coordinate trust box: ±0.00625 rad for each joint and ±1.25 mm for each world-palm coordinate, intersected with the original physical/correction/reach bounds. It is not a 1.25 mm Euclidean sphere. Selection uses the closest of the observed exact-feasible endpoint, major-iterate and terminal checkpoints, with stable ties; it does not claim a globally nearest feasible pose. Raw line-search probes are never eligible. Nine optimization runs report convergence; three reach their 40-iteration limit but already contain a fresh-verified eligible checkpoint.

Every returned projection satisfies the original literal individual fingertip caps, inclusive 10 mm mean gate, native-world targets, floating-root quaternion, original domain and endpoint trust box. The 1 nm inward optimization target is not an acceptance allowance. The strict surface-equivalence and original-cap nonworsening gates are this project's historical criteria, not thresholds prescribed by HUG's published evaluation.

## Recorded results

All twelve saved endpoints acquire an exact-tip-feasible checkpoint. The palm correction from each saved endpoint is only 0.000114–0.039256 mm. These small restoration corrections must not be confused with the endpoint's larger previously optimized displacement from the native anchor.

No restored checkpoint strictly improves the existing exact five-component retained-witness rank while also preserving every orientation pass and every eligible orientation row present in either the original or pooled reference pose. Ten endpoints invalidate protected orientation evidence. The other two are Mug A, which has no saved reliable orientation witness and remains worse by the exact rank.

Counts below retain duplicate bands/studies. They are not independent contacts, rates or surface areas. “Lost eligible / lost passing” compares each restored endpoint against the protected original-or-pooled orientation sets.

| Existing grasp | Endpoint 0: lost eligible / lost passing | Endpoint 1: lost eligible / lost passing | Proximity passes, endpoints 0 and 1 |
|---|---:|---:|---:|
'''+ '\n'.join(lines)+'''

Mug A endpoint 0 satisfies 11/57 proximity witnesses instead of the pooled reference's 10/57, but its maximum normalized violation worsens from 5.0 to 5.445343. Endpoint 1 satisfies 8/57 and has maximum violation 5.289485. The frozen ranking correctly refuses both; an extra passing example cannot compensate for a worse worst violation. No orientation conclusion is available for Mug A's empty retained orientation set.

Common invalidity reasons include closest hand points inside/on the visual object, object queries inside the hand, ambiguous mesh facets, buried closest points, source-band loss and nearest-part changes. These reasons overlap. Exact orientation was not measured at the old tip-infeasible endpoints, so the comparison does **not** establish that the tiny restoration itself caused this invalidity. It reveals that the restored configurations still lack admissible preserved surface evidence.

## Work and verification accounting

There are exactly **1,101 tip-FK callback executions + 161 analytic-Jacobian callback executions = 2,524 projection `mj_forward` calls**, because each callback stages two static forwards. Cache hits do not consume these callback budgets. Scene/source checks and final-source FK calls are separately counted per case. There are **24 exact retained-surface checks**, four per case, below the frozen maximum of five. All declared per-endpoint budgets are respected. Numba is set and asserted to two threads after all inherited imports.

The method performs no physics integration, HUG inference, new native grasp generation, evaluation-label generation, prediction-model training or production promotion. All six returned reference poses remain subject to their previously recorded surface failures. Restoring fingertip feasibility is not a mapping-validity certificate, and these bounded failures do not prove that no acceptable Shadow mapping exists.

## Evidence

- `protocol.json`: frozen source states, source/code hashes, bounds, budgets and review receipt.
- `candidate_targets.json`: all six unchanged returned references and all twelve restoration results.
- `restoration_summary.json`: compact endpoint selection, rank, guard and execution details.
- `endpoint_outcome_diagnostics.json`: saved geometry reasons and correction sizes.
- `traces/*.jsonl.gz`: every FK/Jacobian/cache/checkpoint/refusal event.
- `bundle_manifest.json`: SHA-256 of this local bundle, checked again after writing.

Independent final numerical review is maintained in the separate reviewer bundle; its result is not inferred from optimizer convergence. No full-grid acceptance result is claimed here.
'''
(OUT/'REPORT.md').write_text(report)
exclude={'pycache','compile_cache','__pycache__'}
files=[]
for p in sorted(OUT.rglob('*')):
 if not p.is_file() or any(x in exclude for x in p.relative_to(OUT).parts) or p.name=='bundle_manifest.json':continue
 files.append({'relative_path':str(p.relative_to(OUT)),'sha256':sha(p),'bytes':p.stat().st_size})
manifest={'schema':'hug_feasibility_restoration_local_bundle_v1','root':str(OUT),'files':files,'excluded_runtime_cache_directories':sorted(exclude),'scientific_source_assets_bound_by_protocol_not_duplicated':True,'all_local_hashes_readback_verified':True,'new_eval_data':False}
mp=OUT/'bundle_manifest.json';assert not mp.exists();mp.write_text(json.dumps(manifest,indent=2)+'\n')
readback=json.loads(mp.read_text())
for row in readback['files']:assert sha(OUT/row['relative_path'])==row['sha256'],row['relative_path']
print(json.dumps({'report':str(OUT/'REPORT.md'),'manifest':str(mp),'manifest_sha256':sha(mp),'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files)}))
