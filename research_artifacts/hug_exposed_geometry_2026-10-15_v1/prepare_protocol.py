"""Source-only protocol construction: no geometry, FK, optimization or inference."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
POOL=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/pooled_selection/candidate_targets.json')
WIT=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
CONTRACT=Path('/private/tmp/hug_collision_contract_review_2026-10-15_v1/compiled_contract.json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
old=read(BASE/'protocol.json');pool=read(POOL);witness=read(WIT)
inputs=dict(old['input_sha256'])
for p in [BASE/'protocol.json',POOL,WIT,CONTRACT]+[BASE/n for n in ['constraint_engine.py','constraint_core.py','exact_direction.py','free_palm_kinematics.py','geometry_helpers.py','surface_component_guard.py']]:inputs[str(p)]=sha(p)
source_rows={}
for index in [8,12,16,20,24,28]:
    source_rows[str(index)]={name:[j for j,r in enumerate(witness[key]) if r['source_inventory_index']==index] for name,key in [('proximity_indices','whole_cell_rows'),('orientation_indices','orientation_rows')]}
    c=next(c for c in pool['cases'] if c['source_inventory_index']==index)
    source_rows[str(index)].update(native_sha256=c['native_sha256'],pooled_state_scaled=c['selected_state_scaled'],original_state_scaled=[0.]*27)
protocol={'schema':'hug-exposed-occupied-geometry-diagnostic-v1','execution_authorized_after_independent_review':False,
'cases':[8,12,16,20,24,28],'allowed_cohorts':[[8,12,16,20,24,28],[8,16,24],[12,20,28]],'pooled_candidate_path':str(POOL),
'input_sha256':inputs,'implementation_sha256':{str(HERE/n):sha(HERE/n) for n in ['exposed_classifier.py','test_exposed_classifier.py','run_diagnostic.py','prepare_protocol.py']},'preflight_sha256':{},'source_rows':source_rows,
'source_scope':{'poses_per_case':['original','pooled'],'proximity_rows':178,'orientation_rows':134,'total_row_pose_records':624,'duplicates':'Preserved in source order, no new coordinates. Capped retained witnesses, not full failure set or continuous coverage.','native_surface':'Literal MANO anatomical skin triangles; not asserted to be an exposed union boundary or a non-self-intersecting solid.'},
'geometry_reference':{'hand':'45 actual compiled, object-compatible Shadow colliders; per-mesh convex hull plus exact analytic primitives. Includes forearm/wrist carriers in whole union and burial.','part_subset':'Boundary of whole occupied collider union intersect boundary of any collider of requested anatomical part.','simulator_distinction':'Occupied-union accessibility diagnostic is distinct from MuJoCo pairwise contact generation. No physics/contact-success claim.','object':'Frozen prior object poses and colliders. Queries are existing visual-object surface sites. No new object geometry cause audit; prior visual/collision distinction and saved visual-membership diagnostics preserved.','contract_receipt':str(CONTRACT)},
'numerical_classification_tolerance_m':1e-8,
'method':{'occupancy':'Any component signed distance < -eps certifies strict component/union interior; all >eps certifies exterior. Otherwise boundary/coincident UNRESOLVED; touching component faces may be inside union.','exposure':'At only the nearest global and nearest requested-part component points, own signed-distance residual <=eps and all other signed distances >eps provide a strictly separated exposed-point witness. Any other signed distance <-eps is buried. Coincident/near/own-residual cases unresolved.','outside_whole_union':'Minimum absolute component distance is exact conditional on strict exterior, consistent oracle and no contradictory burial; ties do not imply unique normal.','inside_whole_union':'Maximum component penetration depth is conditional lower bound. An available strictly separated candidate gives an upper bound. Otherwise upper=null.','part_subset':'Minimum individual-component surface distance is lower bound. A strictly separated same-part candidate is upper; if the minimum part candidate is exposed, distance is exact conditional. No exposed point found does not prove subset empty.','oracle_validation':'All SDF versus query-to-closest distance residuals and both candidates own-surface residuals <=eps; nonfinite/malformed oracle aborts; inconsistency refuses bounds and exact claims.','numeric_claim':'Floating reconstructed-geometry diagnostics, not interval-certified machine proofs. No normal uniqueness or new orientation sign claims.','rays':0,'optimizer_calls':0},
'proximity_diagnostic':'A bound-based persistence check is separately labelled conditional: enlarge source cell error by additional eps, apply original strict far / inclusive near mask, preserve historical pass/fail. Failure to prove persistence is NOT evidence of repair.',
'finite_budget':{'robot_poses':12,'retained_row_pose_records':624,'hand_component_query_points':84240,'query_batch_calls':1080,'new_grid_points':0,'rays':0,'optimizers':0,'extra_diagnostic_starts':0},
'pose_gates':'Exact original native-world tip caps, mean <=10mm, original fixed quaternion/physical and q0±.35/translation AABB; fresh source reconstruction and 1e-12 replay agreement. Pose gates are unchanged historical project rules, not claimed HUG published thresholds.',
'concurrency':{'max_processes':2,'numba_threads_per_process':2,'reset_after_all_inherited_imports':True},
'no_acceptance_change':True,'new_eval_data':False,'new_inference':False,'physics_steps':0,'new_fit':False,'production_promotion':False,
'limitations':['No CSG reconstruction or global inside nearest-boundary solution.','No exposure proof from finite probes, nonnegative SDF alone, or removing buried candidates.','No native skin exposure certification.','Neither zero measured discrepancies nor numerical exact claims establish whole-surface mapping validation.']}
(HERE/'protocol.draft.json').write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
print(sha(HERE/'protocol.draft.json'))
