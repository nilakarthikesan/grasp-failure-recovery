"""Run the frozen twelve-pose, 624-row static collision-union diagnostic."""
from pathlib import Path
import argparse,collections,hashlib,json,sys,time
import numpy as np
from exposed_classifier import classify
HERE=Path(__file__).resolve().parent
INHERITED=Path('/private/tmp/hug_feasibility_restoration_2026-10-15_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--indices',nargs='+',type=int,default=[8,12,16,20,24,28]);args=parser.parse_args()
    protocol_path=HERE/'protocol.json';protocol_hash=sha(protocol_path)
    protocol=read(protocol_path);assert protocol['execution_authorized_after_independent_review']
    assert args.indices in protocol['allowed_cohorts']
    def verify_closure():
        assert sha(protocol_path)==protocol_hash
        for group in ('input_sha256','implementation_sha256','preflight_sha256'):
            for path,digest in protocol[group].items():assert sha(path)==digest,(group,path)
    verify_closure()
    cohort_path=HERE/('cohort_'+'_'.join(map(str,args.indices))+'.json')
    assert not cohort_path.exists()
    assert all(not (HERE/f'case_{i}.json').exists() for i in args.indices)
    sys.path.insert(0,str(INHERITED));import constraint_engine as ce
    import numba,mujoco
    numba.set_num_threads(2);assert numba.get_num_threads()==2
    forwards=[0];original_forward=mujoco.mj_forward
    def forward(*a,**k):forwards[0]+=1;return original_forward(*a,**k)
    mujoco.mj_forward=forward
    pool=read(protocol['pooled_candidate_path']);sources={c['source_inventory_index']:c for c in pool['cases']}
    allrecords=[];t0=time.perf_counter()
    for index in args.indices:
        start=time.perf_counter();before=forwards[0];case=ce.Case(index);source=sources[index]
        declared=protocol['source_rows'][str(index)]
        assert source['native_sha256']==case.native_hash==declared['native_sha256']
        saved=read(ce.SAVED)
        assert case.prox_rows==[saved['whole_cell_rows'][j] for j in declared['proximity_indices']]
        assert case.ori_rows==[saved['orientation_rows'][j] for j in declared['orientation_indices']]
        assert np.array_equal(case.caps,source['authoritative_original_tip_caps_sqrt_einsum_m'])
        assert np.array_equal(case.bounds,source['bounds_scaled'])
        assert np.array_equal(case.q0,source['original_qpos_rad'])
        P=len(case.prox_rows);O=len(case.ori_rows);caseposes=[]
        for role,state,diagkey in [('original',np.zeros(27),'baseline_constraint_diagnostics'),('pooled',np.asarray(source['selected_state_scaled']),'selected_constraint_diagnostics')]:
            prior=source[diagkey]
            assert np.array_equal(state,declared[role+'_state_scaled'])
            assert np.array_equal(state,prior['state_scaled'])
            case.stage(state)
            hand=ce.refresh_colliders(case.original_hand,case.model,case.data,case.opos,case.oR)
            assert len(hand)==45 and np.all(state>=case.bounds[:,0]) and np.all(state<=case.bounds[:,1])
            q=case.data.qpos[case.adr];shift=.005*state[24:];aabb=np.asarray(case.domain['world_displacement_AABB_m'])
            assert np.all(q>=case.limits[:,0]) and np.all(q<=case.limits[:,1])
            assert np.all(q>=case.q0-.35) and np.all(q<=case.q0+.35)
            assert np.all(shift>=aabb[:,0]) and np.all(shift<=aabb[:,1])
            errors=ce._link_positions(case.model,case.data,ce.a.URDF)-case.native_tasks
            tips=np.sqrt(np.einsum('ij,ij->i',errors[:5],errors[:5]))
            assert np.all(tips<=case.caps) and float(np.mean(tips))<=.010
            assert np.max(np.abs(errors-np.asarray(prior['task_error_vectors_m'])))<1e-12
            assert np.max(np.abs(case.data.qpos[case.rootadr:case.rootadr+3]-np.asarray(prior['root_target_position_m'])))<1e-12
            assert np.array_equal(case.data.qpos[case.rootadr+3:case.rootadr+7],case.rootquat)
            names=[ce.a.PARTS[i] for i in case.all_parts]
            result=classify(hand,case.all_points,names,ce.a.geom_query,protocol['numerical_classification_tolerance_m'])
            assert result['component_point_queries']==45*(P+O) and result['candidate_point_queries']==90*(P+O)
            assert len(result['rows'])==P+O
            measured=np.asarray([r['part_individual_surface_distance_m'] for r in result['rows']])
            assert np.max(np.abs(measured-np.asarray(prior['part_distance_m'])))<1e-12
            for j,row in enumerate(result['rows']):
                prox=j<P;k=j if prox else j-P;src=case.prox_rows[k] if prox else case.ori_rows[k]
                row.update(row_kind='proximity' if prox else 'orientation',row_index_within_kind=k,
                    saved_witness_json_pointer=('/whole_cell_rows/' if prox else '/orientation_rows/')+str(protocol['source_rows'][str(index)]['proximity_indices' if prox else 'orientation_indices'][k]),
                    source_report=src['source_report'],source_report_sha256=src['source_report_sha256'],source_json_pointer=src['source_json_pointer'])
                if prox:
                    r=src['row'];native_near=r['relationship_change']=='lost_native';band=r['band_mm']/1000
                    error=r['centroid_radius_m']+r['numerical_allowance_m'];eps=protocol['numerical_classification_tolerance_m']
                    low=row['part_exposed_subset_lower_bound_m'];high=row['part_exposed_subset_upper_bound_m']
                    # Extra numerical widening is only diagnostic conservatism; no historical gate is changed.
                    lo_cell=None if low is None else max(0,low-eps-error)
                    hi_cell=None if high is None else high+eps+error
                    persists=bool((native_near and lo_cell is not None and lo_cell>band) or (not native_near and hi_cell is not None and hi_cell<=band))
                    row.update(source_relationship_change=r['relationship_change'],band_mm=r['band_mm'],cell_radius_m=r['centroid_radius_m'],
                        historical_literal_row_passed=bool(prior['proximity']['passed'][k]),
                        conditional_exposed_subset_cell_lower_bound_m=lo_cell,conditional_exposed_subset_cell_upper_bound_m=hi_cell,
                        conditional_whole_cell_counterexample_persists=persists,
                        unproven_persistence_is_not_a_repair=not persists)
                else:
                    fields=prior['exact_orientation_fields'];ori=prior['exact_orientation']
                    row.update(source_native_facing_compatibility=src['row']['native_facing_compatibility'],source_band_mm=src['band_mm'],
                        historical_literal_row_passed=bool(ori['passed'][k]),historical_orientation_valid=bool(fields['valid'][k]),
                        historical_mapped_facing_compatibility=float(fields['score'][k]),
                        historical_visual_closest_membership=int(fields['visual_closest_membership'][k]),
                        historical_invalid_reasons=[n for n,v in fields['reason_masks'].items() if v[k]],
                        new_orientation_sign_claim=False)
            record={'source_inventory_index':index,'native_sha256':case.native_hash,'object_id':source['object_id'],'source_case':source['source_case'],
                'pose_role':role,'state_scaled':state.tolist(),'root_position_m':case.data.qpos[case.rootadr:case.rootadr+3].tolist(),
                'root_quaternion_wxyz':case.rootquat.tolist(),'qpos_rad':case.data.qpos[case.adr].tolist(),'exact_tip_norms_m':tips.tolist(),
                'exact_mean_tip_m':float(np.mean(tips)),'all_original_tip_caps_pass':True,'original_domain_pass':True,
                'hand_geoms':[{'id':g['id'],'part':g['part'],'kind':g['kind']} for g in hand],
                'rows':result['rows'],'component_point_queries':result['component_point_queries'],'candidate_point_queries':result['candidate_point_queries'],
                'historical_part_distance_max_difference_m':float(np.max(np.abs(measured-np.asarray(prior['part_distance_m'])))),
                'counts':{'occupancy':dict(collections.Counter(r['occupancy_status'] for r in result['rows'])),
                    'part_nearest_exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in result['rows'])),
                    'global_nearest_exposure':dict(collections.Counter(r['global_nearest_component_candidate']['exposure_status'] for r in result['rows'])),
                    'proximity_counterexamples_persist_conditional':sum(r.get('conditional_whole_cell_counterexample_persists',False) for r in result['rows']),
                    'oracle_inconsistencies':sum(not r['oracle_consistent'] for r in result['rows']),
                    'exterior_burial_contradictions':sum(r['exterior_burial_contradiction'] for r in result['rows'])}}
            caseposes.append(record);allrecords.append(record)
        assert numba.get_num_threads()==2 and ce.PHYSICS_CALLS==0
        output={'schema':'hug-exposed-geometry-case-v1','protocol_sha256':sha(HERE/'protocol.json'),'poses':caseposes,
            'actual_mj_forward_calls':forwards[0]-before,'physics_steps':ce.PHYSICS_CALLS,'runtime_seconds':time.perf_counter()-start}
        verify_closure()
        with (HERE/f'case_{index}.json').open('x') as f:f.write(json.dumps(plain(output),indent=2,allow_nan=False)+'\n')
        print(json.dumps({'completed_index':index,'runtime_seconds':output['runtime_seconds'],'counts':[p['counts'] for p in caseposes]}),flush=True)
    out={'schema':'hug-exposed-geometry-cohort-v1','protocol_sha256':sha(HERE/'protocol.json'),'indices':args.indices,
         'case_paths':[str(HERE/f'case_{i}.json') for i in args.indices],'case_sha256':{str(HERE/f'case_{i}.json'):sha(HERE/f'case_{i}.json') for i in args.indices},
         'actual_mj_forward_calls':forwards[0],'runtime_seconds':time.perf_counter()-t0,'physics_steps':ce.PHYSICS_CALLS,'effective_numba_threads':numba.get_num_threads(),
         'new_eval_data':False,'new_inference':False,'new_fit':False,'new_object_grids':False,'production_changed':False}
    verify_closure()
    with cohort_path.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
