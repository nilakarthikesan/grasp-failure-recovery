"""Independent saved-output arithmetic/source replay; no runtime geometry calls."""
from pathlib import Path
import collections, hashlib, json
import numpy as np

HERE=Path(__file__).resolve().parent
FIT=Path('/private/tmp/hug_exposed_geometry_2026-10-15_v1')
WIT=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(c,label):
    assert bool(c),label
    checks.append(label)
def close(x,y):
    if x is None or y is None:return x is None and y is None
    return abs(float(x)-float(y))<=1e-14
P=FIT/'protocol.json';p=read(P);digest=sha(P);eps=p['numerical_classification_tolerance_m']
draft=read(FIT/'protocol.draft.json');pre=read(HERE/'diagnostic_preflight_review.json')
check(sha(FIT/'protocol.draft.json')==pre['protocol_draft_sha256'],'reviewed_draft_binding')
for key,value in draft.items():
    if key not in ('execution_authorized_after_independent_review','preflight_sha256'):
        check(p[key]==value,'frozen_field:'+key)
check(p['execution_authorized_after_independent_review'],'frozen_authorization')
bindings={str(P):digest,str(HERE/'diagnostic_preflight_review.json'):sha(HERE/'diagnostic_preflight_review.json')}
for group in ('input_sha256','implementation_sha256','preflight_sha256'):
    for path,h in p[group].items():check(sha(path)==h,'hash:'+path);bindings[path]=h
pool={c['source_inventory_index']:c for c in read(p['pooled_candidate_path'])['cases']};w=read(WIT)
summaries=[];totalrows=totalqueries=totalforwards=0
for index in p['cases']:
    path=FIT/f'case_{index}.json';bindings[str(path)]=sha(path);case=read(path);source=pool[index]
    check(case['protocol_sha256']==digest and case['physics_steps']==0,f'{index}:protocol_no_steps')
    check([r['pose_role'] for r in case['poses']]==['original','pooled'],f'{index}:two_fixed_poses')
    totalforwards+=case['actual_mj_forward_calls']
    for pose in case['poses']:
        role=pose['pose_role'];prefix=f'{index}:{role}:';state=np.asarray(pose['state_scaled']);declared=p['source_rows'][str(index)]
        check(np.array_equal(state,declared[role+'_state_scaled']),prefix+'state')
        prior=source['baseline_constraint_diagnostics' if role=='original' else 'selected_constraint_diagnostics']
        check(pose['native_sha256']==source['native_sha256'],prefix+'native')
        check(np.all(np.asarray(pose['exact_tip_norms_m'])<=np.asarray(source['authoritative_original_tip_caps_sqrt_einsum_m'])),prefix+'exact_old_caps')
        check(np.mean(pose['exact_tip_norms_m'])<=.010 and pose['exact_mean_tip_m']<=.010,prefix+'mean_gate')
        check(np.array_equal(pose['root_quaternion_wxyz'],source['original_root_target_quaternion_wxyz']),prefix+'original_quaternion')
        check(np.max(np.abs(np.asarray(pose['root_position_m'])-prior['root_target_position_m']))<1e-12,prefix+'saved_fk_root_replay')
        bounds=np.asarray(source['bounds_scaled']);check(np.all(state>=bounds[:,0]) and np.all(state<=bounds[:,1]),prefix+'scaled_domain')
        check(np.max(np.abs(np.asarray(pose['qpos_rad'])-(np.asarray(source['original_qpos_rad'])+.025*state[:24])))<1e-12,prefix+'q_decode')
        check(pose['historical_part_distance_max_difference_m']<1e-12,prefix+'historical_part_distance_parity')
        geoms=pose['hand_geoms'];ids=[g['id'] for g in geoms];parts=[g['part'] for g in geoms];byid={g['id']:g for g in geoms}
        check(len(geoms)==45 and len(byid)==45,prefix+'45_distinct_geoms')
        N=len(pose['rows']);totalrows+=N;totalqueries+=pose['component_point_queries']+pose['candidate_point_queries']
        check(pose['component_point_queries']==45*N and pose['candidate_point_queries']==90*N,prefix+'query_counts')
        nprox=len(declared['proximity_indices']);nori=len(declared['orientation_indices']);check(N==nprox+nori,prefix+'row_count')
        for j,r in enumerate(pose['rows']):
            rp=prefix+str(j)+':'
            prox=j<nprox;k=j if prox else j-nprox;kind='whole_cell_rows' if prox else 'orientation_rows'
            ix=declared['proximity_indices' if prox else 'orientation_indices'][k];src=w[kind][ix]
            check(r['saved_witness_json_pointer']==f'/{kind}/{ix}',rp+'source_pointer')
            query=src['row']['centroid_object_m' if prox else 'query_object_m']
            part=src['row']['anatomical_part' if prox else 'native_part']
            check(np.array_equal(r['query_object_m'],query) and r['expected_part']==part,rp+'query_part')
            check(r['source_report_sha256']==src['source_report_sha256'] and r['source_json_pointer']==src['source_json_pointer'],rp+'source_provenance')
            d=np.asarray(r['signed_component_distances_m']);ad=np.abs(d)
            check(d.shape==(45,) and np.isfinite(d).all() and r['component_geom_ids']==ids,rp+'component_values')
            inside=np.flatnonzero(d < -eps);outside=bool(np.all(d>eps))
            occ='STRICT_COMPONENT_INTERIOR' if len(inside) else 'STRICT_UNION_EXTERIOR' if outside else 'BOUNDARY_OR_COINCIDENT_UNRESOLVED'
            check(r['occupancy_status']==occ and r['strict_containing_geom_ids']==[ids[z] for z in inside],rp+'occupancy')
            gi=int(np.argmin(ad));pi=[z for z,v in enumerate(parts) if v==part];mi=pi[int(np.argmin(ad[pi]))]
            check(r['global_individual_surface_distance_m']==ad[gi] and r['part_individual_surface_distance_m']==ad[mi],rp+'min_component_distances')
            gc=r['global_nearest_component_candidate'];pc=r['part_nearest_component_candidate']
            for cand,ci in ((gc,gi),(pc,mi)):
                check(cand['component_index']==ci and cand['geom_id']==ids[ci],rp+'nearest_owner')
                dist=float(np.linalg.norm(np.asarray(cand['point_object_m'])-query))
                check(close(dist,cand['distance_from_query_m']),rp+'candidate_point_distance')
                if cand['sdf_closest_distance_residual_m']>eps:status='ORACLE_DISTANCE_INCONSISTENT'
                elif cand['own_surface_residual_m']>eps:status='OWN_SURFACE_RESIDUAL_UNRESOLVED'
                elif cand['strict_burying_geom_ids']:status='STRICTLY_BURIED_IN_OTHER_COMPONENT'
                elif cand['minimum_other_signed_distance_m'] is None or cand['minimum_other_signed_distance_m']>eps:status='STRICTLY_SEPARATED_EXPOSED_POINT'
                else:status='COINCIDENT_OR_NEAR_BOUNDARY_UNRESOLVED'
                check(cand['exposure_status']==status,rp+'exposure_status')
                check(cand['strict_burying_parts']==sorted({byid[i]['part'] for i in cand['strict_burying_geom_ids']}),rp+'burial_part_labels')
                check(not cand['normal_uniqueness_verified'],rp+'no_normal_claim')
            contradiction=outside and gc['exposure_status']=='STRICTLY_BURIED_IN_OTHER_COMPONENT'
            consistent=(r['maximum_sdf_closest_distance_residual_m']<=eps and gc['own_surface_residual_m']<=eps and pc['own_surface_residual_m']<=eps and not contradiction)
            check(r['oracle_consistent']==consistent and r['exterior_burial_contradiction']==contradiction,rp+'oracle_gate')
            exact=outside and consistent and not contradiction
            wl=float(ad[gi]) if exact else float(np.maximum(-d,0).max()) if len(inside) else 0.
            exposed=[c for c in (gc,pc) if c['exposure_status']=='STRICTLY_SEPARATED_EXPOSED_POINT']
            wu=float(ad[gi]) if exact else min([c['distance_from_query_m'] for c in exposed],default=None)
            pu=min([c['distance_from_query_m'] for c in exposed if c['part']==part],default=None)
            pe=pc['exposure_status']=='STRICTLY_SEPARATED_EXPOSED_POINT' and consistent
            if pe:pu=float(ad[mi])
            pl=float(ad[mi])
            if not consistent:wl=wu=pl=pu=None
            check(close(wl,r['whole_union_boundary_lower_bound_m']) and close(wu,r['whole_union_boundary_upper_bound_m']),rp+'whole_bounds')
            check(close(pl,r['part_exposed_subset_lower_bound_m']) and close(pu,r['part_exposed_subset_upper_bound_m']),rp+'part_bounds')
            check(r['whole_union_distance_exact_conditional']==exact and r['part_exposed_subset_distance_exact_conditional']==pe,rp+'conditional_exact_flags')
            check(not r['interval_arithmetic_certificate'] and not r['normal_uniqueness_verified'],rp+'claim_limits')
            if prox:
                sr=src['row'];band=sr['band_mm']/1000;error=sr['centroid_radius_m']+sr['numerical_allowance_m'];near=sr['relationship_change']=='lost_native'
                nd=sr['native_centroid_distance_m']
                check((nd+error<=band) if near else (max(0,nd-error)>band),rp+'native_whole_cell_relation')
                lo=None if pl is None else max(0,pl-eps-error);hi=None if pu is None else pu+eps+error
                persists=(near and lo is not None and lo>band) or (not near and hi is not None and hi<=band)
                check(close(lo,r['conditional_exposed_subset_cell_lower_bound_m']) and close(hi,r['conditional_exposed_subset_cell_upper_bound_m']),rp+'cell_bounds')
                check(r['conditional_whole_cell_counterexample_persists']==persists and r['unproven_persistence_is_not_a_repair']==(not persists),rp+'persistence')
                check(r['historical_literal_row_passed']==prior['proximity']['passed'][k],rp+'historical_prox_unchanged')
            else:
                check(r['historical_literal_row_passed']==prior['exact_orientation']['passed'][k] and r['historical_orientation_valid']==prior['exact_orientation_fields']['valid'][k],rp+'historical_orientation_unchanged')
                check(r['historical_mapped_facing_compatibility']==prior['exact_orientation_fields']['score'][k] and not r['new_orientation_sign_claim'],rp+'no_new_orientation_claim')
        counts={'occupancy':dict(collections.Counter(r['occupancy_status'] for r in pose['rows'])),
            'part_nearest_exposure':dict(collections.Counter(r['part_nearest_component_candidate']['exposure_status'] for r in pose['rows'])),
            'global_nearest_exposure':dict(collections.Counter(r['global_nearest_component_candidate']['exposure_status'] for r in pose['rows'])),
            'proximity_counterexamples_persist_conditional':sum(r.get('conditional_whole_cell_counterexample_persists',False) for r in pose['rows']),
            'oracle_inconsistencies':sum(not r['oracle_consistent'] for r in pose['rows']),
            'exterior_burial_contradictions':sum(r['exterior_burial_contradiction'] for r in pose['rows'])}
        check(counts==pose['counts'],prefix+'counts')
        summaries.append({'source_inventory_index':index,'pose_role':role,'proximity_witnesses':nprox,'orientation_witnesses':nori,
            'conditional_persistence_by_type':dict(collections.Counter(r['source_relationship_change'] for r in pose['rows'] if r.get('conditional_whole_cell_counterexample_persists'))),
            'counts':counts})
cohortpath=FIT/'cohort_8_12_16_20_24_28.json';c=read(cohortpath);bindings[str(cohortpath)]=sha(cohortpath)
check(c['actual_mj_forward_calls']==totalforwards and c['physics_steps']==0 and c['effective_numba_threads']==2,'execution_totals')
check(totalrows==624 and totalqueries==84240,'finite_output_counts')
for path,h in c['case_sha256'].items():check(sha(path)==h,'cohort_case_hash:'+path)
for path,h in bindings.items():check(sha(path)==h,'lifecycle_after:'+path)
out={'schema':'independent_exposed_geometry_output_review_v1','passed':True,'checks_count':len(checks),'checks':checks,
     'source_bindings':bindings,'per_pose':summaries,'row_pose_records':totalrows,'component_point_evaluations':totalqueries,
     'driver_static_forward_calls':totalforwards,'reviewer_new_geometry_queries':0,'reviewer_new_fk_calls':0,'physics_steps':0,'new_eval_data':False,
     'scope':'Independent source, saved-value arithmetic and lifecycle replay. Component closest-point oracles are not recomputed here; compiled contract and manufactured controls are separate prerequisites. Historical literal criteria unchanged.'}
dest=HERE/'diagnostic_output_review.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'passed':True,'checks_count':len(checks),'path':str(dest),'sha256':sha(dest),'per_pose':summaries},indent=2))
