"""Run frozen endpoint projections; only eligible improvements receive surface follow-up."""
from pathlib import Path
import argparse,copy,gzip,json,time
import numpy as np
import constraint_engine as e
from constraint_engine import Case,plain,read,sha
from restoration_core import project_endpoint,surface_gate,tip_status
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_body_id,_link_positions
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
# Frozen helpers change the Numba mask while importing; set and verify AFTER imports.
e.numba.set_num_threads(2)
OUT=Path(__file__).parent
FORWARD_CALLS=0
_original_forward=e.mujoco.mj_forward
def counted_forward(*args,**kwargs):
    global FORWARD_CALLS
    FORWARD_CALLS+=1
    return _original_forward(*args,**kwargs)
e.mujoco.mj_forward=counted_forward

def write(path,value):
    path=Path(path);assert not path.exists();path.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case-index',type=int,nargs='+',required=True);args=ap.parse_args()
    p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');assert p['execution_authorized_after_independent_review'] and args.case_index in p['allowed_cohorts']
    for section in ('input_sha256','implementation_sha256','preflight_sha256'):
        for path,h in p[section].items():assert sha(path)==h,path
    assert e.numba.get_num_threads()==2
    reference={c['source_inventory_index']:c for c in read(p['pooled_candidate_path'])['cases']}
    suffix='_'.join(map(str,args.case_index));records=[];validator=ShadowPoseValidator(e.a.URDF.parent.parent)
    for index in args.case_index:
        started=time.perf_counter();case_start_forward=FORWARD_CALLS;c=Case(index);cp=p['cases'][str(index)];ref=reference[index]
        assert c.native_hash==cp['native_sha256'] and np.array_equal(c.bounds,np.asarray(ref['bounds_scaled'])) and np.array_equal(c.caps,np.asarray(cp['original_tip_caps_m']))
        def fk(x):
            c.stage(x);q,_=e.decode_state(x,c.q0)
            domain=bool(np.all(x>=c.bounds[:,0]) and np.all(x<=c.bounds[:,1]) and np.all(q>=c.limits[:,0]) and np.all(q<=c.limits[:,1]) and np.all(q>=c.q0-.35) and np.all(q<=c.q0+.35))
            return {'error_vectors_m':(_link_positions(c.model,c.data,e.a.URDF)-c.native_tasks)[:5],'in_domain':domain}
        original_fk=fk(np.zeros(27));assert np.array_equal(tip_status(original_fk['error_vectors_m'],c.caps)['norms_m'],c.caps)
        refx=np.asarray(ref['selected_state_scaled']);reference_exact=c.exact(refx)
        assert reference_exact['tip_status']['passed'] and reference_exact['in_domain'] and tuple(reference_exact['rank'])==tuple(cp['reference_exact_rank'])
        assert np.array_equal(reference_exact['exact_orientation']['passed'],cp['reference_orientation_pass']) and np.array_equal(reference_exact['exact_orientation_fields']['valid'],cp['reference_orientation_valid'])
        pv=np.asarray(cp['protected_orientation_valid'],bool);pp=np.asarray(cp['protected_orientation_pass'],bool)
        endpoint_results=[];eligible=[]
        print(json.dumps({'starting_index':index,'frozen_endpoints':len(cp['endpoints']),'numba_threads':e.numba.get_num_threads()}),flush=True)
        for spec in cp['endpoints']:
            proposal=np.asarray(spec['state_scaled']);assert np.array_equal(proposal,np.asarray(spec['source_terminal_record']['terminal_state_scaled']))
            before_forward=FORWARD_CALLS;before_raw=c.raw_calls;before_exact=c.exact_calls
            trace=OUT/'traces'/f'case_{index}_endpoint_{spec["start_index"]}.jsonl.gz';trace.parent.mkdir(exist_ok=True);assert not trace.exists()
            with gzip.open(trace,'wt') as stream:
                def emit(row):stream.write(json.dumps(plain(row),separators=(',',':'),allow_nan=False)+'\n');stream.flush()
                result=project_endpoint(proposal,c.bounds,c.caps,fk,c.tip_jacobian,p['projection_configuration'],emit)
            projection_forward=FORWARD_CALLS-before_forward
            assert c.raw_calls==before_raw and c.exact_calls==before_exact
            assert projection_forward==2*(result['counts']['FK_executions']+result['counts']['analytic_jacobian_executions'])
            result.update(source_endpoint=spec,trace_path=str(trace),trace_sha256=sha(trace),actual_projection_mj_forward_calls=projection_forward)
            full=None;gate=None
            if result['selected'] is not None:
                x=np.asarray(result['selected']['state_scaled']);full=c.exact(x)
                agreement=float(np.max(np.abs(full['task_error_vectors_m'][:5]-np.asarray(result['fresh_FK']['error_vectors_m']))))
                assert agreement<=1e-12
                gate=surface_gate(full,cp['reference_exact_rank'],pv,pp)
                result['full_FK_agreement_maxabs_m']=agreement
                if gate['accepted_for_limited_coarse_audit']:eligible.append((tuple(full['rank']),spec['start_index'],x.copy(),full))
            result['exact_surface_check']=full;result['improvement_gate']=gate
            endpoint_results.append(plain(result))
            write(OUT/f'case_{index}_endpoint_{spec["start_index"]}.json',result)
            print(json.dumps({'completed_endpoint':spec['start_index'],'index':index,'tip_restored':result['selected'] is not None,'gate':plain(gate),'counts':result['counts']}),flush=True)
        chosen=min(eligible,key=lambda row:(row[0],row[1])) if eligible else None
        selected_x=refx.copy() if chosen is None else chosen[2].copy()
        final=c.exact(selected_x);fresh_gate=None;fresh_refusal=None
        if chosen is not None:
            fresh_gate=surface_gate(final,cp['reference_exact_rank'],pv,pp)
            if not fresh_gate['accepted_for_limited_coarse_audit'] or tuple(final['rank'])!=chosen[0]:
                fresh_refusal={'chosen_endpoint':chosen[1],'fresh_gate':fresh_gate,'fresh_rank':final['rank']};chosen=None;selected_x=refx.copy();final=c.exact(selected_x)
        if chosen is None:assert final['tip_status']['passed'] and final['in_domain'] and tuple(final['rank'])==tuple(cp['reference_exact_rank'])
        q,shift=e.decode_state(selected_x,c.q0)
        validator.data.qpos[:]=reorder_for_mujoco(c.names,q,validator.joint_names);validator.data.qvel[:]=0;e.mujoco.mj_forward(validator.model,validator.data)
        fp=_body_id(validator.model,'palm');Rop=c.T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T;source_R=Rop@validator.data.xmat[fp].reshape(3,3)
        source_links=(_link_positions(validator.model,validator.data,e.a.URDF)-validator.data.xpos[fp])@Rop.T+c.anchor+shift
        scene_links=_link_positions(c.model,c.data,e.a.URDF);fk_error=float(np.linalg.norm(scene_links-source_links,axis=1).max());rotation_error=float(np.max(np.abs(source_R-c.data.xmat[c.palm].reshape(3,3))))
        assert fk_error<1.1e-7 and rotation_error<1e-6
        assert np.array_equal(c.data.qpos[c.rootadr+3:c.rootadr+7],c.rootquat) and c.data.time==0
        assert final['in_domain'] and final['tip_status']['passed'] and c.exact_calls<=5
        record=copy.deepcopy(ref)
        for k in ('legacy_direction_guard_diagnostic_only','solver_audit','selection_provenance','pooling_provenance','selection_source'):record.pop(k,None)
        record.update(qpos_rad=q,selected_state_scaled=selected_x,root_target_position_m=final['root_target_position_m'],root_target_quaternion_wxyz=c.rootquat,
            palm_displacement_world_m=shift,palm_displacement_norm_m=float(np.linalg.norm(shift)),target_palm_world_position_m=final['target_palm_world_position_m'],target_palm_world_rotation=final['target_palm_world_rotation'],source_expected_palm_world_rotation=source_R,
            selected_shadow_wrist_world_position_m=final['wrist_world_position_m'],selected_shadow_wrist_world_rotation=final['wrist_world_rotation'],shadow_wrist_world_displacement_m=final['wrist_world_position_m']-c.original_wrist_pos,floating_forearm_world_displacement_m=final['root_target_position_m']-c.rootpos,
            selected_tip_norms_sqrt_einsum_m=final['tip_status']['norms_m'],selected_fresh_diagnostics={'task_link_error_norm_m':np.linalg.norm(final['task_error_vectors_m'],axis=1)},selected_constraint_diagnostics=final,
            reference_constraint_diagnostics=reference_exact,reference_state_scaled=refx,reference_selection_source=ref.get('selection_source'),selection_source={'method_protocol_sha256':ph,'endpoint_start_index':None if chosen is None else chosen[1],'reference_retained':chosen is None},reference_candidate_binding={'path':p['pooled_candidate_path'],'sha256':sha(p['pooled_candidate_path'])},
            restoration_endpoint_results=endpoint_results,restoration_selected_endpoint=None if chosen is None else chosen[1],fresh_improvement_gate=fresh_gate,fresh_improvement_refusal=fresh_refusal,
            eligible_for_limited_coarse_audit=chosen is not None,audited_pose_role='restored_tip_feasible_surface_improvement' if chosen is not None else 'pooled_reference_retained',constraint_selection_accepted=bool(final['all_witness_constraints_pass']),
            rejection_reason=None if chosen is not None else 'no_restored_projection_improves_reference_rank_without_new_retained_orientation_failure',diagnostic_pose_not_promoted=True,original_fallback=bool(np.array_equal(selected_x,np.zeros(27))),
            source_FK_preflight={'passed':True,'source_scene_task_max_error_m':fk_error,'source_palm_rotation_maxabs_error':rotation_error,'all_five_exact_original_caps_pass':True,'mean_10mm_gate_pass':True,'object_and_root_rotation_unchanged':True},
            runtime_seconds=time.perf_counter()-started,actual_case_mj_forward_calls=FORWARD_CALLS-case_start_forward,exact_surface_checks=c.exact_calls,numba_threads_after_imports=e.numba.get_num_threads(),full_grid_acceptance='NOT_TESTED_BY_RESTORATION',complete_mapping_acceptance=False,new_eval_data=False,new_inference=False,physics_steps=0)
        records.append(plain(record));write(OUT/f'case_{index}_candidate.json',record)
        print(json.dumps({'completed_index':index,'coarse_audit_eligible':chosen is not None,'reference_rank':cp['reference_exact_rank'],'selected_rank':final['rank'],'exact_surface_checks':c.exact_calls,'seconds':record['runtime_seconds']}),flush=True)
    for section in ('input_sha256','implementation_sha256','preflight_sha256'):
        for path,h in p[section].items():assert sha(path)==h,path
    assert e.PHYSICS_CALLS==0 and sha(OUT/'protocol.json')==ph and e.numba.get_num_threads()==2
    write(OUT/f'candidate_cohort_{suffix}.json',{'schema':p['candidate_schema'],'protocol_sha256':ph,'cases':records,'cohort':args.case_index,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'mapping_acceptance':False})
if __name__=='__main__':main()
