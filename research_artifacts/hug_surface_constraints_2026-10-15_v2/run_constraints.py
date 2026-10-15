"""Execute exactly the frozen six-case static witness-constrained study."""
from pathlib import Path
import argparse,gzip,json,time
import numpy as np
import constraint_engine as e
from constraint_engine import Case,read,sha,plain
from constraint_solver import solve
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator,_body_id,_link_positions
from grasp_failure_prediction.evaluation.retargeting import reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

OUT=Path(__file__).parent
def write(path,obj):Path(path).write_text(json.dumps(plain(obj),indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case-index',type=int,nargs='+',required=True);args=ap.parse_args()
    protocol=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json')
    assert protocol['execution_authorized_after_independent_review']
    assert args.case_index in protocol['allowed_cohorts']
    for path,h in protocol['input_sha256'].items():assert sha(path)==h,path
    for path,h in protocol['implementation_sha256'].items():assert sha(path)==h,path
    for path,h in protocol['preflight_sha256'].items():assert sha(path)==h,path
    source_control=read(protocol['source_preflight_path'])
    assert source_control['engine_sha256']==sha(OUT/'constraint_engine.py') and source_control['core_sha256']==sha(OUT/'constraint_core.py')
    assert sorted(r['index'] for r in source_control['records'])==[8,12,16,20,24,28]
    assert all(r['jacobian']['passed'] and r['baseline_exact']['in_domain'] and r['baseline_exact']['tip_status']['passed'] for r in source_control['records'])
    suffix='_'.join(map(str,args.case_index));target=OUT/f'candidate_cohort_{suffix}.json'
    assert not target.exists()
    records=[];start_time=time.perf_counter()
    validator=ShadowPoseValidator(e.a.URDF.parent.parent)
    for index in args.case_index:
        tick=time.perf_counter();case=Case(index)
        assert case.native_hash==protocol['cases'][str(index)]['native_sha256']
        starts=protocol['cases'][str(index)]['frozen_starts']
        print(json.dumps({'starting_index':index,'proximity_rows':len(case.prox_rows),'orientation_rows':len(case.ori_rows)}),flush=True)
        x,final,audit=solve(case,starts,protocol['solver_configuration'],OUT/'traces')
        q,shift=e.decode_state(x,case.q0)
        legacy=case.legacy_guard_diagnostic(x)
        # Independent fixed-source URDF FK vs staged compiled scene.
        validator.data.qpos[:]=reorder_for_mujoco(case.names,q,validator.joint_names);validator.data.qvel[:]=0
        e.mujoco.mj_forward(validator.model,validator.data);fp=_body_id(validator.model,'palm')
        Rop=case.T_wrist[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        source_R=Rop@validator.data.xmat[fp].reshape(3,3)
        source_links=(_link_positions(validator.model,validator.data,e.a.URDF)-validator.data.xpos[fp])@Rop.T+case.anchor+shift
        scene_links=_link_positions(case.model,case.data,e.a.URDF)
        fk_error=float(np.linalg.norm(scene_links-source_links,axis=1).max())
        rotation_error=float(np.max(np.abs(source_R-case.data.xmat[case.palm].reshape(3,3))))
        assert fk_error<1.1e-7 and rotation_error<1e-6
        tip=np.sqrt(np.einsum('ij,ij->i',(scene_links-case.native_tasks)[:5],(scene_links-case.native_tasks)[:5]))
        assert np.array_equal(tip,final['tip_status']['norms_m']) and np.all(tip<=case.caps) and tip.mean()<=.010
        assert np.array_equal(case.data.qpos[case.rootadr+3:case.rootadr+7],case.rootquat) and case.data.time==0
        fallback=bool(np.array_equal(x,np.zeros(27)))
        accepted=bool(final['all_witness_constraints_pass'])
        role='original_fallback' if fallback else ('surface_constraint_selected' if accepted else 'tip_feasible_constraint_refused_diagnostic')
        baseline=audit['baseline_exact']
        record={'source_inventory_index':index,'native_sha256':case.native_hash,'object_id':case.source['object_id'],
            'source_case':Path(case.source['native']['path']).parent.name,'joint_names':case.names,'qpos_rad':q,'original_qpos_rad':case.q0,
            'selected_state_scaled':x,'original_state_scaled':np.zeros(27),'bounds_scaled':case.bounds,'domain':case.domain,
            'root_target_position_m':final['root_target_position_m'],'root_target_quaternion_wxyz':case.rootquat,
            'original_root_target_position_m':case.rootpos,'original_root_target_quaternion_wxyz':case.rootquat,
            'palm_displacement_world_m':shift,'palm_displacement_norm_m':float(np.linalg.norm(shift)),
            'target_palm_world_position_m':final['target_palm_world_position_m'],'target_palm_world_rotation':final['target_palm_world_rotation'],
            'source_expected_palm_world_rotation':source_R,
            'original_shadow_wrist_world_position_m':case.original_wrist_pos,'selected_shadow_wrist_world_position_m':final['wrist_world_position_m'],
            'original_shadow_wrist_world_rotation':case.original_wrist_R,'selected_shadow_wrist_world_rotation':final['wrist_world_rotation'],
            'shadow_wrist_world_displacement_m':final['wrist_world_position_m']-case.original_wrist_pos,
            'floating_forearm_world_displacement_m':final['root_target_position_m']-case.rootpos,
            'authoritative_original_tip_caps_sqrt_einsum_m':case.caps,'selected_tip_norms_sqrt_einsum_m':tip,'prior_original_cap_binding':case.cap_binding,
            'baseline_fresh_diagnostics':{'task_link_error_norm_m':np.linalg.norm(baseline['task_error_vectors_m'],axis=1)},
            'selected_fresh_diagnostics':{'task_link_error_norm_m':np.linalg.norm(final['task_error_vectors_m'],axis=1)},
            'baseline_constraint_diagnostics':baseline,'selected_constraint_diagnostics':final,'solver_audit':audit,
            'constraint_source_rows':{'proximity':case.prox_rows,'orientation':case.ori_rows},
            'legacy_direction_guard_diagnostic_only':legacy,'legacy_guard_used_for_selection':False,
            'audited_pose_role':role,'constraint_selection_accepted':accepted,'diagnostic_pose_not_promoted':True,
            'rejection_reason':None if accepted else 'one_or_more_fixed_witness_constraints_or_exact_orientation_eligibility_conditions_remain_violated',
            'original_fallback':fallback,'all_30_original_caps_scope_is_cohort_not_single_case':True,
            'source_FK_preflight':{'passed':True,'source_scene_task_max_error_m':fk_error,'source_palm_rotation_maxabs_error':rotation_error,
                'all_five_exact_original_caps_pass':True,'mean_10mm_gate_pass':True,'object_and_root_rotation_unchanged':True},
            'orientation_witness_coverage':final['orientation_witness_coverage'],'full_grid_acceptance':'NOT_YET_TESTED',
            'complete_mapping_acceptance':False,'runtime_seconds':time.perf_counter()-tick,'new_eval_data':False,'new_inference':False,'physics_steps':0}
        records.append(plain(record));write(OUT/f'candidate_cohort_{suffix}.partial.json',{'schema':protocol['candidate_schema'],'protocol_sha256':ph,'partial':True,'cases':records})
        print(json.dumps({'completed_index':index,'role':role,'palm_shift_mm':float(np.linalg.norm(shift)*1000),
            'baseline_rank':baseline['rank'],'selected_rank':final['rank'],'counts':audit['counts'],'seconds':record['runtime_seconds']}),flush=True)
    for path,h in protocol['input_sha256'].items():assert sha(path)==h,path
    for path,h in protocol['implementation_sha256'].items():assert sha(path)==h,path
    assert e.PHYSICS_CALLS==0 and sha(OUT/'protocol.json')==ph
    result={'schema':protocol['candidate_schema'],'protocol_sha256':ph,'cases':records,'cohort':args.case_index,
        'runtime_seconds':time.perf_counter()-start_time,'new_eval_data':False,'new_inference':False,'physics_steps':0,
        'production_changed':False,'mapping_acceptance':False,'unchanged_source_hashes_verified':True}
    write(target,result)
    print(json.dumps({'completed_cohort':args.case_index,'path':str(target)}),flush=True)
if __name__=='__main__':main()
