"""Development-only deterministic translated-pose diagnostics, never promotion.
Freeze protocol before ranking. All primary returned poses/refusals remain evidence.
"""
from pathlib import Path
import copy,gzip,hashlib,importlib.util,json,pickle,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco
OUT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('independent_free_palm_reviewer',OUT.parent/'verify_outputs.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
sha=v.sha;read=v.read;check=v.check;same=v.same;near=v.near;norm=v.norm;ident=v.ident

def trace_value(entry):
    raw=gzip.decompress(Path(entry['trace_path']).read_bytes())
    check(sha(entry['trace_path'])==entry['trace_compressed_sha256'],'selected trace compressed SHA')
    check(hashlib.sha256(raw).hexdigest()==entry['trace_uncompressed_sha256'],'selected trace raw SHA')
    return json.loads(raw)

def guard_from_trace(row, trace):
    reasons=[];parts={};baseline=trace_value(row['keyed_direction_evidence']['callbacks'][0])
    for part,ref in row['direction_reference_by_part'].items():
        value=trace['normal_diagnostics'][part];old=baseline['normal_diagnostics'][part]
        check(value['available']==ref['available']==old['available'],'availability consistent')
        if not ref['available']:parts[part]={'available':False};continue
        native=np.asarray(ref['native_score']);weights=np.asarray(ref['weights']);s=np.asarray(value['candidate_score']);valid=np.asarray(value['candidate_valid']);bs=np.asarray(old['candidate_score']);bv=np.asarray(old['candidate_valid'])
        same(native,value['native_score'],'unchanged native direction');same(weights,value['weights'],'unchanged fixed direction mass')
        err=np.where(valid,s-native,2.);berr=np.where(bv,bs-native,2.);sse=float(weights@(err*err));bsse=float(weights@(berr*berr));lost=int((bv & ~valid).sum())
        if lost:reasons.append(part+':original_valid_sites_invalidated')
        if sse>bsse:reasons.append(part+':direction_sse_increased')
        parts[part]={'available':True,'candidate_direction_sse':sse,'baseline_direction_sse':bsse,'original_valid_sites_invalidated_count':lost,'fixed_weight_mass':float(weights.sum())}
    return {'accepted':not reasons,'refusal_reasons':reasons,'parts':parts,'additional_guard_not_scientific_acceptance':True,'zero_sse_increase_allowance':True,'invalid_residual':2.}

def choose_pool(pool):
    guarded=[r for r in pool if r['is_returned_nonfallback_guarded_pose']]
    if guarded:return min(guarded,key=lambda r:(r['objective'],r['method_order'],r['callback_sequence'])),'guarded_selected'
    translated=[r for r in pool if r['has_nonzero_world_palm_shift']]
    if translated:return min(translated,key=lambda r:(r['objective'],r['method_order'],r['callback_sequence'])),'tip_feasible_translated_diagnostic_not_selected'
    original=[r for r in pool if r['is_exact_original_state']]
    check(bool(original),'original callback available')
    return min(original,key=lambda r:(r['method_order'],r['callback_sequence'])),'original_fallback_no_feasible_nonzero'

def numerical_pose(row,state,inventory,urdf,xml,source,historical_caps):
    index=row['source_inventory_index'];case=inventory[index]
    with np.load(case['stored_shadow_command']['path'],allow_pickle=False) as z:
        i=case['stored_shadow_command']['row_index'];names=z['joint_names'].tolist();q0=z['nominal_target_rad'][i];root0=z['root_target_position_m'][i];quat0=z['root_target_quaternion_wxyz'][i]
    check(names==row['joint_names'],'source names');same(q0,row['original_qpos_rad'],'source original q');same(root0,row['original_root_target_position_m'],'source originalroot');same(quat0,row['original_root_target_quaternion_wxyz'],'source originalquat')
    native=pickle.loads(Path(case['native']['path']).read_bytes());wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);wm=wc@np.asarray(native['T_camera_wrist']);world=np.asarray(native['landmarks_3d'])@wc[:3,:3].T+wc[:3,3];targets=world[[4,8,12,16,20,2,6,10,14,18]];rop=wm[:3,:3]@v.MANO_TO_OPERATOR_RIGHT.T
    m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
    with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:d.qpos[:]=z['qpos']
    ja=np.asarray([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]);ra=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);oa=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);palm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');obj=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object');wrist=ident(m,mujoco.mjtObj.mjOBJ_BODY,'wrist')
    objectq=d.qpos[oa:oa+7].copy();d.qpos[ja]=q0;d.qpos[ra:ra+3]=root0;d.qpos[ra+3:ra+7]=quat0;d.qvel[:]=0
    if m.nmocap:d.mocap_pos[0]=root0;d.mocap_quat[0]=quat0
    mujoco.mj_forward(m,d);nominal=d.qpos.copy();objectpos=d.xpos[obj].copy();objectrot=d.xmat[obj].copy();original_points=v.task_points(m,d,xml);original_caps=norm(original_points[:5]-targets[:5]);original_wrist=d.xpos[wrist].copy();original_wrist_R=d.xmat[wrist].reshape(3,3).copy()
    same(original_caps,historical_caps,'actual original caps exactly historical');same(original_caps,row['authoritative_original_tip_caps_sqrt_einsum_m'],'actual original caps exactly source method')
    x=np.asarray(state,float);q=q0+.025*x[:24];shift=.005*x[24:];anchor=wm[:3,3]+shift
    bounds=np.asarray(row['solver_audit']['bounded_correction_domain_scaled']);check(x.shape==(27,) and np.isfinite(x).all() and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]),'diagnostic complete27 domain')
    ranges={j.get('name'):[float(j.find('limit').get(k)) for k in ('lower','upper')] for j in xml.findall('joint') if j.find('limit') is not None};limits=np.asarray([ranges[n] for n in names]);check(np.all(q>=limits[:,0]) and np.all(q<=limits[:,1]) and np.all(q>=q0-.35) and np.all(q<=q0+.35),'diagnostic exact physical/jointdelta bounds')
    # Derive carrier placement independently with zero carrier translation.
    d.qpos[ja]=q;d.qpos[ra:ra+3]=0.;mujoco.mj_forward(m,d);independent_root=anchor-d.xpos[palm]
    # Preserve the declared staging arithmetic so strict cap comparisons do
    # not accidentally compare a differently rounded, newly created pose.
    d.qpos[ra:ra+3]=root0;mujoco.mj_forward(m,d);root=root0+(anchor-d.xpos[palm])
    independent_root_error=near(root,independent_root,1e-12,'independent zero-root carrier agreement')
    if np.array_equal(x,np.zeros(27)):root=root0.copy()
    d.qpos[ra:ra+3]=root
    if m.nmocap:d.mocap_pos[0]=root;d.mocap_quat[0]=quat0
    mujoco.mj_forward(m,d);actual=v.task_points(m,d,xml);errors=actual-targets;tipnorm=norm(errors[:5]);check(np.all(tipnorm<=original_caps) and tipnorm.mean()<=.01,'diagnostic actual exact caps/strict mean')
    # A differently ordered but equivalent root subtraction can alter last bits.
    # Audit zero allowances using the actual published final root; do not copy
    # the trace's cap pass when current reconstructed geometry fails it.
    anchorerr=float(np.linalg.norm(d.xpos[palm]-anchor));check(anchorerr<=1e-12,'actual translated anchor')
    qdict=dict(zip(names,q));source.data.qpos[:]=[qdict[n] for n in source.joint_names];source.data.qvel[:]=0;mujoco.mj_forward(source.model,source.data);sp=ident(source.model,mujoco.mjtObj.mjOBJ_BODY,'palm');source_points=v.task_points(source.model,source.data,xml);pred=(source_points-source.data.xpos[sp])@rop.T+anchor;sourceerr=float(np.linalg.norm(actual-pred,axis=1).max());check(sourceerr<1.1e-7,'independent source scene FK')
    palm_R=d.xmat[palm].reshape(3,3).copy();source_R=rop@source.data.xmat[sp].reshape(3,3);roterr=float(np.max(np.abs(palm_R-source_R)));check(roterr<1e-6,'qdependent source palmR')
    same(d.qpos[oa:oa+7],objectq,'unchanged object q');same(d.xpos[obj],objectpos,'unchanged object xyz');same(d.xmat[obj],objectrot,'unchanged objectrotation');same(d.qpos[ra+3:ra+7],quat0,'fixed actual root quaternion');check(d.time==0,'zero time')
    field={'source_inventory_index':index,'native_sha256':row['native_sha256'],'object_id':row['object_id'],'source_case':row['source_case'],'joint_names':names,'qpos_rad':q.tolist(),'original_qpos_rad':q0.tolist(),'selected_state_scaled':x.tolist(),'root_target_position_m':root.tolist(),'root_target_quaternion_wxyz':quat0.tolist(),'original_root_target_position_m':root0.tolist(),'original_root_target_quaternion_wxyz':quat0.tolist(),'palm_displacement_world_m':shift.tolist(),'palm_displacement_norm_m':float(np.linalg.norm(shift)),'target_palm_world_position_m':anchor.tolist(),'target_palm_world_rotation':palm_R.tolist(),'source_expected_palm_world_rotation':source_R.tolist(),'authoritative_original_tip_caps_sqrt_einsum_m':original_caps.tolist(),'selected_tip_norms_sqrt_einsum_m':tipnorm.tolist(),'baseline_fresh_diagnostics':{'task_link_error_norm_m':np.linalg.norm(original_points-targets,axis=1).tolist()},'selected_fresh_diagnostics':{'task_link_error_norm_m':np.linalg.norm(errors,axis=1).tolist(),'task_link_error_vectors_m':errors.tolist()},'original_actual_wrist_body_world_position_m':original_wrist.tolist(),'diagnostic_actual_wrist_body_world_position_m':d.xpos[wrist].tolist(),'actual_wrist_body_displacement_world_m':(d.xpos[wrist]-original_wrist).tolist(),'actual_forearm_carrier_displacement_world_m':(root-root0).tolist(),'actual_wrist_body_world_rotation':d.xmat[wrist].reshape(3,3).tolist(),'original_actual_wrist_body_world_rotation':original_wrist_R.tolist(),'native_wrist_world_position_m':wm[:3,3].tolist(),'actual_wrist_body_minus_native_wrist_world_m':(d.xpos[wrist]-wm[:3,3]).tolist(),'source_FK_preflight':{'status':'PASS','original_caps_exactly_historical':True,'all_five_exact_original_caps_pass':True,'strict10mmmean_pass':True,'anchor_error_m':anchorerr,'independent_zero_root_carrier_maxabs_error_m':independent_root_error,'fixed_source_vs_scene_max_task_error_m':sourceerr,'source_palm_rotation_maxabs':roterr,'zero_simulation_time':True,'objects_unchanged':True,'root_orientation_bitwise_original':True,'not_surface_acceptance':True},'full_scene_qpos_sha256':v.array_sha(d.qpos),'new_eval_data':False,'physics_steps':0,'new_inference':False}
    return field,errors[:5]

def main():
    start=time.perf_counter();p=read(OUT/'protocol.json');ph=sha(OUT/'protocol.json');check(sha(__file__)==p['script_sha256'],'frozen selection script');check(not (OUT/'candidate_targets.json').exists(),'no output overwrite')
    for path,h in p['input_sha256'].items():check(sha(path)==h,'frozen selection input '+path)
    # Record the first rank/measurement start only after complete closure checks.
    with (OUT/'execution_started.json').open('x') as f:json.dump({'protocol_sha256':ph,'all_inputs_verified_before_ranking':True,'new_eval_data':False},f)
    methods=[]
    for order,method in enumerate(p['methods']):
        proto=read(method['protocol_path']);compact=read(method['candidate_path']);check(compact['protocol_sha256']==sha(method['protocol_path']),'method candidate binds protocol')
        rows={}
        for prov in compact['cohort_provenance']:
            raw=gzip.decompress(Path(prov['gzip_path']).read_bytes());check(sha(prov['gzip_path'])==prov['gzip_sha256'] and hashlib.sha256(raw).hexdigest()==prov['sha256'],'method full cohort bindings')
            full=json.loads(raw)
            for row in full['cases']:check(row['source_inventory_index'] not in rows,'one case per method');rows[row['source_inventory_index']]=row
        check(sorted(rows)==p['case_indices'],'six retained cases per method');methods.append({'method':method,'order':order,'protocol':proto,'rows':rows})
    reference=methods[0]
    for method in methods[1:]:
        for key in ('surface_configuration','direction_configuration','direction_guard_policy','state_coordinates'):
            check(method['protocol'][key]==reference['protocol'][key],'cross-method exact comparable '+key)
        check(method['protocol']['solver_configuration']['f_scale']==reference['protocol']['solver_configuration']['f_scale'],'same robust objective scale')
    inventory=read(p['inventory_path'])['modern_cases'];historical={r['source_inventory_index']:r for r in read(p['historical_candidate_path'])['cases']};urdf=Path(p['source_urdf']);xml=ET.parse(urdf).getroot();source=v.ShadowPoseValidator(urdf.parent.parent)
    outputs=[];selection_reports=[];input_check_count=v.COUNT['checks']
    for index in p['case_indices']:
        base=reference['rows'][index];caps=np.asarray(base['authoritative_original_tip_caps_sqrt_einsum_m']);same(caps,historical[index]['old_original_five_tip_caps_sqrt_einsum_m'],'historical caps before ranking')
        pool=[];trace_reviews=[];mapper_results=[];seen={}
        for method in methods:
            row=method['rows'][index]
            for key in ('native_sha256','joint_names','original_qpos_rad','original_root_target_position_m','original_root_target_quaternion_wxyz','authoritative_original_tip_caps_sqrt_einsum_m','surface_reference_by_part','direction_reference_by_part','baseline_fresh_diagnostics','domain'):
                check(row[key]==base[key],'crossmethod exact percase basis '+key)
            trace_reviews.append({'method':method['method']['name'],'review':v.trace_review(row,caps,p)})
            audit=row['solver_audit'];selected=np.asarray(row['selected_state_scaled']);is_selected=not audit['exact_original_fallback'];mapper_results.append({'method':method['method']['name'],'returned_state_scaled':selected.tolist(),'returned_root_position_m':row['root_target_position_m'],'returned_qpos_rad':row['qpos_rad'],'exact_original_fallback':audit['exact_original_fallback'],'surface_candidate_returned':audit['surface_candidate_returned'],'termination':audit['termination'],'this_returned_pose_is_not_rewritten':True})
            bounds=np.asarray(audit['bounded_correction_domain_scaled'])
            for entry in row['keyed_direction_evidence']['callbacks']:
                trace=trace_value(entry);x=np.asarray(trace['state_scaled']);errors=np.asarray(trace['fingertip_error_vectors_m']);resid=np.asarray(trace['residuals']);check(x.shape==(27,) and np.isfinite(x).all() and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]),'rank only full feasible state domain');check(np.all(norm(errors)<=caps) and norm(errors).mean()<=.01,'rank only exact native cap/full feasible')
                scale=method['protocol']['solver_configuration']['f_scale'];objective=float(scale*np.sum(np.hypot(scale,resid)-scale));check(np.isfinite(objective),'finite comparable full objective')
                key=x.tobytes();guard=guard_from_trace(row,trace)
                record={'method_order':method['order'],'method_name':method['method']['name'],'callback_sequence':entry['callback_sequence'],'state_scaled':x.tolist(),'objective':objective,'has_nonzero_world_palm_shift':bool(np.any(.005*x[24:]!=0.)),'is_exact_original_state':bool(np.array_equal(x,np.zeros(27))),'is_returned_nonfallback_guarded_pose':bool(is_selected and np.array_equal(x,selected)),'direction_guard':guard,'source_trace_binding':copy.deepcopy(entry),'source_method_protocol_sha256':sha(method['method']['protocol_path']),'source_method_candidate_sha256':sha(method['method']['candidate_path']),'source_row':row,'trace_tip_errors_m':errors.tolist()}
                if record['is_returned_nonfallback_guarded_pose']:check(guard['accepted'],'returned guarded candidate actually guard accepted')
                if key in seen:
                    previous=seen[key];same(objective,previous['objective'],'identical state comparable objective');check(guard==previous['direction_guard'],'identical state comparable guard');previous['is_returned_nonfallback_guarded_pose']|=record['is_returned_nonfallback_guarded_pose'];continue
                seen[key]=record;pool.append(record)
        chosen,role=choose_pool(pool)
        field,actual_errors=numerical_pose(chosen['source_row'],chosen['state_scaled'],inventory,urdf,xml,source,np.asarray(historical[index]['old_original_five_tip_caps_sqrt_einsum_m']))
        near(actual_errors,chosen['trace_tip_errors_m'],1e-12,'diagnostic source FK reproduces selected trace')
        field.update(audited_pose_role=role,guarded_mapper_selected_fallback=all(r['exact_original_fallback'] for r in mapper_results),direction_guard_pass=chosen['direction_guard']['accepted'],diagnostic_pose_not_promoted=True,extra_direction_guard=chosen['direction_guard'],guarded_mapper_outputs=mapper_results,selection_source={k:chosen[k] for k in ('method_name','method_order','callback_sequence','objective','source_trace_binding','source_method_protocol_sha256','source_method_candidate_sha256')},scientific_mapping_acceptance='NOT_RUN')
        field['selected_fresh_diagnostics']['objective']=chosen['objective'];field['selected_fresh_diagnostics']['objective_from_bound_full_trace']=True;outputs.append(field)
        selected_key=np.asarray(chosen['state_scaled'],float).tobytes()
        selection_reports.append({'source_inventory_index':index,'audit_role':role,'distinct_fully_scored_states':len(pool),'nonzero_translated_states':sum(r['has_nonzero_world_palm_shift'] for r in pool),'returned_guarded_pose_count':sum(r['is_returned_nonfallback_guarded_pose'] for r in pool),'chosen_method':chosen['method_name'],'chosen_callback_sequence':chosen['callback_sequence'],'chosen_objective':chosen['objective'],'chosen_world_displacement_m':field['palm_displacement_world_m'],'chosen_displacement_norm_m':field['palm_displacement_norm_m'],'direction_guard_pass':chosen['direction_guard']['accepted'],'direction_guard_refusal_reasons':chosen['direction_guard']['refusal_reasons'],'trace_reviews':trace_reviews,'all_pool_candidates':[ {k:r[k] for k in ('method_order','method_name','callback_sequence','state_scaled','objective','has_nonzero_world_palm_shift','is_returned_nonfallback_guarded_pose')} for r in pool],'source_FK_preflight':field['source_FK_preflight'],'original_mapper_outputs_preserved':True})
    for path,h in p['input_sha256'].items():check(sha(path)==h,'selection input unchanged '+path)
    check(v.COUNT['physics_actual']==0,'zero actual integration')
    candidate={'schema':'hug_free_palm_diagnostic_audit_targets_v1','protocol_sha256':ph,'cases':outputs,'new_eval_data':False,'new_inference':False,'physics_steps':0,'production_changed':False,'guarded_mapper_outputs_changed':False,'mapping_acceptance':False,'note':'Development diagnostic poses can be refused by the unchanged additional direction guard. All original selected fallbacks/refusals are retained. Independent unchanged scientific surface checks remain required; no promotion.'}
    (OUT/'candidate_targets.json').write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n');report={'schema':'hug_free_palm_diagnostic_selection_review_v1','protocol_sha256':ph,'candidate_sha256':sha(OUT/'candidate_targets.json'),'passed':True,'scope':'Policy-defined development selection, complete trace arithmetic/comparability, independent original/source/action/FK and literal caps. No surface query/rollout, no production promotion.','checks':v.COUNT,'input_count':len(p['input_sha256']),'cases':selection_reports,'runtime_seconds':time.perf_counter()-start,'new_eval_data':False,'physics_steps':0}
    (OUT/'selection_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'cases':len(outputs),'checks':v.COUNT,'candidate':str(OUT/'candidate_targets.json')}))
if __name__=='__main__':main()
