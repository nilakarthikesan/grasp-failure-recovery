"""Independent stored-input FK, source bounds, exact caps, trace and guard audit.
No primary pose/solver/helper call, surface query, model inference or integration.
"""
from pathlib import Path
import gzip,hashlib,json,pickle,time,xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT
OUT=Path(__file__).parent
COUNT={'checks':0,'physics_actual':0,'physics_selftests':0};TEST=False

def forbid(*args,**kwargs):
    COUNT['physics_selftests' if TEST else 'physics_actual']+=1
    raise RuntimeError('Independent free-palm review forbids integration')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbid)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def check(v,label):
    COUNT['checks']+=1
    if not bool(v):raise AssertionError(label)
def same(a,b,label):check(np.array_equal(np.asarray(a),np.asarray(b)),label)
def near(a,b,eps,label):
    a=np.asarray(a,float);b=np.asarray(b,float)
    check(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),label+' shape/finite')
    error=float(np.max(np.abs(a-b)));check(error<=eps,(label,error,eps));return error
def norm(x):return np.sqrt(np.einsum('ij,ij->i',x,x))
def array_sha(x):return hashlib.sha256(np.asarray(x,dtype=np.float64).tobytes()).hexdigest()
def ident(m,kind,name):
    n=mujoco.mj_name2id(m,kind,name);check(n>=0,'name exists '+name);return n

def task_points(m,d,xml):
    tips=('thtip','fftip','mftip','rftip','lftip');result=[]
    for name in tips+('thmiddle','ffmiddle','mfmiddle','rfmiddle','lfmiddle'):
        if name in tips:
            j=next(j for j in xml.findall('joint') if j.find('child').get('link')==name)
            check(j.get('type')=='fixed','virtual tip fixed '+name)
            o=j.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(o.get('rpy','0 0 0'),sep=' ')
            same(rpy,np.zeros(3),'virtual tip frame unrotated '+name)
            body=ident(m,mujoco.mjtObj.mjOBJ_BODY,j.find('parent').get('link'))
            result.append(d.xpos[body]+d.xmat[body].reshape(3,3)@xyz)
        else:result.append(d.xpos[ident(m,mujoco.mjtObj.mjOBJ_BODY,name)].copy())
    return np.asarray(result)

def independent_reach(xml):
    childmap={j.find('child').get('link'):j for j in xml.findall('joint')};r=[]
    for tip in ('thtip','fftip','mftip','rftip','lftip'):
        node=tip;total=0.;seen=set()
        while node!='palm':
            check(node not in seen and node in childmap,'acyclic palm-to-tip chain');seen.add(node);j=childmap[node]
            check(j.get('type') in ('fixed','revolute','continuous'),'rigid/rotary reach path')
            o=j.find('origin');xyz=np.fromstring(o.get('xyz','0 0 0'),sep=' ') if o is not None else np.zeros(3)
            total+=float(np.sqrt(xyz@xyz));node=j.find('parent').get('link')
        r.append(total)
    return np.asarray(r)

def trace_review(target, caps, p):
    e=target['keyed_direction_evidence'];sa=target['solver_audit'];callbacks=e['callbacks'];trace=[];byseq={};bykey={};baseline=None
    check(len(callbacks)==e['callback_count']==target['full_callback_count_including_fresh_final']==sa['counters']['full'],'full callback counts')
    for i,row in enumerate(callbacks,1):
        check(row['callback_sequence']==i,'callback ordering')
        raw=gzip.decompress(Path(row['trace_path']).read_bytes())
        check(sha(row['trace_path'])==row['trace_compressed_sha256'],'compressed trace hash')
        check(hashlib.sha256(raw).hexdigest()==row['trace_uncompressed_sha256'],'uncompressed trace hash')
        c=json.loads(raw);x=np.asarray(c['state_scaled'],float);r=np.asarray(c['residuals'],float);tips=np.asarray(c['fingertip_error_vectors_m'],float)
        same(x,row['state_scaled'],'trace index full27 vector');check(x.shape==(27,) and np.isfinite(x).all(),'finite27 trace')
        check(array_sha(x)==c['q_float64_sha256'] and array_sha(r)==c['residuals_float64_sha256'] and array_sha(tips)==c['fingertips_float64_sha256'],'complete trace array checksums')
        check(np.all(norm(tips)<=caps) and norm(tips).mean()<=.01,'all full callbacks exact old tip caps')
        if baseline is None:baseline=c;same(x,np.zeros(27),'first callback original')
        key=x.tobytes();check(c['repeat_exact_q_callback']==(key in bykey),'repeat27 key status')
        if key in bykey:
            old=bykey[key];same(r,old['residuals'],'repeat residuals exact');same(tips,old['fingertip_error_vectors_m'],'repeat tips exact');check(c['normal_diagnostics']==old['normal_diagnostics'],'repeat fixed direction evidence')
        bykey[key]=c;byseq[i]=c
        reason=[];part_report={};normal_blocks=[]
        check(set(c['normal_diagnostics'])==set(target['direction_reference_by_part']),'fixed normal part set')
        for part,ref in target['direction_reference_by_part'].items():
            n=c['normal_diagnostics'][part];base=baseline['normal_diagnostics'][part]
            check(n['available']==ref['available'],'normal availability fixed')
            if not ref['available']:continue
            native=np.asarray(ref['native_score']);weights=np.asarray(ref['weights']);score=np.asarray(n['candidate_score']);valid=np.asarray(n['candidate_valid']);bn=np.asarray(base['candidate_score']);bv=np.asarray(base['candidate_valid'])
            same(n['native_score'],native,'native direction targets fixed');same(n['weights'],weights,'normal masses fixed')
            check(valid.dtype.kind=='b' and bv.dtype.kind=='b','boolean normal validity')
            resid=np.where(valid,score-native,2.);bres=np.where(bv,bn-native,2.)
            sse=float(np.dot(weights,resid*resid));bsse=float(np.dot(weights,bres*bres))
            # Arithmetic order exactly follows declared residual normalization.
            nr=np.sqrt(weights*6)*resid/np.sqrt(6);near(nr,n['residual'],1e-14,'fixed complete direction residual')
            normal_blocks.append(np.asarray(n['residual']))
            invalid=bv & ~valid
            if invalid.any():reason.append(part+':original_valid_sites_invalidated')
            if sse>bsse:reason.append(part+':direction_sse_increased')
            part_report[part]=(sse,bsse,int(invalid.sum()))
        start=e['normal_block_start'];count=e['normal_block_count'];same(r[start:start+count],np.concatenate(normal_blocks),'normal evidence binds full residual facing block')
        c['_independent_refusal_reasons']=reason;c['_parts']=part_report
        trace.append({'callback_sequence':i,'native_negative_direction_preserved':True,'full27_sha256':array_sha(x),'is_exact_original_state':bool(np.array_equal(x,np.zeros(27))),'guard_accepted':not reason})
    for assessment in e['guard_assessments']:
        seq=assessment['metadata_callback_sequence'];c=byseq[seq]
        check(assessment['q_float64_sha256']==c['q_float64_sha256'] and assessment['residuals_float64_sha256']==c['residuals_float64_sha256'] and assessment['fingertips_float64_sha256']==c['fingertips_float64_sha256'],'guard exact callback binding')
        check(assessment['accepted']==(not c['_independent_refusal_reasons']),'independent direction guard decision')
        check(assessment['refusal_reasons']==c['_independent_refusal_reasons'],'independent direction refusal reasons')
        for part,(sse,base,invalid) in c['_parts'].items():
            a=assessment['parts'][part];near(a['candidate_direction_sse'],sse,0.,'guard candidateSSE exact');near(a['baseline_direction_sse'],base,0.,'guard baselineSSE exact');check(a['original_valid_sites_invalidated_count']==invalid,'guard invalid counts')
    final=byseq[len(callbacks)];same(final['state_scaled'],target['selected_state_scaled'],'last fresh full callback final state')
    check(target['terminal_direction_guard']['metadata_callback_sequence']==len(callbacks) and target['terminal_direction_guard']['accepted'],'terminal guard fresh and accepted')
    check(e['distinct_exact_q_count']==len(bykey),'distinct27 key count')
    proposal_sweeps=sorted({r['sweep'] for r in sa['proposals']});actual_radii=sorted({r['radius_scaled'] for r in sa['proposals'] if r['radius_scaled'] is not None},reverse=True)
    return {'observed_proposal_sweeps':proposal_sweeps,'observed_pattern_radii_scaled':actual_radii,'seed_proposals_observed':sum(r['sweep']==0 for r in sa['proposals']),'completed_fullscore_proposals':sum('full_assessment' in r for r in sa['proposals']),'guard_refused_proposals':sum(r.get('refusal')=='candidate_guard_refused' for r in sa['proposals']),'declared_maximum_sweeps_not_claimed_as_completed':True,'termination':sa['termination'],'callbacks_replayed':len(callbacks),'guard_assessments_recomputed':len(e['guard_assessments']),'distinct_full27_states':len(bykey),'accepted_nonbaseline_full_callbacks':sum(t['guard_accepted'] and not t['is_exact_original_state'] for t in trace[1:-1]),'all_trace_hashes_verified':True}

def main():
    global TEST
    started=time.perf_counter();p=read(OUT/'verification_protocol.json');ph=sha(OUT/'verification_protocol.json');check(sha(__file__)==p['script_sha256'],'independent script frozen')
    check(not (OUT/'verification_report.json').exists(),'never overwrite prior scientific output')
    for path,h in p['input_sha256'].items():check(sha(path)==h,'input SHA '+path)
    TEST=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('step guard failed')
    TEST=False
    primary=read(p['primary_protocol']);c=read(p['candidate_path']);inv=read(p['inventory_path'])['modern_cases'];prior=read(p['historical_candidate_path']);prior={row['source_inventory_index']:row for row in prior['cases']}
    check(c['protocol_sha256']==sha(p['primary_protocol']),'candidate frozen method');check(c['physics_steps']==0 and not c['new_eval_data'],'candidate static scope')
    urdf=Path(p['source_urdf']);xml=ET.parse(urdf).getroot();ranges={j.get('name'):np.asarray([float(j.find('limit').get(k)) for k in ('lower','upper')]) for j in xml.findall('joint') if j.find('limit') is not None}
    source=ShadowPoseValidator(urdf.parent.parent);srcpalm=ident(source.model,mujoco.mjtObj.mjOBJ_BODY,'palm');reaches=independent_reach(xml)
    records=[];full_cases=[];loaded_cohorts={}
    for compact in c['cases']:
        binding=compact['complete_record_binding'];path=binding['gzip_path']
        check(sha(path)==binding['gzip_sha256'],'complete cohort compressed binding')
        if path not in loaded_cohorts:
            raw=gzip.decompress(Path(path).read_bytes());check(hashlib.sha256(raw).hexdigest()==binding['cohort_sha256'],'complete cohort raw binding')
            loaded_cohorts[path]=json.loads(raw)
        full=next(r for r in loaded_cohorts[path]['cases'] if r['source_inventory_index']==compact['source_inventory_index'])
        for key,value in compact.items():
            if key=='complete_record_binding':continue
            if key=='solver_audit':
                for name,field in value.items():check(field==full[key][name],'compact solver field binds complete trace '+name)
            else:check(value==full[key],'compact field binds complete trace '+key)
        full_cases.append(full)
    check(sorted(row['source_inventory_index'] for row in c['cases'])==[8,12,16,20,24,28],'six exact development cases')
    for row in full_cases:
        index=row['source_inventory_index'];case=inv[index];old=prior[index]
        check(row['native_sha256']==case['native']['sha256']==old['native_sha256'],'native record retained')
        with np.load(case['stored_shadow_command']['path'],allow_pickle=False) as a:
            n=case['stored_shadow_command']['row_index'];names=a['joint_names'].tolist();q0=a['nominal_target_rad'][n];root0=a['root_target_position_m'][n];quat0=a['root_target_quaternion_wxyz'][n]
        check(names==row['joint_names'] and len(names)==24 and len(set(names))==24,'named original joints');same(q0,row['original_qpos_rad'],'actual original action q');same(root0,row['original_root_target_position_m'],'actual original root');same(quat0,row['original_root_target_quaternion_wxyz'],'actual original quaternion');same(quat0,row['root_target_quaternion_wxyz'],'new root orientation bitwise original')
        with Path(case['native']['path']).open('rb') as f:native=pickle.load(f)
        TWC=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False);WM=TWC@np.asarray(native['T_camera_wrist']);nativeworld=np.asarray(native['landmarks_3d'])@TWC[:3,:3].T+TWC[:3,3];targets=nativeworld[[4,8,12,16,20,2,6,10,14,18]];Rop=WM[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        m=mujoco.MjModel.from_binary_path(case['scene_mjb']['path']);d=mujoco.MjData(m)
        with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as a:d.qpos[:]=a['qpos']
        ja=np.asarray([m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]);ra=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')]);oa=int(m.jnt_qposadr[ident(m,mujoco.mjtObj.mjOBJ_JOINT,'object_free')]);palm=ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');obj=ident(m,mujoco.mjtObj.mjOBJ_BODY,'object')
        objectq=d.qpos[oa:oa+7].copy();d.qpos[ja]=q0;d.qpos[ra:ra+3]=root0;d.qpos[ra+3:ra+7]=quat0;d.qvel[:]=0
        if m.nmocap:d.mocap_pos[0]=root0;d.mocap_quat[0]=quat0
        mujoco.mj_forward(m,d);nominal=d.qpos.copy();objectpos=d.xpos[obj].copy();objectR=d.xmat[obj].copy();oldpoints=task_points(m,d,xml);oldcaps=norm(oldpoints[:5]-targets[:5])
        historical=np.asarray(old['old_original_five_tip_caps_sqrt_einsum_m']);same(oldcaps,historical,'old caps exactly historical');same(oldcaps,row['authoritative_original_tip_caps_sqrt_einsum_m'],'no original cap rebasing')
        x=np.asarray(row['selected_state_scaled']);check(x.shape==(27,) and np.isfinite(x).all(),'selected state27');q=q0+.025*x[:24];shift=.005*x[24:];same(q,row['qpos_rad'],'decode exact normalized joint state');same(shift,row['palm_displacement_world_m'],'decode exact world displacement');anchor=WM[:3,3]+shift;same(anchor,row['target_palm_world_position_m'],'native wrist plus world shift')
        bounds=np.asarray(row['solver_audit']['bounded_correction_domain_scaled']);check(bounds.shape==(27,2) and np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]),'selected state inside frozen domain')
        physical=np.asarray([ranges[n] for n in names]);check(np.all(q>=physical[:,0]) and np.all(q<=physical[:,1]),'exact source physical bounds');check(np.all(q>=q0-.35) and np.all(q<=q0+.35),'exact original joint correction bound')
        rad=reaches+1e-6+oldcaps;low=np.max(targets[:5]-rad[:,None],axis=0);high=np.min(targets[:5]+rad[:,None],axis=0);near(row['domain']['outer_ball_radii_m'],rad,1e-15,'independent source reach outer balls');near(row['domain']['palm_world_AABB_m'],np.column_stack([low,high]),1e-15,'independent source reach outer box');check(np.all(anchor>=low) and np.all(anchor<=high),'selected anchor inside necessary outer box')
        # Independently derive carrier translation from zero-root captured-scene FK.
        d.qpos[ja]=q;d.qpos[ra:ra+3]=0.;d.qpos[ra+3:ra+7]=quat0;mujoco.mj_forward(m,d);expected_root=anchor-d.xpos[palm]
        root=np.asarray(row['root_target_position_m']);rooterr=near(root,expected_root,1e-12,'independent q-compensated root')
        d.qpos[ra:ra+3]=root
        if m.nmocap:d.mocap_pos[0]=root;d.mocap_quat[0]=quat0
        mujoco.mj_forward(m,d);points=task_points(m,d,xml);errors=points-targets;tipnorm=norm(errors[:5]);check(np.all(tipnorm<=oldcaps),'exact original-world cap pass');check(tipnorm.mean()<=.01,'strict10mmmean');near(tipnorm,row['selected_tip_norms_sqrt_einsum_m'],0.,'primary final tipnorm exact');near(errors,row['selected_fresh_diagnostics']['task_link_error_vectors_m'],1e-14,'primary task error vectors');anchorerr=near(d.xpos[palm],anchor,1e-12,'actual translated palm anchor');near(d.xmat[palm].reshape(3,3),row['target_palm_world_rotation'],1e-12,'reported actual palm rotation')
        qmap=dict(zip(names,q));source.data.qpos[:]=[qmap[n] for n in source.joint_names];source.data.qvel[:]=0;mujoco.mj_forward(source.model,source.data)
        fixed=task_points(source.model,source.data,xml)-source.data.xpos[srcpalm];expected=fixed@Rop.T+anchor;sourceerr=float(np.linalg.norm(points-expected,axis=1).max());check(sourceerr<1.1e-7,'independent fixed-source captured-scene FK');rot_err=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-Rop@source.data.xmat[srcpalm].reshape(3,3))));check(rot_err<1e-6,'q-dependent source palm orientation')
        same(d.qpos[oa:oa+7],objectq,'object q unchanged');same(d.xpos[obj],objectpos,'object xyz unchanged');same(d.xmat[obj],objectR,'object rotation unchanged');check(d.time==0,'zero simulation time')
        fallback=np.array_equal(x,np.zeros(27));check(fallback==row['solver_audit']['exact_original_fallback'],'explicit fallback status')
        if fallback:same(d.qpos,nominal,'fallback complete original pose bytes')
        gate=row['analytic_jacobian_baseline_check'];near(np.max(np.abs(np.asarray(gate['analytic_jacobian_m_per_scaled_state'])-np.asarray(gate['finite_difference_jacobian_m_per_scaled_state']))),gate['maxabs_difference_m_per_scaled_state'],0.,'analyticJac gate arithmetic');check(gate['passed'] and gate['maxabs_difference_m_per_scaled_state']<=1e-8,'analyticJac gate passed unchanged threshold')
        check(row['mj_forward_calls_in_set_pose']==2*row['set_pose_calls'],'declared staged forward count')
        check(row['solver_audit']['counters']['FK']<=6000 and row['solver_audit']['counters']['analytic_Jacobians']<=6000 and row['solver_audit']['counters']['full']<=128,'separate finite callback budgets')
        trace=trace_review(row,oldcaps,p)
        records.append({'source_inventory_index':index,'native_sha256':row['native_sha256'],'numeric_source_FK_status':'PASS','original_caps_exactly_historical':True,'five_original_caps_pass':(tipnorm<=oldcaps).tolist(),'mean_tip_error_m':float(tipnorm.mean()),'translation_world_m':shift.tolist(),'translation_norm_m':float(np.linalg.norm(shift)),'q_compensated_root_maxabs_error_m':rooterr,'anchor_maxabs_error_m':anchorerr,'source_vs_scene_max_link_error_m':sourceerr,'root_orientation_bitwise_original':True,'joint_correction_and_source_physical_bounds_pass':True,'exact_original_fallback':bool(fallback),'trace_review':trace,'full_scene_qpos_float64_sha256':array_sha(d.qpos),'scientific_mapping_acceptance':False})
    for path,h in p['input_sha256'].items():check(sha(path)==h,'input unchanged '+path)
    check(COUNT['physics_actual']==0,'no actual integration');report={'schema':'independent_free_palm_output_verification_v1','protocol_sha256':ph,'primary_protocol_sha256':sha(p['primary_protocol']),'candidate_sha256':sha(p['candidate_path']),'passed':True,'checks':COUNT,'cases':records,'runtime_seconds':time.perf_counter()-started,'new_surface_queries':0,'new_eval_data':False,'physics_steps':0,'new_inference':False,'scientific_mapping_acceptance':False,'scope':'Independent stored source/action/FK and exact original caps, same27D state/key bindings, full trace arithmetic/guard replay. No optimizer or surface-geometry replay; full scientific surface audit remains separate.'}
    (OUT/'verification_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'checks':COUNT,'cases':len(records),'report':str(OUT/'verification_report.json')}))
if __name__=='__main__':main()
