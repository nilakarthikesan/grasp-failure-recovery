"""One authorized existing-case FK/Jacobian control; no surface calls or fitting."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.spatial.transform import Rotation
from se3_case import Case,base
from se3_math import decode_state,left_compose_root_quaternion

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    protocol_path=Path(args.protocol);protocol=read(protocol_path);ph=sha(protocol_path)
    assert protocol['source_index']==20 and protocol['authorized_scope']=='one_saved_case_static_FK_and_bounded_Jacobian_only'
    for section in ('input_sha256','implementation_sha256'):
        for p,h in protocol[section].items():assert sha(p)==h,p
    out=Path(args.output);assert not out.exists()
    counter=[0];original=base.mujoco.mj_forward
    def forward(*args,**kwargs):counter[0]+=1;return original(*args,**kwargs)
    base.mujoco.mj_forward=forward
    start=time.perf_counter();old=base.Case(20);new=Case(20);checks=[];records=[]
    def check(v,label):assert bool(v),label;checks.append(label)
    pool=next(c for c in read(protocol['pooled_candidate_path'])['cases'] if c['source_inventory_index']==20)
    check(new.native_hash==protocol['native_sha256']==old.native_hash,'source_identity')
    check(np.array_equal(new.caps,old.caps) and np.array_equal(new.caps,pool['authoritative_original_tip_caps_sqrt_einsum_m']),'literal_caps_unchanged')
    check(np.array_equal(new.bounds[:27],old.bounds),'original27_domain_unchanged')
    for label,x27 in [('original',np.zeros(27)),('pooled',np.asarray(pool['selected_state_scaled']))]:
        old.stage(x27);x=np.r_[x27,np.zeros(3)];new.stage(x)
        check(np.array_equal(old.data.qpos,new.data.qpos),label+':zero_rotation_all_qpos_bitwise')
        check(np.array_equal(old.data.xpos,new.data.xpos) and np.array_equal(old.data.xmat,new.data.xmat),label+':zero_rotation_all_body_poses_bitwise')
        norms=np.sqrt(np.einsum('ij,ij->i',(base._link_positions(new.model,new.data,base.a.URDF)-new.native_tasks)[:5],(base._link_positions(new.model,new.data,base.a.URDF)-new.native_tasks)[:5]))
        check(np.all(norms<=new.caps) and norms.mean()<=.010,label+':unchanged_feasible_tips')
    # Frozen controls include a coupled nonzero rotation; no fitted state or surface query.
    for spec in protocol['jacobian_states']:
        x=np.asarray(spec['state_scaled']);check(x.shape==(30,) and np.all(x>=new.bounds[:,0]) and np.all(x<=new.bounds[:,1]),spec['label']+':domain')
        analytic=new.tip_jacobian(x);observed=np.zeros_like(analytic);spans=[];step=protocol['fd_step_scaled']
        def tips(z):new.stage(z);return base._link_positions(new.model,new.data,base.a.URDF)[:5].copy()
        for j in range(30):
            hi=x.copy();lo=x.copy();hi[j]=min(new.bounds[j,1],x[j]+step);lo[j]=max(new.bounds[j,0],x[j]-step)
            span=hi[j]-lo[j];assert span>0
            observed[:,:,j]=(tips(hi)-tips(lo))/span;spans.append(float(span))
        tips(x);err=float(np.max(np.abs(analytic-observed)));check(err<=protocol['jacobian_tolerance_m_per_scaled_state'],spec['label']+':bounded_FD_jacobian')
        q,shift,omega=decode_state(x,new.q0);x27=x[:27];old.stage(x27)
        D=Rotation.from_rotvec(omega).as_matrix();expected=(base._link_positions(old.model,old.data,base.a.URDF)-old.data.xpos[old.palm])@D.T+new.anchor+shift
        actual=base._link_positions(new.model,new.data,base.a.URDF)
        rigiderr=float(np.max(np.abs(expected-actual)));check(rigiderr<1e-12,spec['label']+':independent_rigid_rotation_identity')
        R_expected=D@old.data.xmat[old.palm].reshape(3,3)
        Rerr=float(np.max(np.abs(R_expected-new.data.xmat[new.palm].reshape(3,3))));check(Rerr<1e-12,spec['label']+':palm_rotation_identity')
        check(np.array_equal(new.data.qpos[new.rootadr+3:new.rootadr+7],left_compose_root_quaternion(new.rootquat,omega)),spec['label']+':declared_root_quaternion')
        records.append({'label':spec['label'],'state_scaled':x.tolist(),'maxabs_Jacobian_difference_m_per_scaled_state':err,'joint_block_maxabs':float(np.max(np.abs(analytic[:,:,:24]-observed[:,:,:24]))),'translation_block_maxabs':float(np.max(np.abs(analytic[:,:,24:27]-observed[:,:,24:27]))),'rotation_block_maxabs':float(np.max(np.abs(analytic[:,:,27:]-observed[:,:,27:]))),'rigid_task_identity_maxabs_m':rigiderr,'palm_rotation_identity_maxabs':Rerr,'spans':spans,'analytic':analytic.tolist(),'finite_difference':observed.tolist()})
    check(old.raw_calls==new.raw_calls==old.exact_calls==new.exact_calls==0,'zero_surface_callbacks')
    check(base.PHYSICS_CALLS==0 and base.numba.get_num_threads()==2,'zero_steps_two_threads')
    for section in ('input_sha256','implementation_sha256'):
        for p,h in protocol[section].items():assert sha(p)==h,p
    check(sha(protocol_path)==ph,'lifecycle_hash')
    report={'passed':True,'protocol_sha256':ph,'checks':checks,'records':records,'actual_mj_forward_calls':counter[0],'surface_callbacks':0,'physics_steps':0,'new_fit':False,'new_eval_data':False,'runtime_seconds':time.perf_counter()-start,'scope':'Kinematic control states only; nonzero rotation states are not selected candidates or accuracy results.'}
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'path':str(out),'sha256':sha(out),'checks':len(checks),'actual_mj_forward_calls':counter[0],'records':[{k:r[k] for k in ('label','maxabs_Jacobian_difference_m_per_scaled_state','rigid_task_identity_maxabs_m')} for r in records]},indent=2))

if __name__=='__main__':main()
