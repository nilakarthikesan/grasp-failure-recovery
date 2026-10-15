"""Independent saved-array verification; prohibits MuJoCo/native imports and calls."""
from __future__ import annotations
import hashlib
import importlib.abc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HERE=ROOT/'research_artifacts/manosim_flex_rigid_qualification_mj340_2026-10-15_v1'
OLD=ROOT/'research_artifacts/manosim_flex_primitive_qualification_2026-10-15_v1'
RUN=HERE/'run_v1';PROTOCOL=HERE/'protocol.json'
FREEZE='21f07efb737add48099eef5f1d3c1e2d4f8de73c'
PROTOCOL_SHA='e57f7fe78f8c2a5fbdd79fd9c42147f239a39a522a782f8022d4b2b5f5a35d80'


class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('mujoco','torch','manotorch','mano','hug','grasp_failure_prediction'):
            raise RuntimeError('Saved verifier forbids engine/native imports: '+fullname)
        return None


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)
def near(actual,expected,tolerance,message):
    a=np.asarray(actual);b=np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),message+': shape/nonfinite')
    error=float(np.max(abs(a-b))) if a.size else 0.
    require(error<=tolerance,message+': '+str(error));return error
def frame(name):
    if name=='identity':return np.eye(3),np.zeros(3)
    axis=np.array([1.,-2.,3.]);axis/=np.linalg.norm(axis);angle=.63
    # Independent quaternion-to-matrix path, rather than the worker Rodrigues formula.
    q=np.r_[math.cos(angle/2),axis*math.sin(angle/2)]
    return quat(q),np.array([.123,-.164,.232])
def quat(q):
    w,x,y,z=np.asarray(q,dtype=float)
    require(abs(w*w+x*x+y*y+z*z-1)<1e-14,'Quaternion norm changed')
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def geometry(kind):
    if kind=='triangle':return np.array([[0.,0.,0.],[.04,0.,0.],[0.,.04,0.]]),np.array([[0,1,2]])
    return np.array([[-.04,-.01,0.],[-.02,-.01,0.],[-.03,.01,0.],
        [.02,-.01,0.],[.04,-.01,0.],[.03,.01,0.]]),np.array([[0,1,2],[3,4,5]])
def expected_cases():
    out=[dict(case_id=f'{rid}__{attachment}__{transform}',radius_id=rid,radius_m=radius,
        attachment=attachment,frame=transform,geometry='triangle')
        for rid,radius in [('zero',0.),('positive_1um',1e-6)]
        for attachment in ['single_body','three_bodies'] for transform in ['identity','rigid_nonzero']]
    out += [dict(case_id=f'{rid}__disconnected__{transform}',radius_id=rid,radius_m=radius,
        attachment='single_body',frame=transform,geometry='disconnected')
        for rid,radius in [('zero',0.),('positive_1um',1e-6)] for transform in ['identity','rigid_nonzero']]
    return out
def inventory(path):return dict(sha256=sha(path),bytes=Path(path).stat().st_size)
def bindings(p):
    require(len(p['bindings'])==1022,'Unexpected source/runtime binding count')
    old=read(OLD/'protocol_v2.json');require(len(old['bindings'])==977,'Old closure count changed')
    for name,expected in old['bindings'].items():require(sha(name)==expected,'Preserved old runtime/source changed: '+name)
    for name,expected in p['bindings'].items():require(sha(name)==expected,'Input changed: '+name)
def git(p):
    require(subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()==FREEZE,'HEAD changed')
    for path in [*p['tracked_sources'],str(PROTOCOL)]:
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',FREEZE+':'+str(Path(path).relative_to(ROOT))])
        require(hashlib.sha256(blob).hexdigest()==sha(path),'Git blob changed: '+path)


def jacobian(qpos,case,vertices,point,normal,weights):
    owners=1 if case['attachment']=='single_body' else 3;out=np.zeros((owners+1)*6)
    for i,w in enumerate(weights):
        owner=0 if owners==1 else i;origin=qpos[7*owner:7*owner+3];rotation=quat(qpos[7*owner+3:7*owner+7])
        out[6*owner:6*owner+3]+=w*normal
        # Source path evaluates all body Jacobians at the common contact position.
        for k in range(3):out[6*owner+3+k]+=w*(normal@np.cross(rotation[:,k],point-origin))
    owner=owners;origin=qpos[7*owner:7*owner+3];rotation=quat(qpos[7*owner+3:7*owner+7])
    out[6*owner:6*owner+3]-=normal
    for k in range(3):out[6*owner+3+k]-=normal@np.cross(rotation[:,k],point-origin)
    return out


def verify():
    guard=NoEngine();sys.meta_path.insert(0,guard)
    require('mujoco' not in sys.modules,'Engine was imported before saved-only verification')
    require(sha(PROTOCOL)==PROTOCOL_SHA,'Active protocol hash changed')
    p=read(PROTOCOL);r=read(RUN/'report.json');bindings(p);git(p)
    require(sha(RUN/'report.json')=='3728a2bd58c3375de78975980a97dfb6c06cff2f2885120cf90170630146dc1a','Actual report hash changed')
    require(p['cases']==expected_cases() and [c['case'] for c in r['cases']]==expected_cases(),'Ordered12case identity changed')
    require(r['status']=='COMPLETED_PRIMITIVE_QUALIFICATION_MJ340' and r['failure'] is None and r['forbidden_attempts']==[],'Execution status changed')
    require(r['freeze_commit']==FREEZE and r['protocol_sha256']==PROTOCOL_SHA and Path(r['protocol'])==PROTOCOL,'Freeze/protocol linkage changed')
    expected_counts=dict(model_compile_attempts=12,data_construction_attempts=12,explicit_forward_attempts=84,
        explicit_collision_calls=0,explicit_kinematics_calls=0,native_model_calls=0,physics_steps=0,training_calls=0)
    require(r['operation_counts']==expected_counts,'Operation counts changed')
    progress=read(RUN/'operation_progress.json')
    require(progress['counts']==expected_counts and progress['forbidden_attempts']==[],'Progress counts differ')
    require(0<=r['elapsed_s']<=120 and 0<=progress['elapsed_s']<=120,'Wall-time budget exceeded')
    require(read(RUN/'STARTED.json')==dict(protocol_sha256=PROTOCOL_SHA,freeze_commit=FREEZE,status='STARTED_ONCE'),'Launcher identity changed')
    for key in ['HUG_fidelity_accepted','articulated_MANO_dynamics_validated','collection_or_training_allowed']:
        require(r[key] is False,'Admission must remain closed')
    indexed={};array_receipts={};case_receipts=[];maxima=dict(vertex_m=0.,distance_m=0.,point_m=0.,normal=0.,source_jacobian_replay=0.,expected_jacobian_replay=0.)
    for row in r['artifacts']:
        file=Path(row['path']);require(file.is_absolute() and file.resolve().is_relative_to(RUN) and not file.is_symlink(),'Artifact escaped/aliased')
        require(str(file) not in indexed and inventory(file)=={k:row[k] for k in ('sha256','bytes')},'Artifact changed/duplicate')
        indexed[str(file)]=inventory(file)
    require(len(indexed)==36,'Exact12XML+12NPZ+12scalar inventory required')
    require({str(f.resolve()) for f in RUN.rglob('*.npz')}=={f for f in indexed if f.endswith('.npz')},'Unindexed array')
    for result in r['cases']:
        c=result['case'];folder=RUN/c['case_id'];a=np.load(folder/'readback.npz',allow_pickle=False)
        scalar=dict(result);original=scalar.pop('result_artifact');require(read(original['path'])==scalar,'Case scalar/report disagreement')
        require(result['status']=='COMPILED_AND_MEASURED','Completed measurement required')
        require(result['checks']['status']==('PASS' if c['attachment']=='single_body' else 'FAIL'),'Rigid PASS / multiowner FAIL must be preserved')
        rotation,translation=frame(c['frame']);v,f=geometry(c['geometry']);world=v@rotation.T+translation
        require(np.array_equal(a['expected_faces'],f) and np.array_equal(a['observed_faces'],f),'Original face indices changed')
        near(a['expected_vertices'],world,1e-14,'Expected world vertices')
        error=near(a['observed_vertices'],np.broadcast_to(world,(7,*world.shape)),1e-14,'Observed full vertex sequence')
        maxima['vertex_m']=max(maxima['vertex_m'],error)
        require(error==result['vertex_max_error_m'] or abs(error-result['vertex_max_error_m'])<1e-16,'Vertex error scalar differs')
        owner_ids=np.ones(len(v),dtype=int) if c['attachment']=='single_body' else np.arange(1,4)
        require(np.array_equal(a['owner_body_ids'],owner_ids),'Compiled owner IDs changed')
        require(sha(folder/'model.xml')==p['xml_sha256'][c['case_id']],'Compiled input XML differs from frozen declaration')
        xml=ET.parse(folder/'model.xml').getroot();flex=xml.find('./deformable/flex');bodies=xml.findall('./worldbody/body')
        require(not xml.findall('./asset') and len(bodies)==(2 if c['attachment']=='single_body' else 4),'Unexpected XML assets/bodies')
        require(all(body.find('freejoint') is not None for body in bodies),'Owners are not active free bodies')
        require(float(flex.attrib['radius'])==c['radius_m'] and np.array_equal(np.fromstring(flex.attrib['element'],sep=' ',dtype=int).reshape(-1,3),f),'XML radius/topology changed')
        local=np.fromstring(flex.attrib['vertex'],sep=' ').reshape(-1,3)
        require(local.shape==v.shape,'XML vertex size changed')
        q=a['qpos'];require(q.shape==(7,len(bodies)*7) and a['qvel'].shape==(7,len(bodies)*6),'State dimensions changed')
        require(np.isfinite(q).all() and np.array_equal(a['qvel'],np.zeros_like(a['qvel'])) and np.array_equal(a['warning_counters'],np.zeros_like(a['warning_counters'])),'State/warning guard evidence failed')
        for step in range(7):
            for vi in range(len(v)):
                bi=0 if c['attachment']=='single_body' else vi
                near(quat(q[step,7*bi+3:7*bi+7])@local[vi]+q[step,7*bi:7*bi+3],a['observed_vertices'][step,vi],1e-14,'Independent root/local vertex readback')
        require(len(result['probes'])==6,'Six probes required');counts=[];source_max=expected_max=0.;source_weight_sums=[]
        maxd=maxp=maxn=maxt=maxrot=0.;failures=[]
        for i,record in enumerate(result['probes']):
            probe=record['probe'];require(probe==p['probes'][c['case_id']][i],'Frozen probe changed')
            rows=record['contacts'];require(len(rows)==record['contact_count'],'Contact denominator differs');counts.append(len(rows))
            expected_n=0 if probe['placement'] in ('separated','gap_clearance') else 1
            if probe['placement']!='touching':require(len(rows)==expected_n,'Analytical contact/nonconvex-gap presence failed')
            require(len(rows)<=1,'Single primitive probe produced duplicate contacts')
            near(q[i+1,-7:-4],probe['sphere_center_world'],1e-14,'Sphere placement')
            near(q[i+1,:-7],q[0,:-7],0.,'Triangle poses changed')
            near(q[i+1,-4:],q[0,-4:],0.,'Sphere orientation changed')
            for j,row in enumerate(rows):
                point=np.array(row['position_world']);normal=np.array(row['normal_world'])
                center=np.array(probe['surface_center_local']);expected_point=rotation@(center+[0.,0.,probe['side']*(c['radius_m']+probe['expected_gap_m']/2)])+translation
                maxd=max(maxd,abs(row['distance_m']-probe['expected_gap_m']));maxp=max(maxp,near(point,expected_point,1e-10,'Analytical contact point'))
                maxn=max(maxn,near(normal,-probe['side']*rotation[:,2],1e-10,'Both-sided contact normal'))
                require(abs(row['distance_m']-probe['expected_gap_m'])<=1e-10,'Analytical distance failed')
                face=f[probe['triangle_index']];vertices=world[face]
                # Independent barycentric least-squares on the plane, including partition of unity.
                weights=np.linalg.lstsq(np.vstack((vertices.T,np.ones(3))),np.r_[point,1.],rcond=None)[0]
                # Least squares in 3D includes tiny normal offset; use in-plane transformed XY directly.
                localpoint=rotation.T@(point-translation);sourcev=v[face]
                uv=np.linalg.solve(np.column_stack((sourcev[1,:2]-sourcev[0,:2],sourcev[2,:2]-sourcev[0,:2])),localpoint[:2]-sourcev[0,:2])
                weights=np.r_[1-uv.sum(),uv]
                near(weights,row['barycentric_weights'],1e-13,'Saved geometric barycentric weights')
                expected=np.zeros(result['nv']);owners=1 if c['attachment']=='single_body' else 3
                for k,w in enumerate(weights):
                    bi=0 if owners==1 else k;origin=q[i+1,7*bi:7*bi+3];br=quat(q[i+1,7*bi+3:7*bi+7])
                    expected[6*bi:6*bi+3]+=w*normal
                    for axis in range(3):expected[6*bi+3+axis]+=w*(normal@np.cross(br[:,axis],vertices[k]-origin))
                expected[-6:-3]-=normal
                sr=quat(q[i+1,-4:]);so=q[i+1,-7:-4]
                for axis in range(3):expected[-3+axis]-=normal@np.cross(sr[:,axis],point-so)
                stored_expected=np.asarray(row['expected_vertex_normal_jacobian']);stored_engine=np.asarray(row['engine_normal_jacobian'])
                expected_max=max(expected_max,near(expected,stored_expected,1e-13,'Independent geometric normal Jacobian'))
                require(np.array_equal(a[f'probe_{i}_contact_{j}_expected_vertex_normal_jacobian'],stored_expected),'Expected Jacobian JSON/NPZ differs')
                require(np.array_equal(a[f'probe_{i}_contact_{j}_engine_normal_jacobian'],stored_engine),'Engine Jacobian JSON/NPZ differs')
                # Exact pinned source diagnosis: inverse-distance weights normalized by their sum (L1).
                inverse=1/np.maximum(1e-15,np.linalg.norm(vertices-point,axis=1));source_weights=inverse/inverse.sum()
                source_weight_sums.append(float(source_weights.sum()))
                if row['constraint_active']:
                    require(row['efc_address']>=0,'Active contact lacks row')
                    source=jacobian(q[i+1],c,vertices,point,normal,source_weights)
                    source_max=max(source_max,near(source,stored_engine,1e-12,'Pinned-source inverse-distance L1 Jacobian replay'))
                    columns=row['translation_columns'];maxt=max(maxt,float(max(abs(stored_engine[columns]-expected[columns]))))
                    rotations=sorted(set(range(result['nv']))-set(columns));maxrot=max(maxrot,float(max(abs(stored_engine[rotations]-expected[rotations]))))
                else:require(probe['placement']=='touching' and row['efc_address']==-1,'Unexpected inactive penetrating contact')
        check=result['checks']
        for key,value in [('max_distance_error_m',maxd),('max_contact_point_error_m',maxp),('max_normal_error',maxn),('max_translation_jacobian_error',maxt),('max_vertex_kinematic_rotation_jacobian_error',maxrot)]:
            require(abs(check[key]-value)<1e-13,'Scalar check differs: '+key)
        if owners==1:
            require(maxt<=p['tolerances']['translation_jacobian'] and maxrot<=p['tolerances']['rigid_jacobian'],'Rigid-only derivative failed')
        else:require(maxt>p['tolerances']['translation_jacobian'],'Multiowner translation FAIL disappeared')
        expected_failures=[] if owners==1 else ['translation_jacobian']
        require(check['failures']==expected_failures,'Failure taxonomy changed')
        for key,value in [('distance_m',maxd),('point_m',maxp),('normal',maxn),('source_jacobian_replay',source_max),('expected_jacobian_replay',expected_max)]:maxima[key]=max(maxima[key],value)
        array_receipts[str(folder/'readback.npz')]={name:dict(dtype=str(a[name].dtype),shape=list(a[name].shape),sha256=hashlib.sha256(a[name].tobytes()).hexdigest()) for name in a.files}
        case_receipts.append(dict(case_id=c['case_id'],status='VERIFIED_RIGID_CONTACT_JACOBIAN_PASS' if owners==1 else 'VERIFIED_MULTIOWNER_TRANSLATION_JACOBIAN_FAIL',
            contact_counts=counts,checks=check,source_inverse_distance_L1_jacobian_max_error=source_max,
            source_weight_sum_min=min(source_weight_sums),source_weight_sum_max=max(source_weight_sums),
            original_result_sha256=sha(folder/'result.json')))
    # Reuse only the frozen pure helper after the independent checks above; all engine imports remain blocked.
    spec=importlib.util.spec_from_file_location('frozen_primitive_pure_readback',HERE/'run_qualification.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(module.summarize(r['cases'])==r['radius_assessments'],'Frozen summary differs')
    require(module.summarize_rigid(r['cases'])==r['rigid_radius_assessments'],'Rigid summary differs')
    helper={}
    for radius in ['zero','positive_1um']:
        policy=module.qualified_rigid_radius_policy(RUN/'report.json',sha(RUN/'report.json'),radius)
        require(policy['rigid_only'] is True and policy['locked_articulation_snapshot_only'] is True
            and policy['full_qualification_status']=='NOT_QUALIFIED'
            and policy['articulated_MANO_dynamics_validated'] is False
            and policy['collection_or_training_allowed'] is False,'Rigid helper broadened scope')
        helper[radius]=dict(status='RIGID_ONLY_CAPABILITY_RETURNED_FULL_SCOPE_NOT_QUALIFIED',policy=policy)
    # Compare all actual contacts across the two rigid frames independently of the helper.
    frame_pairs=[]
    for rid in ['zero','positive_1um']:
        for kind,attachment in [('triangle','single_body'),('triangle','three_bodies'),('disconnected','single_body')]:
            pair=[next(x for x in r['cases'] if x['case']['radius_id']==rid and x['case']['geometry']==kind and x['case']['attachment']==attachment and x['case']['frame']==transform) for transform in ['identity','rigid_nonzero']]
            rotation,translation=frame('rigid_nonzero');md=mp=mn=0.
            for left,right in zip(pair[0]['probes'],pair[1]['probes']):
                if left['probe']['placement']!='touching':require(left['contact_count']==right['contact_count'],'Frame contact count changed')
                if left['contact_count']!=right['contact_count']:continue
                for a,b in zip(left['contacts'],right['contacts']):
                    md=max(md,abs(a['distance_m']-b['distance_m']));mp=max(mp,near(a['position_world'],rotation.T@(np.array(b['position_world'])-translation),1e-10,'Frame point'))
                    mn=max(mn,near(a['normal_world'],rotation.T@np.array(b['normal_world']),1e-10,'Frame normal'))
            frame_pairs.append(dict(radius_id=rid,geometry=kind,attachment=attachment,max_distance_error_m=md,max_point_error_m=mp,max_normal_error=mn))
    bindings(p);git(p)
    for name,row in indexed.items():require(inventory(name)==row,'Output changed during verification')
    artifact_inventory={name:inventory(name) for name in [*indexed,str(PROTOCOL),str(RUN/'report.json'),str(RUN/'STARTED.json'),str(RUN/'operation_progress.json'),str(RUN/'worker.log')]}
    source=Path(__file__).read_text();sys.meta_path.remove(guard)
    return dict(schema_version='manosim_flex_rigid_mj340_independent_saved_readback_v1',
        status='VERIFIED_COMPLETED_MJ340_PRIMITIVES_RIGID_ONLY_PASS_FULL_NOT_QUALIFIED',freeze_commit=FREEZE,
        active_protocol_sha256=PROTOCOL_SHA,expected_bindings=1022,preserved_old_bindings=977,all_preserved_old_bindings_verified_before_and_after=True,all_bindings_verified_before=True,
        all_bindings_verified_after=True,selected_Git_blob_count=len(p['tracked_sources'])+1,
        actual_execution_status=r['status'],operation_counts=expected_counts,verification_engine_calls=0,
        cases=case_receipts,frame_pairs=frame_pairs,max_errors=maxima,radius_assessments=r['radius_assessments'],rigid_radius_assessments=r['rigid_radius_assessments'],
        qualification_helper=helper,artifact_inventory=artifact_inventory,array_inventory=array_receipts,
        diagnosis=dict(primary_source='https://github.com/google-deepmind/mujoco/blob/e55fff5dea6f1d5dd7963ca52eecc41d05ad0922/src/engine/engine_core_constraint.c#L144-L181',
            owner_Jacobian_source='https://github.com/google-deepmind/mujoco/blob/e55fff5dea6f1d5dd7963ca52eecc41d05ad0922/src/engine/engine_core_constraint.c',
            finding='MuJoCo3.4.0 element weights are inverse vertex distances normalized by their sum, not barycentric. Same-owner weights now sum to1 and rigid normal Jacobians pass. Multiowner inverse-distance weights still differ from the barycentric vertex-velocity reference; all four corresponding translation failures remain.',
            original_independent_rigid_derivative_reference_unchanged=True,
            geometry_and_contact_checks_passed=True,rigid_normal_Jacobian_qualification_passed=True,multiowner_normal_Jacobian_qualification_failed=True),
        HUG_fidelity_accepted=False,articulated_MANO_dynamics_validated=False,collection_or_training_allowed=False,
        verifier_source_sha256=hashlib.sha256(source.encode()).hexdigest(),verifier_source=source,errors=[])


if __name__=='__main__':
    result=verify();destination=Path(sys.argv[1])
    with destination.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),max_errors=result['max_errors'])))
