"""Freeze-bound static refinement of SIX EXISTING saved HUG proposals.

No new generator proposal, evaluator episode, controller, rollout or label.
Writes diagnostic candidate joint targets only in this temporary directory.
"""
from pathlib import Path
import hashlib, importlib.util, json, pickle, time, warnings
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

GUARDS={"actual":0,"selftests":0,"testing":False}
def forbidden(*args,**kwargs):
    GUARDS["selftests" if GUARDS["testing"] else "actual"]+=1
    raise RuntimeError("Static HUG refinement forbids physics integration")
for api in ("mj_step","mj_step1","mj_step2"):
    setattr(mujoco,api,forbidden)

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OUT=Path(__file__).parent
LIB=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces/audit_other_surfaces.py'
sp=importlib.util.spec_from_file_location('frozen_geometry',LIB)
a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
for api in ("mj_step","mj_step1","mj_step2"):
    setattr(mujoco,api,forbidden)
import numba
numba.set_num_threads(1)
from surface_refinement import RefinementConfig, select_sites, refine_part
from grasp_failure_prediction.evaluation.pose_validation import _link_positions
from grasp_failure_prediction.evaluation.pose_validation import ShadowPoseValidator, _body_id
from grasp_failure_prediction.evaluation.retargeting import default_dex_urdf_root, reorder_for_mujoco
from grasp_failure_prediction.integrations.hug_frames import MANO_TO_OPERATOR_RIGHT

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def refresh_geom(g,model,data,objpos,objR):
    # Local compiled geometry is immutable; no repeated hull reconstruction.
    result=g.copy();gid=g['id']
    result['R']=objR.T@data.geom_xmat[gid].reshape(3,3)
    result['pos']=(data.geom_xpos[gid]-objpos)@objR
    return result

def query_part(colliders,points):
    dist=np.full(len(points),np.inf);cp=np.full_like(points,np.nan);normal=np.zeros_like(points)
    for geom in colliders:
        signed,c,n=a.geom_query(geom,points);candidate=np.abs(signed)
        improve=candidate<dist
        dist[improve]=candidate[improve];cp[improve]=c[improve];normal[improve]=n[improve]
    assert np.isfinite(dist).all() and np.isfinite(cp).all()
    return dist,cp,normal

def main():
    started=time.perf_counter();pp=OUT/'protocol.json';ph=a.sha(pp);protocol=read(pp)
    for path,h in protocol['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in protocol['implementation_sha256'].items():assert a.sha(path)==h,path
    protected=read(OUT/'protected_hashes.json')
    for path,h in protected.items():assert a.sha(path)==h,path
    config=RefinementConfig(**protocol['configuration'])
    validator=ShadowPoseValidator(default_dex_urdf_root())
    GUARDS['testing']=True
    for api in ('mj_step','mj_step1','mj_step2'):
        try:getattr(mujoco,api)(None,None)
        except RuntimeError:pass
        else:raise AssertionError('Static guard failed')
    GUARDS['testing']=False
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
    inventory=read(prior/'existing_hug_inventory.json');fk=read(prior/'static_saved_hug_fk.json');fkmap={c['native_sha256']:c for c in fk['cases']}
    physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper')))
       for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);gw=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    records=[]
    for planned in protocol['cases']:
        case_start=time.perf_counter();index=planned['source_inventory_index'];case=inventory['modern_cases'][index]
        assert case['native']['sha256']==planned['native_sha256']
        meta,tri,labels,norm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(case,gw,fkmap)
        nominal=d.qpos.copy();rootpos=meta['root_target_world_position_m'];rootquat=meta['root_target_world_quaternion_wxyz']
        palm=a.ident(m,mujoco.mjtObj.mjOBJ_BODY,'palm');palmp=d.xpos[palm].copy();palmR=d.xmat[palm].reshape(3,3).copy()
        names=case['stored_shadow_command']['joint_names'];q0=np.array([meta['recovered_original_targets'][n] for n in names]);qnew=q0.copy()
        adr=np.array([m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names],int)
        baseline_links=_link_positions(m,d,a.URDF)
        caseid=case['object_id']+'_'+Path(case['native']['path']).parent.name
        cache=Path(protocol['query_cache_root'])/f'{caseid}_density2mm_queries.npz';assert a.sha(cache)==protocol['query_cache_sha256'][str(cache)]
        with np.load(cache,allow_pickle=False) as data:cached={k:data[k].copy() for k in data.files}
        query=cached['query_object_m'];area=cached['area_mm2'];nd=cached['native_distance_by_part_m']
        fullpath=Path(protocol['original_full_distance_root'])/f'{caseid}_density2mm_full_current.npz'
        binding=read(fullpath.with_suffix('.binding.json'));assert binding['original_cache_sha256']==a.sha(cache)
        assert a.sha(fullpath)==binding['ephemeral_npz_sha256']==protocol['full_cache_sha256'][str(fullpath)]
        with np.load(fullpath,allow_pickle=False) as data:full={k:data[k].copy() for k in data.files}
        assert np.array_equal(full['query_object_m'],query)
        sd=full['current_distance_by_part_m']
        source=pickle.loads(Path(case['native']['path']).read_bytes())
        T_wc=np.load(case['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        T_wm=T_wc@np.asarray(source['T_camera_wrist'],float)
        Rop=T_wm[:3,:3]@MANO_TO_OPERATOR_RIGHT.T
        assert np.linalg.norm(palmp-T_wm[:3,3])<1e-7
        rootadr=int(m.jnt_qposadr[a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        def set_pose(qfull):
            d.qpos[:]=nominal;d.qpos[adr]=qfull
            d.qpos[rootadr:rootadr+3]=rootpos;d.qpos[rootadr+3:rootadr+7]=rootquat
            mujoco.mj_forward(m,d)
            # No fitted translation: compensate only the wrist-palm offset.
            compensated_root=np.asarray(rootpos)-(d.xpos[palm]-T_wm[:3,3])
            d.qpos[rootadr:rootadr+3]=compensated_root
            if m.nmocap:
                d.mocap_pos[0]=compensated_root;d.mocap_quat[0]=rootquat
            mujoco.mj_forward(m,d)
            assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
            assert np.linalg.norm(d.xpos[palm]-T_wm[:3,3])<1e-12
            return compensated_root.copy()
        details={};fingers={'thumb':('THJ',0),'index':('FFJ',1),'middle':('MFJ',2),'ring':('RFJ',3),'little':('LFJ',4)}
        all_joint_ix=[]
        prepared={}
        for part in a.PARTS[:6]:
            pi=a.PARTS.index(part)
            sites,siteweights,sitemeta=select_sites(nd[:,pi],sd[:,pi],area,config)
            points=query[sites];ntri=tri[labels==pi];nn=norm[labels==pi]
            ndpart,ncp,nix=a.nearest_parts(points,ntri,np.zeros(len(ntri),np.int64),1)
            assert np.max(np.abs(ndpart[:,0]-nd[sites,pi]))<1e-12
            prepared[part]={'sites':sites,'weights':siteweights,'meta':sitemeta,'points':points,
                'native_distance':nd[sites,pi], 'native_closest':ncp,
                'native_facing':-np.einsum('ij,ij->i',nn[nix],cached['visual_triangle_outward_normal'][sites]),
                'geoms':[g for g in hand if g['part']==part]}
        print(json.dumps({'starting_case':caseid,'source_inventory_index':index}),flush=True)
        # Stage 1: wrist articulation changes palm orientation while its source
        # anchor POSITION remains fixed. All six anatomical parts participate.
        wristix=np.array([j for j,n in enumerate(names) if n.startswith('WRJ')],int)
        assert len(wristix)==2
        joined_distance=np.concatenate([v['native_distance'] for v in prepared.values()])
        joined_cp=np.concatenate([v['native_closest'] for v in prepared.values()])
        joined_facing=np.concatenate([v['native_facing'] for v in prepared.values()])
        joined_weights=np.concatenate([v['weights']/6 for v in prepared.values()])
        def evaluate_wrist(values):
            qfull=q0.copy();qfull[wristix]=values;set_pose(qfull)
            distances=[];closest=[];facing=[]
            for part,v in prepared.items():
                transformed=[refresh_geom(g,m,d,pos,R) for g in v['geoms']]
                ds,cp,normal=query_part(transformed,v['points'])
                distances.append(ds);closest.append(cp)
                facing.append(-np.einsum('ij,ij->i',normal,cached['visual_triangle_outward_normal'][v['sites']]))
            return np.concatenate(distances),np.concatenate(closest),np.concatenate(facing),_link_positions(m,d,a.URDF)
        original_wrist_fields=evaluate_wrist(q0[wristix])
        # Scene-policy compensation at original q is roundoff only.
        for part,v in prepared.items():
            transformed=[refresh_geom(g,m,d,pos,R) for g in v['geoms']]
            ds,_,_=query_part(transformed,v['points'])
            assert np.max(np.abs(ds-sd[v['sites'],a.PARTS.index(part)]))<1e-7
        wrist_target,wrist_detail=refine_part(q0[wristix],np.array([physical[names[j]] for j in wristix]),
            joined_distance,joined_cp,joined_facing,np.zeros(len(joined_distance),bool),joined_weights,
            baseline_links,evaluate_wrist,config)
        qnew[wristix]=wrist_target
        wrist_detail.update({'joint_names':[names[j] for j in wristix],
           'six_anatomical_parts_equal_total_mass':True,'normal_objective_used':False,
           'palm_anchor_position_m':T_wm[:3,3].tolist(),
           'carrier_compensation':'Original root rotation; translate carrier only by wrist-induced palm-offset difference.'})
        print(json.dumps({'complete_stage':'wrist','case':caseid,'cost_before':wrist_detail['before_robust_cost'],
           'cost_after':wrist_detail['after_robust_cost'],'q_delta_max_rad':wrist_detail['maximum_abs_joint_delta_rad'],
           'nfev':wrist_detail['nfev'],'solver_status':wrist_detail['solver_status']}),flush=True)
        for part,(prefix,linkix) in fingers.items():
            pi=a.PARTS.index(part);ix=np.array([j for j,n in enumerate(names) if n.startswith(prefix)],int)
            assert len(ix) in (4,5);assert not any(j in all_joint_ix for j in ix);all_joint_ix+=ix.tolist()
            set_pose(qnew)
            v=prepared[part]
            sites,siteweights,sitemeta=v['sites'],v['weights'],v['meta']
            points,ncp,nativefacing=v['points'],v['native_closest'],v['native_facing']
            # Facing is audited independently. The nearest-point VECTOR term
            # retains side/offset without claiming a unique mesh normal.
            normal_eligible=np.zeros(len(sites),bool)
            geoms=[g for g in hand if g['part']==part]
            anchors=baseline_links[[linkix,linkix+5]]
            calls={'forward':0}
            def evaluate(values):
                qfull=qnew.copy();qfull[ix]=values;set_pose(qfull);calls['forward']+=1
                assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
                transformed=[refresh_geom(g,m,d,pos,R) for g in geoms]
                distance,closest,normal=query_part(transformed,points)
                facing=-np.einsum('ij,ij->i',normal,cached['visual_triangle_outward_normal'][sites])
                links=_link_positions(m,d,a.URDF)[[linkix,linkix+5]]
                return distance,closest,facing,links
            reproduction=evaluate(q0[ix])
            # Before-finger fields include the frozen wrist-stage correction.
            target,detail=refine_part(q0[ix],np.array([physical[names[j]] for j in ix]),nd[sites,pi],ncp,
                nativefacing,normal_eligible,siteweights,anchors,evaluate,config)
            qnew[ix]=target
            detail.update({'joint_names':[names[j] for j in ix],'part':part,'site_selection':sitemeta,
                'query_indices':sites.tolist(),'query_weights':siteweights.tolist(),
                'native_distance_m':nd[sites,pi].tolist(),'native_closest_point_object_m':ncp.tolist(),
                'after_wrist_before_finger_distance_m':reproduction[0].tolist(),'after_wrist_before_finger_closest_point_object_m':reproduction[1].tolist(),
                'original_mapped_distance_m':sd[sites,pi].tolist(),
                'static_forward_calls':calls['forward'],'normal_objective_used':False,
                'normal_audit_required':True,'not_a_quality_repair':True})
            final=evaluate(target)
            detail['candidate_mapped_distance_m']=final[0].tolist();detail['candidate_mapped_closest_point_object_m']=final[1].tolist()
            details[part]=detail
            write(OUT/f'{caseid}_partial.json',{'parts':details,'partial':True})
            print(json.dumps({'complete_part':part,'case':caseid,'cost_before':detail['before_robust_cost'],
                 'cost_after':detail['after_robust_cost'],'maximum_abs_joint_delta_rad':detail['maximum_abs_joint_delta_rad'],
                 'nfev':detail['nfev'],'solver_status':detail['solver_status']}),flush=True)
        assert np.max(np.abs(qnew-q0))<=config.joint_delta_limit_rad+1e-9
        assert len(all_joint_ix)==22
        final_root=set_pose(qnew)
        assert d.time==0 and np.array_equal(d.qpos[objadr:objadr+7],initial)
        validator.data.qpos[:]=reorder_for_mujoco(names,qnew,validator.joint_names)
        validator.data.qvel[:]=0;mujoco.mj_forward(validator.model,validator.data)
        fixedp=_body_id(validator.model,'palm')
        desiredR=Rop@validator.data.xmat[fixedp].reshape(3,3)
        palm_rotation_error=float(np.max(np.abs(d.xmat[palm].reshape(3,3)-desiredR)))
        assert palm_rotation_error<1e-6
        fixedlinks=_link_positions(validator.model,validator.data,a.URDF)-validator.data.xpos[fixedp]
        expected_world=fixedlinks@Rop.T+T_wm[:3,3]
        actual_world=_link_positions(m,d,a.URDF)
        scene_fixed_error=float(np.linalg.norm(actual_world-expected_world,axis=1).max())
        assert scene_fixed_error<1e-6
        scene_checks=[]
        for name,value in zip(names,qnew,strict=True):
            low,high=physical[name];assert low-1e-9<=value<=high+1e-9
            jid=a.ident(m,mujoco.mjtObj.mjOBJ_JOINT,name);scene_low,scene_high=m.jnt_range[jid]
            difference=max(abs(low-scene_low),abs(high-scene_high));overflow=max(scene_low-value,value-scene_high,0.)
            assert overflow<=difference+1e-9
            scene_checks.append({'joint':name,'source_URDF_range_rad':[low,high],
                 'scene_range_rad':[float(scene_low),float(scene_high)],'representation_delta_rad':float(difference),
                 'strict_scene_overflow_rad':float(overflow),'not_a_production_preflight_rejection':True})
        record={'source_inventory_index':index,'native_sha256':case['native']['sha256'],'object_id':case['object_id'],
            'source_case':Path(case['native']['path']).parent.name,'joint_names':names,'qpos_rad':qnew.tolist(),
            'original_qpos_rad':q0.tolist(),'root_target_position_m':final_root.tolist(),'root_target_quaternion_wxyz':rootquat,
            'original_root_target_position_m':rootpos,'original_root_target_quaternion_wxyz':rootquat,
            'target_palm_world_position_m':T_wm[:3,3].tolist(),'target_palm_world_rotation':desiredR.tolist(),
            'T_world_operator_rotation':Rop.tolist(),'palm_anchor_position_fixed':True,'wrist_stage':wrist_detail,
            'part_diagnostics':details,'scene_bounds_audit':scene_checks,
            'palm_rotation_convention_maxabs_error':palm_rotation_error,
            'fixed_URDF_vs_scene_max_link_error_m':scene_fixed_error,
            'input_sha256':meta['input_sha256'],'runtime_seconds':time.perf_counter()-case_start,
            'original_query_cache_sha256':a.sha(cache),'original_full_cache_sha256':a.sha(fullpath),
            'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0}
        record['candidate_link_world_m']=_link_positions(m,d,a.URDF).tolist()
        record['original_link_world_m']=baseline_links.tolist()
        records.append(record)
        write(OUT/'candidate_targets.partial.json',{'schema':'hug_surface_refinement_candidate_v2',
             'protocol_sha256':ph,'cases':records,'partial':True})
        for path,h in meta['input_sha256'].items():assert a.sha(path)==h,path
    for path,h in protocol['source_sha256'].items():assert a.sha(path)==h,path
    for path,h in protocol['implementation_sha256'].items():assert a.sha(path)==h,path
    for path,h in protected.items():assert a.sha(path)==h,path
    assert GUARDS['actual']==0 and a.sha(pp)==ph
    report={'schema':'hug_surface_refinement_candidate_v2','protocol_sha256':ph,'script_sha256':a.sha(__file__),
        'implementation_sha256':a.sha(OUT/'surface_refinement.py'),'runtime_seconds':time.perf_counter()-started,
        'configuration':protocol['configuration'],'cases':records,'guards':GUARDS,
        'new_eval_data':False,'new_generator_proposal':False,'physics_steps':0,'training':False,'production_changes':False,
        'protected_hashes_unchanged':True,'original_source_hashes_unchanged':True,
        'meaning':'Development candidate translation targets only. No acceptance claim. Frozen strict rule independently audited.',
        'limitations':protocol['limitations']}
    write(OUT/'candidate_targets.json',report)
    print(json.dumps({'completed':True,'seconds':report['runtime_seconds'],'case_count':len(records)}),flush=True)

if __name__=='__main__':main()
