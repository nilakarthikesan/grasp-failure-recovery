"""Static scene reconstruction and fixed saved-witness surface queries."""
import os
for k,v in {'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1',
            'OMP_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','NUMBA_NUM_THREADS':'4'}.items():os.environ[k]=v
from pathlib import Path
import gzip,hashlib,json,pickle,warnings,xml.etree.ElementTree as ET
import numpy as np
import mujoco
import numba
numba.set_num_threads(2)
import exact_direction as direction
from free_palm_kinematics import decode_state,state_bounds,urdf_tip_reaches,stage_translated_palm,virtual_tip_jacobian
from geometry_helpers import refresh_colliders,query_anatomical_part
from constraint_core import proximity_constraints,orientation_constraints,exact_tip_status,diagnostic_rank
from surface_component_guard import freeze_direction_component_guard
from grasp_failure_prediction.evaluation.pose_validation import _link_positions,_body_id,_fixed_tip_offsets,TASK_LINK_NAMES

ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
OLD=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
V1=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1')
SAVED=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
a=direction.a
PHYSICS_CALLS=0
def forbidden(*args,**kwargs):
    global PHYSICS_CALLS
    PHYSICS_CALLS+=1
    raise RuntimeError('No integration is permitted in the static witness study')
for api in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,api,forbidden)

def read(path):
    p=Path(path);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v

class Case:
    def __init__(self,index):
        self.index=index;self.staging_calls=0;self.raw_calls=0;self.exact_calls=0;self.tip_jacobian_calls=0
        inventory=read(ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json')
        fk=read(ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json')
        self.source=inventory['modern_cases'][index];self.native_hash=self.source['native']['sha256']
        with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
        weights=np.asarray(asset['weights']);grouped=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
        (self.meta,self.native_tri,self.native_labels,self.native_normals,self.original_hand,self.objects,
         self.model,self.data,self.initial_object,self.objadr,self.opos,self.oR)=a.load_case(self.source,grouped,{c['native_sha256']:c for c in fk['cases']})
        self.names=self.source['stored_shadow_command']['joint_names']
        self.q0=np.asarray([self.meta['recovered_original_targets'][n] for n in self.names])
        self.nominal=self.data.qpos.copy();self.rootpos=np.asarray(self.meta['root_target_world_position_m'])
        self.rootquat=np.asarray(self.meta['root_target_world_quaternion_wxyz'])
        self.adr=np.asarray([self.model.jnt_qposadr[a.ident(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in self.names])
        self.dofs=np.asarray([self.model.jnt_dofadr[a.ident(self.model,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in self.names])
        self.palm=_body_id(self.model,'palm');self.wrist=_body_id(self.model,'wrist');self.object_body=_body_id(self.model,'object')
        self.rootadr=int(self.model.jnt_qposadr[a.ident(self.model,mujoco.mjtObj.mjOBJ_JOINT,'world_joint')])
        native=pickle.loads(Path(self.source['native']['path']).read_bytes())
        T=np.load(self.source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        self.T_wrist=T@np.asarray(native['T_camera_wrist']);self.anchor=self.T_wrist[:3,3]
        landmarks=np.asarray(native['landmarks_3d'])@T[:3,:3].T+T[:3,3]
        self.native_tasks=landmarks[[4,8,12,16,20,2,6,10,14,18]]
        self.original_wrist_pos=self.data.xpos[self.wrist].copy();self.original_wrist_R=self.data.xmat[self.wrist].reshape(3,3).copy()
        self.original_palm_R=self.data.xmat[self.palm].reshape(3,3).copy()
        spec=_fixed_tip_offsets(a.URDF)
        self.tip_bodies=np.asarray([_body_id(self.model,spec[n][0]) for n in TASK_LINK_NAMES[:5]])
        self.orientation_protocol=read(OLD/'orientation/protocol.json')
        self.visual_tri,self.visual_norm,self.visual_meta=direction.o.visual_mesh(self.meta,self.model,self.data,self.opos,self.oR)
        self.original_errors=_link_positions(self.model,self.data,a.URDF)-self.native_tasks
        self.caps=np.sqrt(np.einsum('ij,ij->i',self.original_errors[:5],self.original_errors[:5]))
        prior_protocol=read(V1/'protocol.json');self.cap_binding=prior_protocol['prior_original_cap_bindings'][str(index)]
        assert sha(self.cap_binding['path'])==self.cap_binding['sha256']
        historical=read(self.cap_binding['path'])['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m']
        assert np.array_equal(self.caps,np.asarray(historical)),'No cap rebasing'
        physical={j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(a.URDF).getroot().findall('joint') if j.find('limit') is not None}
        self.limits=np.asarray([physical[n] for n in self.names])
        self.bounds,self.domain=state_bounds(self.q0,self.limits,self.native_tasks[:5],self.caps,self.anchor,urdf_tip_reaches(a.URDF))
        saved=read(SAVED)
        self.prox_rows=[r for r in saved['whole_cell_rows'] if r['source_inventory_index']==index]
        self.ori_rows=[r for r in saved['orientation_rows'] if r['source_inventory_index']==index]
        self.pq=np.asarray([r['row']['centroid_object_m'] for r in self.prox_rows],float).reshape(-1,3)
        self.pp=np.asarray([a.PARTS.index(r['row']['anatomical_part']) for r in self.prox_rows],int)
        self.radius=np.asarray([r['row']['centroid_radius_m'] for r in self.prox_rows])
        self.allowance=np.asarray([r['row']['numerical_allowance_m'] for r in self.prox_rows])
        self.pband=np.asarray([r['row']['band_mm']/1000 for r in self.prox_rows])
        self.near=np.asarray([r['row']['relationship_change']=='lost_native' for r in self.prox_rows])
        self.oq=np.asarray([r['row']['query_object_m'] for r in self.ori_rows],float).reshape(-1,3)
        self.on=np.asarray([r['row']['object_outward_normal'] for r in self.ori_rows],float).reshape(-1,3)
        self.op=np.asarray([a.PARTS.index(r['row']['native_part']) for r in self.ori_rows],int)
        self.oscore=np.asarray([r['row']['native_facing_compatibility'] for r in self.ori_rows])
        self.osign=np.sign(self.oscore);self.oband=np.asarray([r['band_mm']/1000 for r in self.ori_rows])
        self.all_points=np.concatenate((self.pq,self.oq));self.all_parts=np.concatenate((self.pp,self.op))
        self.groups=[(part,np.flatnonzero(self.all_parts==i)) for i,part in enumerate(a.PARTS) if np.any(self.all_parts==i)]
        native_distance=np.asarray([r['row']['native_centroid_distance_m'] for r in self.prox_rows])
        assert proximity_constraints(native_distance,self.radius,self.allowance,self.pband,self.near)['passed'].all()
        assert all(r['row']['native_sha256']==self.native_hash for r in self.prox_rows)
        assert all(r['native_sha256']==self.native_hash for r in self.ori_rows)
        assert np.all(np.abs(self.oscore)>self.orientation_protocol['facing_dot_numerical_deadzone'])
        self.old_cohort=next(c for c in read(V1/('candidate_cohort_8_16_24.json' if index in (8,16,24) else 'candidate_cohort_12_20_28.json'))['cases'] if c['source_inventory_index']==index)
        self.old_reference=self.old_cohort['direction_reference_by_part']
        self.old_guard=freeze_direction_component_guard(self.old_reference,self.old_cohort['baseline_fresh_diagnostics']['normal_diagnostics'])
        self.old_query_cache=Path(prior_protocol['query_cache_root'])/(self.source['object_id']+'_'+Path(self.source['native']['path']).parent.name+'_density2mm_queries.npz')
        with np.load(self.old_query_cache,allow_pickle=False) as f:self.old_q=f['query_object_m'].copy();self.old_n=f['visual_triangle_outward_normal'].copy()

    def stage(self,x):
        x=np.asarray(x,float);self.staging_calls+=1
        root=stage_translated_palm(self.model,self.data,self.nominal,self.adr,self.rootadr,self.palm,self.rootpos,self.rootquat,self.anchor,self.q0,x)
        _,shift=decode_state(x,self.q0)
        assert self.data.time==0 and np.array_equal(self.data.qpos[self.objadr:self.objadr+7],self.initial_object)
        assert np.array_equal(self.data.xpos[self.object_body],self.opos) and np.array_equal(self.data.xmat[self.object_body].reshape(3,3),self.oR)
        assert np.array_equal(self.data.qpos[self.rootadr+3:self.rootadr+7],self.rootquat)
        assert np.linalg.norm(self.data.xpos[self.palm]-(self.anchor+shift))<1e-12
        return root

    def raw(self,x):
        self.raw_calls+=1;self.stage(x)
        hand=refresh_colliders(self.original_hand,self.model,self.data,self.opos,self.oR)
        distances=np.empty(len(self.all_points));normals=np.empty_like(self.all_points);closest=np.empty_like(self.all_points)
        for part,idx in self.groups:
            ds,cp,n=query_anatomical_part(a,[g for g in hand if g['part']==part],self.all_points[idx])
            distances[idx]=ds;normals[idx]=n;closest[idx]=cp
        P=len(self.pq);score=-np.einsum('ij,ij->i',normals[P:],self.on)
        # Cheap guidance for two exact eligibility conditions. The full old
        # facet/tie/buried/visual-membership tests remain checkpoint gates.
        allpart=np.full((len(self.oq),len(a.PARTS)),np.inf);minimum_signed=np.full(len(self.oq),np.inf)
        for g in hand:
            if not len(self.oq):break
            signed,_,_=a.geom_query(g,self.oq)
            pi=a.PARTS.index(g['part']);allpart[:,pi]=np.minimum(allpart[:,pi],np.abs(signed))
            minimum_signed=np.minimum(minimum_signed,signed)
        if len(self.oq):
            other=allpart.copy();other[np.arange(len(self.oq)),self.op]=np.inf
            part_gap=other.min(1)-distances[P:]-self.orientation_protocol['tie_distance_tolerance_m']
        else:part_gap=np.empty(0)
        exterior=minimum_signed-self.orientation_protocol['query_boundary_tolerance_m']
        guidance=np.concatenate(((part_gap-1e-9)/.001,(exterior-1e-9)/.001))
        prox=proximity_constraints(distances[:P],self.radius,self.allowance,self.pband,self.near)
        ori=orientation_constraints(score,distances[P:],self.osign,self.oband,
            self.orientation_protocol['facing_dot_numerical_deadzone'],self.orientation_protocol['query_boundary_tolerance_m'])
        errors=_link_positions(self.model,self.data,a.URDF)-self.native_tasks
        return {'state_scaled':np.asarray(x,float).copy(),'surface_margin':np.concatenate((prox['solver_margin'],ori['solver_margin'],guidance)),
            'raw_nearest_part_margin_m':part_gap,'raw_query_exterior_margin_m':exterior,
            'proximity':prox,'raw_orientation':ori,'raw_score':score,'part_distance_m':distances,
            'part_closest_object_m':closest,'part_outward_normal':normals,'task_error_vectors_m':errors,'tip_status':exact_tip_status(errors[:5],self.caps)}

    def tip_jacobian(self,x):
        self.tip_jacobian_calls+=1;self.stage(x)
        return virtual_tip_jacobian(self.model,self.data,self.palm,_link_positions(self.model,self.data,a.URDF)[:5],self.tip_bodies,self.dofs)

    def exact(self,x):
        self.exact_calls+=1;raw=self.raw(x)
        hand=refresh_colliders(self.original_hand,self.model,self.data,self.opos,self.oR)
        if len(self.oq):
            fields=direction.mapped_direction_fields(self.oq,self.on,hand,self.visual_tri,self.orientation_protocol,expected_part=self.op)
            ori=orientation_constraints(fields['score'],fields['distance_m'],self.osign,self.oband,
                self.orientation_protocol['facing_dot_numerical_deadzone'],self.orientation_protocol['query_boundary_tolerance_m'],fields['valid'])
        else:
            fields={'valid':np.empty(0,bool),'score':np.empty(0),'distance_m':np.empty(0),'reason_masks':{}}
            ori=orientation_constraints(np.empty(0),np.empty(0),self.osign,self.oband,
                self.orientation_protocol['facing_dot_numerical_deadzone'],self.orientation_protocol['query_boundary_tolerance_m'],fields['valid'])
        q,shift=decode_state(x,self.q0)
        in_domain=bool(np.all(x>=self.bounds[:,0]) and np.all(x<=self.bounds[:,1]) and np.all(q>=self.limits[:,0]) and np.all(q<=self.limits[:,1]) and np.all(q>=self.q0-.35) and np.all(q<=self.q0+.35))
        passed=np.concatenate((raw['proximity']['passed'],ori['passed']))
        exact_margin=np.concatenate((raw['proximity']['margin_m']/.001,ori['sign_margin'],
            ori['band_margin_m']/.001,ori['query_boundary_margin_m']/.001))
        raw.update(exact_orientation_fields=fields,exact_orientation=ori,in_domain=in_domain,
            all_witness_constraints_pass=bool(passed.all()),literal_surface_pass=passed,
            orientation_witness_coverage='PRESENT_CAPPED_EXAMPLES' if len(self.ori_rows) else 'UNAVAILABLE_NO_SAVED_RELIABLE_WITNESS',
            exact_surface_margin_without_optimizer_buffers=exact_margin,
            rank=diagnostic_rank(fields['valid'],exact_margin,passed,x),
            root_target_position_m=self.data.qpos[self.rootadr:self.rootadr+3].copy(),
            target_palm_world_position_m=self.data.xpos[self.palm].copy(),target_palm_world_rotation=self.data.xmat[self.palm].reshape(3,3).copy(),
            wrist_world_position_m=self.data.xpos[self.wrist].copy(),wrist_world_rotation=self.data.xmat[self.wrist].reshape(3,3).copy())
        return raw

    def legacy_guard_diagnostic(self,x):
        self.stage(x);hand=refresh_colliders(self.original_hand,self.model,self.data,self.opos,self.oR);rows={}
        for part,ref in self.old_reference.items():
            if not ref['available']:rows[part]={'available':False};continue
            ix=np.asarray(ref['query_indices'],int)
            fields=direction.mapped_direction_fields(self.old_q[ix],self.old_n[ix],hand,self.visual_tri,self.orientation_protocol,expected_part=a.PARTS.index(part))
            rows[part]={'available':True,'native_score':ref['native_score'],'candidate_score':fields['score'],'candidate_valid':fields['valid']}
        return self.old_guard.assess(rows)
