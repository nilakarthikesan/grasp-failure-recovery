"""Current-scene bridge for the bounded full-field constraint-exchange pilot.

No model, query or optimization runs at import. Original caps/commands/native
targets are read literally. The historical binary supplies ONLY the frozen
object visual reference used by the original orientation eligibility rule.
Every target hand/body/collider is from the current registered scene.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pickle
import sys
import time

import numpy as np


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RepairCase:
    def __init__(self,index,declaration,output_dir,guard_reinstall=None):
        self.index=int(index);self.p=declaration;self.output=Path(output_dir)
        self.output.mkdir(parents=True,exist_ok=True)
        self.counts=dict(raw_callbacks=0,cheap_callbacks=0,tip_jacobians=0,orientation_checks=0,
                         full_field_checkpoints=0,proximity_collider_point_queries=0)
        v1path=Path(declaration['v1_protocol'])
        self.v1=load('repair_bound_full_field_v1',v1path.parent/'audit_full_field.py')
        self.previous=self.v1.read(v1path)
        self.plan=next(c for c in self.previous['cases'] if c['source_inventory_index']==index)
        if len(self.plan['grids'])!=1:raise ValueError('Pilot supports exact one full original grid percase')
        self.grid=self.plan['grids'][0]
        snapshots=v1path.parent/'source_snapshots'
        sys.path.insert(0,str(snapshots))
        import scene_runtime,xml_reference,native_reference,witness_check
        import mujoco
        self.mujoco=mujoco;self.scene_runtime=scene_runtime;self.xml=xml_reference
        self.witness=witness_check
        from grasp_failure_prediction.evaluation import case_runner,retargeting,pose_validation
        from grasp_failure_prediction.evaluation.surface_pose import SurfacePoseAdapter
        from grasp_failure_prediction.integrations.hug import load_hug_prediction
        inventory=self.v1.read(self.previous['inventory'])
        self.source=inventory['modern_cases'][index]
        self.native_hash=self.source['native']['sha256']
        if self.native_hash!=self.plan['native_sha256'] or self.v1.sha(self.source['native']['path'])!=self.native_hash:
            raise ValueError('Native identity changed')
        native=pickle.loads(Path(self.source['native']['path']).read_bytes())
        camera=np.load(self.source['frame_files']['T_world_camera.npy']['path'],allow_pickle=False)
        targets=np.asarray(native['landmarks_3d'])@camera[:3,:3].T+camera[:3,3]
        self.native_targets=targets[[4,8,12,16,20]]
        original=self.plan['poses'][0]
        if original['pose_id']!=f'original_{index}':raise ValueError('Original-current baseline must precede any seed')
        self.names=tuple(original['joint_names']);self.q0=np.asarray(original['qpos_rad'],float)
        self.caps=np.asarray(original['historical_tip_caps_m'],float)
        capfile=self.v1.read(original['historical_cap_report'])
        if not np.array_equal(self.caps,capfile['engineering_guard_independent_audit']['original_tip_caps_sqrt_einsum_m']):
            raise ValueError('Literal historical caps changed; no rebasing')
        payload=self.v1.read(self.source['resolved_case_path'])['case']
        contract=self.v1.read(self.source['frame_files']['scene_contract.json']['path'])
        self.runner,resolved,_=scene_runtime.build_current_runner(payload,declaration['runtime_project_root'],Path(self.previous['source_urdf']).parent.parent)
        if resolved.environment.runner!='grasp_failure_prediction.evaluation.runner:ActuatedShadowRunner':raise ValueError('Wrong current scene')
        scene_runtime.reset_captured_scene(self.runner,payload,contract)
        model,data=self.runner.model,self.runner.data
        source_xml=xml_reference.read_urdf(self.previous['source_urdf'])
        validator=pose_validation.ShadowPoseValidator(Path(self.previous['source_urdf']).parent.parent)
        grasp=load_hug_prediction(Path(self.source['native']['path']))
        mapped=retargeting.RetargetedHandPose(self.names,self.q0,np.zeros((21,3)),np.zeros((10,3)),np.asarray(native['T_camera_wrist']))
        pos,quat=case_runner._hug_world_palm_pose(grasp,mapped,validator,Path(self.source['native']['path']).parent)
        fixed=xml_reference.forward_xml(source_xml,self.names,self.q0)
        expected=native_reference.expected_native_palm(camera,native['T_camera_wrist'],fixed['palm'])['palm_world']
        actual=xml_reference.transform(pos,xml_reference.quaternion_matrix_wxyz(quat))
        if np.max(np.abs(actual-expected))>1e-12:raise ValueError('Independent native/world/palm frame check failed')
        rootpos,rootquat,_=scene_runtime.stage_named_palm(self.runner,self.names,self.q0,actual)
        self.original_root_position=rootpos.copy();self.original_root_quaternion=rootquat.copy()
        domain_source=self.v1.read(declaration['domain_source'])
        domain=next(c for c in domain_source['cases'] if c['source_inventory_index']==index)
        if not np.array_equal(self.q0,domain['original_qpos_rad']) or domain['native_sha256']!=self.native_hash:
            raise ValueError('Bound original translation-domain input changed')
        self.palm_bounds=np.asarray(domain['domain']['world_displacement_AABB_m'],float)
        self.adapter=SurfacePoseAdapter(model,data,joint_names=self.names,source_urdf=self.previous['source_urdf'],
            original_qpos_rad=self.q0,native_tip_targets_world_m=self.native_targets,original_tip_caps_m=self.caps,
            palm_delta_bounds_m=self.palm_bounds,joint_delta_limit_rad=.35)
        self.bounds=self.adapter.scaled_bounds()
        self.original_payload=dict(schema_version='shadow_current_static_named_root_pose_v1',
            joint_targets_rad=dict(zip(self.names,self.q0.tolist(),strict=True)),root_position_m=rootpos.tolist(),
            root_quaternion_wxyz=rootquat.tolist(),source_urdf_sha256=self.adapter.source_urdf_sha256,
            model_geometry_sha256=self.adapter.model_geometry_sha256)
        self.a=load('repair_geometry_base',snapshots/'audit_other_surfaces.py')
        if guard_reinstall is not None:guard_reinstall()
        kernel=load('repair_frozen_explicit_geometry',declaration['geometry_kernel'])
        if guard_reinstall is not None:guard_reinstall()
        kernel.install(self.a)
        if Path(self.a.URDF).resolve()!=Path(self.previous['source_urdf']).resolve():raise ValueError('Geometry source hand differs')
        self.a.load_case=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('Historical target-hand constructor forbidden'))
        self.direction=load('repair_exact_direction',declaration['exact_direction_source'])
        self.direction.a=self.a;self.direction.o.a=self.a
        self.core=load('repair_literal_constraints',declaration['constraint_core_source'])
        if guard_reinstall is not None:guard_reinstall()
        import numba
        numba.set_num_threads(2)
        if numba.get_num_threads()!=2:raise ValueError('Fixed thread budget changed')
        obj=self.runner._object_body_id
        self.object_position=data.xpos[obj].copy();self.object_rotation=data.xmat[obj].reshape(3,3).copy()
        # Immutable object visual criterion reference only. No historical hand
        # collider, pose loader or optimization Case is queried or instantiated.
        old=mujoco.MjModel.from_binary_path(self.source['scene_mjb']['path']);od=mujoco.MjData(old)
        with np.load(self.source['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:od.qpos[:]=z['qpos']
        od.qvel[:]=0.;mujoco.mj_forward(old,od)
        ob=self.a.ident(old,mujoco.mjtObj.mjOBJ_BODY,'object');opos=od.xpos[ob];oR=od.xmat[ob].reshape(3,3)
        if np.max(np.abs(opos-self.object_position))>1e-12 or np.max(np.abs(oR-self.object_rotation))>1e-12:
            raise ValueError('Frozen object-reference frame differs from current captured object')
        visuals=[{'id':g} for g in range(old.ngeom) if witness_check.descendant(old.body_parentid,int(old.geom_bodyid[g]),ob)
                 and not (old.geom_contype[g] or old.geom_conaffinity[g])]
        if len(visuals)!=1:raise ValueError('Exactly one frozen object visual reference expected')
        self.visual_tri,_,self.visual_metadata=self.direction.o.visual_mesh({'disabled_visual_geoms':visuals},old,od,opos,oR)
        del old,od
        saved=self.v1.read(declaration['orientation_saved_witnesses'])
        self.ori_rows=[r for r in saved['orientation_rows'] if r['source_inventory_index']==index]
        if not self.ori_rows or any(r['native_sha256']!=self.native_hash for r in self.ori_rows):
            raise ValueError('Missing/mismatched exact orientation witness protection; no repair admitted')
        self.orientation_protocol=self.v1.read(declaration['orientation_protocol'])
        self.oq=np.asarray([r['row']['query_object_m'] for r in self.ori_rows])
        self.on=np.asarray([r['row']['object_outward_normal'] for r in self.ori_rows])
        self.op=np.asarray([self.a.PARTS.index(r['row']['native_part']) for r in self.ori_rows])
        self.osign=np.sign([r['row']['native_facing_compatibility'] for r in self.ori_rows])
        self.oband=np.asarray([r['band_mm']/1000. for r in self.ori_rows])
        with np.load(self.grid['original_cache'],allow_pickle=False) as old, np.load(self.grid['native_cache'],allow_pickle=False) as native_cache:
            self.query=old['query_object_m'].copy();self.radius=old['cell_radius_m'].copy();self.area=old['area_mm2'].copy()
            self.native_distance=native_cache['native_distance_by_part_m'][:,:6].copy()
            if not np.array_equal(self.query,native_cache['query_object_m']):raise ValueError('Native/cell order changed')
        self.native_fields=self.v1.classifications(self.native_distance,self.radius)
        self.original_hand=self._hand_from_current()
        self.source_receipt=dict(native_sha256=self.native_hash,source_urdf_sha256=self.adapter.source_urdf_sha256,
            model_geometry_sha256=self.adapter.model_geometry_sha256,baseline_state_sha256=self.adapter.baseline_state_sha256,
            original_qpos_rad=self.q0,original_tip_caps_m=self.caps,original_root_position_m=rootpos,
            original_root_quaternion_wxyz=rootquat,palm_delta_bounds_m=self.palm_bounds,scaled_bounds=self.bounds,
            query_count=len(self.query),native_definite_relationships=self.grid['required_relationships'],
            object_visual_reference='bound historical object-only visual triangles; no historical target-hand geometry',
            target_hand_geometry='current registered ActuatedShadowRunner via SurfacePoseAdapter',
            native_targets_rebased=False,historical_caps_rebased=False)
        self._last_full=None

    def _raw_budget(self):
        self.counts['raw_callbacks']+=1
        if self.counts['raw_callbacks']>1500:raise RuntimeError('1500 raw-callback budget exceeded')

    def cheap(self,x):
        self._raw_budget();self.counts['cheap_callbacks']+=1
        return self.adapter.stage(x)

    def tip_jacobian(self,x):
        self._raw_budget();self.counts['tip_jacobians']+=1
        return self.adapter.tip_jacobian(x)

    def _hand_from_current(self):
        m,d=self.adapter.model,self.adapter.data
        root=self.a.ident(m,self.mujoco.mjtObj.mjOBJ_BODY,'forearm');hand=[]
        for g in range(m.ngeom):
            b=int(m.geom_bodyid[g])
            if self.witness.descendant(m.body_parentid,b,root) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                if m.geom_margin[g]!=0 or m.geom_gap[g]!=0:raise ValueError('Hand margin/gap changed')
                rep=self.a.geom_rep(m,d,g,self.object_rotation,self.object_position)
                rep['part']=self.witness.anatomical_part(rep['body']);hand.append(rep)
        if len(hand)!=45:raise ValueError('Current hand collider inventory changed')
        return hand

    def _hand(self):
        d=self.adapter.data;m=self.adapter.model
        if d.time!=0 or np.max(np.abs(d.xpos[self.runner._object_body_id]-self.object_position))>1e-12 or np.max(np.abs(d.xmat[self.runner._object_body_id].reshape(3,3)-self.object_rotation))>1e-12:
            raise ValueError('Static object frame/time changed')
        hand=[]
        for original in self.original_hand:
            g=original.copy();gid=g['id']
            g['R']=self.object_rotation.T@d.geom_xmat[gid].reshape(3,3)
            g['pos']=(d.geom_xpos[gid]-self.object_position)@self.object_rotation
            hand.append(g)
        return hand

    def _part_distances(self,hand,points,part):
        result=np.full(len(points),np.inf)
        for g in hand:
            if g['part']!=part:continue
            signed,_,_=self.a.geom_query(g,points)
            if not np.isfinite(signed).all():raise ValueError('Nonfinite actual collider distance')
            result=np.minimum(result,np.abs(signed));self.counts['proximity_collider_point_queries']+=len(points)
        if not np.isfinite(result).all():raise ValueError('Missing actual anatomical distance')
        return result

    def _orientation_current(self,hand):
        self.counts['orientation_checks']+=1
        fields=self.direction.mapped_direction_fields(self.oq,self.on,hand,self.visual_tri,self.orientation_protocol,expected_part=self.op)
        if not np.isfinite(fields['score']).all() or not np.isfinite(fields['distance_m']).all():
            raise ValueError('Finite exact orientation values required even for invalid sites')
        p=self.orientation_protocol
        result=self.core.orientation_constraints(fields['score'],fields['distance_m'],self.osign,self.oband,
            p['facing_dot_numerical_deadzone'],p['query_boundary_tolerance_m'],fields['valid'])
        return dict(valid=fields['valid'].copy(),passed=result['passed'].copy(),score=fields['score'].copy(),
                    distance_m=fields['distance_m'].copy(),reason_masks=fields['reason_masks'],
                    native_sign=self.osign.copy(),full_orientation_coverage='NOT_PROVED',retained_rows=len(self.ori_rows))

    def orientation(self,x):
        self._raw_budget();self.adapter.stage(x)
        return self._orientation_current(self._hand())

    def raw(self,x,active):
        self._raw_budget();state=self.adapter.stage(x);hand=self._hand()
        cell,part,band=np.asarray(active,int).T
        distances=np.empty(len(cell))
        for j,name in enumerate(self.v1.PARTS):
            ids=np.flatnonzero(part==j)
            if len(ids):distances[ids]=self._part_distances(hand,self.query[cell[ids]],name)
        error=self.radius[cell]+self.v1.ALLOWANCE
        b=np.asarray(self.v1.BANDS)[band]/1000.
        near=self.native_fields['near'][cell,part,band]
        margin=np.where(near,b-error-distances,distances-error-b)
        return dict(surface_margin=margin/.001,task_error_vectors_m=state.tip_errors_world_m,state=state)

    def _full_current(self,state,label):
        self.counts['full_field_checkpoints']+=1
        if self.counts['full_field_checkpoints']>5:raise RuntimeError('Five full-field checkpoint budget exceeded')
        hand=self._hand();distance=np.empty((len(self.query),6))
        for j,name in enumerate(self.v1.PARTS):
            for start in range(0,len(self.query),256):
                distance[start:start+256,j]=self._part_distances(hand,self.query[start:start+256],name)
        fields,summary=self.v1.field_status(self.native_distance,distance,self.radius)
        if int(fields['required'].sum())!=self.grid['required_relationships']:raise ValueError('Full relationship denominator changed')
        error=self.radius[:,None,None]+self.v1.ALLOWANCE;b=np.asarray(self.v1.BANDS)[None,None,:]/1000.
        margin=np.where(fields['native_near'],b-error-distance[:,:,None],distance[:,:,None]-error-b)
        ori=self._orientation_current(hand)
        prefix=f"checkpoint_{self.counts['full_field_checkpoints']:02d}_{label}"
        path=self.output/(prefix+'.npz')
        with path.open('xb') as f:np.savez_compressed(f,distance_m=distance,margin_m=margin,
            orientation_valid=ori['valid'],orientation_passed=ori['passed'],orientation_score=ori['score'],
            orientation_distance_m=ori['distance_m'],**fields)
        result=dict(fields=fields,margin_m=margin,distance_m=distance,state=state,orientation=ori,
            summary=summary,array_path=str(path),array_sha256=self.v1.sha(path),label=label)
        self._last_full=result
        return result

    def original_full(self):
        self._raw_budget();state=self.adapter.stage_serialized(self.original_payload)
        if not np.array_equal(state.joint_qpos_rad,self.q0) or not np.array_equal(state.root_position_m,self.original_root_position) or not np.array_equal(state.root_quaternion_wxyz,self.original_root_quaternion):
            raise ValueError('Original-current baseline must preserve literal named q/root')
        full=self._full_current(state,'literal_original_baseline')
        self.baseline_readback=self._compare_original_full_reference(full)
        self.source_receipt['original_full_field_readback']=self.baseline_readback
        return full

    def _compare_original_full_reference(self,full):
        """Detect staging drift without relaxing any scientific predicate/cap."""
        path=Path(self.p['v1_protocol']).parent/'run_v1'/str(self.index)/f'original_{self.index}_density2mm.npz'
        digest=self.p['bindings'].get(str(path.resolve()))
        if digest is None or self.v1.sha(path)!=digest:
            raise ValueError('Original full-field reference must be explicitly source-bound')
        with np.load(path,allow_pickle=False) as old:
            for name,actual in (('query_object_m',self.query),('cell_radius_m',self.radius),
                                ('native_distance_by_part_m',self.native_distance)):
                if not np.array_equal(actual,old[name]):raise ValueError('Original full-field input changed '+name)
            for name,actual in full['fields'].items():
                if not np.array_equal(actual,old[name]):raise ValueError('Original full-field classification changed '+name)
            reference=old['current_distance_by_part_m']
            if reference.shape!=full['distance_m'].shape or not np.isfinite(reference).all():
                raise ValueError('Original full-field distance reference malformed')
            discrepancy=float(np.max(np.abs(full['distance_m']-reference)))
            if discrepancy>1e-12:raise ValueError('Original full-field numerical distance comparison exceeded 1e-12 m')
        return dict(reference_path=str(path),reference_sha256=digest,all_classification_arrays_exact=True,
                    max_distance_discrepancy_m=discrepancy,numerical_comparison_limit_m=1e-12,
                    scientific_predicate_or_tip_cap_allowance=False)

    def full(self,x,label):
        self._raw_budget();state=self.adapter.stage(x)
        return self._full_current(state,label)

    def finalize(self,result):
        state=result['returned']['state'];payload=state.serialize_pose()
        replay=self.adapter.stage_serialized(payload)
        for name in ('joint_qpos_rad','root_position_m','root_quaternion_wxyz','tip_error_norms_m'):
            if not np.array_equal(getattr(state,name),getattr(replay,name)):
                raise ValueError('Literal final serialized current-scene readback changed '+name)
        return dict(serialized_pose=payload,literal_readback_pass=True,counts=dict(self.counts),
                    adapter_forward_calls=self.adapter.forward_calls,source_receipt=self.source_receipt,
                    full_strict_mapping_accepted=False,no_new_HUG_inference_or_physics=True)
