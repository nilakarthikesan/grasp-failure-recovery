"""Frozen geometry-only engine qualification over the sealed109-state grid."""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SKIN=ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1'
FULL=ROOT/'research_artifacts/manosim_full_surface_reference_mj340_2026-10-15_v2'
BUDGET=dict(maximum_seconds_per_case=600,model_compilations=1,data_constructions=1,
    mujoco_forward_calls=109,continuous_skin_evaluations=109,direct_mujoco_flex_calls=109,
    physics_steps=0,object_geometry_queries=0,actuator_commands=0)
GATES=dict(engine_continuous64_vertex_comparison_m=1e-12,engine_continuous64_normal_component_comparison=1e-10,
    original_source32_vertex_comparison_m=5e-6,original_source32_normal_component_comparison=1e-6,
    original_native_baseline_vertex_comparison_m=5e-6,original_native_baseline_normal_component_comparison=1e-6,
    original_joint_anchor_comparison_m=5e-6,original_rotation_component_comparison=1e-6,
    numerical_limits_are_contact_allowances=False)
FORBIDDEN=dict(physics_integration_attempts=0,object_geometry_query_attempts=0,
    HUG_generator_attempts=0,source_MANO_evaluation_attempts=0,retarget_attempts=0,training_attempts=0)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def declaration():
    """Saved JSON metadata and paths only; never builds XML, skin or models."""
    p=read(SKIN/'protocol.json');priorfull=read(FULL/'protocol.json');cases={}
    for i in (20,28):
        row=read(SKIN/f'run_v1/{i}/result.json')
        artifacts={Path(a['path']).name:a for a in row['artifacts']}
        fullcase=read(FULL/f'run_v1/{i}/result.json')
        include=next(a for a in fullcase['artifacts'] if a['label']=='single_palm_complete_face_flex_include')
        cases[str(i)]=dict(source_inventory_index=i,base_xml=p['cases'][str(i)]['base_xml'],
            old_include=include,sealed_states=artifacts['all109_skin_states.npz'],
            literal_buffers=artifacts['literal_source_skin_buffers.npz'],
            original_compiled_readback=p['cases'][str(i)]['compiled_readback'],
            shape_asset=p['cases'][str(i)]['shape_asset'],expected_shape_sha256=p['cases'][str(i)]['expected_shape_sha256'],
            saved_named_joint_addresses=p['cases'][str(i)]['saved_named_joint_addresses'])
    inputs=[HERE/'skin_engine_bridge.py',HERE/'bridge_case.py',HERE/'run_bridge.py',
        HERE/'test_skin_engine_bridge.py',HERE/'test_run_bridge.py',HERE/'README.md',
        SKIN/'skin_kinematics.py',SKIN/'skin_case.py']
    return dict(budget=BUDGET,gates=GATES,cases=cases,source_inputs=[str(x) for x in inputs],
        motion_grid=p['motion_grid'],runtime_skin=p['runtime_skin'],
        condition='contact_and_actuation_disabled_read_q_skin_geometry_only_v1',
        skin_precision_profile='explicit_float64_of_literal_source_float32_buffers',
        expected_internal_mj_flex_calls_within109_forwards=109,
        expected_total_flex_position_updates_including_direct_calls=218)


def run_case(spec,output_dir,protocol,checkpoint_callback):
    import numpy as np
    import torch
    import mujoco
    if mujoco.__version__!='3.4.0' or np.__version__!='2.2.6' or torch.__version__!='2.11.0':
        raise ValueError('Original isolated runtime precision condition required')
    if protocol['budget']!=BUDGET or protocol['gates']!=GATES:raise ValueError('Frozen bridge budget/gate drift')
    bridge=load('frozen_read_q_engine_geometry',HERE/'skin_engine_bridge.py')
    core=load('frozen_qualified_complete_skin',SKIN/'skin_kinematics.py')
    case=load('frozen_qualified_skin_input_map',SKIN/'skin_case.py')
    if protocol['motion_grid']!=core.motion_grid_spec():raise ValueError('Original sealed motion grid changed')
    torch.set_num_threads(1);out=Path(output_dir);artifacts=[]
    counts={key:0 for key in BUDGET if key!='maximum_seconds_per_case'}
    def count(name):
        if counts[name]>=BUDGET[name]:raise RuntimeError('Before-work bridge budget exhausted: '+name)
        counts[name]+=1;checkpoint_callback(dict(event='operation',name=name,amount=1,counts=dict(counts)))
    def seal(path,label):
        row=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label)
        artifacts.append(row);checkpoint_callback(dict(event='artifact',artifact=row));return row
    def arrays(row):
        if sha(row['path'])!=row['sha256'] or Path(row['path']).stat().st_size!=row['bytes']:raise ValueError('Sealed input changed')
        with np.load(row['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def save(name,label,**a):
        path=out/name
        if path.exists():raise ValueError('Write-once array already exists')
        np.savez_compressed(path,**a);return seal(path,label)
    sealed=arrays(spec['sealed_states']);b=arrays(spec['literal_buffers']);original=arrays(spec['original_compiled_readback'])
    faces=sealed['faces'];qgrid=sealed['actual_integrated_named_qpos']
    if (qgrid.shape!=(109,67) or faces.shape!=(1552,3)
        or not np.array_equal(sealed['pose_ids'],np.asarray([r['pose_id'] for r in protocol['motion_grid']['poses']]))):
        raise ValueError('All original states/faces in exact order required')
    shape=np.load(spec['shape_asset']['path'],allow_pickle=False)
    if (sha(spec['shape_asset']['path'])!=spec['expected_shape_sha256'] or shape.dtype!=np.dtype('float32')
        or shape.shape!=(1,10) or not np.array_equal(shape,b['shape']) or not np.array_equal(shape,sealed['shape'])):
        raise ValueError('Literal originalshape(1,10) must remain exact')
    if sha(spec['old_include']['path'])!=spec['old_include']['sha256']:raise ValueError('Frozen prior flex include changed')
    xml,assembly=bridge.scene_xml(spec['base_xml'],spec['old_include']['path'])
    xmlpath=out/'geometry_only_scene.xml';xmlpath.write_text(xml);seal(xmlpath,'new_condition_old_skeleton_unchanged_all_contacts_actuation_disabled')
    count('model_compilations');model=mujoco.MjModel.from_xml_path(str(xmlpath))
    layout=case.named_model_contract(model,mujoco,spec['saved_named_joint_addresses'])
    count('data_constructions');data=mujoco.MjData(model)
    owner=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,bridge.OWNER))
    flex=int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_FLEX,bridge.FLEX))
    bodyids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'mano_body_{j}') for j in range(16)],dtype=int)
    jointids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in ['wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]],dtype=int)
    if min(bodyids)<1 or min(jointids)<0 or owner<1 or flex!=0:raise ValueError('Actual named anatomy/flex missing')
    if (not np.array_equal(model.body_mass,original['body_mass']) or not np.array_equal(model.body_inertia,original['body_inertia'])):
        raise ValueError('Original skeleton mass/inertia changed')
    rest=np.asarray(layout['body_rest_quaternions_wxyz'])
    arrays64={name:torch.from_numpy(b[name].copy()).double() for name in core.ARRAY_KEYS}
    beta=torch.from_numpy(shape.copy()).double()
    def evaluate(qpos):
        t,q=case.body_inputs(qpos,rest,core)
        state=core.skin_state(core.quaternion_matrices(q),t,beta,arrays64)
        return {name:value.detach().numpy() for name,value in state.items()}
    updater=bridge.ArticulatedSkinUpdater(model,data,owner_body_id=owner,flex_id=flex,faces=faces,
        evaluate_skin=evaluate,flex_update=mujoco.mj_flex,
        before_skin=lambda:count('continuous_skin_evaluations'),before_flex=lambda:count('direct_mujoco_flex_calls'),
        contact_disable_bit=mujoco.mjtDisableBit.mjDSBL_CONTACT,actuation_disable_bit=mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    original_model_hashes=bridge.model_array_hashes(model)
    data.qvel[:]=original['literal_qvel'];data.ctrl[:]=original['retained_ctrl'];data.time=0.
    velocity0=data.qvel.copy();ctrl0=data.ctrl.copy();warnings0=np.array([w.number for w in data.warning],dtype=int)
    if np.any(warnings0):raise ValueError('Fresh model warning counters must be zero')
    actual_vertices=[];actual_normals=[];actual_anchors=[];actual_rotations=[];evaluated=[];qread=[];ownerframes=[]
    source_normal_errors=[];continuous_normal_errors=[];continuous_vertex_errors=[];source_vertex_errors=[]
    anchor_errors=[];rotation_errors=[];triangle_conditions=[]
    for k,pose in enumerate(protocol['motion_grid']['poses']):
        # The fixture alone sets a declared static state. The updater is readonly.
        data.qpos[:]=qgrid[k]
        count('mujoco_forward_calls');mujoco.mj_forward(model,data)
        if (not np.array_equal(data.qpos,qgrid[k]) or not np.array_equal(data.qvel,velocity0)
            or not np.array_equal(data.ctrl,ctrl0) or data.time!=0 or data.ncon!=0
            or np.any(data.qfrc_actuator) or not np.isfinite(data.qacc).all()
            or not np.array_equal(np.array([w.number for w in data.warning]),warnings0)):
            raise ValueError('Static geometry forward state/contact/actuation/warning integrity failure')
        value=updater.update();skin=value['skin_state'];world=value['compiled_vertices_world_m'];normal=value['compiled_face_normals_world']
        if not np.array_equal(np.array([w.number for w in data.warning]),warnings0) or data.ncon!=0:
            raise ValueError('Direct flex update changed warnings/contact count')
        ref64=sealed['candidate_float64_vertices_world_m'][k];ref32=sealed['source_float32_vertices_world_m'][k].astype(np.float64)
        normal64,cond64=bridge.oriented_face_normals(ref64,faces);normal32,cond32=bridge.oriented_face_normals(ref32,faces)
        anchors=data.xanchor[jointids].copy();rotations=data.xmat[bodyids].reshape(16,3,3).copy()
        for a,sh in ((anchors,(16,3)),(rotations,(16,3,3))):bridge.finite(a,sh,'actual compiled anatomy')
        actual_vertices.append(world.copy());actual_normals.append(normal.copy());actual_anchors.append(anchors);actual_rotations.append(rotations)
        evaluated.append(skin['vertices'].copy());qread.append(data.qpos.copy())
        frame=np.eye(4);frame[:3,:3]=value['actual_owner_rotation'];frame[:3,3]=value['actual_owner_position'];ownerframes.append(frame)
        continuous_vertex_errors.append(float(np.max(np.abs(world-ref64))));source_vertex_errors.append(float(np.max(np.abs(world-ref32))))
        continuous_normal_errors.append(float(np.max(np.abs(normal-normal64))));source_normal_errors.append(float(np.max(np.abs(normal-normal32))))
        anchor_errors.append(float(np.max(np.abs(anchors-sealed['candidate_float64_joint_anchors_world_m'][k]))))
        rotation_errors.append(float(np.max(np.abs(rotations-skin['absolute_rotations']))))
        triangle_conditions.append([value['triangle_condition_diagnostic']['maximum_edge_over_double_area_per_m'],cond64['maximum_edge_over_double_area_per_m'],cond32['maximum_edge_over_double_area_per_m']])
        checkpoint_callback(dict(event='progress',state_id=pose['pose_id'],completed_states=k+1))
    save('all109_engine_surface_readbacks.npz','all_fixed_states_actual_compiled_skin_normals_anchors_and_profiles',
        compiled_vertices_world_m=np.asarray(actual_vertices),compiled_oriented_face_normals_world=np.asarray(actual_normals),
        compiled_joint_anchors_world_m=np.asarray(actual_anchors),compiled_body_rotations_world=np.asarray(actual_rotations),
        actual_read_named_qpos=np.asarray(qread),evaluated_continuous64_vertices_world_m=np.asarray(evaluated),
        actual_owner_transforms_world=np.asarray(ownerframes),faces=faces,pose_ids=sealed['pose_ids'],shape=shape,
        source32_vertex_errors_m=np.asarray(source_vertex_errors),continuous64_vertex_errors_m=np.asarray(continuous_vertex_errors),
        source32_normal_component_errors=np.asarray(source_normal_errors),continuous64_normal_component_errors=np.asarray(continuous_normal_errors),
        joint_anchor_errors_m=np.asarray(anchor_errors),joint_rotation_component_errors=np.asarray(rotation_errors),
        triangle_condition_diagnostics=np.asarray(triangle_conditions),
        geom_contype=model.geom_contype.copy(),geom_conaffinity=model.geom_conaffinity.copy(),
        flex_contype=model.flex_contype.copy(),flex_conaffinity=model.flex_conaffinity.copy(),
        body_mass=model.body_mass.copy(),body_inertia=model.body_inertia.copy(),disableflags=np.asarray(model.opt.disableflags),
        actual_joint_qpos_addresses=model.jnt_qposadr.copy(),actual_joint_dof_addresses=model.jnt_dofadr.copy(),
        actual_joint_types=model.jnt_type.copy(),retained_qvel=velocity0,retained_ctrl=ctrl0,
        warning_counts=np.array([w.number for w in data.warning]))
    if bridge.model_array_hashes(model)!=original_model_hashes:raise ValueError('Nonflex model array changed during fixture')
    if counts!={k:v for k,v in BUDGET.items() if k!='maximum_seconds_per_case'}:raise ValueError('Exact fixed operation quotas incomplete')
    baseline_normal,_=bridge.oriented_face_normals(sealed['original_native_target_world_m'],faces)
    baseline_vertex_error=float(np.max(np.abs(actual_vertices[0]-sealed['original_native_target_world_m'])))
    baseline_normal_error=float(np.max(np.abs(actual_normals[0]-baseline_normal)))
    bridge_ok=(max(continuous_vertex_errors)<=GATES['engine_continuous64_vertex_comparison_m'] and
        max(continuous_normal_errors)<=GATES['engine_continuous64_normal_component_comparison'] and
        max(anchor_errors)<=GATES['original_joint_anchor_comparison_m'] and max(rotation_errors)<=GATES['original_rotation_component_comparison'])
    source_ok=(max(source_vertex_errors)<=GATES['original_source32_vertex_comparison_m'] and
        max(source_normal_errors)<=GATES['original_source32_normal_component_comparison'])
    baseline_ok=(baseline_vertex_error<=GATES['original_native_baseline_vertex_comparison_m'] and
        baseline_normal_error<=GATES['original_native_baseline_normal_component_comparison'])
    return dict(status='COMPLETED',counts=counts,forbidden_counts=FORBIDDEN,artifacts=artifacts,
        geometry_only_bridge_check_passed=bridge_ok,strict_source32_vertex_and_normal_check_passed=source_ok,
        original_native_baseline_vertex_and_normal_check_passed=baseline_ok,
        source32_comparison_status='NUMERICAL_PASS' if source_ok else 'NUMERICAL_FAIL',
        original_native_baseline_comparison_status='NUMERICAL_PASS' if baseline_ok else 'NUMERICAL_FAIL',
        metrics=dict(maximum_engine_continuous64_vertex_error_m=max(continuous_vertex_errors),
            maximum_engine_continuous64_normal_component_error=max(continuous_normal_errors),
            maximum_engine_source32_vertex_error_m=max(source_vertex_errors),maximum_engine_source32_normal_component_error=max(source_normal_errors),
            maximum_joint_anchor_error_m=max(anchor_errors),maximum_joint_rotation_component_error=max(rotation_errors),
            original_native_baseline_vertex_error_m=baseline_vertex_error,original_native_baseline_normal_component_error=baseline_normal_error,
            maximum_triangle_condition_per_m=float(np.max(triangle_conditions))),
        scene_assembly=assembly,named_layout=layout,nonflex_model_array_hashes=original_model_hashes,
        direct_mj_flex_calls=109,expected_internal_flex_calls_within_forward=109,
        static_state_fixture_sets_qpos_only=True,updater_never_writes_qpos_qvel_ctrl=True,
        field_queries_status='NOT_RUN',contact_orientation_over_motion_status='NOT_RUN',physical_contact_status='NOT_RUN',
        stock_flex_contact_forces_permitted=False,geometry_condition_physics_disabled=True,
        accepted_mapping=False,native_contact_qualified=False,new_collection_or_training_allowed=False,physical_outcome=None)
