"""Exact pinned HUG/MANO camera-float32 graph in a new source-state domain."""
from __future__ import annotations
import hashlib,importlib.util,json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BRIDGE=ROOT/'research_artifacts/manosim_skin_engine_bridge_2026-10-15_v2'
SKIN=ROOT/'research_artifacts/manosim_articulated_skin_2026-10-15_v1'
BUDGET=dict(maximum_seconds_per_case=600,coordinate_preflight_states=109,model_compilations=1,data_constructions=1,
    source_layer_constructions=1,source_forward_calls=109,mujoco_forward_calls=109,direct_mujoco_flex_calls=109,
    physics_steps=0,object_geometry_queries=0,actuator_commands=0)
GATES=dict(engine_source_vertex_comparison_m=1e-12,engine_source_normal_component_comparison=1e-10,
    original_native_baseline_vertex_comparison_m=5e-6,original_native_baseline_normal_component_comparison=1e-6,
    original_joint_anchor_comparison_m=5e-6,original_rotation_component_comparison=1e-6,
    float32_axisangle_rotation_roundtrip_comparison=1e-6,numerical_limits_are_contact_allowances=False)
FORBIDDEN=dict(physics_integration_attempts=0,object_geometry_query_attempts=0,HUG_generator_attempts=0,
    retarget_attempts=0,training_attempts=0,old64_jacobian_force_application_attempts=0)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def declaration():
    p=read(BRIDGE/'protocol.json');skinp=read(SKIN/'protocol.json');cases={}
    for i in (20,28):
        old=p['cases'][str(i)];sp=skinp['cases'][str(i)];sr=read(SKIN/f'run_v1/{i}/result.json')
        cases[str(i)]=dict(source_inventory_index=i,base_xml=old['base_xml'],old_include=old['old_include'],
            decoded_reference=sp['decoded_reference'],shape_asset=old['shape_asset'],expected_shape_sha256=old['expected_shape_sha256'],
            original_compiled_readback=old['original_compiled_readback'],saved_named_joint_addresses=old['saved_named_joint_addresses'],
            saved_rest_quaternions=sr['simulator_state_binding']['body_rest_quaternions_wxyz'],
            mano_assets_root=sp['mano_assets_root'],dependency_dirs=sp['dependency_dirs'])
    sources=[HERE/n for n in ('source_coordinates.py','source_case.py','run_source_pipeline.py','test_source_coordinates.py','test_source_case.py','test_run_source_pipeline.py','README.md','inspect_old_gs_coordinates.py','old_gs_coordinate_diagnosis.json')]
    sources += [BRIDGE/'skin_engine_bridge.py',SKIN/'skin_case.py']
    return dict(budget=BUDGET,gates=GATES,cases=cases,source_inputs=[str(x) for x in sources],
        motion_grid=skinp['motion_grid'],runtime_skin=skinp['runtime_skin'],
        state_domain='mano_source_postAA32_exp64_camera64_named_state_v1',
        operational_geometry_profile='pinned_ManoLayer_AA32_beta32_then_t32_then_cameraWorld64_v1',
        previous_GS64_states_and_profiles_reused_as_new_targets=False,
        old64_jacobian_qualified_for_this_profile=False,expected_internal_flex_calls_within_forward=109)


def require_motion_grid(value):
    """Exact frozen 109 delta/ID metadata, not the old GS64 absolute states."""
    original=read(SKIN/'protocol.json')['motion_grid']
    if value!=original:raise ValueError('Original finite109 delta grid/IDs must remain exact')
    return True


def evaluate_pinned_source(layer,qpos,rest,camera,original_aa32,beta32,coordinates,torch,before_source):
    """One actual pinned source forward; source t addition remains Torch32."""
    import numpy as np
    recovered=coordinates.recover_source_inputs(qpos,rest,camera,original_aa32)
    if np.max(recovered['float32_axisangle_rotation_roundtrip_errors'])>GATES['float32_axisangle_rotation_roundtrip_comparison']:
        raise ValueError('Source input coordinate rotation roundtrip failed')
    aa=torch.from_numpy(recovered['axis_angles_float32'][None].copy())
    t=torch.from_numpy(recovered['translation_camera_float32'].copy())
    before_source()
    with torch.no_grad():out=layer(aa,beta32)
    vertices_wrist32=out.verts[0]
    translated32=vertices_wrist32+t
    transforms=out.transforms_abs[0]
    anchors_camera32=transforms[:,:3,3]+t
    vertices=coordinates.world_from_camera32(translated32.detach().numpy(),camera)
    anchors=coordinates.world_from_camera32(anchors_camera32.detach().numpy(),camera)
    sourceR=transforms[:,:3,:3].detach().numpy()
    rotations=np.einsum('ij,njk->nik',camera[:3,:3],sourceR,optimize=False)
    return dict(vertices=vertices,joint_anchors_world=anchors,absolute_rotations=rotations,
        source_record=recovered,source_wrist_vertices32=vertices_wrist32.detach().numpy(),
        translated_camera_vertices32=translated32.detach().numpy(),source_camera_rotations32=sourceR,
        translated_camera_anchors32=anchors_camera32.detach().numpy())


def run_case(spec,output_dir,protocol,checkpoint_callback):
    import numpy as np
    import torch
    import mujoco
    if (mujoco.__version__,np.__version__,torch.__version__)!=('3.4.0','2.2.6','2.11.0'):raise ValueError('Frozen source/runtime required')
    if protocol['budget']!=BUDGET or protocol['gates']!=GATES:raise ValueError('Source budget/gate changed')
    coord=load('frozen_native_source_coordinates',HERE/'source_coordinates.py')
    bridge=load('frozen_contact_disabled_geometry_updater',BRIDGE/'skin_engine_bridge.py')
    oldcase=load('frozen_named_state_metadata',SKIN/'skin_case.py')
    if protocol['state_domain']!=coord.DOMAIN or protocol['previous_GS64_states_and_profiles_reused_as_new_targets'] is not False:
        raise ValueError('New source-state domain must remain distinct')
    require_motion_grid(protocol['motion_grid'])
    torch.set_num_threads(1);outdir=Path(output_dir);artifacts=[]
    counts={k:0 for k in BUDGET if k!='maximum_seconds_per_case'}
    def count(name):
        if counts[name]>=BUDGET[name]:raise RuntimeError('Before-work source budget exhausted '+name)
        counts[name]+=1;checkpoint_callback(dict(event='operation',name=name,amount=1,counts=dict(counts)))
    def seal(path,label):
        row=dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label)
        artifacts.append(row);checkpoint_callback(dict(event='artifact',artifact=row));return row
    def save(name,label,**arrays):
        path=outdir/name
        if path.exists():raise ValueError('Write-once artifact already exists')
        np.savez_compressed(path,**arrays);return seal(path,label)
    def arrays(row):
        if sha(row['path'])!=row['sha256'] or Path(row['path']).stat().st_size!=row['bytes']:raise ValueError('Bound source input changed')
        with np.load(row['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    decoded=arrays(spec['decoded_reference']);old=arrays(spec['original_compiled_readback'])
    aa32=coord.array(decoded['actual_full_poses'],(48,),'float32','original source post-AA')
    t32=coord.array(decoded['params_translation_m'],(3,),'float32','original source translation')
    camera=coord.array(decoded['camera_world'],(4,4),'float64','original camera')
    rest=np.asarray(spec['saved_rest_quaternions'],dtype=np.float64)
    shape=np.load(spec['shape_asset']['path'],allow_pickle=False)
    if sha(spec['shape_asset']['path'])!=spec['expected_shape_sha256'] or shape.dtype!=np.dtype('float32') or shape.shape!=(1,10) or shape.tobytes()!=decoded['shape'].tobytes():raise ValueError('Literal originalshape32 row required')
    # No model/source constructor or forward is allowed before this guard.
    q0,initialization=coord.source_initialized_qpos(aa32,t32,camera,rest)
    qgrid=coord.motion_states(q0,protocol['motion_grid']['poses'])
    first=coord.recover_source_inputs(qgrid[0],rest,camera,aa32)
    byte_guard=coord.require_original_bytes(first,aa32,t32)
    preflight=[]
    for q in qgrid:
        count('coordinate_preflight_states');r=coord.recover_source_inputs(q,rest,camera,aa32)
        if np.max(r['float32_axisangle_rotation_roundtrip_errors'])>GATES['float32_axisangle_rotation_roundtrip_comparison']:raise ValueError('Preflight coordinate roundtrip refused')
        preflight.append(r)
    save('source_initialized_domain.npz','new_domain_originalbytes_and109_coordinate_preflight_no_model_lookup',
        source_initialized_q0=q0,source_initialized_qgrid=qgrid,original_axisangles_float32=aa32,
        original_camera_translation_float32=t32,original_cameraWorld64=camera,shape=shape,rest_quaternions=rest,
        recovered_axisangles_float32=np.asarray([r['axis_angles_float32'] for r in preflight]),
        recovered_camera_translation_float32=np.asarray([r['translation_camera_float32'] for r in preflight]),
        branch_indices=np.asarray([r['equivalent_branch_indices'] for r in preflight]),
        coordinate_rotation_roundtrip_errors=np.asarray([r['float32_axisangle_rotation_roundtrip_errors'] for r in preflight]),
        faces=decoded['original_native_faces'],original_native_target_world_m=decoded['native_vertices_world_m'],
        pose_ids=np.asarray([r['pose_id'] for r in protocol['motion_grid']['poses']]))
    if sha(spec['old_include']['path'])!=spec['old_include']['sha256']:raise ValueError('Complete original flex include changed')
    xml,assembly=bridge.scene_xml(spec['base_xml'],spec['old_include']['path'])
    path=outdir/'source_pipeline_geometry_only_scene.xml';path.write_text(xml);seal(path,'original_skeleton_new_source_camera_profile_all_contacts_actuation_disabled')
    count('model_compilations');model=mujoco.MjModel.from_xml_path(str(path))
    layout=oldcase.named_model_contract(model,mujoco,spec['saved_named_joint_addresses'])
    if not np.array_equal(np.asarray(layout['body_rest_quaternions_wxyz']),rest):raise ValueError('Actual rest quaternions differ from preflight source domain')
    count('data_constructions');data=mujoco.MjData(model)
    if not np.array_equal(model.body_mass,old['body_mass']) or not np.array_equal(model.body_inertia,old['body_inertia']):raise ValueError('Original inertials changed')
    for name in reversed(spec['dependency_dirs']):
        if name not in sys.path:sys.path.insert(0,name)
    from manotorch.manolayer import ManoLayer
    count('source_layer_constructions');layer=ManoLayer(rot_mode='axisang',side='right',center_idx=0,
        use_pca=False,flat_hand_mean=True,mano_assets_root=spec['mano_assets_root']).cpu().float().eval()
    if layer.th_hands_mean.dtype!=torch.float32 or bool((layer.th_hands_mean!=0).any()):raise ValueError('Original flat-zero mean source required')
    save('literal_source_buffers.npz','literal_pinned_source_buffers_for_separately_qualified_future_camera_graph',
        template=layer.th_v_template[0].detach().numpy(),shapedirs=layer.th_shapedirs.detach().numpy(),
        posedirs=layer.th_posedirs.detach().numpy(),joint_regressor=layer.th_J_regressor.detach().numpy(),
        weights=layer.th_weights.detach().numpy(),shape=shape,faces=decoded['original_native_faces'],
        source_faces=layer.th_faces.detach().numpy(),native_face_labels=old['native_face_labels'],
        original_cameraWorld64=camera,rest_quaternions=rest,source_initialized_q0=q0,
        original_postAA32=aa32,original_camera_translation32=t32)
    beta=torch.from_numpy(shape.copy());source_records=[]
    def evaluate(q):
        # Read actual state again; never substitute precomputed source inputs.
        r=coord.recover_source_inputs(q,rest,camera,aa32)
        if not source_records:coord.require_original_bytes(r,aa32,t32)
        state=evaluate_pinned_source(layer,q,rest,camera,aa32,beta,coord,torch,lambda:count('source_forward_calls'))
        source_records.append(state);return state
    owner=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,bridge.OWNER)
    flex=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_FLEX,bridge.FLEX)
    bodyids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'mano_body_{j}') for j in range(16)],int)
    jointids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ['wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]],int)
    faces=decoded['original_native_faces']
    updater=bridge.ArticulatedSkinUpdater(model,data,owner_body_id=owner,flex_id=flex,faces=faces,evaluate_skin=evaluate,
        flex_update=mujoco.mj_flex,before_skin=lambda:None,before_flex=lambda:count('direct_mujoco_flex_calls'),
        contact_disable_bit=mujoco.mjtDisableBit.mjDSBL_CONTACT,actuation_disable_bit=mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    model_hashes=bridge.model_array_hashes(model);data.qvel[:]=old['literal_qvel'];data.ctrl[:]=old['retained_ctrl'];data.time=0.
    qvel0=data.qvel.copy();ctrl0=data.ctrl.copy();warnings0=np.asarray([w.number for w in data.warning],int)
    if np.any(warnings0):raise ValueError('Fresh warning counters must be zero')
    worlds=[];normal_values=[];anchors=[];rotations=[];inputs=[];locals_readback=[];frames=[]
    vertex_errors=[];normal_errors=[];anchor_errors=[];rotation_errors=[]
    for k,row in enumerate(protocol['motion_grid']['poses']):
        data.qpos[:]=qgrid[k];count('mujoco_forward_calls');mujoco.mj_forward(model,data)
        if (not np.array_equal(data.qpos,qgrid[k]) or not np.array_equal(data.qvel,qvel0) or not np.array_equal(data.ctrl,ctrl0)
            or data.time!=0 or data.ncon!=0 or np.any(data.qfrc_actuator) or not np.isfinite(data.qacc).all()
            or not np.array_equal(np.asarray([w.number for w in data.warning]),warnings0)):raise ValueError('Static source-state forward integrity failure')
        value=updater.update();state=value['skin_state'];world=value['compiled_vertices_world_m'];normal=value['compiled_face_normals_world']
        source_normal,_=bridge.oriented_face_normals(state['vertices'],faces)
        ac=data.xanchor[jointids].copy();ar=data.xmat[bodyids].reshape(16,3,3).copy()
        bridge.finite(ac,(16,3),'actual anatomy anchors');bridge.finite(ar,(16,3,3),'actual anatomy rotations')
        if not np.array_equal(np.asarray([w.number for w in data.warning]),warnings0) or data.ncon!=0:raise ValueError('Source flex refresh changed warnings/contact')
        worlds.append(world.copy());normal_values.append(normal.copy());anchors.append(ac);rotations.append(ar);inputs.append(data.qpos.copy())
        locals_readback.append(model.flex_vert.copy());frame=np.eye(4);frame[:3,:3]=value['actual_owner_rotation'];frame[:3,3]=value['actual_owner_position'];frames.append(frame)
        vertex_errors.append(float(np.max(np.abs(world-state['vertices']))));normal_errors.append(float(np.max(np.abs(normal-source_normal))))
        anchor_errors.append(float(np.max(np.abs(ac-state['joint_anchors_world']))));rotation_errors.append(float(np.max(np.abs(ar-state['absolute_rotations']))))
        checkpoint_callback(dict(event='progress',state_id=row['pose_id'],completed_states=k+1))
    target=decoded['native_vertices_world_m'];target_normal,_=bridge.oriented_face_normals(target,faces)
    baseline_vertex=float(np.max(np.abs(worlds[0]-target)));baseline_normal=float(np.max(np.abs(normal_values[0]-target_normal)))
    save('all109_native_source_engine_readbacks.npz','exact_source32_camera_graph_and_actual_compiled_articulation',
        actual_named_qpos=np.asarray(inputs),compiled_vertices_world_m=np.asarray(worlds),compiled_normals_world=np.asarray(normal_values),
        compiled_joint_anchors_world=np.asarray(anchors),compiled_body_rotations_world=np.asarray(rotations),
        actual_model_flex_local_vertices=np.asarray(locals_readback),actual_owner_transforms=np.asarray(frames),
        source_vertices_world=np.asarray([s['vertices'] for s in source_records]),source_joint_anchors_world=np.asarray([s['joint_anchors_world'] for s in source_records]),
        source_body_rotations_world=np.asarray([s['absolute_rotations'] for s in source_records]),
        source_wrist_vertices_float32=np.asarray([s['source_wrist_vertices32'] for s in source_records]),
        source_translated_camera_vertices_float32=np.asarray([s['translated_camera_vertices32'] for s in source_records]),
        source_camera_rotations_float32=np.asarray([s['source_camera_rotations32'] for s in source_records]),
        source_translated_camera_anchors_float32=np.asarray([s['translated_camera_anchors32'] for s in source_records]),
        actual_recovered_axisangles_float32=np.asarray([s['source_record']['axis_angles_float32'] for s in source_records]),
        actual_recovered_translations_float32=np.asarray([s['source_record']['translation_camera_float32'] for s in source_records]),
        coordinate_rotation_roundtrip_errors=np.asarray([s['source_record']['float32_axisangle_rotation_roundtrip_errors'] for s in source_records]),
        engine_source_vertex_errors=np.asarray(vertex_errors),engine_source_normal_errors=np.asarray(normal_errors),
        joint_anchor_errors=np.asarray(anchor_errors),joint_rotation_errors=np.asarray(rotation_errors),
        faces=faces,shape=shape,original_native_target_world=target,geom_contype=model.geom_contype.copy(),geom_conaffinity=model.geom_conaffinity.copy(),
        flex_contype=model.flex_contype.copy(),flex_conaffinity=model.flex_conaffinity.copy(),disableflags=np.asarray(model.opt.disableflags),
        body_mass=model.body_mass.copy(),body_inertia=model.body_inertia.copy(),retained_qvel=qvel0,retained_ctrl=ctrl0,warning_counts=warnings0)
    if len(source_records)!=109 or bridge.model_array_hashes(model)!=model_hashes or counts!={k:v for k,v in BUDGET.items() if k!='maximum_seconds_per_case'}:raise ValueError('Exact source operation/immutability inventory incomplete')
    checks=dict(original_input_byte_identity=True,engine_source_vertices=max(vertex_errors)<=GATES['engine_source_vertex_comparison_m'],
        engine_source_normals=max(normal_errors)<=GATES['engine_source_normal_component_comparison'],
        original_native_q0_vertices=baseline_vertex<=GATES['original_native_baseline_vertex_comparison_m'],
        original_native_q0_normals=baseline_normal<=GATES['original_native_baseline_normal_component_comparison'],
        actual_anchors=max(anchor_errors)<=GATES['original_joint_anchor_comparison_m'],actual_rotations=max(rotation_errors)<=GATES['original_rotation_component_comparison'])
    return dict(status='COMPLETED',counts=counts,forbidden_counts=FORBIDDEN,artifacts=artifacts,checks=checks,source_pipeline_geometry_check_passed=all(checks.values()),
        source_state_initialization=initialization,original_input_byte_guard=byte_guard,scene_assembly=assembly,named_layout=layout,
        nonflex_model_array_hashes=model_hashes,metrics=dict(maximum_engine_source_vertex_error_m=max(vertex_errors),maximum_engine_source_normal_component_error=max(normal_errors),
            original_native_baseline_vertex_error_m=baseline_vertex,original_native_baseline_normal_component_error=baseline_normal,
            maximum_joint_anchor_error_m=max(anchor_errors),maximum_joint_rotation_component_error=max(rotation_errors)),
        old64_jacobian_qualified_for_this_profile=False,contact_jacobian_status='NOT_QUALIFIED',physical_contact_status='NOT_RUN',contact_orientation_over_motion_status='NOT_RUN',
        stock_flex_contact_forces_permitted=False,operational_profile='pinned_ManoLayer_AA32_beta32_then_t32_then_cameraWorld64_v1',state_domain=coord.DOMAIN,
        accepted_mapping=False,native_contact_qualified=False,new_collection_or_training_allowed=False,physical_outcome=None)
