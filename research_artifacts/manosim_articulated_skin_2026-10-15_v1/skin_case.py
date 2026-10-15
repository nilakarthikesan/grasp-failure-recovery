"""Bounded source-MANO parity and local-tangent qualification, never dynamics.

Import/declaration read metadata only. Native buffer loading, source skinning,
model compilation and mj_integratePos are confined to run_case after freezing.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT/'research_artifacts/manosim_full_surface_reference_mj340_2026-10-15_v2'
ORIGINAL = Path('/Users/nilakarthikesan/grasp-failure-prediction')
DEPS = ORIGINAL/'runs/hug_integration_check/hug-deps'
HUG = ORIGINAL/'runs/hug_integration_check/hug'
SCHEMA = 'manosim_complete_skin_tangent_qualification_v1'
BUDGET = dict(maximum_seconds_per_case=600, physics_steps=0, model_compilations=1,
    data_constructions=0, mujoco_forward_calls=0, mujoco_integratePos_calls=109,
    source_layer_constructions=1, source_axisangle_evaluations=1,
    source_matrix_skin_evaluations=109, candidate_float32_evaluations=109,
    candidate_float64_evaluations=109, jacobian_forward_evaluations=51,
    finite_difference_evaluations=102, vjp_evaluations=1)
GATES = dict(source_float32_numerical_comparison_m=5e-6,
    saved_native_reconstruction_comparison_m=5e-6,
    continuous64_to_source32_comparison_m=5e-6,
    integrate_pos_quaternion_component_comparison=1e-12,
    finite_difference_step=1e-6, derivative_absolute_comparison=1e-8,
    derivative_relative_comparison=1e-6, virtual_work_absolute_comparison=1e-10,
    virtual_work_relative_comparison=1e-10, numerical_limits_are_contact_allowances=False)
FORBIDDEN = dict(physics_integration_attempts=0, mujoco_data_attempts=0,
    mujoco_forward_attempts=0, object_geometry_query_attempts=0,
    HUG_generator_attempts=0, retarget_attempts=0, training_attempts=0)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def load_skin():
    name = 'frozen_complete_skin_qualification_core'
    spec = importlib.util.spec_from_file_location(name, HERE/'skin_kinematics.py')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def declaration():
    """Only paths/JSON/library metadata; no asset array/model evaluation."""
    p = read(PRIOR/'protocol.json')
    cases = {}
    for index in (20,28):
        result_path = PRIOR/f'run_v1/{index}/result.json'
        result = read(result_path)
        cases[str(index)] = dict(source_inventory_index=index,
            decoded_reference=p['cases'][str(index)]['inputs']['decoded_reference'],
            shape_asset=p['cases'][str(index)]['inputs']['shape_asset'],
            compiled_readback=result['source_receipt']['compiled_readback_artifact'],
            saved_named_joint_addresses=result['source_receipt']['named_joint_addresses'],
            prior_case=dict(path=str(result_path),sha256=sha(result_path)),
            base_xml=p['base_xml'], mano_assets_root=str(HUG/'assets/mano_models'),
            dependency_dirs=[str(DEPS), str(ORIGINAL/'runs/hug_integration_check/deps')],
            expected_shape_sha256=p['cases'][str(index)]['expected_shape_sha256'])
    sources = [HERE/'skin_kinematics.py',HERE/'skin_case.py',HERE/'test_skin_kinematics.py',
        HERE/'test_skin_case.py',HERE/'README_SKIN.md',HERE/'manufactured_control_lineage.json',
        HUG/'src/models/mano.py',HUG/'assets/mano_models/models/MANO_RIGHT.pkl']
    sources += list((DEPS/'manotorch').rglob('*.py'))
    # Bind installed Torch implementation and linked CPU libraries in addition
    # to explicit version metadata. Imports alone evaluate no native hand.
    import torch
    sources += [Path(module.__file__) for name,module in list(sys.modules.items())
        if name.startswith('torch') and getattr(module,'__file__',None)
        and Path(module.__file__).is_file()]
    sources += list(Path(torch.__file__).parent.rglob('*.py'))
    sources += list((Path(torch.__file__).parent/'lib').glob('*.dylib'))
    return dict(budget=BUDGET, gates=GATES,
        source_inputs=sorted({str(x.resolve()) for x in sources}), cases=cases,
        motion_grid=load_skin().motion_grid_spec(),
        runtime_skin=dict(torch=torch.__version__,torch_threads=1,
            source_precision='literal_source_float32', derivative_precision='explicit_float64_cast_of_literal_source_buffers',
            numpy='2.2.6',mujoco='3.4.0'))


def finite_array(value, shape, name):
    import numpy as np
    a = np.asarray(value)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite literal '+name+' '+str(shape)+' required')
    return a


def named_model_contract(model, mujoco, expected_rows):
    """Actual compiled metadata; no Data, FK, forward or contact calculation."""
    import numpy as np
    if (model.nq,model.nv,model.nu,model.njnt) != (67,51,51,19):
        raise ValueError('Exact67qpos/51DOFs/51actuators/19joints required')
    names = ['wrist_tx','wrist_ty','wrist_tz','wrist_rot']+[f'mano_joint_{j}' for j in range(1,16)]
    rows,rest = [],[]
    bodies = [int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'mano_body_{j}')) for j in range(16)]
    if min(bodies) < 1: raise ValueError('All16 anatomical body names required')
    for k,name in enumerate(names):
        joint = int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name))
        if joint < 0: raise ValueError('Missing named joint '+name)
        row = dict(name=name,qpos_address=int(model.jnt_qposadr[joint]),
            dof_address=int(model.jnt_dofadr[joint]),type=int(model.jnt_type[joint]))
        wanted = dict(name=name,qpos_address=k if k<3 else 3+4*(k-3),
            dof_address=k if k<3 else 3+3*(k-3),type=2 if k<3 else 1)
        if row != wanted: raise ValueError('Named tangent/address/type mismatch '+name)
        if int(model.jnt_bodyid[joint]) != bodies[0 if k<4 else k-3]:
            raise ValueError('Named joint anatomical body mismatch')
        rows.append(row)
    if rows != expected_rows: raise ValueError('Actual named layout differs from frozen saved layout')
    for j,b in enumerate(bodies):
        expected_parent = 0 if j == 0 else bodies[load_skin().PARENTS[j]]
        if int(model.body_parentid[b]) != expected_parent: raise ValueError('MANO parent tree changed')
        rest.append(finite_array(model.body_quat[b],(4,),'body rest quaternion').copy())
    rest = np.asarray(rest)
    # This bounded source model has root at world origin and canonical world
    # slides. General transformed slide frames need a separately qualified map.
    if (not np.array_equal(np.asarray(model.body_pos[bodies[0]]),np.zeros(3))
            or not np.array_equal(rest[0],np.array([1.,0.,0.,0.]))):
        raise ValueError('This qualification requires literal original root rest frame')
    slides = np.asarray([model.jnt_axis[k] for k in range(3)])
    if not np.array_equal(slides,np.eye(3)): raise ValueError('Canonical named world slide axes required')
    if not np.array_equal(np.asarray(model.jnt_pos[:4]),np.zeros((4,3))):
        raise ValueError('Original root joint anchors required')
    return dict(named_joint_addresses=rows,body_ids=bodies,
        body_rest_quaternions_wxyz=rest.tolist(),slide_basis_world=slides.T.tolist(),
        nq=67,nv=51,nu=51,engine_tangent_convention='local_ball_right_increment_world_slides',
        geometric_simulator_articulation_status='NOT_RUN',physical_contact_status='NOT_RUN')


def body_inputs(qpos,rest_quaternions,skin):
    """Copy named67-state to51D skin coordinates; never mutates native qpos."""
    import numpy as np
    import torch
    q = finite_array(qpos,(67,),'saved named qpos')
    rest = torch.tensor(finite_array(rest_quaternions,(16,4),'rest quaternions'),dtype=torch.float64)
    quats = torch.tensor(q[3:].reshape(16,4).copy(),dtype=torch.float64)
    return torch.tensor(q[:3].copy(),dtype=torch.float64),skin.quaternion_product(rest,quats)


def expected_integrated_qpos(qpos,delta,skin):
    import numpy as np
    import torch
    q = finite_array(qpos,(67,),'qpos'); d = finite_array(delta,(51,),'delta')
    result = q.copy(); result[:3] += d[:3]
    before = torch.tensor(q[3:].reshape(16,4),dtype=torch.float64)
    increment = skin.rotation_increment(torch.tensor(d[3:].reshape(16,3),dtype=torch.float64))
    result[3:] = skin.quaternion_product(before,increment).numpy().reshape(64)
    return result


def baseline_checks(source_vertices,continuous_vertices,original_target):
    """Join the independent saved-source and named-state chains explicitly."""
    import numpy as np
    source=finite_array(source_vertices,(778,3),'source baseline skin')
    continuous=finite_array(continuous_vertices,(778,3),'continuous baseline skin')
    target=finite_array(original_target,(778,3),'unchanged native skin')
    a=float(np.max(np.abs(source-target)));b=float(np.max(np.abs(continuous-target)))
    limit=GATES['saved_native_reconstruction_comparison_m']
    return dict(source32_named_state_baseline_matches_saved=a<=limit,
                continuous64_named_state_baseline_matches_saved=b<=limit), dict(
        baseline_source32_to_original_saved_m=a,baseline_continuous64_to_original_saved_m=b)


def run_case(spec, output_dir, protocol, checkpoint_callback):
    import numpy as np
    import torch
    import mujoco
    if mujoco.__version__ != '3.4.0' or np.__version__ != '2.2.6':
        raise ValueError('Frozen MuJoCo3.4.0/baseNumPy2.2.6 required')
    if protocol['budget'] != BUDGET or protocol['gates'] != GATES:
        raise ValueError('Declared skin budgets/gates changed')
    skin = load_skin()
    if protocol['motion_grid'] != skin.motion_grid_spec(): raise ValueError('Fixed109 motion grid changed')
    torch.set_num_threads(1)
    counts = {name:0 for name in BUDGET if name not in ('maximum_seconds_per_case',)}
    artifacts = []; output = Path(output_dir)
    def count(name):
        if counts[name] >= BUDGET[name]: raise RuntimeError('Operation budget exhausted: '+name)
        counts[name] += 1
        checkpoint_callback(dict(event='operation',name=name,amount=1,counts=dict(counts)))
    def save(name,label,**arrays):
        path = output/name
        if path.exists(): raise ValueError('Write-once artifact exists')
        np.savez_compressed(path,**arrays)
        row = dict(path=str(path.resolve()),sha256=sha(path),bytes=path.stat().st_size,label=label)
        artifacts.append(row); checkpoint_callback(dict(event='artifact',artifact=row))
        return row
    for key in ('decoded_reference','shape_asset','compiled_readback'):
        if sha(spec[key]['path']) != spec[key]['sha256']: raise ValueError('Bound saved input changed: '+key)
    with np.load(spec['decoded_reference']['path'],allow_pickle=False) as a:
        decoded = {name:a[name].copy() for name in a.files}
    with np.load(spec['compiled_readback']['path'],allow_pickle=False) as a:
        q0 = finite_array(a['literal_qpos'],(67,),'literal saved qpos').copy()
    beta_np = np.load(spec['shape_asset']['path'],allow_pickle=False)
    if (beta_np.dtype != np.dtype('float32') or beta_np.shape != (1,10)
            or not np.array_equal(beta_np,decoded['shape'])
            or sha(spec['shape_asset']['path']) != spec['expected_shape_sha256']):
        raise ValueError('Exact originalfloat32 beta(1,10) required')
    q0_before = q0.copy()
    count('model_compilations'); model = mujoco.MjModel.from_xml_path(spec['base_xml'])
    contract = named_model_contract(model,mujoco,spec['saved_named_joint_addresses'])
    rest = np.asarray(contract['body_rest_quaternions_wxyz'])
    for path in reversed(spec['dependency_dirs']):
        if path not in sys.path: sys.path.insert(0,path)
    # This is the independent pinned source implementation, not the candidate.
    from manotorch.manolayer import ManoLayer
    count('source_layer_constructions')
    source = ManoLayer(rot_mode='axisang',side='right',center_idx=0,use_pca=False,
        flat_hand_mean=True,mano_assets_root=spec['mano_assets_root']).cpu().eval()
    if source.kintree_parents[1:] != list(skin.PARENTS[1:]): raise ValueError('Source parent tree mismatch')
    arrays32 = dict(template=source.th_v_template[0].detach().clone(),
        shapedirs=source.th_shapedirs.detach().clone(),posedirs=source.th_posedirs.detach().clone(),
        joint_regressor=source.th_J_regressor.detach().clone(),weights=source.th_weights.detach().clone())
    if arrays32['template'].shape != (778,3): raise ValueError('All778 source vertices required')
    beta32 = torch.from_numpy(beta_np.copy())
    arrays64 = {key:value.double() for key,value in arrays32.items()}; beta64 = beta32.double()
    buffer_row = save('literal_source_skin_buffers.npz','literal_float32_source_buffers_and_exact_beta',
        **{key:value.numpy() for key,value in arrays32.items()},shape=beta_np,
        source_faces=source.th_faces.numpy(),closed_faces=decoded['original_native_faces'],
        camera_world=decoded['camera_world'],source_actual_full_poses=decoded['actual_full_poses'])
    # Independent original source AA evaluation must reproduce the saved target.
    count('source_axisangle_evaluations')
    with torch.no_grad():
        original = source(torch.from_numpy(decoded['actual_full_poses'].copy()).reshape(1,48),beta32)
    camera = finite_array(decoded['camera_world'],(4,4),'saved camera world')
    native_camera = original.verts[0].numpy()+decoded['params_translation_m']
    original_world = np.einsum('ij,kj->ki',camera[:3,:3],native_camera,optimize=False)+camera[:3,3]
    original_target = finite_array(decoded['native_vertices_world_m'],(778,3),'unchanged saved native target')
    native_error = float(np.max(np.abs(original_world-original_target)))
    grid = protocol['motion_grid']['poses']
    source_verts,low_verts,high_verts,positions,quats,engine_qpos = [],[],[],[],[],[]
    integration_errors=[]; source_errors=[]; precision_errors=[]; candidate_joints=[]
    t0,qbody0 = body_inputs(q0,rest,skin)
    for row in grid:
        delta = np.asarray(row['delta51'],dtype=np.float64)
        q_expected = expected_integrated_qpos(q0,delta,skin)
        actual = q0.copy(); count('mujoco_integratePos_calls')
        mujoco.mj_integratePos(model,actual,delta,1.)
        finite_array(actual,(67,),'actual integrated qpos')
        integration_errors.append(float(np.max(np.abs(actual-q_expected))))
        t,q = body_inputs(actual,rest,skin); rotations64=skin.quaternion_matrices(q)
        r32=rotations64.float(); t32=t.float()
        count('source_matrix_skin_evaluations')
        with torch.no_grad(): oracle = source.skinning_layer(r32[None],beta32)['verts'][0]+t32
        count('candidate_float32_evaluations')
        low=skin.skin_state(r32,t32,beta32,arrays32)
        count('candidate_float64_evaluations')
        high=skin.skin_state(rotations64,t,beta64,arrays64)
        source_verts.append(oracle.numpy()); low_verts.append(low['vertices'].detach().numpy())
        high_verts.append(high['vertices'].detach().numpy()); candidate_joints.append(high['joint_anchors_world'].detach().numpy())
        source_errors.append(float(torch.max(torch.abs(low['vertices']-oracle))))
        precision_errors.append(float(torch.max(torch.abs(high['vertices']-oracle.double()))))
        positions.append(t.numpy()); quats.append(q.numpy()); engine_qpos.append(actual.copy())
        checkpoint_callback(dict(event='progress',state_id=row['pose_id'],completed_states=len(source_verts)))
    save('all109_skin_states.npz','all_fixed_motion_source32_candidate32_continuous64_and_actual_tangents',
        source_float32_vertices_world_m=np.asarray(source_verts),candidate_float32_vertices_world_m=np.asarray(low_verts),
        candidate_float64_vertices_world_m=np.asarray(high_verts),candidate_float64_joint_anchors_world_m=np.asarray(candidate_joints),
        wrist_world_translation_m=np.asarray(positions),local_body_quaternions_wxyz=np.asarray(quats),
        actual_integrated_named_qpos=np.asarray(engine_qpos),saved_literal_qpos=q0,
        delta51=np.asarray([row['delta51'] for row in grid]),pose_ids=np.asarray([row['pose_id'] for row in grid]),
        faces=decoded['original_native_faces'],source_reconstructed_original_world_m=original_world,
        original_native_target_world_m=original_target,shape=beta_np,
        integration_component_errors=np.asarray(integration_errors),source32_errors_m=np.asarray(source_errors),
        continuous64_vs_source32_errors_m=np.asarray(precision_errors),body_rest_quaternions_wxyz=rest,
        joint_axes=np.asarray(model.jnt_axis),joint_qpos_addresses=np.asarray(model.jnt_qposadr),
        joint_dof_addresses=np.asarray(model.jnt_dofadr),joint_types=np.asarray(model.jnt_type),
        body_parent_ids=np.asarray(model.body_parentid),body_positions=np.asarray(model.body_pos))
    #51 complete forward-mode evaluations and102 independently perturbed skins.
    jac = skin.skin_jacobian(t0,qbody0,beta64,arrays64,
        evaluation_callback=lambda:count('jacobian_forward_evaluations'))
    if not bool(torch.isfinite(jac).all()): raise ValueError('All51JVP columns must be finite')
    step = GATES['finite_difference_step']; fd_columns=[]
    for column in range(51):
        delta = torch.zeros(51,dtype=torch.float64);delta[column]=step
        count('finite_difference_evaluations');plus=skin.tangent_vertices(t0,qbody0,delta,beta64,arrays64)
        count('finite_difference_evaluations');minus=skin.tangent_vertices(t0,qbody0,-delta,beta64,arrays64)
        fd_columns.append((plus-minus)/(2*step))
    fd = torch.stack(fd_columns,dim=-1)
    forces = torch.sin(torch.arange(778*3,dtype=torch.float64).reshape(778,3)/37)/100
    velocity = torch.sin(torch.arange(51,dtype=torch.float64)/11)/10
    count('vjp_evaluations');tau=skin.skin_force_vjp(t0,qbody0,beta64,arrays64,forces)
    explicit_tau=torch.einsum('vcd,vc->d',jac,forces)
    skin_work=(torch.einsum('vcd,d->vc',jac,velocity)*forces).sum()
    joint_work=tau@velocity
    jac_error=torch.abs(jac-fd)
    jac_ok=bool(torch.all(jac_error <= GATES['derivative_absolute_comparison']+
        GATES['derivative_relative_comparison']*torch.abs(fd)))
    transpose_error=float(torch.max(torch.abs(tau-explicit_tau)))
    work_error=float(torch.abs(skin_work-joint_work))
    work_limit=GATES['virtual_work_absolute_comparison']+GATES['virtual_work_relative_comparison']*abs(float(skin_work))
    save('all51_tangent_derivatives.npz','full_pose_corrective_LBS_JVP_FD_VJP_virtual_work',
        jacobian_m_per_named_tangent=jac.detach().numpy(),central_difference=fd.detach().numpy(),
        fixed_vertex_forces_world=forces.numpy(),fixed_named_velocity=velocity.numpy(),
        generalized_force_vjp=tau.detach().numpy(),explicit_J_transpose_force=explicit_tau.detach().numpy(),
        skin_virtual_power=np.asarray(float(skin_work)),joint_virtual_power=np.asarray(float(joint_work)))
    if not np.array_equal(q0,q0_before): raise ValueError('Saved native qpos mutated')
    if counts != {k:v for k,v in BUDGET.items() if k!='maximum_seconds_per_case'}:
        raise ValueError('Declared fixed operation inventory incomplete')
    all_vertices=np.asarray(high_verts)
    baseline_ok,baseline_metrics=baseline_checks(source_verts[0],all_vertices[0],original_target)
    checks=dict(source_original_native_reconstruction=native_error<=GATES['saved_native_reconstruction_comparison_m'],
        all109_source_float32_parity=max(source_errors)<=GATES['source_float32_numerical_comparison_m'],
        all109_continuous64_precision_comparison=max(precision_errors)<=GATES['continuous64_to_source32_comparison_m'],
        all109_actual_engine_tangent_integration=max(integration_errors)<=GATES['integrate_pos_quaternion_component_comparison'],
        all51_derivative_columns=jac_ok,force_transpose=transpose_error<=work_limit,
        virtual_work=work_error<=work_limit,saved_native_qpos_unchanged=True,
        nonzero_finger_motion=bool(np.max(np.abs(all_vertices[1:91]-all_vertices[0]))>0))
    checks.update(baseline_ok)
    return dict(status='COMPLETED',counts=counts,forbidden_counts=dict(FORBIDDEN),
        artifacts=artifacts,checks=checks,check_passed=all(checks.values()),
        precision_comparisons=dict(original_source_vs_saved_native_m=native_error,
            max_source32_candidate32_m=max(source_errors),max_continuous64_source32_m=max(precision_errors),
            **baseline_metrics,
            max_actual_integrate_pos_component_error=max(integration_errors),
            max_derivative_absolute_error=float(jac_error.max()),vjp_transpose_error=transpose_error,virtual_work_error=work_error),
        simulator_state_binding=contract,source_buffers_artifact=buffer_row,
        runtime=dict(torch=torch.__version__,torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),mujoco=mujoco.__version__,numpy=np.__version__),
        motion_states=109,simulator_articulated_geometry_status='NOT_RUN',contact_orientation_over_motion_status='NOT_RUN',
        physical_contact_status='NOT_RUN',contact_force_application_status='NOT_RUN',
        accepted_mapping=False,native_contact_qualified=False,new_collection_or_training_allowed=False,physical_outcome=None)
