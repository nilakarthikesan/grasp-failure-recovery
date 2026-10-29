"""Pinned Wuji Hand 2.1 retarget and force-limited synthetic lift demonstration."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

COMMIT = '5c4dfa16917355d073e2e1d7e3d03897b6cf9508'
MODEL_PATH = 'hand2/hand2_1/body/mjcf/right.xml'
BASE_URL = f'https://raw.githubusercontent.com/wuji-technology/wuji-description/{COMMIT}/'
FINGERS = ('thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky')
DEFAULT = dict(approach_s=1.5, close_s=1.0, lift_s=1.5, hold_s=1.0,
               approach_height_m=.12, lift_command_m=.10, success_lift_m=.05,
               retarget_max_nfev=90, root_translation_bound_m=.08,
               root_rotation_bound_rad=.9, penetration_allowance_m=.003,
               maximum_mean_tip_error_m=.05, maximum_tip_error_m=.08,
               video_fps=20, video_width=480, video_height=480)

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fetch_assets(root):
    """Download only the pinned XML and its 22 referenced meshes; never overwrite."""
    root = Path(root)
    marker = root/'asset_manifest.json'
    if marker.exists():
        receipt = json.loads(marker.read_text())
        if receipt['commit'] != COMMIT:
            raise ValueError('asset revision mismatch')
        for name, digest in receipt['sha256'].items():
            if sha(root/name) != digest:
                raise ValueError('asset bytes changed: '+name)
        return receipt
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('incomplete/nonempty asset directory cannot be overwritten')
    root.mkdir(parents=True, exist_ok=True)
    def get(relative, destination):
        with urllib.request.urlopen(BASE_URL+relative, timeout=60) as response:
            content = response.read()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    get(MODEL_PATH, root/'right.xml')
    model = ET.parse(root/'right.xml').getroot()
    names = [mesh.attrib['file'] for mesh in model.findall('asset/mesh')]
    if len(names) != 22 or len(set(names)) != 22 or any(Path(n).name != n for n in names):
        raise ValueError('unexpected official mesh closure')
    def mesh(name):
        get('hand2/hand2_1/body/meshes/right/'+name, root/'meshes'/name)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(mesh, names))
    receipt = {'commit': COMMIT, 'model': MODEL_PATH, 'mesh_count': len(names),
               'source': BASE_URL, 'sha256': {str(p.relative_to(root)):sha(p)
                  for p in [root/'right.xml', *[root/'meshes'/n for n in names]]}}
    dump(marker, receipt)
    return receipt

def quat_wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3,0,1,2]]

def rigid_fit(source, target):
    """Proper rigid Kabsch; no scaling of the human source targets."""
    a, b = np.asarray(source), np.asarray(target)
    u, _, vt = np.linalg.svd((a-a.mean(0)).T @ (b-b.mean(0)))
    correction = np.eye(3); correction[2,2] = np.linalg.det(vt.T@u.T)
    rotation = vt.T@correction@u.T
    return rotation, b.mean(0)-rotation@a.mean(0)

def load_proposal(scene, proposal):
    with np.load(Path(proposal)/'proposal.npz', allow_pickle=False) as archive:
        points = np.asarray(archive['landmarks_3d'], dtype=np.float64)
        native_wrist = np.asarray(archive['T_camera_wrist'], dtype=np.float64)
    transform = np.load(Path(scene)/'T_world_camera.npy', allow_pickle=False)
    if points.shape != (21,3) or native_wrist.shape != (4,4) or transform.shape != (4,4):
        raise ValueError('invalid HUG landmarks/wrist/camera shapes')
    if not all(np.isfinite(a).all() for a in (points,native_wrist,transform)):
        raise ValueError('nonfinite HUG input')
    for value in (native_wrist, transform):
        if not (np.allclose(value[3],[0,0,0,1]) and
                np.allclose(value[:3,:3].T@value[:3,:3],np.eye(3),atol=1e-5) and
                np.isclose(np.linalg.det(value[:3,:3]),1.,atol=1e-5)):
            raise ValueError('invalid rigid input transform')
    # HUG points are ALREADY in camera coordinates: do not apply wrist pose again.
    return points@transform[:3,:3].T+transform[:3,3], native_wrist, transform

def build_model(scene, assets, config):
    source = ET.parse(Path(assets)/'right.xml').getroot()
    root = ET.parse(Path(scene)/'scene.xml').getroot()
    key = root.find('keyframe')
    if key is None:
        raise ValueError('source scene must contain settled capture keyframe')
    cube_pose = np.fromstring(key.find('key').attrib['qpos'], sep=' ')
    cube_velocity = np.fromstring(key.find('key').attrib['qvel'], sep=' ')
    if cube_pose.shape!=(7,) or cube_velocity.shape!=(6,):
        raise ValueError('source capture keyframe must hold one free cube pose/velocity')
    root.remove(key)
    # Honor explicit native hand inertias; cube still derives its explicit mass.
    root.find('compiler').set('inertiafromgeom','auto')
    asset = root.find('asset')
    if asset is None:
        asset=ET.SubElement(root,'asset')
    for element in source.find('asset'):
        element=copy.deepcopy(element)
        if element.tag=='mesh':
            element.set('file',str((Path(assets)/'meshes'/element.attrib['file']).resolve()))
        asset.append(element)
    hand=copy.deepcopy(source.find('worldbody/body'))
    hand.set('mocap','true')
    hand.set('pos','0 0 .5')
    # The official root and wrist transform are retained exactly.
    for joint in hand.iter('joint'):
        joint.attrib.setdefault('armature','.0002')
        joint.attrib.setdefault('damping','0')
    root.find('worldbody').append(hand)
    for section in ('actuator','contact'):
        root.append(copy.deepcopy(source.find(section)))
    visual=root.find('visual/global')
    visual.set('offwidth',str(config['video_width']))
    visual.set('offheight',str(config['video_height']))
    xml=ET.tostring(root,encoding='unicode')
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    data.qpos[:7]=cube_pose
    data.qvel[:6]=cube_velocity
    mujoco.mj_forward(model,data)
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,x.attrib['name'])
            for x in source.findall('.//joint') if 'name' in x.attrib]
    addresses=np.array([model.jnt_qposadr[j] for j in joints])
    limits=np.array([model.jnt_range[j] for j in joints])
    tips=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'r_'+f+'_tip') for f in FINGERS])
    # Explicit anatomical correspondence, not an asserted surface equivalence.
    body_names=['r_wrist']
    point_types=[('body',mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'r_wrist'))]
    for f,tip in zip(FINGERS,tips):
        for suffix in ('proximal','middle','distal'):
            name='r_'+f+'_'+suffix
            point_types.append(('body',mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)))
            body_names.append(name)
        point_types.append(('site',int(tip)))
    hand_geoms=np.flatnonzero((model.geom_group==2))
    return model,data,xml,dict(addresses=addresses,limits=limits,tips=tips,points=point_types,
                             hand_geoms=hand_geoms,body_names=body_names)

def robot_points(data, mapping):
    return np.array([data.xpos[index] if kind=='body' else data.site_xpos[index]
                     for kind,index in mapping['points']])

def retarget(model,data,mapping,targets,config):
    data.mocap_pos[0]=0; data.mocap_quat[0]=[1,0,0,0]
    data.qpos[mapping['addresses']]=0
    mujoco.mj_forward(model,data)
    anchors=np.array([0,5,9,13,17])
    rotation,translation=rigid_fit(robot_points(data,mapping)[anchors],targets[anchors])
    lower,upper=mapping['limits'].T
    q0=np.clip(np.zeros(20),lower+1e-6,upper-1e-6)
    x0=np.r_[translation,np.zeros(3),q0]
    tb,rb=config['root_translation_bound_m'],config['root_rotation_bound_rad']
    bounds=(np.r_[translation-tb,[-rb]*3,lower+1e-7],
            np.r_[translation+tb,[rb]*3,upper-1e-7])
    weights=np.ones((21,1)); weights[[4,8,12,16,20]]=2.; weights[0]=.5
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table')
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube')
    def apply(x):
        data.mocap_pos[0]=x[:3]
        data.mocap_quat[0]=quat_wxyz(Rotation.from_rotvec(x[3:6]).as_matrix()@rotation)
        data.qpos[mapping['addresses']]=x[6:]
        mujoco.mj_forward(model,data)
    def residual(x):
        apply(x)
        collision=[]
        for g in mapping['hand_geoms']:
            for other in (table,cube):
                distance=mujoco.mj_geomDistance(model,data,int(g),other,.01,None)
                collision.append(min(0.,distance+config['penetration_allowance_m'])*3.)
        return np.r_[((robot_points(data,mapping)-targets)*weights).ravel(),
                     collision,.001*(x[6:]-q0),.05*(x[:3]-translation),.001*x[3:6]]
    start=time.monotonic()
    fit=least_squares(residual,x0,bounds=bounds,max_nfev=config['retarget_max_nfev'],
                      ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-4)
    apply(fit.x)
    fitted=robot_points(data,mapping)
    errors=np.linalg.norm(fitted-targets,axis=1)
    tip_errors=errors[[4,8,12,16,20]]
    report={'optimizer_success':bool(fit.success),'optimizer_status':int(fit.status),
            'optimizer_message':str(fit.message),'function_evaluations':int(fit.nfev),
            'elapsed_s':time.monotonic()-start,'joint_names':[model.joint(int(j)).name
                 for j in range(model.njnt) if int(model.jnt_qposadr[j]) in mapping['addresses']],
            'joint_positions_rad':fit.x[6:].tolist(),'joint_limits_rad':mapping['limits'].tolist(),
            'root_mount_position_world_m':data.mocap_pos[0].tolist(),
            'root_mount_orientation_wxyz':data.mocap_quat[0].tolist(),
            'root_translation_change_from_rigid_initialization_m':(fit.x[:3]-translation).tolist(),
            'root_rotation_change_rotvec_rad':fit.x[3:6].tolist(),
            'landmark_errors_m':errors.tolist(),'tip_errors_m':tip_errors.tolist(),
            'mean_tip_error_m':float(tip_errors.mean()),'maximum_tip_error_m':float(tip_errors.max()),
            'source_scale':1.,'exact_source_fidelity_claimed':False,
            'correspondence':'MANO21: wrist; each proximal/middle/distal origin and native fingertip site',
            'frame_method':'literal camera-to-world landmarks; proper rigid anatomical initialization; bounded root+joint fit; no Shadow rotation',
            'native_mount_to_wrist_preserved':True}
    groups={'hand_object':[],'hand_table':[],'hand_self':[]}
    hand=set(map(int,mapping['hand_geoms']))
    for contact in data.contact:
        a,b=int(contact.geom1),int(contact.geom2)
        if cube in (a,b) and (a in hand or b in hand):groups['hand_object'].append(float(contact.dist))
        if table in (a,b) and (a in hand or b in hand):groups['hand_table'].append(float(contact.dist))
        if a in hand and b in hand:groups['hand_self'].append(float(contact.dist))
    report['final_pose_contact_diagnostics']={name:{'contact_count':len(values),
        'minimum_signed_contact_distance_m':min(values) if values else None,
        'maximum_penetration_m':max(0.,-min(values)) if values else 0.}
        for name,values in groups.items()}
    report['collision_scope']='active native MuJoCo collision pairs after source exclusions; diagnostic only, not exact source surface-fidelity acceptance'
    report['joint_limits_satisfied']=bool(np.all(fit.x[6:]>=lower) and np.all(fit.x[6:]<=upper))
    return report,data.mocap_pos[0].copy(),data.mocap_quat[0].copy(),fit.x[6:].copy(),fitted

def run_trial(model,data,mapping,root_position,root_quat,grasp_q,config,output):
    """Move only ideal wrist fixture and actuator commands; cube remains free."""
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cube')
    cube_geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube')
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table')
    initial=data.qpos[:7].copy()
    initial_velocity=data.qvel[:6].copy()
    open_q=np.clip(np.zeros(20),mapping['limits'][:,0],mapping['limits'][:,1])
    # Reset before trial only. All later object motion comes from mj_step.
    mujoco.mj_resetData(model,data)
    data.qpos[:7]=initial
    data.qvel[:6]=initial_velocity
    data.qpos[mapping['addresses']]=open_q
    data.mocap_pos[0]=root_position+[0,0,config['approach_height_m']]
    data.mocap_quat[0]=root_quat
    data.ctrl[:]=np.clip(open_q,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
    mujoco.mj_forward(model,data)
    dt=float(model.opt.timestep); times=[]; poses=[]; qpos=[]; commands=[]; forces=[]; contact_rows=[]
    video_path=Path(output)/'video.mp4'
    writer=imageio.get_writer(video_path,fps=config['video_fps'],codec='libx264',quality=7,
                              macro_block_size=None)
    sample_every=max(1,round(1/(config['video_fps']*dt)))
    durations=np.cumsum([config[k] for k in ('approach_s','close_s','lift_s','hold_s')])
    first_image=None; phases=[]
    physics_times=[];physics_poses=[];physics_qpos=[];physics_qvel=[];physics_commands=[];physics_forces=[]
    hold_lifts=[];maximum_lift=0.
    option=mujoco.MjvOption(); option.geomgroup[2]=0; option.sitegroup[:]=0
    try:
        with mujoco.Renderer(model,height=config['video_height'],width=config['video_width']) as renderer:
            for step in range(int(round(durations[-1]/dt))):
                t=step*dt
                if t<durations[0]:
                    phase='approach'; alpha=t/durations[0]
                    pos=root_position+[0,0,config['approach_height_m']*(1-alpha)]
                    command=open_q
                elif t<durations[1]:
                    phase='close'; alpha=(t-durations[0])/config['close_s']
                    pos=root_position; command=(1-alpha)*open_q+alpha*grasp_q
                elif t<durations[2]:
                    phase='lift'; alpha=(t-durations[1])/config['lift_s']
                    pos=root_position+[0,0,config['lift_command_m']*alpha]; command=grasp_q
                else:
                    phase='hold';pos=root_position+[0,0,config['lift_command_m']];command=grasp_q
                if phase=='hold':hold_lifts.append(float(data.qpos[2]-initial[2]))
                data.mocap_pos[0]=pos;data.mocap_quat[0]=root_quat
                data.ctrl[:]=np.clip(command,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
                mujoco.mj_step(model,data)
                if not all(np.isfinite(x).all() for x in (data.qpos,data.qvel,data.qacc)):
                    raise RuntimeError('nonfinite physics state')
                if any(w.number for w in data.warning):
                    raise RuntimeError('MuJoCo runtime warning')
                current_lift=float(data.qpos[2]-initial[2])
                maximum_lift=max(maximum_lift,current_lift)
                if phase=='hold':hold_lifts.append(current_lift)
                physics_times.append(float(data.time));physics_poses.append(data.qpos[:7].copy())
                physics_qpos.append(data.qpos.copy());physics_qvel.append(data.qvel.copy())
                physics_commands.append(data.ctrl.copy());physics_forces.append(data.actuator_force.copy())
                if step%sample_every==0 or step==int(round(durations[-1]/dt))-1:
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera='capture',scene_option=option)
                    frame=renderer.render().copy();writer.append_data(frame)
                    if first_image is None:first_image=frame
                    pairs=[(int(c.geom1),int(c.geom2),float(c.dist)) for c in data.contact]
                    hand=set(map(int,mapping['hand_geoms']))
                    hand_cube=sum(cube_geom in (a,b) and (a in hand or b in hand) for a,b,_ in pairs)
                    hand_table=sum(table in (a,b) and (a in hand or b in hand) for a,b,_ in pairs)
                    contact_rows.append([hand_cube,hand_table])
                    times.append(float(data.time));poses.append(data.xpos[cube].copy())
                    qpos.append(data.qpos.copy());commands.append(data.ctrl.copy());forces.append(data.actuator_force.copy());phases.append(phase)
            renderer.update_scene(data,camera='capture',scene_option=option)
            imageio.imwrite(Path(output)/'final.png',renderer.render())
    finally:
        writer.close()
    if first_image is not None:imageio.imwrite(Path(output)/'initial.png',first_image)
    poses=np.array(poses); times=np.array(times); hold=np.array(phases)=='hold'
    lift=poses[:,2]-initial[2]
    hold_min=float(min(hold_lifts)) if hold_lifts else 0.
    success=bool(hold_lifts and hold_min>=config['success_lift_m'])
    contacts=np.array(contact_rows)
    np.savez_compressed(Path(output)/'trajectory.npz',time_s=times,object_positions_m=poses,
        qpos=np.array(qpos),commanded_targets=np.array(commands),actuator_force_nm=np.array(forces),
        hand_object_table_contact_counts=contacts,phase=np.array(phases))
    np.savez_compressed(Path(output)/'physics_trace.npz',time_s=np.array(physics_times),
        object_pose_wxyz=np.array(physics_poses),qpos=np.array(physics_qpos),qvel=np.array(physics_qvel),
        commanded_targets=np.array(physics_commands),actuator_force_nm=np.array(physics_forces))
    events=[]
    for index,phase in enumerate(phases):
        if index==0 or phases[index-1]!=phase:
            events.append({'time_s':float(times[index]),'event':'phase_start','phase':phase})
        for column,label in ((0,'hand_object_contact'),(1,'hand_table_contact')):
            previous=bool(contacts[index-1,column]) if index else False
            current=bool(contacts[index,column])
            if current!=previous:events.append({'time_s':float(times[index]),'event':label+'_start' if current else label+'_end'})
    dump(Path(output)/'events.json',{'contact_event_resolution_s':sample_every*dt,
        'contact_events_are_sampled_not_exact_onset':True,'physics_step_s':dt,'events':events})
    return {'status':'completed','success':success,
            'failure_type':None if success else ('no_object_contact' if not contacts[:,0].any() else 'lift_or_hold_failure'),
            'metrics':{'maximum_lift_m':maximum_lift,'minimum_hold_lift_m':hold_min,
                       'final_lift_m':float(lift[-1]),'required_lift_m':config['success_lift_m'],
                       'required_hold_s':config['hold_s'],'hand_table_contact_samples':int((contacts[:,1]>0).sum()),
                       'hand_object_contact_samples':int((contacts[:,0]>0).sum()),'saved_samples':len(times),
                       'physics_steps':len(physics_times),'hold_check':'all physics steps including hold boundary',
                       'maximum_actuator_force_nm':np.abs(np.array(physics_forces)).max(0).tolist()},
            'video_path':str(video_path),'trajectory_path':str(Path(output)/'trajectory.npz')}

def run(scene,proposal,assets,output,config=None):
    cfg={**DEFAULT,**(config or {})}; output=Path(output)
    if output.exists():raise FileExistsError('output must be a new directory')
    output.mkdir(parents=True)
    started=time.monotonic()
    result={'status':'runtime_failed','success':None,'failure_type':None,'metrics':{}}
    try:
        receipt=fetch_assets(assets)
        targets,native_wrist,camera=load_proposal(scene,proposal)
        model,data,xml,mapping=build_model(scene,assets,cfg)
        report,p,q,joints,fitted=retarget(model,data,mapping,targets,cfg)
        report.update(asset_commit=COMMIT,assets_sha256=receipt['sha256'],config=cfg,
                      proposal_sha256=sha(Path(proposal)/'proposal.npz'),
                      scene_xml_sha256=sha(Path(scene)/'scene.xml'),
                      ideal_wrist_fixture=True,native_actuator_force_limits_preserved=True,
                      source_global_physics_overridden_by_approved_scene=True)
        report['initial_cube_velocity_m_s_rad_s']=data.qvel[:6].tolist()
        report['scene_contact_override']=json.loads((Path(scene)/'scene_config.json').read_text())['contact']
        report['contact_policy']='approved cube-scene geom friction/contact defaults also apply to native hand collision geoms; native collision geometry and assembly exclusions unchanged'
        dump(output/'report.json',report)
        (output/'scene.xml').write_text(xml)
        np.savez_compressed(output/'retarget.npz',source_landmarks_world_m=targets,
                            robot_landmarks_world_m=fitted,joint_targets_rad=joints,
                            root_mount_position_m=p,root_mount_quaternion_wxyz=q)
        if report['mean_tip_error_m']>cfg['maximum_mean_tip_error_m'] or report['maximum_tip_error_m']>cfg['maximum_tip_error_m']:
            result.update(status='validation_failed',failure_type='retarget_residual_exceeds_demo_guard',
                          metrics={'mean_tip_error_m':report['mean_tip_error_m'],'maximum_tip_error_m':report['maximum_tip_error_m']})
        else:
            result=run_trial(model,data,mapping,p,q,joints,cfg,output)
    except (ValueError,FileNotFoundError) as error:
        result.update(status='validation_failed',success=None,failure_type=type(error).__name__,error=str(error))
    except Exception as error:
        result.update(status='runtime_failed',success=None,failure_type=type(error).__name__,error=str(error))
    result.update(elapsed_s=time.monotonic()-started,report_path=str(output/'report.json'),
                  scope='synthetic physics demonstration; force-limited fingers with ideal moving wrist; no hardware/exact-fidelity claim')
    dump(output/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('scene','proposal','assets','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--config')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text()) if args.config else None
    result=run(args.scene,args.proposal,args.assets,args.output,config)
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
