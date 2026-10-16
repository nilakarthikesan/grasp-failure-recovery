"""Authored native MANOSim capsule target/command coordinates, never admission.

Pure source/array operations only: no MuJoCo/MANO constructors, FK or stepping.
Ball position controls are principal quaternion-log transmission lengths, not
Euler angles or quaternion components. They match pinned MuJoCo3.4 joint
transmission semantics; its moment arms remain the authored gear axes.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

UPSTREAM_REVISION='c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac'
SOURCE_DOMAIN='mano_source_postAA32_exp64_camera64_named_state_v1'
COMMAND_DOMAIN='manosim_authored_capsule_principal_ball_position_v1'
JOINT_NAMES=tuple('wrist_t'+x for x in 'xyz')+('wrist_rot',)+tuple(f'mano_joint_{i}' for i in range(1,16))
BODY_NAMES=tuple(f'mano_body_{i}' for i in range(16))
ACTUATOR_NAMES=tuple('wrist_motor_t'+x for x in 'xyz')+tuple('wrist_motor_r'+x for x in 'xyz')+tuple(f'mano_motor_{i}_{x}' for i in range(1,16) for x in 'xyz')
PARENTS=(-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
PI_GUARD_RAD=1e-6 # predeclared command-chart exclusion, never a fidelity allowance
MJ_MINVAL=1e-15
COORDINATE_SHA256='88f9a9b7b6710fc251563d208d727bc55d205e541bbeaed25d655996a0a83433'
INERTIA_RULE='mujoco_3_4_0_mjuu_eig3_postcondition_v1'
JACOBI_EPS=1e-12
INERTIA_SOURCE_SHA256='f9d5ef77317707039f12658e620ff3393f1eb8a0e7087111e4740207f2ee2522'
CLOSED_FLAGS=dict(accepted_mapping=False,collection_or_training_allowed=False,physical_model_qualified=False)
_COMPILED_ARRAYS=('body_parentid','body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_qposadr','jnt_dofadr','jnt_limited','qpos0','dof_damping','dof_armature','dof_M0','actuator_trntype','actuator_trnid','actuator_gear','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_gainprm','actuator_biasprm','actuator_ctrllimited','actuator_ctrlrange','actuator_forcelimited','actuator_forcerange','geom_type','geom_sameframe','geom_bodyid','geom_pos','geom_quat','geom_size','geom_dataid','mesh_vert','mesh_face','mesh_pos','mesh_quat','geom_solref','geom_solimp','geom_friction','geom_priority','geom_contype','geom_conaffinity','body_gravcomp','actuator_actrange')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _digest(a):
    a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def _canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _literal(x,shape,dtype,name):
    a=np.asarray(x)
    if a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():raise ValueError('Literal finite '+name+' required')
    return a
def _readonly(x):
    a=np.array(x,copy=True);a.flags.writeable=False;return a
def _strict_json(path):
    def pairs(rows):
        d={}
        for k,v in rows:
            if k in d:raise ValueError('Duplicate receipt key')
            d[k]=v
        return d
    def reject(v):raise ValueError('Nonfinite receipt value')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=reject)
def _numbers(text,n):
    a=np.asarray([float(x) for x in text.split()],dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Malformed authored numeric attribute')
    return a

def _rotation(q):
    q=_literal(np.asarray(q),(4,),'float64','proper source/compiled quaternion')
    if abs(float(q@q)-1)>1e-12:raise ValueError('Proper source/compiled quaternion required')
    w,x,y,z=q;scale=2/float(q@q)
    return np.array([[1-scale*(y*y+z*z),scale*(x*y-z*w),scale*(x*z+y*w)],
        [scale*(x*y+z*w),1-scale*(x*x+z*z),scale*(y*z-x*w)],
        [scale*(x*z-y*w),scale*(y*z+x*w),1-scale*(x*x+y*y)]])


def inertia_representation_receipt(source_fullinertia,principal_inertia,inertia_quaternion):
    """Validate the compiler's approximate eigenframe, never retune source I.

    Pinned user_util.cc611-700 stops at max offdiag<1e-12 OR cosine>
    1-1e-12. The latter implies b<=diagonal_gap*t/(1-t*t), where
    t*t < cosine**(-2)-1. gamma32 encloses the finite normal 2x2 Schur
    arithmetic; outward rounding makes the coefficient conservative.
    gamma128*||I||inf encloses the small quaternion/matrix arithmetic.
    These are representation postconditions, not physical calibration gates.
    """
    f=_literal(np.asarray(source_fullinertia),(6,),'float64','source full inertia')
    principal=_literal(np.asarray(principal_inertia),(3,),'float64','compiled principal inertia')
    if np.any(principal<=0):raise ValueError('Positive compiled principal inertia required')
    I=np.array([[f[0],f[3],f[4]],[f[3],f[1],f[5]],[f[4],f[5],f[2]]])
    R=_rotation(inertia_quaternion)
    D=np.einsum('ki,kl,lj->ij',R,I,R,optimize=False)
    diagonal=np.diag(D);off=D-np.diag(diagonal)
    unit=np.finfo(np.float64).eps/2
    gamma32=32*unit/(1-32*unit);gamma128=128*unit/(1-128*unit)
    # A computed c>fl(1-kEigEPS) has exact c above this lower envelope.
    lower=np.nextafter((1-JACOBI_EPS)/(1+gamma32),0.)
    square_lower=np.nextafter(lower*lower,0.)
    t2_upper=np.nextafter(np.nextafter(1/square_lower,np.inf)-1,np.inf)
    t_upper=np.nextafter(np.sqrt(t2_upper),np.inf)
    coefficient=np.nextafter(t_upper/(1-t2_upper)*(1+gamma32),np.inf)
    scale=float(np.max(np.sum(np.abs(I),axis=1)));arithmetic=gamma128*scale
    gap=float(np.max(diagonal)-np.min(diagonal))
    off_bound=max(JACOBI_EPS,float(coefficient)*gap)+arithmetic
    diag_error=float(np.max(np.abs(diagonal-principal)));off_error=float(np.max(np.abs(off)))
    reconstructed=np.einsum('ik,k,jk->ij',R,principal,R,optimize=False)
    if not np.isfinite([scale,arithmetic,gap,off_bound,diag_error,off_error]).all() or scale<=0:raise ValueError('Finite inertia representation arithmetic required')
    if diag_error>arithmetic:raise ValueError('Authored compiled inertia principal diagonal differs')
    if off_error>off_bound:raise ValueError('Authored compiled inertia exceeds pinned Jacobi stop envelope; nonconvergence not accepted')
    return dict(rule=INERTIA_RULE,primary_source_sha256=INERTIA_SOURCE_SHA256,kEigEPS=JACOBI_EPS,
        source_fullinertia_kg_m2=f.tolist(),compiled_principal_kg_m2=principal.tolist(),compiled_iquat=np.asarray(inertia_quaternion).tolist(),
        source_in_compiled_frame_kg_m2=D.tolist(),diagonal_error_kg_m2=diag_error,diagonal_arithmetic_bound_kg_m2=arithmetic,
        maximum_offdiagonal_kg_m2=off_error,jacobi_offdiagonal_bound_kg_m2=off_bound,diagonal_span_kg_m2=gap,
        cosine_stop_coefficient=float(coefficient),gamma32=gamma32,gamma128=gamma128,
        maximum_reconstructed_tensor_error_kg_m2=float(np.max(np.abs(reconstructed-I))),
        compiler_maximum_iterations=500,actual_compiler_iteration_count=None,
        iteration_limit_is_not_assumed_converged=True,output_eigenframe_postconditions_passed=True,source_inertia_changed=False)


def _coordinates():
    path=Path(__file__).resolve().parents[3]/'research_artifacts/manosim_native_source_pipeline_2026-10-15_v1/source_coordinates.py'
    if _sha(path)!=COORDINATE_SHA256:raise ValueError('Qualified source coordinate implementation changed')
    spec=importlib.util.spec_from_file_location('qualified_capsule_source_coordinates',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@dataclass(frozen=True)
class CapsuleTarget:
    qpos:np.ndarray
    shape_digest:str
    rest_digest:str
    source_receipt:dict
    def metadata(self):
        return dict(schema_version='manosim_capsule_source_target_v1',source_domain=SOURCE_DOMAIN,
            qpos_sha256=_digest(self.qpos),shape_digest=self.shape_digest,rest_digest=self.rest_digest,
            source_receipt=self.source_receipt,pose_offset_applied=False,**CLOSED_FLAGS)


def target_from_source(post_axisangles_float32,translation_camera_float32,camera_world,rest_quaternions,*,shape_float32):
    """Same qualified post-AA32/camera64 state; no MANO or engine evaluation."""
    shape=_literal(shape_float32,(1,10),'float32','saved source shape row')
    coord=_coordinates();q,receipt=coord.source_initialized_qpos(post_axisangles_float32,translation_camera_float32,camera_world,rest_quaternions)
    recovered=coord.recover_source_inputs(q,rest_quaternions,camera_world,post_axisangles_float32)
    guard=coord.require_original_bytes(recovered,post_axisangles_float32,translation_camera_float32)
    source=dict(coordinate_source_sha256=COORDINATE_SHA256,initialization=receipt,original_input_byte_guard=guard,
        original_AA32_sha256=_digest(post_axisangles_float32),original_translation32_sha256=_digest(translation_camera_float32),camera64_sha256=_digest(camera_world))
    return CapsuleTarget(_readonly(q),_digest(shape),_digest(rest_quaternions),source)


def inspect_authored_assets(capsule_xml,shape_asset,builder_report_binding,*,expected_shape_sha256,native_shape_float32):
    """Verify only durable generated19-asset bytes and write-time producer receipt.

    Archived upstream/temp dependency closure is not asserted live or reverified.
    The caller separately binds the completed independent producer receipt.
    """
    if set(builder_report_binding)!={'path','sha256'}:raise ValueError('Exact builder binding required')
    report_path=Path(builder_report_binding['path']).resolve()
    if _sha(report_path)!=builder_report_binding['sha256']:raise ValueError('Builder report changed')
    report=_strict_json(report_path);root=Path(report['generated_root']).resolve()
    prefix='assets/literal_native/mano_rhand/'
    expected={'assets/mano_rhand_shape.npy',prefix+'mesh_hand.xml',prefix+'capsule_hand.xml'}|{prefix+f'meshes/mano_bone_{i}.obj' for i in range(16)}
    if (report.get('schema_version')!='manosim_literal_native_shape_build_report_v1' or report.get('status')!='COMPLETE'
        or report.get('failure') is not None or report.get('upstream_revision')!=UPSTREAM_REVISION
        or set(report.get('generated_files',{}))!=expected or report.get('shape_asset_sha256')!=expected_shape_sha256):raise ValueError('Exact completed literal-shape build required')
    shape_path=Path(shape_asset).resolve();xml=Path(capsule_xml).resolve()
    if shape_path!=root/'assets/mano_rhand_shape.npy' or xml!=root/(prefix+'capsule_hand.xml'):raise ValueError('Generated capsule/shape selection differs')
    shape=np.load(shape_path,allow_pickle=False);native=_literal(native_shape_float32,(1,10),'float32','native shape')
    if _sha(shape_path)!=expected_shape_sha256 or shape.dtype!=np.float32 or shape.shape!=(1,10) or shape.tobytes()!=native.tobytes():raise ValueError('Literal native shape identity required')
    rows={}
    for a in report['artifacts']:
        p=Path(a['path']).resolve()
        if p in rows:raise ValueError('Duplicate producer artifact path')
        rows[p]=a
    files={}
    for relative,digest in report['generated_files'].items():
        p=root/relative
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):raise ValueError('Generated asset escaped durable root')
        a=rows.get(p.resolve())
        if a is None or a['sha256']!=digest or a['bytes']!=p.stat().st_size or _sha(p)!=digest:raise ValueError('Generated write-time asset bytes changed')
        files[str(p)]=digest
    tree=ET.parse(xml).getroot()
    if tree.tag!='mujoco' or tree.findall('.//include') or tree.findall('.//deformable') or tree.find('compiler').get('angle')!='radian':raise ValueError('Standalone original radian capsule XML required')
    bodies={b.get('name'):b for b in tree.findall('./worldbody//body')}
    joints={j.get('name'):j for j in tree.findall('./worldbody//joint')}
    acts={a.get('name'):a for a in tree.findall('./actuator/*')}
    if len(tree.findall('./worldbody//body'))!=16 or len(tree.findall('./worldbody//joint'))!=19 or len(tree.findall('./actuator/*'))!=51 or len(bodies)!=16 or set(bodies)!=set(BODY_NAMES) or len(joints)!=19 or set(joints)!=set(JOINT_NAMES) or len(acts)!=51 or set(acts)!=set(ACTUATOR_NAMES):raise ValueError('Exact named16/19/51 authored inventory required')
    defaults=tree.findall('./default/geom')
    if len(defaults)!=1 or defaults[0].get('solimp')!='0.5 0.95 0.001' or defaults[0].get('solref')!='0.04 1':raise ValueError('Authored contact compliance changed')
    body_records=[]
    for i,name in enumerate(BODY_NAMES):
        b=bodies[name];geoms=b.findall('geom');inertial=b.find('inertial')
        if len(geoms)!=1 or inertial is None or geoms[0].get('type')!=('mesh' if i==0 else 'capsule'):raise ValueError('Original mesh-palm/15capsule and explicit inertials required')
        if i==0 and (geoms[0].get('priority')!='1' or geoms[0].get('solimp')!='0.005 0.9 0.025 0.5 2'):raise ValueError('Authored softer palm changed')
        owned=[JOINT_NAMES[j] for j in (range(4) if i==0 else [i+3])]
        if [j.get('name') for j in b.findall('joint')]!=owned:raise ValueError('Authored joint ownership/order changed')
        for namej in owned:
            j=joints[namej];ball=namej not in JOINT_NAMES[:3]
            expected_damping=5. if i==0 else .01
            if j.get('type')!=('ball' if ball else 'slide') or j.get('limited')!='false' or float(j.get('damping','0'))!=expected_damping:raise ValueError('Authored unlimited joint/damping changed')
            if ball and float(j.get('armature','0'))!=(.01 if i==0 else .001):raise ValueError('Authored armature changed')
        body_records.append(dict(name=BODY_NAMES[i],attributes=dict(b.attrib),mass=float(inertial.get('mass')),inertial=dict(inertial.attrib),geom=dict(geoms[0].attrib)))
    actuator_records=[]
    for k,name in enumerate(ACTUATOR_NAMES):
        a=acts[name];joint=JOINT_NAMES[k] if k<3 else JOINT_NAMES[3+(k-3)//3]
        gear=np.array([1.,0,0,0,0,0]) if k<3 else np.eye(6)[(k-3)%3]
        parsed=_numbers(a.get('gear','1 0 0'),len(a.get('gear','1 0 0').split()))
        parsed=np.pad(parsed,(0,6-len(parsed))) if len(parsed)<=6 else parsed
        kp=300. if k<3 else (50. if k<6 else 30.);ratio=1.5 if 3<=k<6 else 1.
        if a.tag!='position' or a.get('joint')!=joint or not np.array_equal(parsed,gear) or float(a.get('kp'))!=kp or float(a.get('dampratio'))!=ratio:raise ValueError('Authored51 gear/gain/damping settings changed')
        actuator_records.append(dict(name=name,joint=joint,gear=gear.tolist(),kp=kp,dampratio=ratio))
    receipt=dict(schema_version='manosim_authored_capsule_durable_assets_v1',builder_report=dict(path=str(report_path),sha256=_sha(report_path)),
        files=files,shape_asset_sha256=expected_shape_sha256,shape_array_digest=_digest(shape),capsule_xml_sha256=_sha(xml),body_records=body_records,
        actuators=actuator_records,joints={name:dict(joints[name].attrib) for name in JOINT_NAMES},default_contact=dict(defaults[0].attrib),total_serialized_mass_kg=sum(b['mass'] for b in body_records),
        upstream_revision=UPSTREAM_REVISION,archived_temporary_dependency_closure_live_verified=False,**CLOSED_FLAGS)
    receipt['closure_sha256']=_canonical(receipt);return receipt


def _model_signature(model):
    h=hashlib.sha256()
    for key in _COMPILED_ARRAYS:
        a=np.ascontiguousarray(getattr(model,key));h.update(key.encode());h.update(_digest(a).encode())
    return h.hexdigest()

@dataclass(frozen=True)
class CompiledCapsuleContract:
    joint_ids:tuple
    qpos_addresses:tuple
    dof_addresses:tuple
    body_ids:tuple
    actuator_ids:tuple
    gears:np.ndarray
    rest_quaternions:np.ndarray
    model_signature:str
    asset_closure_sha256:str
    shape_digest:str
    inertia_representations:tuple
    def metadata(self):
        return dict(schema_version='manosim_authored_capsule_compiled_contract_v2',joint_names=list(JOINT_NAMES),actuator_names=list(ACTUATOR_NAMES),
            joint_ids=list(self.joint_ids),qpos_addresses=list(self.qpos_addresses),dof_addresses=list(self.dof_addresses),body_ids=list(self.body_ids),actuator_ids=list(self.actuator_ids),
            gears=self.gears.tolist(),rest_quaternions=self.rest_quaternions.tolist(),model_signature=self.model_signature,asset_closure_sha256=self.asset_closure_sha256,
            command_domain=COMMAND_DOMAIN,shape_digest=self.shape_digest,inertia_representations=list(self.inertia_representations),**CLOSED_FLAGS)


def validate_compiled_contract(model,resolve_name,assets,rest_quaternions):
    """Check existing compiled arrays only. resolve_name(kind,name) performs lookup."""
    if (model.nq,model.nv,model.nu,model.njnt)!=(67,51,51,19):raise ValueError('Exact native67/51/51/19 required')
    if assets.get('schema_version')!='manosim_authored_capsule_durable_assets_v1' or assets.get('closure_sha256')!=_canonical({k:v for k,v in assets.items() if k!='closure_sha256'}):raise ValueError('Asset receipt content changed')
    for path,digest in assets['files'].items():
        if _sha(path)!=digest:raise ValueError('Durable generated asset changed')
    rest=_literal(rest_quaternions,(16,4),'float64','qualified rest quaternions')
    body=tuple(int(resolve_name('body',n)) for n in BODY_NAMES);joint=tuple(int(resolve_name('joint',n)) for n in JOINT_NAMES);act=tuple(int(resolve_name('actuator',n)) for n in ACTUATOR_NAMES)
    if min(body)<=0 or len(set(body))!=16 or set(joint)!=set(range(19)) or set(act)!=set(range(51)):raise ValueError('Missing/aliased authored names')
    qadr=tuple(int(model.jnt_qposadr[j]) for j in joint);dadr=tuple(int(model.jnt_dofadr[j]) for j in joint)
    expectedq=tuple(range(3))+tuple(3+4*i for i in range(16));expectedd=tuple(range(3))+tuple(3+3*i for i in range(16))
    if qadr!=expectedq or dadr!=expectedd or not np.array_equal(model.body_quat[list(body)],rest):raise ValueError('Actual named state/rest domain differs')
    if np.max(np.abs(np.sum(rest*rest,axis=1)-1))>1e-12 or np.any(model.jnt_limited[list(joint)]):raise ValueError('Proper literal rest and unlimited authored joints required')
    inertia_receipts=[]
    for i,bid in enumerate(body):
        if int(model.body_parentid[bid])!=(0 if i==0 else body[PARENTS[i]]):raise ValueError('Anatomical tree differs')
        original=assets['body_records'][i];attrs=original['attributes'];inertial=original['inertial'];geom=original['geom']
        if set(attrs)&{'euler','axisangle','xyaxes','zaxis'}:raise ValueError('Authored explicit quaternion/default identity frame required')
        if np.max(np.abs(_rotation(model.body_quat[bid])-_rotation(_numbers(attrs.get('quat','1 0 0 0'),4))))>128*np.finfo(float).eps:
            raise ValueError('Authored body rest frame changed')
        if not np.array_equal(model.body_pos[bid],_numbers(attrs.get('pos','0 0 0'),3)) or not np.array_equal(model.body_ipos[bid],_numbers(inertial.get('pos','0 0 0'),3)):
            raise ValueError('Authored body/inertial origins changed')
        if not np.isfinite(model.body_mass[bid]) or model.body_mass[bid]!=original['mass'] or np.any(np.asarray(model.body_inertia[bid])<=0):raise ValueError('Authored positive mass/inertia required')
        inertia_receipts.append(dict(body_name=BODY_NAMES[i],body_id=bid,**inertia_representation_receipt(_numbers(inertial['fullinertia'],6),model.body_inertia[bid],model.body_iquat[bid])))
        gids=np.flatnonzero(np.asarray(model.geom_bodyid)==bid)
        if len(gids)!=1:raise ValueError('Exactly one authored collider per anatomical body required')
        gid=int(gids[0])
        if int(model.geom_type[gid])!=(7 if i==0 else 3):raise ValueError('Authored mesh-palm/capsule collider type changed')
        if i==0:
            mesh_id=int(resolve_name('mesh','mano_bone_0'))
            if mesh_id<0 or int(model.geom_dataid[gid])!=mesh_id:raise ValueError('Authored palm mesh binding changed')
            if set(geom)&{'euler','axisangle','xyaxes','zaxis'}:raise ValueError('Authored palm quaternion/default identity frame required')
            # Pinned3.4 user_objects.cc3422-3428: compiler accumulates mesh frame
            # into authored geom frame. This verifies placement, not mesh fidelity.
            source_R=_rotation(_numbers(geom.get('quat','1 0 0 0'),4))
            mesh_pos=_literal(np.asarray(model.mesh_pos[mesh_id]),(3,),'float64','compiled mesh position')
            expected_pos=_numbers(geom.get('pos','0 0 0'),3)+np.einsum('ij,j->i',source_R,mesh_pos,optimize=False)
            expected_R=np.einsum('ij,jk->ik',source_R,_rotation(model.mesh_quat[mesh_id]),optimize=False)
            bound=128*np.finfo(float).eps
            if (np.max(np.abs(np.asarray(model.geom_pos[gid])-expected_pos))>bound*max(1.,float(np.max(np.abs(expected_pos))))
                or np.max(np.abs(_rotation(model.geom_quat[gid])-expected_R))>bound):raise ValueError('Authored compiled palm mesh frame changed')
        else:
            ends=_numbers(geom['fromto'],6).reshape(2,3);radius=float(geom['size'])
            if model.geom_size[gid,0]!=radius:raise ValueError('Authored capsule radius changed')
            qw,qx,qy,qz=np.asarray(model.geom_quat[gid]);axis=np.array([2*(qx*qz+qy*qw),2*(qy*qz-qx*qw),1-2*(qx*qx+qy*qy)])
            half=float(model.geom_size[gid,1]);actual_ends=np.array([model.geom_pos[gid]-half*axis,model.geom_pos[gid]+half*axis])
            discrepancy=min(np.max(np.abs(actual_ends-ends)),np.max(np.abs(actual_ends-ends[::-1])))
            if discrepancy>128*np.finfo(float).eps*max(1.,float(np.max(np.abs(ends)))):raise ValueError('Authored capsule endpoints changed')
    for k,jid in enumerate(joint):
        if int(model.jnt_type[jid])!=(2 if k<3 else 1) or int(model.jnt_bodyid[jid])!=body[0 if k<4 else k-3]:raise ValueError('Named type/anatomical owner differs')
        source_joint=assets['joints'][JOINT_NAMES[k]]
        expected_pos=_numbers(source_joint.get('pos','0 0 0'),3)
        width=1 if k<3 else 3
        if (not np.array_equal(model.jnt_pos[jid],expected_pos) or not np.all(np.asarray(model.dof_damping[dadr[k]:dadr[k]+width])==float(source_joint['damping']))
            or not np.all(np.asarray(model.dof_armature[dadr[k]:dadr[k]+width])==float(source_joint.get('armature','0')))):
            raise ValueError('Authored anchor/damping/armature changed')
        if k>=3 and not np.array_equal(model.qpos0[qadr[k]:qadr[k]+4],np.array([1.,0,0,0])):raise ValueError('Authored ball reference changed')
        if k<3 and (not np.array_equal(model.jnt_axis[jid],np.eye(3)[k]) or model.qpos0[qadr[k]]!=0):raise ValueError('Original world-slide convention differs')
    if not np.array_equal(model.body_pos[body[0]],np.zeros(3)) or not np.array_equal(model.jnt_pos[joint[3]],np.zeros(3)):raise ValueError('Literal wrist anchor must be root origin')
    gears=[]
    for k,aid in enumerate(act):
        a=assets['actuators'][k];jid=joint[k] if k<3 else joint[3+(k-3)//3];gear=np.asarray(a['gear'])
        gain=np.asarray(model.actuator_gainprm[aid]);bias=np.asarray(model.actuator_biasprm[aid])
        if (int(model.actuator_trntype[aid])!=0 or int(model.actuator_trnid[aid,0])!=jid or not np.array_equal(model.actuator_gear[aid],gear)
            or int(model.actuator_gaintype[aid])!=0 or int(model.actuator_biastype[aid])!=1 or int(model.actuator_dyntype[aid])!=0
            or not np.isfinite(gain).all() or not np.isfinite(bias).all() or gain[0]!=a['kp'] or bias[0]!=0 or bias[1]!=-a['kp'] or bias[2]>=0 or model.actuator_ctrllimited[aid] or model.actuator_forcelimited[aid]):raise ValueError('Authored scalar position transmission/gain differs')
        dof=dadr[k] if k<3 else dadr[3+(k-3)//3]+(k-3)%3
        mass=float(model.dof_M0[dof]);expected_kv=2*a['dampratio']*np.sqrt(a['kp']*mass)
        if not np.isfinite(mass) or mass<=0 or abs(float(bias[2])+expected_kv)>64*np.finfo(float).eps*expected_kv:raise ValueError('Authored compiler-derived actuator damping changed')
        gears.append(gear)
    for key in _COMPILED_ARRAYS:
        if not np.isfinite(np.asarray(getattr(model,key))).all():raise ValueError('Nonfinite compiled mapping/model operand')
    return CompiledCapsuleContract(joint,qadr,dadr,body,act,_readonly(gears),_readonly(rest),_model_signature(model),assets['closure_sha256'],assets['shape_array_digest'],tuple(inertia_receipts))


def verify_compiled_contract(model,contract):
    if _model_signature(model)!=contract.model_signature:raise ValueError('Compiled capsule contract changed')
    return True


def principal_ball_coordinates(quaternions):
    q=_literal(quaternions,(16,4),'float64','named unit ball quaternions')
    norm=np.sqrt(np.einsum('ij,ij->i',q,q,optimize=False))
    if np.max(np.abs(norm-1))>1e-12:raise ValueError('Unit literal ball qpos required; no pose fitting')
    normalized=q/norm[:,None] # Same representation normalization as mj_transmission.
    length=np.sqrt(np.einsum('ij,ij->i',normalized[:,1:],normalized[:,1:],optimize=False))
    axis=np.zeros((16,3));nonzero=length>=MJ_MINVAL;axis[nonzero]=normalized[nonzero,1:]/length[nonzero,None]
    sine=length.copy();sine[~nonzero]=0.
    speed=2*np.arctan2(sine,normalized[:,0]);wrapped=speed>np.pi;speed[wrapped]-=2*np.pi
    rotvec=axis*speed[:,None];margin=np.pi-np.abs(speed)
    if np.any(margin<=PI_GUARD_RAD):raise ValueError('Principal ball actuator chart near pi; declared branch refusal')
    return rotvec,margin,wrapped,float(np.max(np.abs(normalized-q)))

@dataclass(frozen=True)
class CapsuleCommand:
    ctrl:np.ndarray
    principal_rotvecs:np.ndarray
    pi_margin_rad:np.ndarray
    wrapped_input_quaternions:np.ndarray
    representation_normalization_max_change:float
    model_signature:str
    def metadata(self):
        return dict(schema_version='manosim_authored_capsule_command_v1',command_domain=COMMAND_DOMAIN,ctrl_sha256=_digest(self.ctrl),
            minimum_pi_margin_rad=float(np.min(self.pi_margin_rad)),pi_guard_rad=PI_GUARD_RAD,wrapped_input_quaternions=self.wrapped_input_quaternions.tolist(),
            representation_normalization_max_change=self.representation_normalization_max_change,model_signature=self.model_signature,
            quaternion_components_used_as_ctrl=False,euler_angles_used=False,clipping_applied=False,gravity_feedforward_or_offset_applied=False,
            ball_moment_arm_is_authored_gear_not_full_log_jacobian=True,**CLOSED_FLAGS)


def command_from_qpos(qpos,contract):
    q=_literal(qpos,(67,),'float64','native named qpos');rv,margin,wrapped,error=principal_ball_coordinates(q[3:].reshape(16,4))
    canonical=np.empty(51);canonical[:3]=q[:3]*contract.gears[:3,0]
    canonical[3:]=np.einsum('ij,ij->i',np.repeat(rv,3,axis=0),contract.gears[3:,:3],optimize=False)
    ctrl=np.empty(51);ctrl[list(contract.actuator_ids)]=canonical
    return CapsuleCommand(_readonly(ctrl),_readonly(rv),_readonly(margin),_readonly(wrapped),error,contract.model_signature)


def command_from_target(target,contract):
    if target.shape_digest!=contract.shape_digest or target.rest_digest!=_digest(contract.rest_quaternions):raise ValueError('Target shape/rest differs from compiled capsule')
    return command_from_qpos(target.qpos,contract)


def require_continuous_principal_targets(qpos_sequence,contract):
    """Finite sampled command-chart check only; no continuous/dynamic certificate."""
    q=np.asarray(qpos_sequence)
    if q.ndim!=2 or q.shape[1:]!=(67,) or q.dtype!=np.float64 or len(q)<1:raise ValueError('Finite named qpos sequence required')
    commands=[command_from_qpos(row,contract) for row in q]
    for a,b in zip(commands,commands[1:]):
        if np.any(np.linalg.norm(a.principal_rotvecs-b.principal_rotvecs,axis=1)>=np.pi):raise ValueError('Sampled principal command branch jump refused; no hidden unwrap')
    return commands
