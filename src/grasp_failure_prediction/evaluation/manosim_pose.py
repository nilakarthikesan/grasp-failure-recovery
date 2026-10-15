"""Named, static upstream MANOSim placement; no HUG decoding or control execution.

Inputs are an explicit WORLD wrist joint-anchor transform and fifteen absolute
parent-to-child BODY rotations. A ball qpos is the extra rotation after the
body's MJCF rest quaternion, so those rotations are not blindly flattened into
qpos. Non-root body origins may be bone centroids: joint anchors use jnt_pos.

The exact upstream right-hand topology/actuator names are required, but unrelated
scene coordinates may precede/follow it. The adapter owns scratch data restored
from one frozen full integration state, changes named hand qpos/velocities only,
and calls mj_forward at time zero. ctrl is never a pose vector or written here.
Source/shape hashes assert asset identity, not proof of geometry generation from
betas, continuous skin equivalence, biomechanics, tracking or physical fidelity.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

UPSTREAM_REVISION = 'c96e2d5cd70838a2b025b60b85a7e4e06af1b3ac'
EXPECTED_SHAPE_SHA256 = '3e9a85085ae67f36745ce647b904acb4e3620e8ac46378755498291fcf010ddf'
POSE_DOMAIN = 'upstream_manosim_right_unlimited_ball_static_v1'
ROTATION_CONVENTION = 'parent_to_child_body_rotation_v1'
POSE_SCHEMA = 'manosim_static_literal_named_qpos_v1'
PARENTS = (-1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14)
BODY_NAMES = tuple(f'mano_body_{i}' for i in range(16))
BALL_NAMES = ('wrist_rot',) + tuple(f'mano_joint_{i}' for i in range(1,16))
SLIDE_NAMES = ('wrist_tx','wrist_ty','wrist_tz')
LOCAL_NAMES = BALL_NAMES[1:]
JOINT_NAMES = SLIDE_NAMES + BALL_NAMES
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
TOL = 1e-12
_MODEL_ARRAYS = ('names','body_parentid','body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
 'body_mocapid','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_qposadr','jnt_dofadr','jnt_limited','jnt_range','qpos0',
 'geom_type','geom_bodyid','geom_pos','geom_quat','geom_size','geom_dataid','geom_contype','geom_conaffinity',
 'mesh_vert','mesh_face','mesh_vertadr','mesh_faceadr','mesh_vertnum','mesh_facenum','dof_damping','dof_armature',
 'actuator_trntype','actuator_trnid','actuator_gear','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_ctrlrange')


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _array(value,shape,name):
    a=np.array(value,dtype=float,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'Finite {shape} {name} required')
    return a

def _readonly(value):
    a=np.array(value,copy=True);a.flags.writeable=False;return a

def _rotation(value,name):
    r=_array(value,(3,3),name)
    defect=float(np.max(np.abs(np.einsum('ji,jk->ik',r,r,optimize=False)-np.eye(3))))
    if defect>TOL or abs(float(np.linalg.det(r))-1.)>TOL:raise ValueError('Proper orthonormal rotation required: '+name)
    return r

def _transform(value,name):
    t=_array(value,(4,4),name)
    if not np.array_equal(t[3],[0.,0.,0.,1.]):raise ValueError('Literal homogeneous bottom row required')
    _rotation(t[:3,:3],name+' rotation');return t

def _quat(value):
    q=_array(value,(4,),'wxyz quaternion')
    if abs(float(np.sqrt(np.einsum('i,i->',q,q)))-1.)>TOL:raise ValueError('Unit literal wxyz quaternion required; no input normalization')
    return q

def _quat_matrix(q):
    w,x,y,z=_quat(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def _matrix_quat(r):
    """Convert validated rotation, retaining full float64 conversion precision."""
    r=_rotation(r,'quaternion conversion rotation');trace=float(np.trace(r))
    if trace>0:
        s=2*np.sqrt(1+trace);q=np.array([s/4,(r[2,1]-r[1,2])/s,(r[0,2]-r[2,0])/s,(r[1,0]-r[0,1])/s])
    else:
        k=int(np.argmax(np.diag(r)))
        if k==0:
            s=2*np.sqrt(1+r[0,0]-r[1,1]-r[2,2]);q=np.array([(r[2,1]-r[1,2])/s,s/4,(r[0,1]+r[1,0])/s,(r[0,2]+r[2,0])/s])
        elif k==1:
            s=2*np.sqrt(1+r[1,1]-r[0,0]-r[2,2]);q=np.array([(r[0,2]-r[2,0])/s,(r[0,1]+r[1,0])/s,s/4,(r[1,2]+r[2,1])/s])
        else:
            s=2*np.sqrt(1+r[2,2]-r[0,0]-r[1,1]);q=np.array([(r[1,0]-r[0,1])/s,(r[0,2]+r[2,0])/s,(r[1,2]+r[2,1])/s,s/4])
    raw=q.copy();raw_norm=float(np.sqrt(np.einsum('i,i->',q,q)));q/=raw_norm
    if q[np.flatnonzero(q)[0]]<0:q=-q;raw=-raw
    return q,dict(raw_norm=raw_norm,normalization_max_change=float(np.max(np.abs(q-raw))),
        applied_norm=float(np.sqrt(np.einsum('i,i->',q,q))),rotation_max_error=float(np.max(np.abs(_quat_matrix(q)-r))),
        dtype='float64',decimal_rounding_applied=False)

def _expected_shape_digest(value):
    # The expectation must come from a separately frozen protocol; never infer it
    # from the candidate shape file and thereby admit every shape automatically.
    return _sha256_digest(EXPECTED_SHAPE_SHA256 if value is None else value)


def _sha256_digest(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Explicit expected shape must be a lowercase SHA-256 digest')
    return value


def asset_shape_identity(shape_asset,*,expected_shape_sha256=None):
    path=Path(shape_asset).resolve();digest=_sha(path)
    if digest!=_expected_shape_digest(expected_shape_sha256):raise ValueError('Upstream/HUG right-hand shape asset identity mismatch')
    return dict(schema_version='manosim_right_shape_asset_identity_v1',shape_asset_sha256=digest,
        handedness='right',linear_units='m',upstream_revision=UPSTREAM_REVISION)

_GENERATED_HAND_PREFIX='assets/literal_native/mano_rhand/'
_GENERATED_SHAPE_PATH='assets/mano_rhand_shape.npy'
_GENERATED_FILES=frozenset({_GENERATED_SHAPE_PATH,
    _GENERATED_HAND_PREFIX+'mesh_hand.xml',_GENERATED_HAND_PREFIX+'capsule_hand.xml',
    *(_GENERATED_HAND_PREFIX+f'meshes/mano_bone_{i}.obj' for i in range(16))})


def _strict_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate builder report JSON key: '+key)
            result[key]=value
        return result
    def constant(value):raise ValueError('Nonfinite builder report JSON constant: '+value)
    def number(value):
        result=float(value)
        if not np.isfinite(result):raise ValueError('Nonfinite builder report JSON number')
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=constant,parse_float=number)


def _builder_receipt(binding,root,xml,shape,expected,consumed,add):
    if not isinstance(binding,Mapping) or set(binding)!={'path','sha256'}:
        raise ValueError('Exact builder provenance path/sha256 binding required')
    path=Path(binding['path'])
    if not path.is_absolute() or not path.is_file() or _sha(path)!=_sha256_digest(binding['sha256']):
        raise ValueError('Builder provenance receipt bytes changed or path invalid')
    report=_strict_json(path)
    if (not isinstance(report,Mapping) or report.get('schema_version')!='manosim_literal_native_shape_build_report_v1'
            or report.get('status')!='COMPLETE' or report.get('failure') is not None
            or report.get('upstream_revision')!=UPSTREAM_REVISION or report.get('shape_asset_sha256')!=expected
            or not isinstance(report.get('generated_root'),str) or not Path(report['generated_root']).is_absolute()
            or Path(report['generated_root']).resolve()!=root):
        raise ValueError('Builder provenance report shape/root/schema/status mismatch')
    generated=report.get('generated_files')
    if not isinstance(generated,Mapping) or set(generated)!=_GENERATED_FILES:
        raise ValueError('Builder provenance requires exact nineteen generated assets')
    xml_relative=str(xml.relative_to(root));shape_relative=str(shape.relative_to(root))
    expected_consumed={_GENERATED_HAND_PREFIX+xml.name,
        *(_GENERATED_HAND_PREFIX+f'meshes/mano_bone_{i}.obj' for i in range(16))}
    if (xml_relative!=_GENERATED_HAND_PREFIX+xml.name or shape_relative!=_GENERATED_SHAPE_PATH
            or consumed!=expected_consumed or generated[_GENERATED_SHAPE_PATH]!=expected):
        raise ValueError('Generated XML/mesh/shape closure differs from literal build inventory')
    artifacts=report.get('artifacts')
    if not isinstance(artifacts,list):raise ValueError('Builder write-time artifact inventory required')
    indexed={}
    for row in artifacts:
        if (not isinstance(row,Mapping) or set(row)!={'path','sha256','bytes','label'}
                or not isinstance(row['path'],str) or not Path(row['path']).is_absolute()
                or type(row['bytes']) is not int or row['bytes']<0 or not isinstance(row['label'],str) or not row['label']):
            raise ValueError('Exact builder write-time artifact binding required')
        artifact=Path(row['path']).resolve()
        if artifact in indexed or not artifact.is_relative_to(root):raise ValueError('Duplicated/outside builder artifact path')
        _sha256_digest(row['sha256']);indexed[artifact]=row
    for relative,digest in generated.items():
        candidate=add(root/relative)
        if _sha(candidate)!=_sha256_digest(digest):raise ValueError('Generated MANOSim asset changed: '+relative)
        row=indexed.get(candidate)
        if row is None or row['sha256']!=digest or row['bytes']!=candidate.stat().st_size:
            raise ValueError('Generated write-time asset binding mismatch: '+relative)
    # Re-read the receipt after all files, preventing a moving manifest admission.
    if _sha(path)!=binding['sha256']:raise ValueError('Builder provenance receipt changed during validation')
    return dict(path=str(path.resolve()),sha256=binding['sha256'])


def source_asset_closure(xml_path,shape_asset,*,source_root,extra_source_paths=(),expected_shape_sha256=None,builder_provenance=None):
    """Hash complete XML/include/file-asset closure, plus the explicit shape file.

    This reads files only. It does not compile a model or infer the shape used by
    a previous mesh build. Paths outside source_root and unhandled file-bearing
    elements fail closed rather than being silently omitted. Canonical assets
    retain the v1 payload. An explicit shape expectation plus a complete frozen
    builder receipt is required for the generated-asset v2 domain; hashes assert
    identity and provenance, not geometry generation correctness or fidelity.
    """
    root=Path(source_root).resolve();xml=Path(xml_path).resolve();shape=Path(shape_asset).resolve()
    expected=_expected_shape_digest(expected_shape_sha256)
    if builder_provenance is not None and expected_shape_sha256 is None:
        raise ValueError('Generated assets require a protocol-provided explicit expected shape SHA-256')
    if builder_provenance is None and expected!=EXPECTED_SHAPE_SHA256:
        raise ValueError('Noncanonical shape requires frozen builder provenance')
    if xml.name not in ('mesh_hand.xml','capsule_hand.xml') or xml.parent.name!='mano_rhand':
        raise ValueError('Literal upstream right-hand mano_rhand mesh_hand/capsule_hand source path required')
    files={};seen=set();compiler=None
    def add(path):
        path=Path(path).resolve()
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Source asset outside declared existing closure: '+str(path))
        files[str(path)]=_sha(path);return path
    def visit(path):
        nonlocal compiler
        path=add(path)
        if path in seen:raise ValueError('Repeated/cyclic XML include is unsupported')
        seen.add(path);tree=ET.parse(path).getroot()
        for c in tree.iter('compiler'):
            if compiler is not None:raise ValueError('Multiple XML compiler domains are unsupported')
            compiler=c
        for node in tree.iter():
            name=node.get('file')
            if name is None:continue
            if node.tag=='include':visit(path.parent/name)
            elif node.tag in ('mesh','texture','hfield','skin'):
                directory='' if compiler is None else compiler.get('assetdir','')
                if compiler is not None:directory=compiler.get('meshdir' if node.tag in ('mesh','skin') else 'texturedir',directory)
                add(xml.parent/directory/name)
            else:raise ValueError('Unhandled file-bearing source element: '+node.tag)
    visit(xml)
    if compiler is None or compiler.get('angle')!='radian':raise ValueError('Explicit radian XML compiler domain required')
    consumed={str(Path(path).relative_to(root)) for path in files}
    provenance=None
    if builder_provenance is not None:
        provenance=_builder_receipt(builder_provenance,root,xml,shape,expected,consumed,add)
    add(shape)
    for path in extra_source_paths:add(path)
    identity=asset_shape_identity(shape,expected_shape_sha256=expected)
    portable={str(Path(path).relative_to(root)):digest for path,digest in sorted(files.items())}
    payload=dict(schema_version='manosim_static_source_asset_closure_v1',upstream_revision=UPSTREAM_REVISION,
        source_variant=xml.name,xml_relative_path=str(xml.relative_to(root)),files=portable,asset_shape_identity=identity)
    if provenance is not None:
        payload.update(schema_version='manosim_static_source_asset_closure_v2',
            expected_shape_sha256=expected,builder_provenance=provenance)
    return dict(**payload,source_root=str(root),xml_path=str(xml),shape_asset_path=str(shape),
        absolute_files=dict(sorted(files.items())),closure_sha256=_json_hash(payload),
        identity_is_not_mesh_generation_or_fidelity_proof=True)

def _model_hash(model):
    h=hashlib.sha256()
    for key in _MODEL_ARRAYS:
        a=np.ascontiguousarray(getattr(model,key));h.update(key.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    h.update(np.asarray(model.opt.gravity).tobytes());h.update(np.asarray(model.opt.timestep).tobytes());return h.hexdigest()

def _id(model,kind,name):
    value=int(mujoco.mj_name2id(model,kind,name))
    if value<0:raise ValueError('Missing literal upstream name: '+name)
    return value

def _descendant(model,body,root):
    while body>0:
        if body==root:return True
        body=int(model.body_parentid[body])
    return False

@dataclass(frozen=True)
class ManoSimPoseState:
    named_qpos: dict
    bone_joint_names: tuple
    joint_anchor_local_m: np.ndarray
    joint_anchor_world_m: np.ndarray
    independent_joint_anchor_world_m: np.ndarray
    independent_body_origin_world_m: np.ndarray
    independent_world_body_rotations: np.ndarray
    body_origin_world_m: np.ndarray
    local_body_rotations: np.ndarray
    world_body_rotations: np.ndarray
    wrist_anchor_world_transform: np.ndarray
    root_body_world_transform: np.ndarray
    palm_geometry_world_transforms: tuple
    independent_readback: dict
    quaternion_precision: dict
    joint_addresses: tuple
    actuator_addresses: tuple
    asset_shape_identity: dict
    source_closure_sha256: str
    model_sha256: str
    baseline_state_sha256: str

    def serialize_pose(self):
        return dict(schema_version=POSE_SCHEMA,domain=POSE_DOMAIN,units='m',handedness='right',
            qpos_convention='named_scalar_slide_metres_and_ball_wxyz_quaternions',
            ctrl_convention='not_commanded_frozen_baseline_retained',
            named_qpos={name:float(value) if np.ndim(value)==0 else np.asarray(value).tolist() for name,value in self.named_qpos.items()},
            asset_shape_identity=dict(self.asset_shape_identity),source_closure_sha256=self.source_closure_sha256,
            model_sha256=self.model_sha256,baseline_state_sha256=self.baseline_state_sha256)


class ManoSimStaticPoseAdapter:
    def __init__(self,model,baseline_data,*,source_assets,asset_shape_identity,domain=POSE_DOMAIN,expected_shape_sha256=None):
        if domain!=POSE_DOMAIN:raise ValueError('Unknown MANOSim pose domain')
        self.model=model;self.sources=json.loads(json.dumps(source_assets));self.shape=json.loads(json.dumps(asset_shape_identity))
        self.expected_shape_sha256=_expected_shape_digest(expected_shape_sha256)
        if self.sources.get('schema_version')=='manosim_static_source_asset_closure_v2' and expected_shape_sha256 is None:
            raise ValueError('Generated adapter requires protocol-provided explicit expected shape SHA-256')
        self._check_identity(self.shape);self._verify_sources()
        self.model_sha256=_model_hash(model)
        if baseline_data.time!=0:raise ValueError('Frozen static baseline must have time zero')
        self.bodies=np.array([_id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in BODY_NAMES])
        self.balls=np.array([_id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in BALL_NAMES])
        self.slides=np.array([_id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in SLIDE_NAMES])
        if (np.any(model.jnt_type[self.balls]!=mujoco.mjtJoint.mjJNT_BALL)
                or np.any(model.jnt_type[self.slides]!=mujoco.mjtJoint.mjJNT_SLIDE)):
            raise ValueError('Literal upstream ball/slide joint types required')
        hand_joints=set(map(int,np.r_[self.slides,self.balls]))
        if set(np.flatnonzero([_descendant(model,int(model.jnt_bodyid[j]),int(self.bodies[0])) for j in range(model.njnt)]))!=hand_joints:
            raise ValueError('Extra/aliased hand joints are not the upstream domain')
        descendants={i for i in range(1,model.nbody) if _descendant(model,i,int(self.bodies[0]))}
        if descendants!=set(map(int,self.bodies)):raise ValueError('Exactly the sixteen upstream hand bodies required')
        for i,body in enumerate(self.bodies):
            expected=0 if i==0 else self.bodies[PARENTS[i]]
            if model.body_parentid[body]!=expected or model.jnt_bodyid[self.balls[i]]!=body:
                raise ValueError('Upstream named bone hierarchy/ownership required')
        if np.any(model.jnt_bodyid[self.slides]!=self.bodies[0]) or np.max(self.slides)>=self.balls[0]:
            raise ValueError('Wrist slides must precede wrist ball on world-root body')
        if not np.allclose(model.jnt_axis[self.slides],np.eye(3),rtol=0,atol=TOL):raise ValueError('Missing/aliased named wrist translation axes')
        if np.any(model.jnt_limited[np.r_[self.slides,self.balls]]):raise ValueError('Unknown limited-joint domain; this profile is the upstream unlimited layout')
        self.qadr={name:int(model.jnt_qposadr[j]) for name,j in zip(JOINT_NAMES,np.r_[self.slides,self.balls],strict=True)}
        self.dadr={name:int(model.jnt_dofadr[j]) for name,j in zip(JOINT_NAMES,np.r_[self.slides,self.balls],strict=True)}
        self.actuators=self._actuator_contract()
        self._state=np.empty(mujoco.mj_stateSize(model,STATE));mujoco.mj_getState(model,baseline_data,self._state,STATE)
        if not np.isfinite(self._state).all():raise ValueError('Finite frozen full integration state required')
        self._state=_readonly(self._state);self.baseline_state_sha256=hashlib.sha256(self._state.tobytes()).hexdigest()
        self._ctrl=_readonly(baseline_data.ctrl);self.data=mujoco.MjData(model);self.forward_calls=0
        self.address_records=tuple(dict(name=name,joint_id=int(j),qpos_address=self.qadr[name],dof_address=self.dadr[name],
            qpos_width=1 if name in SLIDE_NAMES else 4,dof_width=1 if name in SLIDE_NAMES else 3)
            for name,j in zip(JOINT_NAMES,np.r_[self.slides,self.balls],strict=True))

    def _check_identity(self,identity):
        expected=dict(schema_version='manosim_right_shape_asset_identity_v1',shape_asset_sha256=self.expected_shape_sha256,
            handedness='right',linear_units='m',upstream_revision=UPSTREAM_REVISION)
        if identity!=expected or self.sources.get('asset_shape_identity')!=expected:raise ValueError('Wrong shape/side/units/upstream identity')

    def _verify_sources(self):
        source=self.sources
        version=source.get('schema_version')
        if version not in ('manosim_static_source_asset_closure_v1','manosim_static_source_asset_closure_v2'):
            raise ValueError('Versioned source asset closure required')
        if version=='manosim_static_source_asset_closure_v2' and source.get('expected_shape_sha256')!=self.expected_shape_sha256:
            raise ValueError('Generated closure expected shape differs from adapter protocol')
        for path,digest in source['absolute_files'].items():
            if _sha(path)!=digest:raise ValueError('Bound MANOSim source asset changed: '+path)
        portable={str(Path(path).relative_to(source['source_root'])):digest for path,digest in sorted(source['absolute_files'].items())}
        payload={key:source[key] for key in ('schema_version','upstream_revision','source_variant','xml_relative_path','asset_shape_identity')}
        payload['files']=portable
        if version=='manosim_static_source_asset_closure_v2':
            payload.update(expected_shape_sha256=source['expected_shape_sha256'],builder_provenance=source['builder_provenance'])
        if portable!=source['files'] or _json_hash(payload)!=source['closure_sha256']:raise ValueError('MANOSim source closure identity changed')
        reconstructed=source_asset_closure(source['xml_path'],source['shape_asset_path'],source_root=source['source_root'],
            extra_source_paths=source['absolute_files'],expected_shape_sha256=self.expected_shape_sha256,
            builder_provenance=source.get('builder_provenance') if version=='manosim_static_source_asset_closure_v2' else None)
        if reconstructed['absolute_files']!=source['absolute_files'] or reconstructed['closure_sha256']!=source['closure_sha256']:
            raise ValueError('Source XML/asset dependency closure is incomplete')
        if hasattr(self,'_state') and hashlib.sha256(self._state.tobytes()).hexdigest()!=self.baseline_state_sha256:
            raise ValueError('Frozen baseline integration state changed')
        if hasattr(self,'model_sha256') and _model_hash(self.model)!=self.model_sha256:raise ValueError('Bound compiled MANOSim model changed')

    def _actuator_contract(self):
        m=self.model;records=[];used=set()
        for name,joint in zip(SLIDE_NAMES,self.slides,strict=True):
            aname='wrist_motor_t'+name[-1];aid=_id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,aname)
            records.append((aname,aid,int(joint),np.array([1.,0,0,0,0,0]),'scalar_slide'))
        for i,joint in enumerate(self.balls):
            for axis,label in enumerate('xyz'):
                aname=('wrist_motor_r'+label) if i==0 else f'mano_motor_{i}_{label}'
                aid=_id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,aname);gear=np.zeros(6);gear[axis]=1
                records.append((aname,aid,int(joint),gear,'ball_joint_scalar_gear_axis'))
        for name,aid,joint,gear,kind in records:
            if (aid in used or m.actuator_trntype[aid]!=mujoco.mjtTrn.mjTRN_JOINT
                    or int(m.actuator_trnid[aid,0])!=joint or not np.array_equal(m.actuator_gear[aid],gear)
                    or m.actuator_gaintype[aid]!=mujoco.mjtGain.mjGAIN_FIXED or m.actuator_biastype[aid]!=mujoco.mjtBias.mjBIAS_AFFINE
                    or m.actuator_gainprm[aid,0]<=0 or m.actuator_biasprm[aid,1]!=-m.actuator_gainprm[aid,0]):
                raise ValueError('Missing/aliased/mismatched upstream scalar actuator axes: '+name)
            used.add(aid)
        hand_ids=set(map(int,np.r_[self.slides,self.balls]))
        actual={a for a in range(m.nu) if m.actuator_trntype[a]==mujoco.mjtTrn.mjTRN_JOINT and int(m.actuator_trnid[a,0]) in hand_ids}
        if actual!=used:raise ValueError('Unexpected aliased hand actuator')
        return tuple(dict(name=n,ctrl_address=a,joint_id=j,gear=g.tolist(),coordinate_kind=k,
            ctrl_written=False,qpos_quaternion_is_not_ctrl=True) for n,a,j,g,k in records)

    def _independent(self,named):
        m=self.model;origins=[];rotations=[];anchors=[]
        for i,body in enumerate(self.bodies):
            rest=_quat_matrix(m.body_quat[body]);q=_quat_matrix(named[BALL_NAMES[i]]);c=m.jnt_pos[self.balls[i]]
            if i==0:
                delta=np.array([named[name]-m.qpos0[self.qadr[name]] for name in SLIDE_NAMES])
                translated=m.body_pos[body]+np.einsum('ij,j->i',rest,delta,optimize=False)
                anchor=translated+np.einsum('ij,j->i',rest,c,optimize=False);r=np.einsum('ij,jk->ik',rest,q,optimize=False)
            else:
                p=PARENTS[i];pre=np.einsum('ij,jk->ik',rotations[p],rest,optimize=False)
                translated=origins[p]+np.einsum('ij,j->i',rotations[p],m.body_pos[body],optimize=False)
                anchor=translated+np.einsum('ij,j->i',pre,c,optimize=False);r=np.einsum('ij,jk->ik',pre,q,optimize=False)
            origins.append(anchor-np.einsum('ij,j->i',r,c,optimize=False));rotations.append(r);anchors.append(anchor)
        return np.asarray(origins),np.asarray(rotations),np.asarray(anchors)

    def stage(self,wrist_world_transform,local_body_rotations,*,asset_shape_identity,units='m',handedness='right',rotation_convention=ROTATION_CONVENTION,domain=POSE_DOMAIN):
        if units!='m' or handedness!='right' or rotation_convention!=ROTATION_CONVENTION or domain!=POSE_DOMAIN:
            raise ValueError('Unknown units/side/rotation convention/domain')
        self._check_identity(asset_shape_identity);target=_transform(wrist_world_transform,'explicit world wrist-anchor transform')
        if not isinstance(local_body_rotations,Mapping) or set(local_body_rotations)!=set(LOCAL_NAMES):raise ValueError('Exactly fifteen literal named local rotations required')
        local={name:_rotation(local_body_rotations[name],name) for name in LOCAL_NAMES};m=self.model
        rest=_quat_matrix(m.body_quat[self.bodies[0]]);c=m.jnt_pos[self.balls[0]]
        displacement=target[:3,3]-m.body_pos[self.bodies[0]]-np.einsum('ij,j->i',rest,c,optimize=False)
        slide_delta=np.linalg.solve(rest,displacement)
        named={name:float(slide_delta[i]+m.qpos0[self.qadr[name]]) for i,name in enumerate(SLIDE_NAMES)};precision={}
        for i,name in enumerate(BALL_NAMES):
            requested=target[:3,:3] if i==0 else local[name]
            initial=_quat_matrix(m.body_quat[self.bodies[i]])
            q,record=_matrix_quat(np.einsum('ji,jk->ik',initial,requested,optimize=False))
            named[name]=q;precision[name]=record
        state=self._apply(named,precision)
        if (np.max(np.abs(state.wrist_anchor_world_transform-target))>TOL
                or any(np.max(np.abs(state.local_body_rotations[i]-local[name]))>TOL for i,name in enumerate(LOCAL_NAMES,1))):
            raise ValueError('Requested wrist/local rotation readback changed')
        return state

    def stage_serialized(self,payload):
        expected={'schema_version','domain','units','handedness','qpos_convention','ctrl_convention','named_qpos','asset_shape_identity',
            'source_closure_sha256','model_sha256','baseline_state_sha256'}
        if not isinstance(payload,Mapping) or set(payload)!=expected or payload['schema_version']!=POSE_SCHEMA:
            raise ValueError('Exact named static pose schema required')
        if (payload['domain']!=POSE_DOMAIN or payload['units']!='m' or payload['handedness']!='right'
                or payload['qpos_convention']!='named_scalar_slide_metres_and_ball_wxyz_quaternions'
                or payload['ctrl_convention']!='not_commanded_frozen_baseline_retained'):
            raise ValueError('Unknown serialized pose convention/domain')
        self._check_identity(payload['asset_shape_identity'])
        for key,actual in (('source_closure_sha256',self.sources['closure_sha256']),('model_sha256',self.model_sha256),('baseline_state_sha256',self.baseline_state_sha256)):
            if payload[key]!=actual:raise ValueError('Serialized pose source/model/baseline identity differs')
        values=payload['named_qpos']
        if not isinstance(values,Mapping) or set(values)!=set(JOINT_NAMES):raise ValueError('Literal named qpos inventory required; no flat vectors or aliases')
        named={name:float(_array(values[name],(),name)) for name in SLIDE_NAMES}
        named.update({name:_quat(values[name]) for name in BALL_NAMES})
        precision={name:dict(literal_roundtrip=True,input_norm=float(np.linalg.norm(named[name])),normalization_applied=False,decimal_rounding_applied=False) for name in BALL_NAMES}
        return self._apply(named,precision)

    def _apply(self,named,precision):
        self._verify_sources();m=self.model;d=self.data
        mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,self._state.copy(),STATE)
        for name in SLIDE_NAMES:d.qpos[self.qadr[name]]=named[name];d.qvel[self.dadr[name]]=0
        for name in BALL_NAMES:d.qpos[self.qadr[name]:self.qadr[name]+4]=named[name];d.qvel[self.dadr[name]:self.dadr[name]+3]=0
        warnings_before=d.warning.number.copy()
        self.forward_calls+=1;mujoco.mj_forward(m,d)
        if np.any(d.warning.number>warnings_before):raise ValueError('MuJoCo warning during static pose readback')
        if d.time!=0 or not np.array_equal(d.ctrl,self._ctrl) or not all(np.isfinite(a).all() for a in (d.qpos,d.qvel,d.qacc,d.ctrl)):raise ValueError('Static time/control/finite-state contract changed')
        for name in JOINT_NAMES:
            actual=d.qpos[self.qadr[name]] if name in SLIDE_NAMES else d.qpos[self.qadr[name]:self.qadr[name]+4]
            if not np.array_equal(actual,named[name]):raise ValueError('Literal named qpos changed during forward')
        origins,world,anchors=self._independent(named)
        actual_o=d.xpos[self.bodies].copy();actual_r=d.xmat[self.bodies].reshape(16,3,3).copy();actual_a=d.xanchor[self.balls].copy()
        if not all(np.isfinite(value).all() for value in (origins,world,anchors,actual_o,actual_r,actual_a)):
            raise ValueError('Nonfinite independent/actual joint-anchor or body-transform readback')
        local=np.empty_like(actual_r)
        for i in range(16):local[i]=actual_r[i] if i==0 else np.einsum('ji,jk->ik',actual_r[PARENTS[i]],actual_r[i],optimize=False)
        errors=dict(anchor_max_error_m=float(np.max(np.abs(anchors-actual_a))),body_origin_max_error_m=float(np.max(np.abs(origins-actual_o))),
            world_rotation_max_error=float(np.max(np.abs(world-actual_r))),time_s=float(d.time),integration_steps=0,ctrl_written=False,
            units='m',body_origin_substitution_used=False,independent_reference='compiled rest poses, ball pivot offsets and named qpos; separate recursive SE(3) arithmetic')
        if max(errors['anchor_max_error_m'],errors['body_origin_max_error_m'],errors['world_rotation_max_error'])>TOL:
            raise ValueError('Independent joint-anchor/body kinematic readback differs')
        wrist=np.eye(4);wrist[:3,:3]=actual_r[0];wrist[:3,3]=actual_a[0]
        root=np.eye(4);root[:3,:3]=actual_r[0];root[:3,3]=actual_o[0]
        geoms=[]
        for gid in np.flatnonzero(m.geom_bodyid==self.bodies[0]):
            t=np.eye(4);t[:3,:3]=d.geom_xmat[gid].reshape(3,3);t[:3,3]=d.geom_xpos[gid]
            geoms.append(dict(geom_id=int(gid),world_transform=t.tolist(),meaning='actual root-body/palm geom frame, not a substituted wrist joint anchor'))
        self._verify_sources()
        return ManoSimPoseState(named_qpos={name:float(value) if np.ndim(value)==0 else _readonly(value) for name,value in named.items()},
            bone_joint_names=BALL_NAMES,joint_anchor_local_m=_readonly(m.jnt_pos[self.balls]),joint_anchor_world_m=_readonly(actual_a),
            independent_joint_anchor_world_m=_readonly(anchors),independent_body_origin_world_m=_readonly(origins),independent_world_body_rotations=_readonly(world),
            body_origin_world_m=_readonly(actual_o),local_body_rotations=_readonly(local),world_body_rotations=_readonly(actual_r),
            wrist_anchor_world_transform=_readonly(wrist),root_body_world_transform=_readonly(root),palm_geometry_world_transforms=tuple(geoms),
            independent_readback=errors,quaternion_precision=precision,joint_addresses=self.address_records,actuator_addresses=self.actuators,
            asset_shape_identity=dict(self.shape),source_closure_sha256=self.sources['closure_sha256'],model_sha256=self.model_sha256,baseline_state_sha256=self.baseline_state_sha256)
