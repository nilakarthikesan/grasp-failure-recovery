"""Prospectively frozen analytic contact benchmark; no engine work at import/prepare."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import warnings

RUNTIME_ROOT = Path(__file__).resolve().parents[2]/'runs/manosim_mujoco_3_4_0_runtime_v1'
sys.path.insert(0, str(RUNTIME_ROOT))
loaded = sys.modules.get('mujoco')
if loaded is not None and (getattr(loaded, '__version__', None) != '3.4.0'
        or not Path(loaded.__file__).resolve().is_relative_to(RUNTIME_ROOT.resolve())):
    raise RuntimeError('Already-loaded engine is not the isolated MuJoCo3.4.0 runtime')
import numpy as np
if np.__version__ != '2.2.6': raise RuntimeError('Unchanged base NumPy2.2.6 required')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIMITIVE = ROOT/'research_artifacts/manosim_flex_rigid_qualification_mj340_2026-10-15_v1'
SCHEMA = 'manosim_analytic_contact_dynamics_mj340_v1'
REVISION = 'e55fff5dea6f1d5dd7963ca52eecc41d05ad0922'
BUDGET = dict(models=14, data=14, initial_forwards=14, steps=10650, contact_force_reads=10652,
              maximum_seconds=120, outer_seconds=130)
SOLVER = dict(type='Newton',integrator='Euler',cone='elliptic',iterations=100,tolerance=1e-12,
    solref=[-10000.,-200.],solimp=[.9,.9,.001,.5,2.],mu=.5,impratio=1.,noslip_iterations=0)
FLAGS = dict(native_contact_law_qualified=False, articulated_MANO_dynamics_validated=False,
    real_hand_physical_calibration=False, collection_or_training_allowed=False)
FORBIDDEN_IMPORTS = ('torch', 'manotorch', 'manopth', 'smplx', 'hug', 'transformers', 'grasp_failure_prediction')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


checks = load('analytic_contact_checks', HERE/'analytic_checks.py')
factory = load('analytic_contact_factory', HERE/'contact_factory.py')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant '+value)
    def number(value):
        result=float(value)
        if not math.isfinite(result):reject(value)
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject,parse_float=number)
def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def plan():
    result = [dict(id='jacobian_flex', kind='jacobian', geometry='flex', dt_s=.001, steps=0, gravity=0.),
              dict(id='jacobian_solid', kind='jacobian', geometry='plane', dt_s=.001, steps=0, gravity=0.)]
    definitions = [('freefall', 'none', .25), ('support_flex', 'flex', 1.), ('support_solid', 'plane', 1.),
                   ('friction_low', 'flex', .5), ('friction_high', 'flex', .5), ('pair_impact', 'pair', .3)]
    for name, geometry, duration in definitions:
        for dt in (.001, .0005):
            result.append(dict(id=f'{name}_{int(round(1/dt))}hz', kind=name, geometry=geometry,
                dt_s=dt, steps=int(round(duration/dt)), gravity=0. if geometry == 'pair' else -checks.GRAVITY_M_S2))
    return result


def frame():
    axis=np.array([1.,-2.,3.]);axis/=np.linalg.norm(axis); angle=.63
    k=np.array([[0.,-axis[2],axis[1]],[axis[2],0.,-axis[0]],[-axis[1],axis[0],0.]])
    return np.eye(3)+math.sin(angle)*k+(1-math.cos(angle))*(k@k), np.array([.123,-.164,.232]), np.r_[math.cos(angle/2),axis*math.sin(angle/2)]

def numbers(values): return ' '.join(format(float(x), '.17g') for x in np.ravel(values))

def xml(case, model_id=factory.STANDARD_MODEL):
    factory.require_current_model(model_id)
    if case not in plan(): raise ValueError('Exact finite fixture schedule required')
    static=case['kind']=='jacobian'; rotation, translation, quat=frame() if static else (np.eye(3),np.zeros(3),np.array([1.,0,0,0]))
    m,r=checks.MASS,checks.RADIUS_M; condim=1 if case['geometry']=='pair' else 3
    contact=f'condim="{condim}" friction=".5 0 0" solref="-10000 -200" solimp=".9 .9 .001 .5 2" margin="0" gap="0"'
    def sphere(name, position, mass=m, slides=False):
        joints=('<joint name="sx" type="slide" axis="1 0 0"/><joint name="sy" type="slide" axis="0 1 0"/><joint name="sz" type="slide" axis="0 0 1"/>') if slides else f'<freejoint name="{name}_free"/>'
        inertia=.4*mass*r*r
        return f'<body name="{name}" pos="{numbers(position)}" quat="{numbers(quat)}">{joints}<inertial pos="0 0 0" mass="{mass:.17g}" diaginertia="{numbers([inertia]*3)}"/><geom name="{name}_geom" type="sphere" size=".01" {contact}/></body>'
    vertices=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
    floor=''; deform=''
    if case['geometry'] in ('flex','plane'):
        body_start=f'<body name="floor" pos="{numbers(translation)}" quat="{numbers(quat)}">'
        if static and case['geometry']=='flex': body_start+='<freejoint name="floor_free"/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>'
        if case['geometry']=='plane': floor=body_start+f'<geom name="floor_geom" type="plane" size="2 2 .1" {contact}/></body>'
        else:
            floor=body_start+'</body>'
            deform=f'<deformable><flex name="rigid_patch" dim="2" radius="0" body="floor" vertex="{numbers(vertices)}" element="0 1 2"><contact {contact} internal="false" selfcollide="none"/></flex></deformable>'
    center=np.array([0.,-1/3,0.]); height=(r-2e-6 if static else r-checks.rest_penetration() if case['kind'].startswith('friction') else r+.02 if case['kind'].startswith('support') else .5)
    position=rotation@(center+[0,0,height])+translation
    if case['geometry']=='pair': bodies=sphere('sphere',[-.03,0,0])+sphere('sphere2',[.03,0,0],mass=.4)
    else: bodies=floor+sphere('sphere',position,slides=case['kind'].startswith('friction'))
    return (f'<mujoco model="{case["id"]}"><compiler angle="radian" inertiafromgeom="false"/>'
        f'<option timestep="{case["dt_s"]:.17g}" gravity="0 0 {case["gravity"]:.17g}" solver="Newton" integrator="Euler" jacobian="dense" cone="elliptic" iterations="100" tolerance="1e-12" impratio="1" noslip_iterations="0"><flag energy="enable"/></option>'
        f'<worldbody>{bodies}</worldbody>{deform}</mujoco>')


def prior():
    module=load('contact_prior_rigid_qualification',PRIMITIVE/'run_qualification.py')
    path=PRIMITIVE/'run_v1/report.json'; report=read(path); protocol=Path(report['protocol'])
    p,_=module.validate(protocol,report['freeze_commit'],require_head=False)
    policy=module.qualified_rigid_radius_policy(path,sha(path),'zero')
    independent_path=PRIMITIVE/'run_v1/independent_readback.json'; independent=read(independent_path)
    if (independent.get('schema_version')!='manosim_flex_rigid_mj340_independent_saved_readback_v1'
            or independent.get('status')!='VERIFIED_COMPLETED_MJ340_PRIMITIVES_RIGID_ONLY_PASS_FULL_NOT_QUALIFIED'
            or independent.get('active_protocol_sha256')!=sha(protocol)
            or independent.get('all_bindings_verified_before') is not True or independent.get('all_bindings_verified_after') is not True):
        raise ValueError('Exact independent rigid primitive verification required')
    files={str(protocol):sha(protocol),str(path):sha(path),str(independent_path):sha(independent_path)}
    for name,row in independent['artifact_inventory'].items():
        if sha(name)!=row['sha256'] or Path(name).stat().st_size!=row['bytes']: raise ValueError('Prior artifact changed')
        files[name]=row['sha256']
    return module,p,dict(files=files,radius_policy=policy,prior_protocol_sha256=sha(protocol))


def sources(): return [HERE/name for name in ('run_contact_dynamics.py','analytic_checks.py','contact_factory.py','test_contact_dynamics.py','README.md','primary_sources.json','controls.json')]

def prepare(path):
    path=Path(path).resolve()
    if path!=HERE/'protocol.json' or path.exists() or (HERE/'run_v1').exists(): raise ValueError('Fresh exact protocol path required before execution')
    module,old,evidence=prior(); controls=read(HERE/'controls.json')
    if controls.get('exit_code')!=0 or controls.get('integration_calls')!=0: raise ValueError('Successful manufactured controls required')
    bindings=dict(old['bindings']); bindings.update(evidence['files'])
    bindings.update({str(p.resolve()):sha(p) for p in sources()+module.runtime_files()})
    selected=set(old['tracked_sources'])|{str(p.resolve()) for p in sources()}|{name for name in evidence['files'] if Path(name).is_relative_to(ROOT) and Path(name).suffix in ('.py','.json','.md')}
    p=dict(schema_version=SCHEMA,prepared_utc=datetime.now(timezone.utc).isoformat(),runtime=module.runtime(),
        upstream_revision=REVISION,plan=plan(),budget=BUDGET,limits=checks.LIMITS,contact_model_id=factory.STANDARD_MODEL,
        solver=SOLVER,
        xml_sha256={c['id']:hashlib.sha256(xml(c).encode()).hexdigest() for c in plan()},prior=evidence,
        bindings=bindings,tracked_sources=sorted(selected),output_root=str(HERE/'run_v1'),raw_root=str(ROOT/'runs/manosim_contact_dynamics_2026-10-15_v1/run_v1/raw'),**FLAGS)
    write(path,p); return p


def validate(protocol,commit):
    p=read(protocol); module,old,evidence=prior()
    if (Path(protocol).resolve()!=HERE/'protocol.json' or p['schema_version']!=SCHEMA or p['plan']!=plan()
            or p['budget']!=BUDGET or p['limits']!=checks.LIMITS or p['solver']!=SOLVER or p['prior']!=evidence
            or p['runtime']!=module.runtime() or p['upstream_revision']!=REVISION
            or p['contact_model_id']!=factory.STANDARD_MODEL
            or p['output_root']!=str(HERE/'run_v1') or p['raw_root']!=str(ROOT/'runs/manosim_contact_dynamics_2026-10-15_v1/run_v1/raw')
            or any(p.get(key) is not False for key in FLAGS)):
        raise ValueError('Frozen analytic method/runtime/roots changed')
    expected=dict(old['bindings']);expected.update(evidence['files']);expected.update({str(f.resolve()):sha(f) for f in sources()+module.runtime_files()})
    if p['bindings']!=expected: raise ValueError('Exact source/runtime/prior closure changed')
    for name,digest in p['bindings'].items():
        if sha(name)!=digest: raise ValueError('Bound file changed: '+name)
    if p['xml_sha256']!={c['id']:hashlib.sha256(xml(c).encode()).hexdigest() for c in plan()}: raise ValueError('Fixture XML changed')
    frozen=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',commit+'^{commit}'],text=True).strip()
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=frozen: raise ValueError('Exact HEAD hold required')
    selected=set(old['tracked_sources'])|{str(f.resolve()) for f in sources()}|{name for name in evidence['files'] if Path(name).is_relative_to(ROOT) and Path(name).suffix in ('.py','.json','.md')}
    if p['tracked_sources']!=sorted(selected): raise ValueError('Selected Git closure changed')
    for name in sorted(selected|{str(Path(protocol).resolve())}):
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',frozen+':'+str(Path(name).relative_to(ROOT))])
        if hashlib.sha256(blob).hexdigest()!=sha(name): raise ValueError('Committed scientific source changed')
    return p,frozen


class Budget:
    def __init__(self,output):
        self.start=time.monotonic();self.output=Path(output);self.counts={key:0 for key in ('models','data','initial_forwards','steps','contact_force_reads')};self.forbidden=[]
    def charge(self,key):
        if time.monotonic()-self.start>BUDGET['maximum_seconds']: raise RuntimeError('120second scientific budget exceeded')
        if key not in self.counts or self.counts[key]>=BUDGET[key]: self.reject('exhausted_'+key)
        self.counts[key]+=1;self.save()
    def save(self):
        path=self.output/'progress.tmp';path.write_text(json.dumps(dict(counts=self.counts,forbidden=self.forbidden,elapsed_s=time.monotonic()-self.start))+'\n');path.replace(self.output/'progress.json')
    def reject(self,key): self.forbidden.append(key);self.save();raise RuntimeError('Forbidden operation '+key)


class Imports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN_IMPORTS: raise RuntimeError('Native/model/training import forbidden: '+fullname)


def state(data):
    values=(data.qpos,data.qvel,data.qacc,data.ctrl)
    if not all(np.isfinite(v).all() for v in values) or not math.isfinite(float(data.time)): raise RuntimeError('Nonfinite engine state')
    if np.max(np.abs(data.qpos),initial=0)>10 or np.max(np.abs(data.qvel),initial=0)>100 or np.max(np.abs(data.qacc),initial=0)>1e6: raise RuntimeError('Bounded fixture state escaped')


class Engine:
    def __init__(self,module,budget):
        self.mj,self.b=module,budget
        self.original={name:getattr(module,name) for name in ('MjModel','MjData','mj_forward','mj_step','mj_contactForce')}
        self.allowed=[xml(c) for c in plan()];self.models=set();self.pairs=set();self.initialized=set();self.saved={};self.live_models=[];self.live_data=[]
    def compile(self,text):
        if not self.allowed or text!=self.allowed[0]: self.b.reject('unplanned_XML')
        self.allowed.pop(0);self.b.charge('models')
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always');model=self.original['MjModel'].from_xml_string(text)
        if emitted:raise RuntimeError('Compiler emitted warning')
        self.models.add(id(model))
        self.live_models.append(model) # prevent Python-id reuse between finite fixtures
        return model
    def data(self,model):
        if id(model) not in self.models or any(pair[0]==id(model) for pair in self.pairs):self.b.reject('unregistered_or_duplicate_data')
        self.b.charge('data');data=self.original['MjData'](model);self.pairs.add((id(model),id(data)));self.live_data.append(data);return data
    def call(self,kind,model,data):
        if (id(model),id(data)) not in self.pairs:self.b.reject('unregistered_model_data')
        if kind not in ('initial_forwards','steps'):self.b.reject('unknown_pipeline_operation')
        pair=(id(model),id(data))
        if (kind=='initial_forwards' and pair in self.initialized) or (kind=='steps' and pair not in self.initialized):self.b.reject('unordered_pipeline_operation')
        state(data);before=float(data.time);warn=data.warning.number.copy();ctrl=data.ctrl.copy();applied=data.qfrc_applied.copy()
        if np.any(warn):raise RuntimeError('Nonzero warning baseline before engine operation')
        self.b.charge(kind)
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter('always');self.original['mj_step' if kind=='steps' else 'mj_forward'](model,data)
        state(data)
        if emitted or np.any(warn) or not np.array_equal(warn,data.warning.number):raise RuntimeError('Engine numerical/runtime warning')
        if not np.array_equal(ctrl,data.ctrl) or not np.array_equal(applied,data.qfrc_applied):raise RuntimeError('Unexpected controller/applied-force mutation')
        expected=model.opt.timestep if kind=='steps' else 0.
        if abs((data.time-before)-expected)>1e-12: raise RuntimeError('Engine clock increment changed')
        if kind=='initial_forwards':self.initialized.add(pair)
    def force(self,model,data,index):
        if (id(model),id(data)) not in self.pairs or index!=0 or data.ncon!=1:self.b.reject('unregistered_contact_force_read')
        self.b.charge('contact_force_reads');out=np.zeros(6);self.original['mj_contactForce'](model,data,index,out)
        return checks.finite(out,(6,))
    def __enter__(self):
        for name in ('MjModel','MjData','mj_forward','mj_step','mj_step1','mj_step2','mj_forwardSkip','mj_kinematics','mj_fwdPosition','mj_fwdVelocity','mj_fwdAcceleration','mj_collision','mj_contactForce','set_mjcb_control','set_mjcb_passive'):
            if hasattr(self.mj,name):
                self.saved[name]=getattr(self.mj,name)
                setattr(self.mj,name,lambda *a,_name=name,**k:self.b.reject(_name))
        return self
    def __exit__(self,*args):
        for name,value in self.saved.items():setattr(self.mj,name,value)


def body_specs(model,case,mj):
    names=['sphere','sphere2'] if case['geometry']=='pair' else ['sphere']; result=[]
    for name,mass in zip(names,[checks.MASS,.4],strict=False):
        bid=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
        if bid<=0:raise ValueError('Declared analytic body missing')
        jid=int(model.body_jntadr[bid]);q=int(model.jnt_qposadr[jid]);v=int(model.jnt_dofadr[jid])
        if abs(model.body_mass[bid]-mass)>1e-15:raise ValueError('Declared analytic mass changed')
        result.append(dict(name=name,id=bid,q=q,v=v,mass=mass,slides=case['kind'].startswith('friction')))
    return result


def contact(model,data,mj,engine):
    if data.ncon>1:raise RuntimeError('One-contact analytic inventory exceeded')
    result=dict(active=False,present=bool(data.ncon),gap_m=0.,force_contact=np.zeros(3),force_world=np.zeros(3),
        J=np.zeros((3,model.nv)),gradient=np.zeros(3),efc_force=np.zeros(3),regularizer=np.zeros(3),reference_acceleration=np.zeros(3),frame=np.zeros((3,3)),dim=0,bodies=(-1,-1),point=np.zeros(3),kkt=checks.cone_residual([0.],[0.],0.))
    if not data.ncon:return result
    c=data.contact[0];result['gap_m']=float(c.dist)
    if not math.isfinite(c.dist) or c.dist<-.005:raise RuntimeError('Nonfinite/escaped contact penetration')
    result['point']=checks.finite(c.pos,(3,)).copy();basis=checks.finite(np.asarray(c.frame).reshape(3,3),(3,3))
    if not np.array_equal(np.asarray(c.solref),SOLVER['solref']) or not np.array_equal(np.asarray(c.solimp),SOLVER['solimp']):raise RuntimeError('Declared contact compliance changed')
    result['frame']=basis.copy()
    bodies=[]
    for side in range(2):
        if c.geom[side]>=0: bodies.append(int(model.geom_bodyid[c.geom[side]]))
        elif c.flex[side]>=0: bodies.append(int(model.flex_vertbodyid[int(model.flex_vertadr[c.flex[side]])]))
        else:raise RuntimeError('Unknown contact ownership')
    result['bodies']=tuple(bodies);address=int(c.efc_address);dim=int(c.dim)
    if dim not in (1,3):raise RuntimeError('Only declared1D/3D contact allowed')
    if dim==3 and not np.array_equal(np.asarray(c.friction)[:2],[.5,.5]):raise RuntimeError('Declared isotropic friction changed')
    if address<0:return result
    if address+dim>data.nefc:raise RuntimeError('Constraint row escaped')
    force=engine.force(model,data,0)[:3];j=np.asarray(data.efc_J).reshape(data.nefc,model.nv)[address:address+dim].copy()
    ef=np.asarray(data.efc_force)[address:address+dim].copy();r=np.asarray(data.efc_R)[address:address+dim];aref=np.asarray(data.efc_aref)[address:address+dim]
    gradient=j@data.qacc+r*ef-aref
    if not np.allclose(force[:dim],ef,rtol=0,atol=1e-10):raise RuntimeError('Contact force/elliptic constraint coordinates differ')
    result.update(active=True,dim=dim,force_contact=force.copy(),force_world=basis.T@force,
                  kkt=checks.cone_residual(ef,gradient,.5 if dim==3 else 0.))
    result['J'][:dim]=checks.finite(j);result['gradient'][:dim]=checks.finite(gradient)
    result['efc_force'][:dim]=checks.finite(ef);result['regularizer'][:dim]=checks.finite(r);result['reference_acceleration'][:dim]=checks.finite(aref)
    return result


def static_measure(model,data,case,mj,engine):
    c=contact(model,data,mj,engine)
    if not c['active'] or c['dim']!=3:raise RuntimeError('Declared penetrating3D contact missing')
    positions=[];rotations=[];addresses=[]
    for bid in c['bodies']:
        positions.append(data.xpos[bid].copy());rotations.append(data.xmat[bid].reshape(3,3).copy())
        addresses.append(int(model.jnt_dofadr[model.body_jntadr[bid]]) if model.body_jntnum[bid] else None)
    expected=checks.contact_jacobian(np.asarray(data.contact[0].frame).reshape(3,3),positions,rotations,addresses,c['point'],model.nv)
    velocity=np.linspace(-.31,.43,model.nv);f=np.array([.73,-.11,.19])
    a=float(f@(c['J']@velocity));b=float((expected.T@f)@velocity)
    error=float(np.max(np.abs(expected-c['J'])));work=abs(a-b)
    force_error=float(np.max(np.abs(data.qfrc_constraint-expected.T@c['force_contact'])))
    return dict(status='PASS' if error<=checks.LIMITS['jacobian'] and work<=checks.LIMITS['virtual_work_W'] and force_error<=1e-8 else 'FAIL',
        jacobian_max_error=error,virtual_work_error_W=work,generalized_contact_force_error=force_error,
        engine_J=c['J'],independent_J=expected,contact_frame=np.asarray(data.contact[0].frame).copy(),physical_outcome=None)


def sample(model,data,bodies,c,pre_q,pre_v,case):
    force_balance=work=0.;momentum=np.zeros(3);angular=np.zeros(3);energy=0.;positions=[];velocities=[];actual_forces=[];actual_torques=[]
    for b in bodies:
        q,v,m=b['q'],b['v'],b['mass'];v0=pre_v[v:v+3];v1=data.qvel[v:v+3];position=data.qpos[q:q+3].copy()
        if b['slides']:position+=model.body_pos[b['id']]
        contact_force=(int(c['bodies'][1]==b['id'])-int(c['bodies'][0]==b['id']))*c['force_world']
        external=np.array([data.qfrc_applied[v],data.qfrc_applied[v+1],data.qfrc_applied[v+2]])
        total=contact_force+external+[0,0,m*case['gravity']]
        ferr,werr=checks.discrete_force_work(m,v0,v1,total,case['dt_s']);force_balance=max(force_balance,ferr);work=max(work,werr)
        momentum+=m*v1;angular+=np.cross(position,m*v1);energy+=.5*m*(v1@v1)-m*case['gravity']*position[2]
        actual_force=data.qfrc_constraint[v:v+3].copy();actual_forces.append(actual_force)
        torque=np.zeros(3)
        if not b['slides']:
            rot=checks.quaternion_matrix(data.qpos[q+3:q+7]);omega=rot@data.qvel[v+3:v+6]
            inertia=.4*m*checks.RADIUS_M**2;angular+=inertia*omega;energy+=.5*inertia*(omega@omega)
            torque=data.xmat[b['id']].reshape(3,3)@data.qfrc_constraint[v+3:v+6]
        actual_torques.append(torque+np.cross(data.xpos[b['id']],actual_force))
        positions.append(position);velocities.append(v1.copy())
    measured_pair=case['kind']=='pair_impact'
    reaction=float(np.linalg.norm(np.sum(actual_forces,axis=0))) if measured_pair else 0.
    torque=float(np.linalg.norm(np.sum(actual_torques,axis=0))) if measured_pair else 0.
    if measured_pair:
        before=[pre_q[b['q']:b['q']+3] for b in bodies]
        gap0=float(np.linalg.norm(before[1]-before[0])-2*checks.RADIUS_M)
        gap1=float(np.linalg.norm(positions[1]-positions[0])-2*checks.RADIUS_M)
    elif case['geometry'] in ('flex','plane'):
        b=bodies[0];z0=pre_q[b['q']+2]+(model.body_pos[b['id'],2] if b['slides'] else 0.)
        gap0=float(z0-checks.RADIUS_M);gap1=float(positions[0][2]-checks.RADIUS_M)
    else:gap0=gap1=0.
    if min(gap0,gap1)<-.005:raise RuntimeError('Analytic known-surface penetration exceeded5mm')
    return dict(pre_time_s=float(data.time-case['dt_s']),endpoint_time_s=float(data.time),pre_qpos=pre_q,pre_qvel=pre_v,
        endpoint_qpos=data.qpos.copy(),endpoint_qvel=data.qvel.copy(),start_qacc=data.qacc.copy(),start_qfrc_constraint=data.qfrc_constraint.copy(),
        start_engine_energy_J=data.energy.copy(),start_qfrc_applied=data.qfrc_applied.copy(),contact_present=c['present'],constraint_active=c['active'],contact_gap_m=c['gap_m'],
        contact_force_world_N=c['force_world'],contact_force_contact_N=c['force_contact'],contact_J=c['J'],cone_gradient=c['gradient'],
        analytic_gap_start_m=gap0,analytic_gap_endpoint_m=gap1,analytic_gap_defined=case['geometry']!='none',
        contact_frame=c['frame'],contact_point_world_m=c['point'],contact_body_ids=np.asarray(c['bodies']),contact_dimension=c['dim'],
        constraint_force=c['efc_force'],constraint_regularizer=c['regularizer'],constraint_reference_acceleration_m_s2=c['reference_acceleration'],
        cone_primal_N=c['kkt']['primal_N'],cone_dual_m_s2=c['kkt']['dual_m_s2'],cone_complementarity_N_m_s2=c['kkt']['complementarity_N_m_s2'],
        cone_force_scale_N=c['kkt']['force_scale_N'],cone_gradient_scale_m_s2=c['kkt']['gradient_scale_m_s2'],
        force_balance_N=force_balance,discrete_impulse_work_error_J=work,linear_momentum_kg_m_s=momentum,total_angular_momentum_kg_m2_s=angular,
        independent_energy_J=energy,positions_world_m=np.asarray(positions),velocities_world_m_s=np.asarray(velocities),
        action_reaction_N=reaction,angular_action_reaction_N_m=torque,action_reaction_measured=measured_pair,
        actual_body_contact_forces_world_N=np.asarray(actual_forces),actual_body_contact_torques_about_world_origin_N_m=np.asarray(actual_torques),warning_counts=data.warning.number.copy(),
        solver_niter=np.asarray(data.solver_niter).copy())


def assess(case,arrays):
    errors=[];limits=checks.LIMITS;tail=arrays['endpoint_time_s']>=arrays['endpoint_time_s'][-1]-.1
    for key,limit in [('force_balance_N',limits['force_balance_N']),('discrete_impulse_work_error_J',limits['discrete_work_J']),
                      ('action_reaction_N',limits['action_reaction_N']),('angular_action_reaction_N_m',limits['angular_action_reaction_N_m'])]:
        if float(np.max(arrays[key]))>limit:errors.append(key)
    fscale=arrays['cone_force_scale_N'];gscale=arrays['cone_gradient_scale_m_s2']
    if np.any(arrays['cone_primal_N']>limits['cone_primal_relative']*fscale):errors.append('cone_primal')
    if np.any(arrays['cone_dual_m_s2']>limits['cone_dual_relative']*gscale):errors.append('cone_dual')
    if np.any(arrays['cone_complementarity_N_m_s2']>limits['cone_complementarity_relative']*fscale*gscale):errors.append('cone_complementarity')
    metrics={key:float(np.max(arrays[key])) for key in ('force_balance_N','discrete_impulse_work_error_J','cone_primal_N','cone_dual_m_s2','cone_complementarity_N_m_s2')}
    if case['kind']=='freefall':
        z,v=checks.euler_freefall(.5,0.,case['dt_s'],case['steps']);actual=arrays['positions_world_m'][-1,0,2];av=arrays['velocities_world_m_s'][-1,0,2]
        metrics.update(discrete_z_reference_m=z,discrete_v_reference_m_s=v,position_error_m=abs(actual-z),velocity_error_m_s=abs(av-v))
        if abs(actual-z)>limits['discrete_freefall_m'] or abs(av-v)>limits['discrete_freefall_m_s']:errors.append('freefall_discrete_reference')
    if case['kind'].startswith('support'):
        penetration=float(-np.mean(arrays['contact_gap_m'][tail]));normal=float(np.mean(np.linalg.norm(arrays['contact_force_world_N'][tail],axis=1)))
        velocity=float(np.max(np.abs(arrays['velocities_world_m_s'][tail])))
        metrics.update(rest_penetration_reference_m=checks.rest_penetration(),mean_penetration_m=penetration,mean_support_N=normal,settled_speed_m_s=velocity)
        if abs(penetration-checks.rest_penetration())>limits['support_penetration_m']:errors.append('analytic_soft_penetration')
        if abs(normal-checks.MASS*checks.GRAVITY_M_S2)>limits['support_force_relative']*checks.MASS*checks.GRAVITY_M_S2:errors.append('gravity_support')
        if velocity>limits['settled_velocity_m_s']:errors.append('settling')
        initial_energy=checks.MASS*checks.GRAVITY_M_S2*(checks.RADIUS_M+.02)
        if float(np.max(arrays['independent_energy_J']))>initial_energy*(1+limits['impact_energy_gain_fraction']):errors.append('drop_impact_passivity')
    if case['kind'].startswith('friction'):
        force=(.25 if case['kind']=='friction_low' else 1.5)*checks.MU*checks.MASS*checks.GRAVITY_M_S2
        velocity=float(np.mean(arrays['velocities_world_m_s'][tail,0,0]));expected=checks.regularized_creep(force)
        metrics.update(applied_lateral_N=force,mean_lateral_velocity_m_s=velocity,regularized_creep_reference_m_s=expected)
        if case['kind']=='friction_low' and abs(velocity-expected)>max(limits['regularized_creep_floor_m_s'],limits['regularized_creep_relative']*abs(expected)):errors.append('regularized_creep_reference')
        if case['kind']=='friction_high':
            acceleration=float(np.mean(arrays['start_qacc'][tail,0]));expected_acc=.5*checks.MU*checks.GRAVITY_M_S2
            metrics.update(mean_sliding_acceleration_m_s2=acceleration,sliding_acceleration_reference_m_s2=expected_acc)
            if acceleration<=0 or abs(acceleration-expected_acc)>limits['sliding_acceleration_relative']*expected_acc:errors.append('sliding_reference')
    if case['kind']=='pair_impact':
        drift=float(np.max(np.linalg.norm(arrays['linear_momentum_kg_m_s'],axis=1)));angular=float(np.max(np.linalg.norm(arrays['total_angular_momentum_kg_m2_s'],axis=1)))
        metrics.update(momentum_drift_kg_m_s=drift,angular_momentum_drift_kg_m2_s=angular,energy_gain_J=float(np.max(arrays['independent_energy_J'])-.006))
        if drift>limits['momentum_kg_m_s']:errors.append('isolated_pair_momentum')
        if angular>limits['angular_momentum_kg_m2_s']:errors.append('isolated_pair_angular_momentum')
        if metrics['energy_gain_J']>limits['impact_energy_gain_fraction']*.006:errors.append('impact_passivity')
    return dict(status='PASS' if not errors else 'FAIL',failures=errors,metrics=metrics,physical_outcome=None,**FLAGS)


class Measurement:
    def __init__(self,case,raw):self.case=case;self.raw=Path(raw);self.rows=[];self.artifacts=[];self.data=None
    def artifact(self,path,label):
        row=dict(path=str(Path(path).resolve()),sha256=sha(path),bytes=Path(path).stat().st_size,label=label);self.artifacts.append(row);return row
    def save(self,name,values,label):
        path=self.raw/name
        with path.open('xb') as stream:np.savez_compressed(stream,**values)
        return self.artifact(path,label)
    def arrays(self):return {key:np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
    def run(self,engine,mj):
        case=self.case;source=xml(case);path=self.raw/'fixture.xml';path.write_text(source);self.artifact(path,'exact_analytic_fixture_XML')
        model=engine.compile(source);data=engine.data(model);self.data=data;bodies=body_specs(model,case,mj)
        if model.nu or np.any(model.dof_damping) or np.any(model.dof_armature):raise ValueError('No actuators, damping or armature permitted in analytic fixtures')
        if (model.opt.timestep!=case['dt_s'] or not np.array_equal(model.opt.gravity,[0.,0.,case['gravity']])
                or int(model.opt.solver)!=int(mj.mjtSolver.mjSOL_NEWTON) or int(model.opt.integrator)!=int(mj.mjtIntegrator.mjINT_EULER)):
            raise ValueError('Compiled solver/clock/gravity differs from analytic method')
        if case['kind']=='pair_impact':data.qvel[bodies[0]['v']]=.2;data.qvel[bodies[1]['v']]=-.1
        if case['kind'].startswith('friction'):
            data.qfrc_applied[bodies[0]['v']]=(.25 if case['kind']=='friction_low' else 1.5)*checks.MU*checks.MASS*checks.GRAVITY_M_S2
        engine.call('initial_forwards',model,data)
        self.save('compiled_fixture.npz',dict(body_mass_kg=model.body_mass.copy(),body_inertia_kg_m2=model.body_inertia.copy(),
            body_pos_m=model.body_pos.copy(),body_quaternion_wxyz=model.body_quat.copy(),body_parentid=model.body_parentid.copy(),
            joint_type=model.jnt_type.copy(),joint_bodyid=model.jnt_bodyid.copy(),joint_qpos_address=model.jnt_qposadr.copy(),joint_dof_address=model.jnt_dofadr.copy(),
            dof_damping=model.dof_damping.copy(),dof_armature=model.dof_armature.copy(),initial_qpos=data.qpos.copy(),initial_qvel=data.qvel.copy(),
            initial_mass_matrix_packed=data.qM.copy(),gravity_m_s2=model.opt.gravity.copy(),timestep_s=np.asarray(model.opt.timestep)), 'compiled_analytic_mass_layout_solver_initial_state')
        if case['kind']=='jacobian':
            measured=static_measure(model,data,case,mj,engine)
            numeric={key:measured.pop(key) for key in ('engine_J','independent_J','contact_frame')}
            numeric.update(qpos=data.qpos.copy(),qvel=data.qvel.copy(),qfrc_constraint=data.qfrc_constraint.copy(),time_s=np.asarray(data.time))
            self.save('static_readback.npz',numeric,'full3D_static_J_virtual_work_readback')
            return dict(kind=case['kind'],assessment=measured,valid_endpoint_samples=0)
        for _ in range(case['steps']):
            pre_q,pre_v=data.qpos.copy(),data.qvel.copy()
            engine.call('steps',model,data)
            c=contact(model,data,mj,engine)
            self.rows.append(sample(model,data,bodies,c,pre_q,pre_v,case))
        values=self.arrays()
        if not all(np.isfinite(value).all() for value in values.values()):raise RuntimeError('Nonfinite emitted numeric trace')
        self.save('trajectory.npz',values,'prestep_contact_and_endpoint_state_separate_clocks')
        summary=assess(case,values)
        return dict(kind=case['kind'],assessment=summary,valid_endpoint_samples=len(self.rows),
            final_position_world_m=values['positions_world_m'][-1].tolist(),final_velocity_world_m_s=values['velocities_world_m_s'][-1].tolist(),
            maximum_penetration_m=float(max(0.,-np.min(values['analytic_gap_start_m']),-np.min(values['analytic_gap_endpoint_m']))),
            energy_min_J=float(np.min(values['independent_energy_J'])),energy_max_J=float(np.max(values['independent_energy_J'])),
            force_clock='completed step start t; contacts/forces/qacc are NOT endpoint measurements',
            state_clock='step endpoint t+dt; solver_niter and warnings belong to preceding solve')
    def preserve_partial(self):
        if self.rows:self.save('partial_trajectory.npz',self.arrays(),'valid_prefix_before_unexpected_failure_unqualified')
        if self.data is not None:
            self.save('failed_state_diagnostics.npz',{key:np.asarray(getattr(self.data,key)).copy() for key in ('qpos','qvel','qacc','ctrl')},'quarantined_diagnostic_may_contain_nonfinite_values')


def convergence(results):
    rows=[]
    for kind in ('freefall','support_flex','support_solid','friction_low','friction_high','pair_impact'):
        pair=[r for r in results if r['case']['kind']==kind]
        if len(pair)!=2 or any(r['status']!='MEASURED' for r in pair):rows.append(dict(kind=kind,status='NOT_RUN'));continue
        a,b=[r['measurement'] for r in pair]
        position=float(np.max(np.abs(np.asarray(a['final_position_world_m'])-np.asarray(b['final_position_world_m']))))
        velocity=float(np.max(np.abs(np.asarray(a['final_velocity_world_m_s'])-np.asarray(b['final_velocity_world_m_s']))))
        if kind=='freefall':
            # Semi-implicit Euler's known position bias changes exactly with h.
            expected=.5*checks.GRAVITY_M_S2*.25*(.001-.0005)
            passed=abs(position-expected)<=1e-10 and velocity<=1e-10
            limits=dict(position_difference_reference_m=expected,position_reference_atol_m=1e-10,velocity_m_s=1e-10)
        else:
            pmax=5e-6 if kind.startswith('support') else max(1e-4,.01*.5*(.5*checks.MU*checks.GRAVITY_M_S2)*.5**2) if kind=='friction_high' else 1e-4
            vmax=max(1e-3,.01*.5*checks.MU*checks.GRAVITY_M_S2*.5) if kind=='friction_high' else 1e-3
            passed=position<=pmax and velocity<=vmax;limits=dict(position_m=pmax,velocity_m_s=vmax)
        rows.append(dict(kind=kind,status='PASS' if passed else 'FAIL',maximum_position_difference_m=position,maximum_velocity_difference_m_s=velocity,limits=limits))
    return rows


def matched_solid_reference(results):
    rows=[]
    for dt in (.001,.0005):
        pair=[next((r for r in results if r['case']['kind']==kind and r['case']['dt_s']==dt),None) for kind in ('support_flex','support_solid')]
        if any(r is None or r['status']!='MEASURED' for r in pair):rows.append(dict(dt_s=dt,status='NOT_RUN'));continue
        a,b=[r['measurement'] for r in pair]
        position=float(np.max(np.abs(np.asarray(a['final_position_world_m'])-np.asarray(b['final_position_world_m']))))
        force=abs(a['assessment']['metrics']['mean_support_N']-b['assessment']['metrics']['mean_support_N'])
        passed=position<=5e-6 and force<=.01*checks.MASS*checks.GRAVITY_M_S2 and all(r['measurement']['assessment']['status']=='PASS' for r in pair)
        rows.append(dict(dt_s=dt,status='PASS' if passed else 'FAIL',final_position_difference_m=position,
            mean_support_force_difference_N=force,limits=dict(position_m=5e-6,force_N=.01*checks.MASS*checks.GRAVITY_M_S2)))
    return rows


def verify_artifacts(rows,raw):
    seen=set();raw=Path(raw).resolve()
    for row in rows:
        path=Path(row['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(raw) or path.resolve() in seen or sha(path)!=row['sha256'] or path.stat().st_size!=row['bytes']:
            raise ValueError('Output artifact changed/escaped/duplicated')
        seen.add(path.resolve())
    if set(raw.rglob('*.npz'))!={p for p in seen if p.suffix=='.npz'}:raise ValueError('Unindexed numeric artifacts')


def worker(protocol,commit):
    p,frozen=validate(protocol,commit);output=Path(p['output_root']);raw=Path(p['raw_root'])
    budget=Budget(output);budget.save();results=[];failure=None;active=None;measurement=None;artifacts=[];finder=Imports()
    import mujoco
    if mujoco.__version__!='3.4.0' or not Path(mujoco.__file__).resolve().is_relative_to(RUNTIME_ROOT):raise RuntimeError('Wrong imported engine')
    for name in ('get_mjcb_control','get_mjcb_passive'):
        if hasattr(mujoco,name) and getattr(mujoco,name)() is not None:raise RuntimeError('Custom force callbacks forbidden in stock fixture protocol')
    sys.meta_path.insert(0,finder)
    try:
        with Engine(mujoco,budget) as engine:
            for case in plan():
                active=case;folder=raw/case['id'];folder.mkdir();measurement=Measurement(case,folder)
                measured=measurement.run(engine,mujoco)
                result=dict(case=case,status='MEASURED',measurement=measured,artifacts=list(measurement.artifacts),physical_outcome=None,**FLAGS)
                write(folder/'result.json',result);scalar=output/(case['id']+'.json')
                with scalar.open('xb') as stream:stream.write((folder/'result.json').read_bytes())
                results.append(dict(result,result_path=str(scalar),result_sha256=sha(scalar),raw_result_path=str(folder/'result.json'),raw_result_sha256=sha(folder/'result.json')))
                artifacts.extend(measurement.artifacts);active=None;measurement=None
        if budget.counts['models']!=14 or budget.counts['data']!=14 or budget.counts['initial_forwards']!=14 or budget.counts['steps']!=10650:raise RuntimeError('Completed counts disagree with exact schedule')
        verify_artifacts(artifacts,raw)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
        if active is not None:
            folder=raw/active['id']
            for path in (folder/'result.json',output/(active['id']+'.json')):
                if path.exists():path.rename(path.with_name('failed_'+path.name))
            partial_error=None
            try:
                if measurement:measurement.preserve_partial();artifacts.extend(measurement.artifacts)
            except BaseException as partial:partial_error=dict(type=type(partial).__name__,message=str(partial))
            results.append(dict(case=active,status='HARD_STOPPED_PARTIAL',valid_endpoint_samples=len(measurement.rows) if measurement else 0,partial_persistence_error=partial_error,physical_outcome=None))
    finally:
        if finder in sys.meta_path:sys.meta_path.remove(finder)
        budget.save()
    seen={r['case']['id'] for r in results};results.extend(dict(case=c,status='NOT_RUN',physical_outcome=None) for c in plan() if c['id'] not in seen)
    final_binding_error=None
    try:validate(protocol,frozen)
    except BaseException as error:final_binding_error=dict(type=type(error).__name__,message=str(error));failure=failure or final_binding_error
    if time.monotonic()-budget.start>BUDGET['maximum_seconds']:
        failure=failure or dict(type='RuntimeError',message='120second worker budget exceeded before publication')
    comparisons=convergence(results)
    references=matched_solid_reference(results)
    qualified=failure is None and all(r['measurement']['assessment']['status']=='PASS' for r in results) and all(r['status']=='PASS' for r in comparisons+references)
    report=dict(schema_version=SCHEMA,status='HARD_STOP' if failure else 'COMPLETED_ANALYTIC_BENCHMARK',
        numerical_standard_contact_qualification='PASS' if qualified else 'NOT_QUALIFIED',
        protocol_sha256=sha(protocol),freeze_commit=frozen,cases=results,timestep_comparisons=comparisons,matched_solid_reference=references,
        operation_counts=budget.counts,forbidden_attempts=budget.forbidden,elapsed_s=time.monotonic()-budget.start,
        failure=failure,final_binding_error=final_binding_error,artifacts=artifacts,
        unindexed_array_paths=[str(path) for path in raw.rglob('*.npz') if str(path.resolve()) not in {r['path'] for r in artifacts}],
        physical_outcome=None,**FLAGS)
    write(output/'report.json',report);return report


def run(protocol,commit):
    p,frozen=validate(protocol,commit);output=Path(p['output_root']);raw=Path(p['raw_root'])
    if output.exists() or raw.exists():raise FileExistsError('Never retry or overwrite a dynamics qualification')
    output.mkdir(parents=True);raw.mkdir(parents=True)
    write(output/'STARTED.json',dict(protocol_sha256=sha(protocol),freeze_commit=frozen,status='STARTED_ONCE'))
    try:
        with (output/'worker.log').open('x') as log:
            child=subprocess.run([sys.executable,str(Path(__file__).resolve()),'_worker','--protocol',str(Path(protocol).resolve()),'--freeze-commit',frozen],stdout=log,stderr=subprocess.STDOUT,timeout=BUDGET['outer_seconds'])
        if child.returncode or not (output/'report.json').exists():raise RuntimeError('Worker failed or timed out before completed report')
        report=read(output/'report.json');validate(protocol,frozen)
        for row in report['cases']:
            if row['status']=='MEASURED' and (sha(row['result_path'])!=row['result_sha256'] or sha(row['raw_result_path'])!=row['raw_result_sha256'] or Path(row['result_path']).read_bytes()!=Path(row['raw_result_path']).read_bytes()):raise ValueError('Per-case scalar result changed')
        if report['failure'] is None:verify_artifacts(report['artifacts'],raw)
        if report['status']=='HARD_STOP':raise RuntimeError('Worker hard-stopped; evidence retained')
        return report
    except BaseException as error:
        write(output/'launcher_failure.json',dict(type=type(error).__name__,message=str(error),progress=read(output/'progress.json') if (output/'progress.json').exists() else None,physical_outcome=None,**FLAGS));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','run','_worker'));parser.add_argument('--protocol',type=Path,default=HERE/'protocol.json');parser.add_argument('--freeze-commit')
    args=parser.parse_args()
    if args.stage=='prepare':result=prepare(args.protocol)
    else:
        if not args.freeze_commit:parser.error('--freeze-commit is required')
        result=run(args.protocol,args.freeze_commit) if args.stage=='run' else worker(args.protocol,args.freeze_commit)
    print(json.dumps(dict(schema_version=SCHEMA,status=result.get('status','PREPARED_NOT_RUN'))))


if __name__=='__main__':main()
