"""Freeze source files and existing literal commands; no FK/model construction."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco
from xml_reference import read_urdf

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
HISTORY=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/static_saved_hug_fk.json'
INVENTORY=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals/existing_hug_inventory.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    target=HERE/'protocol.draft.json';assert not target.exists(),'Keep each prepared version explicit'
    h=json.loads(HISTORY.read_text());iv=json.loads(INVENTORY.read_text())
    assert sha(INVENTORY)==h['input_inventory_sha256']
    assert len(h['cases'])==len(iv['modern_cases'])==32
    history={c['native_sha256']:c for c in h['cases']};assert len(history)==32
    source=read_urdf(h['task_link_urdf_path']);assert sha(source['path'])==h['task_link_urdf_sha256']
    bindings={};cases=[]
    def bind(path,expected=None):
        path=str(Path(path).resolve());actual=sha(path)
        assert expected is None or actual==expected,path
        bindings[path]=actual;return path,actual
    for path in [HISTORY,INVENTORY,source['path'],*HERE.glob('*.py'),HERE/'manufactured_controls.log']:
        bind(path)
    for path in [ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py',
                 ROOT/'src/grasp_failure_prediction/evaluation/retargeting.py',
                 ROOT/'src/grasp_failure_prediction/evaluation/__init__.py',
                 ROOT/'src/grasp_failure_prediction/evaluation/schema.py',
                 ROOT/'src/grasp_failure_prediction/integrations/hug_frames.py',
                 Path(mujoco.__file__)]:bind(path)
    # Bind all referenced visual AND collision assets, because production loader
    # packages both before URDF's discardvisual policy chooses compiled geoms.
    from xml.etree import ElementTree as ET
    for mesh in ET.parse(source['path']).getroot().findall('.//mesh'):
        bind(Path(source['path']).parent/mesh.get('filename'))
    for index,c in enumerate(iv['modern_cases']):
        native=c['native']['sha256'];assert native in history
        cb={}
        for f in [c['native'],c['stored_shadow_command'],c['scene_mjb'],*c['frame_files'].values()]:
            p,d=bind(f['path'],f['sha256']);cb[p]=d
        command=c['stored_shadow_command'];old=history[native]
        assert old['stored_shadow_command']['sha256']==command['sha256']
        assert old['stored_shadow_command']['row_index']==command['row_index']
        with np.load(command['path'],allow_pickle=False) as z:
            i=command['row_index'];names=z['joint_names'].tolist();q=z['nominal_target_rad'][i]
            p=z['root_target_position_m'][i];quat=z['root_target_quaternion_wxyz'][i]
        assert len(names)==len(q)==24 and len(set(names))==24
        assert [j['joint_name'] for j in old['joint_limit_checks']]==names
        assert np.array_equal(q,np.array([j['value_rad'] for j in old['joint_limit_checks']]))
        assert q.shape==(24,) and p.shape==(3,) and quat.shape==(4,) and all(np.isfinite(v).all() for v in [q,p,quat])
        cases.append({'source_inventory_index':index,'native_sha256':native,'command_path':command['path'],
                      'command_sha256':command['sha256'],'row_index':i,'joint_names':names,'qpos_rad':q.tolist(),
                      'root_position_m':p.tolist(),'root_quaternion_wxyz':quat.tolist(),'input_bindings':cb})
    protocol={'schema':'hug_current_execution_bridge_protocol_v1','status':'DRAFT_NO_REAL_GEOMETRY_EXECUTED',
              'scope':'Engineering command-realization bridge for all32 immutable saved nominal legacy Shadow commands, not MANO anatomical equivalence, learned-generator quality, dynamics, new static-profile solves or SE3 candidate validation.',
              'historical_static_report':str(HISTORY),'inventory_path':str(INVENTORY),'source_urdf':source['path'],
              'current_pose_validation_source':str(ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py'),
              'bindings':bindings,'cases':cases,'case_count':32,
              'current_loader':'unmodified current production ShadowPoseValidator + load_shadow_model(floating_root=True)',
              'independent_reference':'XML parent/child tree, explicit local-axis Rodrigues and RzRyRx, direct stored carrier override',
              'current_compiled_joint_limit_policy':'both fixed and floating models require limited=True and ranges matching source XML within1e-12rad',
              'source_joint_limit_policy':'literal source XML lower<=storedvalue<=upper; zero tolerance; no clipping/reordering by positional index',
              'geometry_policy':'all source collision primitives; all source mesh vertices compared to compiled vertex sets plus full hull extrema retained in actual support graph; complete current hand geom inventory; contact masks1/1 margin0 gap0 no extra pair/plugin/flex/exclusion',
              'engineering_tolerances':{'current_position_m':1e-10,'current_rotation_matrix':1e-10,'primitive_size_m':1e-12,
                                        'source_compiled_joint_limit_rad':1e-12,'mesh_vertex_set_m':1e-6,'historical_replay_position_m':1e-10},
              'tolerance_rationale':{'current_double_kinematics':'1e-10m and matrix entries are prospective engineering roundoff checks for same double-valued command/source chain; not mapping-error allowances.',
                                     'mesh':'1micrometer prospective bound for float32 compiled vertex centering/rotation on meter-scale-or-smaller source assets; actual residuals retained. Mutual vertex-set bound transfers to convex hulls when compiled support graph retains all full hull extremes. This is a numerical comparison, not an interval certificate or physical shape-accuracy claim.',
                                     'historical':'Exact existing MJB and commands replayed with same MuJoCo version;1e-10m checks immutable old point evidence. Current-vs-old differences are reported without an equality acceptance gate.'},
              'finite_budget':{'current_fixed_compiles':1,'current_floating_compiles':1,'historical_mjb_loads':32,
                               'current_fixed_forward_calls':32,'current_floating_forward_calls':32,'historical_forward_calls':32,
                               'maximum_total_forward_calls':96,'physics_steps':0,'retargeting_solves':0,'hand_object_distance_queries':0},
              'output':'one immutable per-case JSON and final summary; source hashes verified before/after plus per-case immutable source readback',
              'runtime':{'mujoco_version':mujoco.__version__,'numpy_version':np.__version__},
              'manufactured_controls':'eleven synthetic controls passed; failed setup-path and fixed-body-fusion preparation logs preserved; no HUG model/assets used by those controls',
              'no_new_eval_data':True,'no_new_proposals':True,'authorization_pending':True}
    target.write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'draft':str(target),'sha256':sha(target),'bindings':len(bindings),'cases':len(cases)}))

if __name__=='__main__':main()
