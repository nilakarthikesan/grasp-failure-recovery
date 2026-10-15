"""Bind all source/assets/immutable inputs without constructing actual scenes."""
from pathlib import Path
import hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
import mujoco
HERE=Path(__file__).resolve().parent
W=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
R=Path('/Users/nilakarthikesan/grasp-failure-prediction')
B=Path('/private/tmp/hug_current_execution_bridge_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 target=HERE/'protocol.draft.json';assert not target.exists()
 base=json.loads((B/'protocol.json').read_text());iv=json.loads(Path(base['inventory_path']).read_text());bindings={}
 def bind(path,expected=None):
  path=Path(path).resolve();h=sha(path);assert expected is None or h==expected,str(path);bindings[str(path)]=h
 for path,h in base['bindings'].items():bind(path,h)
 bind(B/'protocol.json')
 for name in ['xml_reference.py','bridge_model.py']:bind(B/name)
 for path in [Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/protocol.json'),Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/results.json'),Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/bundle_manifest.json'),Path('/private/tmp/hug_current_scene_bridge_review_2026-10-15_v1/inertia_precision_patch_review.json'),Path('/private/tmp/hug_object_inertia_precision_fix_2026-10-15_v1/production.patch')]:bind(path)
 for path in HERE.iterdir():
  if path.is_file() and path.suffix in ('.py','.log','.txt'):bind(path)
 for path in (W/'src/grasp_failure_prediction/evaluation').rglob('*.py'):bind(path)
 for path in (W/'src/grasp_failure_prediction/evaluation/configs').glob('*.yaml'):bind(path)
 for name in ('hug.py','hug_frames.py'):bind(W/'src/grasp_failure_prediction/integrations'/name)
 cases=[];assets={}
 for c in base['cases']:
  c=dict(c);v=iv['modern_cases'][c['source_inventory_index']];local=dict(c['input_bindings']);rpath=Path(v['resolved_case_path']);bind(rpath);local[str(rpath.resolve())]=sha(rpath);payload=json.loads(rpath.read_text())['case']
  dirs=[R/payload['grasp']['observation_path'],Path(v['native']['path']).parent]
  names={'rgb.png','depth.png','depth_m.npy','depth.npy','object_mask.png','intrinsics.npy','T_world_camera.npy','scene_contract.json','observation_manifest.json','inference_report.json'}
  for directory in dirs:
   for path in directory.iterdir():
    if path.is_file() and path.name in names:bind(path);local[str(path.resolve())]=sha(path)
  g=payload['object']['geometry']
  if g is not None:
   path=R/g['mjcf_path'];bind(path);local[str(path.resolve())]=sha(path);root=ET.parse(path).getroot();compiler=root.find('compiler')
   for mesh in root.findall('asset/mesh'):
    mp=(path.parent/compiler.get('meshdir','.')/mesh.get('file')).resolve();bind(mp);local[str(mp)]=sha(mp)
   assets[v['geometry_content_hash']]=str(path)
  else:assets[v['geometry_content_hash']]=None
  c['input_bindings']=local;cases.append(c)
 protocol={'schema':'hug_current_assembled_scene_bridge_protocol_v2','status':'DRAFT_NO_REAL_SCENES_EXECUTED','scope':'Current registered ActuatedShadowRunner assembled scene: all32 existing native camera/wrist frames plus literal historical nominal named-q commands, correct current source URDF and original object assets/captured poses. Verifies declared static source/scene/input realization, not native surface equivalence, new retargeter profile solves, tracking, force-close, lifting, labels, generator quality or prediction adequacy.',
  'bindings':bindings,'source_urdf':base['source_urdf'],'inventory_path':base['inventory_path'],'runtime_asset_root':str(R),'current_pose_validation_source':base['current_pose_validation_source'],'cases':cases,'case_count':32,'distinct_geometry_assets':assets,
  'engineering_tolerances':base['engineering_tolerances'],'tolerance_rationale':base['tolerance_rationale'],'object_inertia_mass_absolute_tolerance':1e-12,
  'native_frame_reference':'Independent raw native T_camera_wrist, camera multiplication, explicit fixed MANO-to-operator matrix, source-XML fixed-palm rotation. Matrix-to-quaternion positive-trace/otherwise-largest-diagonal convention normalized explicitly; raw nonorthogonality and normalization changes retained. Current helper calls actual fixed ShadowPoseValidator with saved q. Compare normalized declared pose within engineering roundoff checks; no new origin calibration.',
  'native_geometry_origin_choice':'Current helper anchors Shadow palm to HUG wrist; this implementation correspondence is checked, not newly proved anatomically exact.',
  'placement_reference':'Actual current runner._root_pose_for_palm then _set_kinematic_targets, mocap and servo target writes, one forward. Independent XML forearm-to-palm inversion predicts root and every world link; historical stored root differences reported without equality acceptance gate.',
  'object_reference':'Independent bounded single-body explicit MJCF/OBJ reference: imported visual oriented triangle connectivity, collision convex support vertices, local/body/world transforms, masks/friction/margin/gap/group, mass and inertia. Defaults/includes/reference frames outside supported form refused. Captured scene binding and exact original asset content hashes required.',
  'controller_reference':'Named targets and source limits, actuator gain/bias/dynamics/transmission/gear, ranges/gains, active identity weld, mocap, table shape/frame/masks/friction/margin/gap and timing options. Static configuration only; no feedback tracking claim.',
  'finite_budget':{'distinct_registered_runner_instances':4,'fixed_validator_instances':1,'cases':32,'forward_calls_per_case':4,'maximum_total_forward_calls':128,'physics_steps':0,'retarget_solves':0,'inference_calls':0,'explicit_surface_distance_queries':0,'full_surface_grid_audits':0},
  'runtime':{'python_executable':sys.executable,'numpy_version':np.__version__,'mujoco_version':mujoco.__version__},'manufactured_controls':'16 meaningful synthetic tests PASS: positive current importer, named placement, native frame conversion and current actuation; intentional geometry/frame/winding/actuation/weld/table/default corruption refusals. Earlier preparation mismatch retained in bound v1 bundle; current16 controls rerun on corrected production object importer before this freeze.',
  'execution_guards':'Require separate bound authorization; assert hashes before/after and per-case source read; patch mj_step/1/2, runner execute/_advance, inference and retarget to raise; assert simulation time0, empty trace and exact forward counts.',
  'no_new_eval_data':True,'no_new_native_proposals':True,'historical_surface_failures_reclassified':False,'authorization_pending':True}
 previous=json.loads(Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/protocol.json').read_text())
 assert protocol['engineering_tolerances']==previous['engineering_tolerances'] and protocol['object_inertia_mass_absolute_tolerance']==previous['object_inertia_mass_absolute_tolerance']
 assert sha(HERE/'scene_checks.py')==sha(Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/scene_checks.py'))
 protocol['previous_protocol_sha256']=sha(Path('/private/tmp/hug_current_scene_bridge_2026-10-15_v1/protocol.json'))
 protocol['prospective_change']='Production import now represents copied fullinertia via source-preserving symmetric eigendecomposition and17digit diagonal/proper quaternion. Same32 inputs and exact unchanged checker/criteria/budget; original24/32 pass and milk inertia failures remain immutable.'
 target.write_text(json.dumps(protocol,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(target),'sha256':sha(target),'bindings':len(bindings),'cases':len(cases),'geometries':len(assets)}))
if __name__=='__main__':main()
