"""Freeze completed immutable inputs before independent new-anchor arithmetic."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json
B=Path(__file__).parent
AUTHOR=Path('/private/tmp/hug_calibrated_anchor_radial_control_2026-10-15_v1')
CANONICAL=Path('/private/tmp/hug_canonical_mcp_independent_verification_2026-10-14_v1')
PRIOR=Path('/private/tmp/hug_radial_control_independent_verification_2026-10-14_v1/runtime_v2')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root-reviewed',action='store_true');args=ap.parse_args();assert args.root_reviewed
 assert not (B/'protocol.json').exists() and not (B/'report.json').exists()
 author=read(AUTHOR/'protocol.json');canonical=read(CANONICAL/'protocol.json');prior=read(PRIOR/'protocol.json')
 assert sha(AUTHOR/'report.json')=='6e908767f32877bb8dbad3cb4794fd938fb2090b3d0f357eb942e8279b1dbd55'
 assert sha(CANONICAL/'report.json')=='8b79219b1b83f8ff75ab390010bec7bf6a84c7e7f144d3e59b33712bcaefa5b8'
 inputs={}
 for table in [author['input_sha256'],canonical['input_sha256']]:
  for path,h in table.items():
   assert path not in inputs or inputs[path]==h,('Conflicting frozen input hash',path)
   inputs[path]=h
 for path in [AUTHOR/'protocol.json',AUTHOR/'report.json',AUTHOR/'radial_control.py',AUTHOR/'METHOD.md',CANONICAL/'protocol.json',CANONICAL/'report.json',CANONICAL/'verify_calibration_and_poses.py',PRIOR/'protocol.json',PRIOR/'report.json',PRIOR/'verify_radial_control.py',B/'REVIEW_METHOD.md']:
  inputs[str(path)]=sha(path)
 for path,h in inputs.items():assert sha(path)==h,path
 ast.parse((B/'verify_radial_control.py').read_text())
 p={'schema':'hug_canonical_new_anchor_independent_radial_arithmetic_verification_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
  'before_first_independent_new_anchor_radial_margin':True,'execution_authorized_by_root_review':True,
  'review_script_sha256':sha(B/'verify_radial_control.py'),'author_protocol':str(AUTHOR/'protocol.json'),'author_report':str(AUTHOR/'report.json'),'author_script':str(AUTHOR/'radial_control.py'),
  'input_sha256':inputs,'independent_C_FK_report':str(CANONICAL/'report.json'),'urdf':prior['urdf'],'prior_independent_cell_geometry_report':prior['prior_independent_cell_geometry_report'],'query_cache_root':prior['query_cache_root'],
  'source_URDF_compiled_corner_comparison_allowance_m':prior['source_URDF_compiled_corner_comparison_allowance_m'],
  'source_corner_allowance_is_not_added_to_shell_budget':True,'shell_distance_transform_allowances_unchanged':True,'precision_digits':90,
  'scope':'All seven inherited palm cells and eight recorded boxes in the two saved Mug proposals at their single source-only canonical fixed new anchors. Independent C/native/world/object anchor composition and all shell/whole-cell margins. Six source/cap refusals retained. No arbitrary-translation or alternative-calibration claim.',
  'limitations':['MANO canonical neutral descriptor inherited; frame algebra and six-case FK independently verified separately.','Mug A native cell geometry/radii inherited from frozen prior study, not regenerated here. Mug B two cells are bound to prior independent geometry.','Conditional numeric allowances, not formal directed rounding. Nonpositive margins remain UNRESOLVED.','No inference, fitting, new geometry queries, physics, eval data, training or production changes.'],
  'new_geometry_queries':0,'physics_steps':0,'new_eval_data':False,'full_surface_audit_authorized':False}
 with (B/'protocol.json').open('x') as f:json.dump(p,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'protocol_sha256':sha(B/'protocol.json'),'script_sha256':p['review_script_sha256'],'bound_input_count':len(inputs)},indent=2))
if __name__=='__main__':main()
