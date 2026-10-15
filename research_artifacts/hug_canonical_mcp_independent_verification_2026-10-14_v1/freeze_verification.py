"""Bind completed primary outputs before independent C/FK execution."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json

OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary-dir',type=Path,required=True);ap.add_argument('--root-reviewed',action='store_true');args=ap.parse_args()
    assert args.root_reviewed,'Final independent source/protocol review is required before freeze'
    assert not (OUT/'protocol.json').exists() and not (OUT/'report.json').exists()
    primary_dir=args.primary_dir.resolve();primary_path=primary_dir/'protocol.json';primary=json.loads(primary_path.read_bytes())
    assert primary['execution_authorized_by_root_and_independent_review']
    for required in ['calibration.json','candidate_targets.json']:
        assert (primary_dir/required).is_file(),required
    candidate=json.loads((primary_dir/'candidate_targets.json').read_bytes())
    assert len(candidate['cases'])==6 and sorted(c['source_inventory_index'] for c in candidate['cases'])==[8,12,16,20,24,28]
    old_independent=ROOT/'research_artifacts/hug_guarded_geometry_validation_2026-10-14_v1'
    previous=json.loads((old_independent/'protected_hashes.json').read_text())
    inputs={}
    for table in [previous,primary['source_sha256'],primary['implementation_sha256']]:
        for path,h in table.items():
            assert path not in inputs or inputs[path]==h,('Conflicting historical/source input hash',path)
            inputs[path]=h
    for path in [primary_path,primary_dir/'calibration.json',primary_dir/'candidate_targets.json',primary_dir/'METHOD.md',Path(primary['parent_exported_calibration_path']),old_independent/'protocol.json',old_independent/'candidate_validation_preflight_only.json.gz',old_independent/'candidate_validation_preflight_only.json.gz.binding.json']:
        inputs[str(path)]=sha(path)
    for path,h in inputs.items():assert sha(path)==h,path
    script=OUT/'verify_calibration_and_poses.py';ast.parse(script.read_text())
    protocol={'schema':'hug_canonical_MCP_independent_verification_protocol_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
        'before_first_independent_C_or_FK_calculation':True,'script_sha256':sha(script),'execution_authorized_by_root_review':True,
        'primary_protocol':str(primary_path),'calibration_path':str(primary_dir/'calibration.json'),'candidate_path':str(primary_dir/'candidate_targets.json'),
        'prior_actual_preflight':str(old_independent/'candidate_validation_preflight_only.json.gz'),
        'prior_actual_preflight_protocol_sha256':sha(old_independent/'protocol.json'),'input_sha256':inputs,'case_indices':[8,12,16,20,24,28],
        'field_reproduction_allowance_m':1e-12,
        'frame_policy':'C maps neutral palm P0 to MANO. Independent frames/linear solve from serialized primary canonical descriptors and source rigid MCP origins; R_BP0 from fresh sourceURDFneutral FK. Fixed carrier R_WB=R_WM*C.R*R_BP0.T; actualpalm R_WB*R_BP(q); source rigid T_BF from XML handles fused forearm. No secondA and no transformed joint0 subtraction.',
        'native_groundtruth_policy':'Original camera-frame landmarks composed directly with saved T_world_camera; C never changes scientific WORLD targets. Reconstruct original scene/caps independently before applying new target.',
        'old_tip_caps_policy':'Exact original authoritative five tip norms remain mandatory with zero acceptance allowance; calibrated q0 or newplacement errors cannot replace them.',
        'root_alignment_policy':'Replace old wrist=palmequality with independently reconstructed calibrated p_anchor and q-dependent actualpalmR, retain explicit displacement. Independently derive root from captured floating zero-root FK; do not assume fixedsourcebaseB equals floatingforearm frame. Retain primary source-to-compiled translation compensation separately.',
        'scope':'Allsix existing development proposals retained, every numerical/engineering eligibility flag independently recomputed. No optimizer replay, MANO neutral rerun, object surface query, fullsurface audit, inference, physics episode or evaldata.',
        'limitations':['Neutral MANO geometry is inherited from the separately frozen primary source-template forward, not independently regenerated here. Its frame algebra and sourceURDF proxy descriptors are independently reconstructed.','A source-only canonical anatomy proxy is a new mapping hypothesis, not known anatomical ground truth.','Numeric/engineering source gates are independently checked here; separate new-anchor radial control and explicit root review are required before any full surface audit. Passing these gates would not establish strict mapping acceptance.','Original source/rule/query histories remain immutable. Native failure equivalence, continuouscontact and unseenobject validity remain untested.'],
        'strict_surface_audit_execution_authorized':False,'new_inference':False,'physics_steps':0,'new_eval_data':False,'production_changes':False}
    with (OUT/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(script),'primary_protocol_sha256':sha(primary_path),'calibration_sha256':sha(primary_dir/'calibration.json'),'candidate_sha256':sha(primary_dir/'candidate_targets.json'),'bound_input_count':len(inputs)},indent=2))
if __name__=='__main__':main()
