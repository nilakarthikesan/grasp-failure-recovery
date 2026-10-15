"""Read text/JSON only; record source lines and existing historical evidence."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
RUNTIME=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources=[
    ('HUG_wrist_definition',RUNTIME/'hug/src/models/mano.py',[(34,43),(133,146)]),
    ('Dex_explicit_origin_correspondence',RUNTIME/'deps/dex_retargeting/configs/teleop/shadow_hand_right.yml',[(1,15)]),
    ('project_vector_transform',ROOT/'src/grasp_failure_prediction/evaluation/retargeting.py',[(44,63),(65,75),(148,163)]),
    ('project_paired_axis_transform',ROOT/'src/grasp_failure_prediction/integrations/hug_frames.py',[(4,12),(23,28)]),
    ('project_explicit_palm_placement',ROOT/'src/grasp_failure_prediction/evaluation/case_runner.py',[(195,222)]),
    ('runner_actual_root_compensation',ROOT/'src/grasp_failure_prediction/evaluation/runner.py',[(476,492)]),
    ('URDF_palm_wrist_and_rigid_MCP_origins',ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf',[(102,110),(182,188),(214,220),(346,353),(479,486),(613,621)]),
    ('earlier_proxy_diagnostic_source',ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/palm/diagnose_palm_anchor.py',[(50,51),(78,95),(365,401)]),
]
bindings=[]
for role,path,ranges in sources:
    text=path.read_text().splitlines()
    bindings.append({'role':role,'path':str(path),'sha256':sha(path),'line_number_basis':'one-based inclusive in this SHA-bound file',
        'line_excerpts':[{'first_line':a,'last_line':b,'text':'\n'.join(text[a-1:b])} for a,b in ranges]})
report_path=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/palm/palm_anchor_report.json'
prior=json.loads(report_path.read_bytes())
assert prior['guards']['calibration_applied'] is False and prior['guards']['retarget_optimizer_loaded_or_used'] is False
original=prior['global_translation_diagnostics']['tips5']['original_point_error_m']['mean']
hypothetical=prior['cross_point_family_translation_diagnostic']['fingertip_errors_if_MCP_mean_algebraically_removed_m']['mean']
assert hypothetical>original
history={'path':str(report_path),'sha256':sha(report_path),'scope':prior['scope'],
    'original_mean_tip_error_m':original,'hypothetical_MCP_mean_subtraction_tip_error_m':hypothetical,
    'source_field_original':'global_translation_diagnostics.tips5.original_point_error_m.mean',
    'source_field_hypothetical':'cross_point_family_translation_diagnostic.fingertip_errors_if_MCP_mean_algebraically_removed_m.mean',
    'interpretation':'Pure saved-residual subtraction, not an executed geometry placement or successful calibration. Mean error worsens; no offset is applied.',
    'calibration_applied':False,'prior_palm_origin_native_wrist_max_error_m':prior['software_anchor_summary']['palm_origin_vs_native_wrist_m']['max'],
    'prior_wrist_palm_length_m':prior['software_anchor_summary']['wrist_to_palm_length_m']['mean']}
result={'schema':'hug_anchor_convention_source_audit_evidence_v1','created_utc':datetime.now(timezone.utc).isoformat(),
    'source_bindings':bindings,'historical_proxy_diagnostic':history,'new_geometry_queries':0,'new_FK_or_model_executions':0,
    'new_inference':False,'physics_steps':0,'new_eval_data':False,'shared_source_or_docs_edited':False,
    'conclusion':'The inspected current wrist0-to-palm convention is explicit and source-consistent; this read-only audit demonstrates no new frame/origin bug and does not establish anatomical or surface equivalence.',
    'canonical_calibration_verification_status':'PLAN_ONLY_NOT_RUN','collector_sha256':sha(__file__)}
with (OUT/'source_evidence.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'source_file_count':len(bindings),'historical_report_sha256':history['sha256'],'original_mean_tip_mm':original*1000,'hypothetical_MCP_subtracted_mean_tip_mm':hypothetical*1000,'source_evidence_sha256':sha(OUT/'source_evidence.json')},indent=2))
if any(sha(x['path'])!=x['sha256'] for x in bindings):raise AssertionError('Source changed during text audit')
