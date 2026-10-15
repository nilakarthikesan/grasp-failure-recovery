"""Read-only provenance audit. Standard-library text/hash/git reads only."""
from pathlib import Path
import json,hashlib,subprocess,datetime
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction');OUT=Path(__file__).parent
entries=[]
def quote(key,rel,start,end):
 p=ROOT/rel;raw=p.read_bytes();lines=raw.decode().splitlines();end=min(end,len(lines));assert 1<=start<=end<=len(lines)
 row={'id':key,'path':str(p),'repository_relative_path':rel,'sha256':hashlib.sha256(raw).hexdigest(),'line_start':start,'line_end':end,'excerpt':'\n'.join(lines[start-1:end])};entries.append(row);return key
quote('acceptance_claim','docs/HUG_MAPPING_ACCEPTANCE.md',33,60)
quote('acceptance_contract','docs/HUG_MAPPING_ACCEPTANCE.md',85,139)
quote('orientation_definition','docs/HUG_MAPPING_ACCEPTANCE.md',177,217)
quote('papers_do_not_certify_choices','docs/HUG_MAPPING_ACCEPTANCE.md',237,264)
quote('engineering_guard','docs/HUG_STATIC_VALIDATION_PROTOCOL.md',90,136)
quote('future_adequacy','docs/HUG_STATIC_VALIDATION_PROTOCOL.md',334,362)
quote('earliest_located_bands','research_artifacts/hug_mapping_refinement_2026-10-14/surface/protocol.json',1,23)
quote('frozen_rule','research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/acceptance/acceptance_rule.json',1,62)
quote('source_interval_code','research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/sampling/cell_intervals.py',1,60)
quote('mean_guard_config','src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml',14,28)
quote('relative_caps_intent','docs/HUG_CONSTRAINED_REFINEMENT.md',10,32)
quote('caps_earlier_allowance','src/grasp_failure_prediction/evaluation/constrained_surface_refinement.py',287,320)
quote('caps_later_literal','src/grasp_failure_prediction/evaluation/feasible_surface_refinement.py',149,168)
quote('caps_not_absolute_accuracy','docs/HUG_FEASIBLE_REFINEMENT.md',70,90)
quote('trust_not_accuracy','research_artifacts/hug_surface_refinement_2026-10-14/prototype/REPORT.md',86,96)
quote('trust_config','research_artifacts/hug_surface_refinement_2026-10-14/prototype/surface_refinement.py',12,28)
quote('literal_boundary_implementation','research_artifacts/hug_surface_constraints_audit_2026-10-15_v1/validate_candidate.py',34,52)
quote('normal_eligibility_and_sign','research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/check_orientation.py',80,134)
quote('strict_rule_historical_decision','docs/RESEARCH_DECISIONS.md',1795,1847)
quote('caps_historical_motivation','docs/RESEARCH_DECISIONS.md',1884,1900)
quote('caps_historical_adoption','docs/RESEARCH_DECISIONS.md',1940,1965)
quote('identity_control_limit','research_artifacts/hug_surface_constraints_review_2026-10-15_v1/identity_interval_control.py',19,45)
quote('latest_unchanged_failure','docs/RESEARCH_DECISIONS.md',2464,2470)
# Exact immutable-source protocol values retained without regenerating metrics.
protocols={}
for rel in ['research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/orientation/protocol.json','research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14/sampling/protocol.json','research_artifacts/hug_surface_constraints_review_2026-10-15_v1/identity_interval_control.json','research_artifacts/hug_surface_constraints_audit_2026-10-15_v1/manifest.json']:
 p=ROOT/rel;protocols[rel]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
commits={}
for short in ['4f972f0','2d8e000','ee943e3','d577818','b8e3a16','8c9d725']:
 value=subprocess.check_output(['git','show','-s','--format=%H%n%aI%n%s',short],cwd=ROOT,text=True).splitlines();commits[short]={'commit':value[0],'author_time':value[1],'subject':value[2]}
patch=subprocess.check_output(['git','show','4f972f0','--','src/grasp_failure_prediction/evaluation/configs/execution_protocols.yaml','src/grasp_failure_prediction/evaluation/case_runner.py'],cwd=ROOT)
(OUT/'initial_10mm_gate.patch').write_bytes(patch)
record={'schema':'hug_criteria_provenance_readonly_audit_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checkout':str(ROOT),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'scope':'Read-only local documentation, code, immutable protocol and git provenance. No independent reread of external papers is claimed; their relation below is explicitly the local source-attribution record.','excerpts':entries,'protocol_receipts':protocols,'commit_provenance':commits,'initial_10mm_patch_sha256':hashlib.sha256(patch).hexdigest(),'new_geometry_queries':0,'new_inference':False,'physics_steps':0,'new_eval_data':False,'shared_source_or_document_edits':False}
(OUT/'evidence.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'excerpts':len(entries),'protocol_receipts':len(protocols),'git_commits':len(commits),'evidence_sha256':hashlib.sha256((OUT/'evidence.json').read_bytes()).hexdigest()},indent=2))
