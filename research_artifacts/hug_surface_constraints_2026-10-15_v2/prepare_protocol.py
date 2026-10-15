"""Freeze a single search-guidance repair; preserve all scientific predicates."""
from pathlib import Path
import copy,hashlib,json
OUT=Path(__file__).parent;V1=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
p=copy.deepcopy(read(V1/'protocol.json'))
for k in ('reviewed_draft_sha256','independent_preexecution_review','frozen_at_utc','root_execution_authorization'):p.pop(k,None)
p['schema']='hug_saved_surface_witness_constraint_protocol_v2';p['candidate_schema']='hug_saved_surface_witness_constraint_candidates_v2'
p['execution_authorized_after_independent_review']=False
p['implementation_sha256']={str(OUT/Path(path).name):sha(OUT/Path(path).name) for path in p['implementation_sha256']}
p['input_sha256'].update({str(q):sha(q) for q in [V1/'protocol.json',V1/'candidate_targets.json',V1/'fit_summary.json',V1/'optimizer_refusal_summary.json',OUT/'code_delta.json']})
p['preflight_sha256']={str(q):sha(q) for q in [OUT/'all_six_source_preflight.json',OUT/'solver_tests.log',Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v1/manufactured_constraint_controls.json')]}
p['source_preflight_path']=str(OUT/'all_six_source_preflight.json')
p['preflight_control_counts']={'inherited_independent_core_controls':36,'same_solver_controls_rerun':8,'source_cases_rerun':6,'source_analytic_jacobian_cases_rerun':6,'additional_far_guidance_controls':'bound by independent review receipt before freeze'}
p['version_reason']={'previous_protocol_sha256':sha(V1/'protocol.json'),'previous_candidate_sha256':sha(V1/'candidate_targets.json'),'evidence':'Several V1 optimizer terminals report convergence with epigraph5.000001 at native-far cells where d<r+eps, making inherited max(0,d-(r+eps))-band search residual flat. These are optimizer convergence reports, not scientific feasibility.','single_search_change':'For native-far rows only, inner solver margin uses ((d-(r+eps))-band-inward_target)/1mm with positive-band precondition. All literal final lower=max(0,d-(r+eps)),upper=d+(r+eps),passed,margin and exact ranking remain unchanged.','equivalence_scope':'For b>0, max(0,d-error)>b iff d-error>b. The unclamped surrogate supplies a derivative in d<error; actual geometry distance can still be nonsmooth. This does not prove joint feasibility or optimizer success.','unchanged':'Source witnesses, original caps, starts, 27D domain/root, all exact orientation eligibility, final selection, budgets and no-integration restrictions remain identical to V1.','no_more_versions':'One independently reviewed correction only; stop after the same declared two starts per case and no third retuning this task.'}
p['constraints']['optimizer_surrogate']+=' V2 native-far search residual only is unclamped d-(radius+old_eps)-band to remove the artificial zero gradient of the interval clamp; exact literal final predicates and ranking remain V1.'
p['cross_version_selection']='Separate independently frozen audit policy may choose among each version selected diagnostic pose using the unchanged exact diagnostic_rank, stable V1 tie; no full-grid outcomes enter selection. All V1/V2 evidence remains separately versioned.'
target=OUT/'protocol.draft.json';assert not target.exists();target.write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'draft':str(target),'sha256':sha(target)}))
