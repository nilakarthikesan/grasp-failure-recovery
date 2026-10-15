"""Read-only arithmetic analysis of completed diagnostic and old major traces."""
from pathlib import Path
import hashlib,json
import numpy as np

OUT=Path(__file__).parent
CANONICAL=Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    report=read(OUT/'report.json');assert report['full_geometry_calls']==60 and report['guards']['actual']==0
    case_diagnostics=[]
    for case in report['records']:
        b=case['baseline'];base_fields=b['normal_fields_for_diagnosis'];rows=[]
        for joint in ['WRJ1','FFJ1','THJ1']:
            step_results=[]
            for step in [1e-5,1e-6,1e-7]:
                bysign={r['sign']:r['diagnostics'] for r in case['probes'] if r['joint_name']==joint and r['step_rad']==step}
                minus,plus=bysign[-1],bysign[1]
                changes={}
                for sign,probe in bysign.items():
                    fields=probe['normal_fields_for_diagnosis']
                    changes[str(sign)]={
                        'invalid_bin_count':probe['normal_invalid_bin_count'],
                        'nearest_geom_id_change_count':int(np.count_nonzero(np.asarray(fields['nearest_geom_id'])!=base_fields['nearest_geom_id'])),
                        'part_change_count':int(np.count_nonzero(np.asarray(fields['part_index'])!=base_fields['part_index'])),
                        'tie_count_change_count':int(np.count_nonzero(np.asarray(fields['tie_count'])!=base_fields['tie_count'])),
                        'validity_change_count':int(np.count_nonzero(np.asarray(fields['valid'])!=base_fields['valid'])),
                        'max_outward_normal_change':float(np.linalg.norm(np.asarray(fields['outward_normal'])-base_fields['outward_normal'],axis=1).max()),
                        'max_facing_score_change':float(np.max(np.abs(np.asarray(fields['score'])-base_fields['score']))),
                        'tip_error_increase_m':(np.asarray(probe['task_link_error_norm_m'][:5])-b['task_link_error_norm_m'][:5]).tolist(),
                    }
                step_results.append({'step_rad':step,'forward_slope_cost_per_rad':(plus['objective']-b['objective'])/step,
                    'backward_slope_cost_per_rad':(b['objective']-minus['objective'])/step,
                    'central_slope_cost_per_rad':(plus['objective']-minus['objective'])/(2*step),
                    'central_component_slopes':{k:(plus['component_soft_l1_cost'][k]-minus['component_soft_l1_cost'][k])/(2*step) for k in b['component_soft_l1_cost']},
                    'normal_feature_changes':changes})
            slopes=np.asarray([r['central_slope_cost_per_rad'] for r in step_results]);fwdback=np.asarray([abs(r['forward_slope_cost_per_rad']-r['backward_slope_cost_per_rad']) for r in step_results])
            rows.append({'joint_name':joint,'steps':step_results,'central_slope_range_cost_per_rad':float(np.ptp(slopes)),
                         'finest_forward_backward_gap_cost_per_rad':float(fwdback[-1]),
                         'largest_forward_backward_gap_cost_per_rad':float(fwdback.max()),
                         'normal_invalidity_or_identity_changes_observed':any(c['invalid_bin_count'] or c['nearest_geom_id_change_count'] or c['part_change_count'] or c['tie_count_change_count'] or c['validity_change_count'] for r in step_results for c in r['normal_feature_changes'].values())})
        invalidity_events=[]
        available_parts=[part for part,d in b['normal_diagnostics'].items() if d['available']]
        for probe in case['probes']:
            d=probe['diagnostics'];fields=d['normal_fields_for_diagnosis']
            for bin_index in np.flatnonzero(~np.asarray(fields['valid'])):
                part=available_parts[int(bin_index)//64]
                native_score=b['normal_diagnostics'][part]['native_score'][int(bin_index)%64]
                weight=1/(6*64)
                old_residual=np.sqrt(weight)*(base_fields['score'][bin_index]-native_score)
                new_residual=np.sqrt(weight)*2
                bin_penalty_cost_increase=.2*(np.hypot(.2,new_residual)-np.hypot(.2,old_residual))
                invalidity_events.append({'joint_name':probe['joint_name'],'step_rad':probe['step_rad'],'sign':probe['sign'],
                    'bin_index':int(bin_index),'anatomical_part':part,'native_score':native_score,
                    'baseline_score':base_fields['score'][bin_index],'inactive_candidate_score':fields['score'][bin_index],
                    'invalid_reasons':[key for key,mask in fields['reason_masks'].items() if mask[bin_index]],
                    'invalid_bin_cost_increase_vs_original_bin':float(bin_penalty_cost_increase),
                    'total_facing_cost_increase_vs_original':d['component_soft_l1_cost']['facing']-b['component_soft_l1_cost']['facing']})
        raw=case['raw_final']
        case_diagnostics.append({'source_inventory_index':case['source_inventory_index'],'baseline_objective':b['objective'],
            'baseline_component_soft_l1_cost':b['component_soft_l1_cost'],
            'baseline_facing_fraction_of_total_cost':b['component_soft_l1_cost']['facing']/b['objective'],
            'raw_final_objective':raw['objective'],'raw_final_component_soft_l1_cost':raw['component_soft_l1_cost'],
            'raw_final_facing_fraction_of_total_cost':raw['component_soft_l1_cost']['facing']/raw['objective'],
            'raw_final_invalid_direction_bins':raw['normal_invalid_bin_count'],
            'raw_final_per_part_invalid_bin_count':{k:v.get('invalid_bin_count',0) for k,v in raw['normal_diagnostics'].items()},
            'coordinate_stencils':rows,'observed_invalidity_events':invalidity_events})
    traces=[]
    for index in [8,12,16,20,24,28]:
        cohort=read(CANONICAL/f'candidate_cohort_{index}.json');case=cohort['cases'][0];audit=case['solver_audit'];major=audit['major_iteration_assessments'];q0=np.asarray(case['original_qpos_rad']);baseline=audit['baseline']['objective'];first=major[0]
        tip_fail_counts=np.sum([~np.asarray(row['assessment']['allowance_individual_tip_nonworsening_pass']) for row in major],axis=0)
        tip_increases=np.asarray([row['assessment']['tip_error_increase_m'] for row in major]);maxinc=tip_increases.max(1);best_guard_index=int(np.argmin(maxinc))
        first_delta=np.asarray(first['qpos_rad'])-q0
        traces.append({'source_inventory_index':index,'case_sha256':sha(CANONICAL/f'candidate_cohort_{index}.json'),
            'solver_status':audit['solver_status'],'solver_success':audit['solver_success'],'solver_iterations':audit['iterations'],
            'recorded_major_callback_count':len(major),'lower_cost_recorded_major_count':sum(row['assessment']['objective']<baseline-1e-12 for row in major),
            'feasible_improving_recorded_major_count':sum(row['eligible_lower_objective_candidate'] for row in major),
            'first_recorded_major_max_abs_delta_rad':float(np.abs(first_delta).max()),
            'first_recorded_major_joints_at_035rad_trust_edge_count':int(np.count_nonzero(np.abs(np.abs(first_delta)-.35)<1e-6)),
            'first_recorded_major_tip_error_increase_m':first['assessment']['tip_error_increase_m'],
            'per_tip_nonworsening_refusal_count':tip_fail_counts.tolist(),
            'best_max_tip_increase_recorded_major_callback_number':major[best_guard_index]['major_iteration'],
            'minimum_over_majors_max_tip_error_increase_m':float(maxinc[best_guard_index]),
            'best_guard_major_objective':major[best_guard_index]['assessment']['objective'],
            'baseline_facing_sse':case['baseline_fresh_diagnostics']['component_sse']['facing'],
            'baseline_distance_plus_cp_sse':sum(case['baseline_fresh_diagnostics']['component_sse'][k] for k in ['distance','closest_vector']),
            'FK_only_callback_supplied':audit['FK_only_fingertip_callback_supplied'],
            'full_FK_callback_agreement_max_error_m':audit['full_FK_callback_agreement_max_error_m'],
            'evaluation_cache_hits':audit['evaluation_cache_hits'],'fingertip_cache_hits':audit['fingertip_cache_hits']})
    summary={'schema':'hug_frozen_objective_conditioning_analysis_v1','report_sha256':sha(OUT/'report.json'),
        'protocol_sha256':report['protocol_sha256'],'analysis_script_sha256':sha(__file__),
        'bounded_objective_stencils':case_diagnostics,'all_six_original_major_trace_analysis':traces,
        'guards':report['guards'],'full_geometry_calls':report['full_geometry_calls'],
        'optimizer_run':False,'new_eval_data':False,'physics_steps':0,
        'supported_claims':[
            'In all six recorded first major callbacks, many joint values hit the 0.35rad correction bound and nonlinear fingertip errors greatly worsen.',
            'At 1e-6 and 1e-7 rad, original q0 slopes on the selected nine case-coordinate combinations converge with no normal validity/identity change. Positive 1e-5 rad also retains valid sites.',
            'For MugB case 28, negative 1e-5 rad WRJ1 and THJ1 each invalidate thumb bin 283 through the mapped-mesh competing-facet safeguard; the fixed penalty creates a measured cost jump and poor backward/central slopes at that step.',
            'Distance/closest-vector terms dominate baseline scalar cost at the three representative cases; raw final candidates lower those terms while worsening direction cost and making some original fixed direction bins invalid.',
            'The documented byte-keyed cache and zero full/FK error are consistent with faithful cheap constraint evaluation; cache hits do not make a candidate feasible.'
        ],'unsupported_claims':[
            'No global smoothness, complete24-coordinate gradient conditioning or full-trajectory derivative diagnosis is established.',
            'No global feasible correction or impossibility certificate, and no source-label/physical-context equivalence, is established.',
            'The observed nearby invalidity jump does not establish causation of all 6 solver failures. Original forward q0 derivative steps on these 3coordinates avoidit; other coordinates and later steps remain unmeasured.',
            'No scientific acceptance criterion or direction weight may be changed on the basis of this diagnostic without a separately declared development protocol.'
        ]}
    write(OUT/'analysis.json',summary)
    lines=['# HUG objective diagnostic (2026-10-14)','',
        'Read-only diagnosis of the completed original constrained attempt. A frozen 60-call stencil used the literal original V2.1 callback on saved cases 8/20/28; no optimizer, generated grasp, physics step, outcome label or eval data.',
        '', '## What the recorded old fit actually did', '',
        'All six first recorded major callbacks move many joints to their 0.35 rad correction limit. This large local-SQP departure worsens nonlinear tip errors; intermediate local-SQP steps are not promised to be feasible. The best maximum tip increase over recorded majors is still above the declared 10 nm final allowance in every case. This is evidence of a failed local procedure, not a proof that no feasible mapping correction exists.', '']
    for t in traces:
        lines.append(f"- Case {t['source_inventory_index']}: {t['first_recorded_major_joints_at_035rad_trust_edge_count']} joints at 0.35 rad on first callback; {t['lower_cost_recorded_major_count']}/{t['recorded_major_callback_count']} callbacks improve scalar cost; zero eligible improving callbacks. Smallest recorded maximum tip increase {t['minimum_over_majors_max_tip_error_increase_m']*1e6:.6f} µm.")
    lines+=['','## Tiny objective probes','',
        'For WRJ1, FFJ1 and THJ1 at each representative q0, symmetric 1e-6 and 1e-7 rad steps produced no normal invalidity/part/geometry/tie identity change; slopes converge. Positive 1e-5 rad steps also retain valid sites. The two negative 1e-5 rad MugB changes described below are exceptions. This narrow probe distinguishes a measured nearby objective jump from proof that this jump caused all six solver failures; other coordinates and later feature switches remain untested.', '']
    for c in case_diagnostics:
        lines.append(f"- Case {c['source_inventory_index']}: facing cost {c['baseline_component_soft_l1_cost']['facing']:.8f}/{c['baseline_objective']:.8f} ({100*c['baseline_facing_fraction_of_total_cost']:.3f}%) at q0; raw final facing cost {c['raw_final_component_soft_l1_cost']['facing']:.8f} with {c['raw_final_invalid_direction_bins']} invalid bins. Central slopes at 1e-7 rad: "+', '.join(f"{r['joint_name']} {r['steps'][-1]['central_slope_cost_per_rad']:.8f}" for r in c['coordinate_stencils'])+'cost/rad.')
    for c in case_diagnostics:
        for event in c['observed_invalidity_events']:
            lines.append(f"- Case {c['source_inventory_index']} {event['joint_name']} {event['sign']*event['step_rad']:.0e} rad invalidates {event['anatomical_part']} bin {event['bin_index']} ({', '.join(event['invalid_reasons'])}); facing cost jumps {event['total_facing_cost_increase_vs_original']:.8f}. Native score {event['native_score']:.8f}; original mapped score {event['baseline_score']:.8f}; unavailable candidate score {event['inactive_candidate_score']:.8f} is correctly replaced by the fixed 2.0 residual before weighting.")
    lines+=['','## Implication for the next development attempt','',
        'Preserve strict per-tip eligibility and the independent mapping audit. Use a much smaller feasibility-aware development step/line-search instead of assuming that local SQP trial poses remain feasible; independently recompute all accepted poses. The objective stencil does not justify weakening the constraints or retuning away bad native grasp proposals.',
        '', 'The original objective retains fixed invalid-site penalties. The actual penalty is piecewise and jumps when eligibility changes; this is both a mathematical property of the definition and measured in the two negative MugB perturbations. Raw endpoints contain further invalid bins, but those endpoint observations do not identify causal transitions along the solver trajectory.',
        '', f"Protocol SHA256: `{report['protocol_sha256']}`. Geometry report SHA256: `{summary['report_sha256']}`. All {report['full_geometry_calls']} full geometry calls were static; integration attempts {report['guards']['actual']}, deliberate guard selftests {report['guards']['selftests']}. Source hashes remained unchanged.", '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    files=['protocol.json','diagnose_objective.py','extracted_literal_setup.py','run.log','report.json','summarize_objective.py','analysis.json','REPORT.md']
    manifest={'schema':'hug_frozen_objective_conditioning_manifest_v1','files':[{'path':f,'sha256':sha(OUT/f),'bytes':(OUT/f).stat().st_size} for f in files]}
    write(OUT/'manifest.json',manifest)

if __name__=='__main__':main()
