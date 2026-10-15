"""Read-only bounded static diagnostic; no integration or historical edits."""
import sys,json,pathlib,tempfile,hashlib
import numpy as np
import mujoco

project=pathlib.Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
sys.path.insert(0,str(project/'scripts'))
import run_hug_build_collection as c
evidence=project/'research_artifacts/hug_build_collection_2026-10-15_v2'
protocol=c.read_protocol(evidence)
from grasp_failure_prediction.evaluation.mapping_validation import static_only
from grasp_failure_prediction.evaluation.approach_execution import plan_grasp_approach

schedule=json.loads((evidence/'study_schedule.json').read_text())
summary=json.loads((evidence/'study_summary.json').read_text())
generation=json.loads((evidence/'generation.json').read_text())
by={s['attempt_id']:s for s in schedule['scenarios']}
aggregate=[]
for offset in (0.,.02):
    rows=[r for r in summary['attempts'] if by[r['attempt_id']]['object_id']=='hug_milk_box' and by[r['attempt_id']]['table_offset_m']==offset]
    aggregate.append(dict(offset_m=offset,attempts=len(rows),refused=sum(r['status']=='APPROACH_REFUSED' for r in rows),
                          completed=sum(r['status']=='COMPLETED' for r in rows)))
refused=[dict(attempt_id=r['attempt_id'],scenario=by[r['attempt_id']]) for r in summary['attempts']
         if by[r['attempt_id']]['object_id']=='hug_milk_box' and r['status']=='APPROACH_REFUSED']
proposal=next(p for p in generation['proposals'] if p['slot_id']=='proposal_0023')
rows=[]
with static_only() as counts:
    for offset in (0.,.02):
        with tempfile.TemporaryDirectory(prefix='milk-static-translation-') as tmp:
            context,pose,palm,quat=c._case_and_context(protocol,proposal,pathlib.Path(tmp))
            runner=c._new_runner(protocol,context)
            scenario=dict(by['study_attempt_0079']);scenario['table_offset_m']=offset
            applied,actual=c._apply_scenario(runner,context,scenario,palm)
            plan=plan_grasp_approach(runner,pose,applied,quat,
                clearance=runner.clearance_configuration,approach=runner.approach_configuration)
            rows.append(dict(offset_m=offset,actual=actual,intended_world_palm=applied.tolist(),
                object_world_position=runner.data.xpos[runner._object_body_id].tolist(),
                table_world_position=runner.data.geom_xpos[runner._table_geom_id].tolist(),
                plan=plan.as_record()))
    delta={key:(np.asarray(rows[1][key])-np.asarray(rows[0][key])).tolist()
           for key in ('intended_world_palm','object_world_position','table_world_position')}
result=dict(schema_version='hug_milk_translation_static_diagnostic_v1',source_protocol_sha256=c.sha(evidence/'preparation.json'),
    study_summary_sha256=c.sha(evidence/'study_summary.json'),generation_sha256=c.sha(evidence/'generation.json'),
    driver_sha256=c.sha(project/'scripts/run_hug_build_collection.py'),proposal_slot='proposal_0023',
    native_sha256=proposal['native_sha256'],observed_milk_counts=aggregate,observed_refusals=refused,
    recomputed_static_cases=rows,world_translation_deltas=delta,api_counts=counts,
    source_mutations=0,new_dynamic_outcomes=0,scope='engineering diagnostic only; no protocol repairs or relabeling')
dest=pathlib.Path('/private/tmp/milk_translation_static_diagnostic.json')
with dest.open('x') as out:json.dump(result,out,indent=2,sort_keys=True,allow_nan=False);out.write('\n')
print(json.dumps(dict(result_path=str(dest),counts=aggregate,translation_deltas=delta,api_counts=counts,
    plans=[dict(offset_m=r['offset_m'],feasible=r['plan']['feasible'],selected=r['plan']['selected_candidate_id'],
        wrist=r['plan']['clearance_plan'],candidates=[dict(candidate_id=p['candidate_id'],status=p['status'],
            reason=p['rejection_reason'],min_table=p.get('path',{}).get('minimum_table_clearance_m'),
            min_object=p.get('path',{}).get('minimum_proximal_object_distance_m')) for p in r['plan']['candidates']])
           for r in rows]),indent=2))
