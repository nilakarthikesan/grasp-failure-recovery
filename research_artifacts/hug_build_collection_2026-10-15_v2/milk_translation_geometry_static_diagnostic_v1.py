"""Bounded static geometry/query check; all physics entry points blocked."""
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
from grasp_failure_prediction.evaluation.table_clearance import _descendant,_lowest_world_z

generation=json.loads((evidence/'generation.json').read_text())
schedule=json.loads((evidence/'study_schedule.json').read_text())
proposal=next(p for p in generation['proposals'] if p['slot_id']=='proposal_0023')
scenario=next(s for s in schedule['scenarios'] if s['attempt_id']=='study_attempt_0079')

def arr_sha(a):
    a=np.ascontiguousarray(a)
    return c.digest(dict(dtype=a.dtype.str,shape=list(a.shape),bytes_sha256=hashlib.sha256(a.tobytes()).hexdigest()))

cases=[];raw={}
with static_only() as counts:
    for ci,offset in enumerate((0.,.02)):
        with tempfile.TemporaryDirectory(prefix='milk-static-geometry-') as tmp:
            context,pose,palm,quat=c._case_and_context(protocol,proposal,pathlib.Path(tmp))
            runner=c._new_runner(protocol,context)
            chosen=dict(scenario);chosen['table_offset_m']=offset
            palm,actual=c._apply_scenario(runner,context,chosen,palm)
            plan=plan_grasp_approach(runner,pose,palm,quat,clearance=runner.clearance_configuration,approach=runner.approach_configuration)
            record=plan.as_record();candidate=next(p for p in record['candidates'] if p['candidate_id']=='palm_minus_y')
            direction=np.asarray(candidate['direction_world'])
            near_open=np.asarray(candidate['path']['near_open_targets_rad'])
            model=runner.model
            hands=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in runner._hand_body_ids and (model.geom_contype[g] or model.geom_conaffinity[g])]
            proximal=[g for g in hands if not _descendant(model,int(model.geom_bodyid[g]),runner._palm_body_id)]
            objects=sorted(runner.object_collision_geom_ids)
            relevant=sorted(set(hands+objects));pairs=[(g,o) for g in proximal for o in objects]
            addresses=[int(model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in runner._hand_joint_names]
            scratch=mujoco.MjData(model);scratch.qpos[:]=runner.data.qpos
            root=int(runner._root_qpos_address)
            position=np.asarray(record['clearance_plan']['root_position_m'])
            quaternion=np.asarray(record['clearance_plan']['root_quaternion_wxyz'])
            geometries=[dict(geom_id=g,name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g),
                body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g])),
                geom_type=int(model.geom_type[g]),data_id=int(model.geom_dataid[g])) for g in relevant]
            intrinsic={name:arr_sha(getattr(model,name)[relevant]) for name in ('geom_type','geom_dataid','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_margin','geom_gap')}
            intrinsic.update({name:arr_sha(getattr(model,name)) for name in ('mesh_vert','mesh_face','mesh_normal','mesh_texcoord') if hasattr(model,name)})
            points=[];rotations=[];distances=[];table=[];commands=[]
            for alpha in np.linspace(0.,1.,33):
                scratch.qpos[addresses]=near_open
                scratch.qpos[root:root+7]=np.r_[position-(1-alpha)*runner.protocol.parameters.pregrasp_distance_m*direction,quaternion]
                mujoco.mj_forward(model,scratch)
                points.append(scratch.geom_xpos[relevant].copy());rotations.append(scratch.geom_xmat[relevant].copy())
                distances.append([float(mujoco.mj_geomDistance(model,scratch,g,o,10.,None)) for g,o in pairs])
                table.append(float(scratch.geom_xpos[runner._table_geom_id,2]+model.geom_size[runner._table_geom_id,2]))
                commands.append(scratch.qpos[root:root+7].copy())
            raw.update({f'positions_{ci}':np.asarray(points),f'rotations_{ci}':np.asarray(rotations),
                        f'distances_{ci}':np.asarray(distances),f'table_top_{ci}':np.asarray(table),f'root_command_{ci}':np.asarray(commands)})
            cases.append(dict(offset_m=offset,plan_feasible=plan.feasible,geom_ids=relevant,hand_geom_ids=hands,proximal_geom_ids=proximal,
                object_geom_ids=objects,geometries=geometries,query_pairs=[list(p) for p in pairs],intrinsic_geometry_hashes=intrinsic,
                near_open_targets_rad=near_open.tolist(),direction_world=direction.tolist(),
                query_distances_m=np.asarray(distances).tolist(),table_top_z_m=table,
                raw_array_hashes={k:arr_sha(v) for k,v in raw.items() if k.endswith('_'+str(ci))}))

expected=np.array([0.,0.,.02])
position_error=raw['positions_1']-raw['positions_0']-expected
rotation_error=raw['rotations_1']-raw['rotations_0']
dist_delta=raw['distances_1']-raw['distances_0']
unexpected=[]
for ai,pi in np.argwhere(abs(dist_delta)>1e-10):
    pair=cases[0]['query_pairs'][int(pi)]
    unexpected.append(dict(alpha_index=int(ai),alpha=float(ai/32),pair=pair,
        geoms=[g for g in cases[0]['geometries'] if g['geom_id'] in pair],
        offset0_distance_m=float(raw['distances_0'][ai,pi]),offset002_distance_m=float(raw['distances_1'][ai,pi]),
        pair_position_translation_residual_max_m=float(np.max(abs(position_error[ai,[cases[0]['geom_ids'].index(g) for g in pair]]))),
        pair_rotation_residual_max=float(np.max(abs(rotation_error[ai,[cases[0]['geom_ids'].index(g) for g in pair]])))))
raw_root=pathlib.Path(protocol['raw_output'])/'milk_translation_geometry_diagnostic_v1'
raw_root.mkdir(exist_ok=False)
np.savez_compressed(raw_root/'static_geometry_queries.npz',**raw)
result=dict(schema_version='hug_milk_translation_geometry_diagnostic_v1',proposal_slot='proposal_0023',native_sha256=proposal['native_sha256'],
    source_protocol_sha256=c.sha(evidence/'preparation.json'),generation_sha256=c.sha(evidence/'generation.json'),
    driver_sha256=c.sha(project/'scripts/run_hug_build_collection.py'),cases=cases,
    all_geom_position_translation_residual_max_m=float(abs(position_error).max()),
    all_geom_rotation_residual_max=float(abs(rotation_error).max()),
    same_query_pairs=cases[0]['query_pairs']==cases[1]['query_pairs'],same_geometry_intrinsics=cases[0]['intrinsic_geometry_hashes']==cases[1]['intrinsic_geometry_hashes'],
    same_near_open_targets=cases[0]['near_open_targets_rad']==cases[1]['near_open_targets_rad'],same_direction=cases[0]['direction_world']==cases[1]['direction_world'],
    table_top_translation_residual_max_m=float(np.max(abs(raw['table_top_1']-raw['table_top_0']-.02))),
    query_distance_max_difference_m=float(abs(dist_delta).max()),noninvariant_queries=unexpected,api_counts=counts,
    raw_artifacts={str(raw_root/'static_geometry_queries.npz'):c.sha(raw_root/'static_geometry_queries.npz')},
    source_mutations=0,new_dynamic_outcomes=0,existing_labels_modified=False,
    interpretation='same hand/object query geometry under joint translation can return zero; current frozen protocol retains conservative static refusals; no relabeling or approach-code repair')
dest=pathlib.Path('/private/tmp/milk_translation_geometry_static_diagnostic.json')
with dest.open('x') as handle:json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False);handle.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)},indent=2))
