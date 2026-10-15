"""Bounded static diagnostic of the frozen HUG objective; no optimizer."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, sys, time

OUT = Path(__file__).parent
CANONICAL = Path('/private/tmp/hug_constrained_refinement_runtime_v2_1_2026-10-14')
DRIVER = CANONICAL / 'run_constrained_refinement.py'
sys.path.insert(0, str(CANONICAL))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def build_setup_module():
    # Reuse literal canonical setup/evaluate statements. Exclude orchestration,
    # optimizer invocation, all-cohort protection and result mutations.
    source = ast.parse(DRIVER.read_text())
    main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    first = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'configuration' for t in n.targets))
    loop_i = next(i for i,n in enumerate(main.body) if isinstance(n, ast.For)
                  and isinstance(n.target, ast.Name) and n.target.id == 'index')
    statements = copy.deepcopy(main.body[first:loop_i])
    loop_body = copy.deepcopy(main.body[loop_i].body)
    end = next(i for i,n in enumerate(loop_body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'returned' for t in ast.walk(n)))
    loop_body = loop_body[:end]
    evaluator = next(n for n in loop_body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    extra_i = next(i for i,n in enumerate(evaluator.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'last_extra' for t in n.targets))
    extra = ast.parse("""
last_extra['component_soft_l1_cost'] = {k: float(configuration.f_scale * np.sum(np.hypot(configuration.f_scale, v) - configuration.f_scale)) for k, v in components.items()}
last_extra['normal_fields_for_diagnosis'] = {key: normal_fields[key] for key in ('valid', 'score', 'part_index', 'closest_object_m', 'outward_normal', 'nearest_geom_id', 'tie_count', 'reason_masks')} if normal_fields is not None else {}
""").body
    evaluator.body[extra_i + 1:extra_i + 1] = extra
    statements += loop_body
    statements += ast.parse("""
def snapshot(q):
    evaluation = evaluate(q, emit=False)
    result = jsonable(last_extra)
    result['residuals'] = evaluation.residuals.tolist()
    return result
return {'q0': q0, 'baseline': jsonable(last_extra), 'snapshot': snapshot,
        'caseid': caseid, 'native_sha256': case['native']['sha256'],
        'input_sha256': meta['input_sha256'], 'joint_names': names,
        'original_full_callback_count': full_callback_count,
        'counter': lambda: full_callback_count, 'time': lambda: float(d.time),
        'object_unchanged': lambda: bool(np.array_equal(d.qpos[objadr:objadr+7],initial))}
""").body
    fn = ast.FunctionDef(name='build_diagnostic_case', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg='p'), ast.arg(arg='index')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=statements,
        decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    (OUT/'extracted_literal_setup.py').write_text(ast.unparse(tree) + '\n')
    spec = importlib.util.spec_from_file_location('canonical_diagnostic_driver', DRIVER)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    exec(compile(tree, str(OUT/'extracted_literal_setup.py'), 'exec'), mod.__dict__)
    return mod

def main():
    started=time.perf_counter(); protocol=read(OUT/'protocol.json'); ph=sha(OUT/'protocol.json')
    for path,h in protocol['source_sha256'].items(): assert sha(path)==h,path
    mod=build_setup_module(); p=read(CANONICAL/'protocol.json')
    records=[]
    for index in protocol['case_indices']:
        setup=mod.build_diagnostic_case(p,index); baseline=setup['baseline']; q0=setup['q0']
        old=read(CANONICAL/f'candidate_cohort_{index}.json')['cases'][0]
        assert abs(baseline['objective']-old['baseline_fresh_diagnostics']['objective'])<1e-12
        for k,v in baseline['component_sse'].items(): assert abs(v-old['baseline_fresh_diagnostics']['component_sse'][k])<1e-12
        probes=[]
        for joint in protocol['joint_names']:
            joint_index=setup['joint_names'].index(joint)
            for step in protocol['steps_rad']:
                for sign in [-1,1]:
                    q=q0.copy();q[joint_index]+=sign*step; tick=time.perf_counter()
                    result=setup['snapshot'](q)
                    probes.append({'joint_name':joint,'joint_index':joint_index,'step_rad':step,'sign':sign,
                                   'seconds':time.perf_counter()-tick,'diagnostics':result})
        raw=setup['snapshot'](mod.np.asarray(old['raw_candidate_qpos_rad']))
        assert abs(raw['objective']-old['solver_audit']['raw_assessment']['objective'])<1e-10
        for path,h in setup['input_sha256'].items():assert sha(path)==h,path
        assert setup['counter']()==20 and setup['time']()==0 and setup['object_unchanged']()
        records.append({'source_inventory_index':index,'caseid':setup['caseid'],'native_sha256':setup['native_sha256'],
                        'q0_rad':q0.tolist(),'baseline':baseline,'probes':probes,'raw_final':raw,
                        'full_geometry_calls':setup['counter'](),'static_time_s':setup['time'](),
                        'object_state_unchanged':setup['object_unchanged'](),'input_sha256':setup['input_sha256']})
        write(OUT/'report.partial.json',{'partial':True,'protocol_sha256':ph,'records':records})
        print(json.dumps({'case_completed':index,'calls':setup['counter'](),'elapsed_seconds':time.perf_counter()-started}),flush=True)
    for path,h in protocol['source_sha256'].items(): assert sha(path)==h,path
    assert sha(OUT/'protocol.json')==ph and mod.COUNTS['actual']==0
    report={'schema':'hug_frozen_objective_conditioning_diagnostic_v1','protocol_sha256':ph,
            'records':records,'guards':mod.COUNTS,'full_geometry_calls':sum(r['full_geometry_calls'] for r in records),
            'runtime_seconds':time.perf_counter()-started,'source_hashes_unchanged':True,
            'new_eval_data':False,'new_generator_proposals':False,'physics_steps':0,'optimizer_run':False,
            'scope':'Three old saved HUG grasps, three coordinates and three symmetric tiny perturbations; not full gradient or feasibility proof.'}
    write(OUT/'report.json',report)
    print(json.dumps({'complete':True,'full_geometry_calls':report['full_geometry_calls'],'runtime_seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
