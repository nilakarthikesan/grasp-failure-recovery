"""Conditional one-shot fine-scale extension; execute only after all six V1 fallbacks."""
from pathlib import Path
import json,hashlib,shutil
V1=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v1');V2=Path('/private/tmp/hug_free_palm_fit_2026-10-15_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=json.loads((V1/'fit_summary.json').read_text());targets=json.loads((V1/'candidate_targets.json').read_text())
assert len(s['cases'])==6 and all(not c['selected_new_pose'] for c in s['cases'])
assert all(not c['solver_audit']['surface_candidate_returned'] for c in targets['cases'])
assert not V2.exists();V2.mkdir()
for name in ('run_free_palm.py','free_palm_solver.py','free_palm_kinematics.py','keyed_direction_evidence.py','surface_component_guard.py','geometry_helpers.py','exact_direction.py','merge_free_palm.py','test_free_palm.py','test_keyed_direction_evidence.py','protected_hashes.json','freeze_protocol.py','archive_bundle.py','summarize_exploration.py'):
    shutil.copyfile(V1/name,V2/name)
p=V2/'free_palm_kinematics.py';text=p.read_text();a=text.index('def initial_seeds(bounds):');b=text.index('def virtual_tip_jacobian(',a)
text=text[:a]+'''def initial_seeds(bounds):
    result=[];labels=[]
    for distance_m in (.00125,.000625,.0003125,.00015625):
        for axis in range(3):
            for sign in (1,-1):
                x=np.zeros(27);x[24+axis]=sign*distance_m/TRANSLATION_SCALE_M
                result.append(np.clip(x,bounds[:,0],bounds[:,1]));labels.append(f'fine_translation_{distance_m:g}m_axis{axis}_sign{sign:+d}')
        for sx in (1,-1):
            for sy in (1,-1):
                for sz in (1,-1):
                    x=np.zeros(27);x[24:]=np.asarray([sx,sy,sz])/np.sqrt(3)*distance_m/TRANSLATION_SCALE_M
                    result.append(np.clip(x,bounds[:,0],bounds[:,1]));labels.append(f'fine_translation_{distance_m:g}m_diagonal_{sx:+d}_{sy:+d}_{sz:+d}')
    return np.asarray(result),labels

'''+text[b:];p.write_text(text)
p=V2/'run_free_palm.py';text=p.read_text().replace("p['method_version']==1 and p['method_stage']=='free_palm_normalized_pattern_v1'","p['method_version']==2 and p['method_stage']=='free_palm_fine_translation_seed_v2'").replace("p['solver_configuration']['maximum_pattern_sweeps']==4","p['solver_configuration']['maximum_pattern_sweeps']==0").replace("p['solver_configuration']['maximum_full_geometry_evaluations']==128","p['solver_configuration']['maximum_full_geometry_evaluations']==64").replace("assert seeds.shape==(26,27)","assert seeds.shape==(56,27)").replace("'normalized_seed_or_pattern_after_FK_restore'","'fine_translation_seed_after_FK_restore'").replace("'method_version':1,'method_stage':'free_palm_normalized_pattern_v1'","'method_version':2,'method_stage':'free_palm_fine_translation_seed_v2'");p.write_text(text)
p=V2/'test_free_palm.py';text=p.read_text().replace('seeds.shape==(26,27) and len(labels)==26','seeds.shape==(56,27) and len(labels)==56').replace('[.5,0,0]','[.25,0,0]').replace('[-.5,0,0]','[-.25,0,0]').replace('np.linalg.norm(seeds[-8:,24:],axis=1),1','np.linalg.norm(seeds[-8:,24:],axis=1),.03125');text=text.replace("assert np.allclose(np.linalg.norm(seeds[-8:,24:],axis=1),.03125)","assert np.allclose(np.linalg.norm(seeds[-8:,24:],axis=1),.03125)\n    for block, expected in zip(np.split(seeds,4),(.25,.125,.0625,.03125),strict=True):\n        assert np.array_equal(block[:,:24],np.zeros((14,24)))\n        assert np.allclose(np.linalg.norm(block[:,24:],axis=1),expected,rtol=1e-14,atol=0)");p.write_text(text)
p=json.loads((V1/'protocol.json').read_text())
for key in ('frozen_utc','final_review_binding','root_review','manufactured_controls'):p.pop(key,None)
p.update(method_version=2,method_stage='free_palm_fine_translation_seed_v2',execution_authorized_by_root_review=False,draft_only_not_executed=True,
    actual_geometry_measured_before_freeze=False,direction_helper_path=str(V2/'exact_direction.py'),
    preceding_study={'protocol_path':str(V1/'protocol.json'),'protocol_sha256':sha(V1/'protocol.json'),
        'summary_path':str(V1/'fit_summary.json'),'summary_sha256':sha(V1/'fit_summary.json'),
        'candidate_path':str(V1/'candidate_targets.json'),'candidate_sha256':sha(V1/'candidate_targets.json'),
        'trigger':'All six V1 selections were exact original fallbacks. Root authorized one finite previously unvisited fine-scale/direction extension. This is an explicitly exploratory follow-up, not independent held-out validation.'},
    maximum_full_surface_evaluations_per_case=64,
    predeclared_seeds={'count':56,'order':'For each distance1.25,.625,.3125,.15625mm: six XYZaxis +,- then eight normalized cube diagonals signs+,- lexicographic. All original joints and original anchor, clamp to original conservative domain.',
        'units':'optimization hyperparameters only, not accuracy tolerances or measured witness gaps','seeds_frozen_before_first_case_surface_callback':True},
    selection_policy='Exactly56 fine translation seeds restored under original fingertip caps and same direction guard, zero pattern sweeps, at most64fullcallbacks, fresh terminal checks. Stop this declared extension regardless of result.',
    optimization_order='Four fixed fine-scale groups,6axes+8diagonals each. Original q0/native caps never rebased. No broadseed repetition and no additional pattern directions.',
    callback_budgets='Per case max64fullcallbacks incl terminal reserves,6000cheapFK and6000analyticJac callbacks separately,256restorations. Pre-solver Jacobian gate/final independent FK counted separately.',
    scope='Single frozen fine-scale static follow-up on the same six existing development grasps. No inference, physics, eval outcomes or training.')
p['solver_configuration']['maximum_pattern_sweeps']=0;p['solver_configuration']['maximum_full_geometry_evaluations']=64
for key in ('protocol','summary','candidate'):
    path=p['preceding_study'][key+'_path'];p['source_sha256'][path]=p['preceding_study'][key+'_sha256']
p['implementation_sha256']={str(V2/Path(path).name):sha(V2/Path(path).name) for path in p['implementation_sha256']}
p['protected_snapshot_sha256']=sha(V2/'protected_hashes.json')
(V2/'protocol.before_tests.json').write_text(json.dumps(p,indent=2,allow_nan=False)+'\n')
print(json.dumps({'prepared':str(V2),'actual_fine_geometry_before_freeze':False}))
