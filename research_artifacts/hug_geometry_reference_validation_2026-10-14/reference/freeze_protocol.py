from pathlib import Path
import hashlib,json,datetime
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
DEP=Path('/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps/dex_retargeting')
OUT=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals'
inventory=json.loads((prior/'existing_hug_inventory.json').read_text())
p={
 'schema':'hug-static-reference-retargeting-v1','date':'2026-10-14','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'Static development comparison on32existing HUG poses; no inference, rollout, eval labels, predictor fitting, calibration, acceptance thresholds or scientific success filtering.',
 'source_inventory_path':str(prior/'existing_hug_inventory.json'),'source_inventory_sha256':sha(prior/'existing_hug_inventory.json'),
 'prior_fk_report_path':str(prior/'static_saved_hug_fk.json'),'prior_fk_report_sha256':sha(prior/'static_saved_hug_fk.json'),
 'cases':[{'source_inventory_index':i,'native_sha256':c['native']['sha256'],'object_id':c['object_id'],'seed':c['native']['seed'],'observation_group_id':c['observation_group_id']} for i,c in enumerate(inventory['modern_cases'])],
 'reference_method':'dex-retargeting0.5.0 released five-finger DexPilot adaptation; distinct objective, not original Allegro implementation or HUG gold-standard mapping.',
 'released_config_path':str(DEP/'configs/teleop/shadow_hand_right_dexpilot.yml'),'released_config_sha256':sha(DEP/'configs/teleop/shadow_hand_right_dexpilot.yml'),
 'production_urdf_sha256':sha(ROOT/'third_party/dex-urdf/robots/hands/shadow_hand/shadow_hand_right.urdf'),
 'methods':{
 'stored_current':{'origin':'previous recorded24named nominal target at closing endpoint, before force-close','optimizer_run':False,'scientific_primary_current':True},
 'current_recomputed':{'origin':'actual production ShadowHandRetargeter fresh percase','scale':1.0,'filter':None,'repeated_solve_calls':20,'purpose':'control for reproducing saved nominal target; discrepancy retained rather than overwritten'},
 'dexpilot_shipped':{'origin':'released YAML+defaults unchanged','scale':1.2,'low_pass_alpha':0.2,'repeated_solve_calls':20,'purpose':'primary external reference, using20static repeats to match current wrapper solve-count policy; raw and filtered20both retained','targets':'10intertip+5ee_link-to-tip vectors; ee_link coincides withpalm origin under actual URDF','project_dist_m':0.03,'escape_dist_m':0.05,'norm_delta':0.004,'huber_delta':0.02},
 'dexpilot_metric_static':{'origin':'predeclared static sensitivity to teleoperation scaling and smoothing','changes_from_shipped':{'scale':1.0,'filter':None},'repeated_solve_calls':20,'purpose':'descriptive sensitivity, not a calibrated or selected replacement'}},
 'initialization':'Fresh method instance pernative pose; packaged joint-limit midpoint; no cross-case warm start/projection/filter state. WRJ1/WRJ2 optimized with remaining22joints, no frozen-wrist alteration.',
 'frames':'Use same saved camera landmarks after inverseT_camera_wrist then proper MANO_TO_OPERATOR_RIGHT rotation. Independent fixedURDF FK yields palm translation/rotation. Apply actual production mapping policy: nativewrist becomesShadowpalm origin and R_world_operator composes with eachconfiguration fixedURDF palm rotation; derive floating carrier analytically from that targetpalm and actual q-specific identity-root FK. No adjustable worldtranslation/rotation, same object/camera geometry and initial state.',
 'fixed_landmark_correspondence':{'tips':[4,8,12,16,20],'middles':[2,6,10,14,18],'fingers':['thumb','index','middle','ring','little']},
 'measurements':['Perlink/perfinger native-to-target displacement inworld, initialobject and operatorbasis','10unordered intertip vector residuals/length changes/angular differences','Reference-vs-stored target point and24joint changes','FixedURDF MuJoCo versus independentPinocchio targetlink FK','Original stored root andpoints reproduce; regenerated current discrepancy retained','Solver returncode/evaluations/fallbacks/projectedpair flags/raw-vs-filtered result, physical-limit clipping and displacement'],
 'solver_reporting':'Wrap actual NLopt optimize to capture result and exceptions, allowing library fallback behavior to finish while explicitly recording failure; no silent solver-success assertion. Raw values must finite24 and within physicalmargin1.01e-3rad before clipping; invalid outcomes retained as rejected without successful-q FK.',
 'guards':['mj_step','mj_step1','mj_step2'],'guard_self_test':True,'invariants':['allscene time0','initial object stateunchanged','allinput/code/config/sourceasset hashesunchanged','identical24named joint convention','no new learned model inference'],
 'surface_followup':'Root may use produced static configurations for six existing visual-surface cases on identical frozenqueries. This report contains no contact metrics and does not infer contact quality frompoint error.',
 'limitations':['Existing32development poses are eight observationgroups/fourobjects with oneMANO shape; not independent held-out validation.','Different hand morphology and reference point definitions remain approximation.','DexPilot projects close tip pairs intentionally; this can improve pinch execution but can also change an originally poor proposal.','Released vector and DexPilot objectives neither constrain object surfaces nor preserve all native contact relationships.','No scientific pass threshold, no statistical independence of160fingertips, no successful lifting criterion.','Fixed anchoring policy is controlled, not certified anatomically accurate bybaseline comparison.'],
 'code_and_config_sha256':{str(p):sha(p) for p in [ROOT/'src/grasp_failure_prediction/evaluation/retargeting.py',ROOT/'src/grasp_failure_prediction/evaluation/pose_validation.py',ROOT/'src/grasp_failure_prediction/evaluation/case_runner.py',ROOT/'src/grasp_failure_prediction/integrations/hug.py',ROOT/'src/grasp_failure_prediction/integrations/hug_frames.py',DEP/'optimizer.py',DEP/'seq_retarget.py',DEP/'retargeting_config.py',DEP/'robot_wrapper.py',DEP/'configs/teleop/shadow_hand_right.yml',DEP/'configs/teleop/shadow_hand_right_dexpilot.yml']}
}
(OUT/'protocol.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
print(json.dumps({'protocol':str(OUT/'protocol.json'),'sha256':sha(OUT/'protocol.json'),'case_count':len(p['cases'])}))
