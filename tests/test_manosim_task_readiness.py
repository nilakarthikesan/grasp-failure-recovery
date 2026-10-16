"""Manufactured native task bridge records; no scientific fits or engine work."""
from pathlib import Path
import importlib.util,sys
import numpy as np
import pytest
from grasp_failure_prediction.prediction import manosim_task_readiness as bridge
from grasp_failure_prediction.prediction import manosim_training as training

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_readiness_manufactured_fixtures',ROOT/'tests/test_manosim_training.py');fixtures=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixtures;spec.loader.exec_module(fixtures)

@pytest.fixture(autouse=True)
def no_engine_or_fit(monkeypatch):
 import mujoco
 def no(*a,**kw):raise AssertionError('Readiness controls prohibit native work and fitting')
 for name in ('MjModel','MjData','mj_forward','mj_kinematics','mj_step','mj_step1','mj_step2'):monkeypatch.setattr(mujoco,name,no)
 monkeypatch.setattr(training,'fit_fold',no);monkeypatch.setattr(training,'fit_logistic',no)


def setup(tmp_path,changes=None):
 cp,sp,c=fixtures.collection(tmp_path,changes);splits=fixtures.put(tmp_path/'splits.json',training.freeze_splits(sp));return cp,sp,c,splits


def test_unqualified_actual_recording_format_keeps_training_closed(tmp_path):
 cp,sp,c,splits=setup(tmp_path);r=bridge.export_readiness(cp,splits,tmp_path/'readiness')
 assert r['scientific_fits_performed']==0 and not r['scientific_training_allowed'] and r['trainable_collection'] is None
 assert r['readiness']['scheduled']==6 and r['readiness']['eligible']==6 and r['models']==list(training.MODELS)
 assert r['feature_contract']['channels']==169 and all(row['cutoff_s']<.84 for row in r['eligible_rows'])


def test_only_exact_external_independent_qualification_opens_ready_pointer(tmp_path,monkeypatch):
 cp,sp,c,splits=setup(tmp_path);qualified=fixtures.qualify(cp,sp,c,monkeypatch)
 r=bridge.export_readiness(cp,splits,tmp_path/'ready',qualifications={c.contract_sha256:qualified})
 assert r['scientific_training_allowed'] and r['trainable_collection'] and not r['adequacy_guaranteed']
 assert r['maximum_candidate_fits_per_fold']==24 and all(f['status']=='READY' for f in r['implemented_category_training_folds'])


def test_fixture_pass_and_mapping_runtime_labels_cannot_become_negatives(tmp_path):
 cp,sp,c,splits=setup(tmp_path,{'a0':{'mapping':'refused_guard'},'a1':{'status':'runtime_error'}})
 fake=fixtures.put(tmp_path/'fixture.json',dict(schema_version='manosim_capsule_execution_v3_report',status='COMPLETED',qualification='PASS',scientific_training_allowed=True))
 r=bridge.export_readiness(cp,splits,tmp_path/'closed',qualifications={c.contract_sha256:fake})
 assert not r['scientific_training_allowed'] and r['readiness']['excluded']==2
 assert {r['reason'] for r in r['excluded_rows']}=={'mapping_or_setup_refusal','runtime_or_unexecuted'}


def test_earliest_failure_at_or_before_cutoff_excluded_unknown_time_excluded(tmp_path):
 cp,sp,c,splits=setup(tmp_path,{'a1':{'outcome':dict(success=False,failure_type='drop',label_rule_id='native_task_label_v1',first_failure_time_s=.8)},'a3':{'outcome':dict(success=False,failure_type='drop',label_rule_id='native_task_label_v1')}})
 r=bridge.export_readiness(cp,splits,tmp_path/'closed');assert r['readiness']['eligible']==4
 assert {r['reason'] for r in r['excluded_rows']}=={'already_failed_at_or_before_cutoff','unknown_failure_time'}


def test_singleclass_roles_close_pointer_even_when_task_evidence_qualified(tmp_path,monkeypatch):
 changes={f'a{i}':{'status':'runtime_error'} for i in (1,3,5)};cp,sp,c,splits=setup(tmp_path,changes);fixtures.qualify(cp,sp,c,monkeypatch)
 r=bridge.export_readiness(cp,splits,tmp_path/'closed');assert r['readiness']['qualified_native_task_evidence'] and not r['scientific_training_allowed'] and r['trainable_collection'] is None
 assert all(f['status']=='SKIPPED_EMPTY_OR_SINGLE_CLASS_ROLE' for f in r['implemented_category_training_folds'])


def test_missing_contract_qualification_and_existing_output_refused(tmp_path):
 cp,sp,c,splits=setup(tmp_path)
 with pytest.raises(ValueError):bridge.export_readiness(cp,splits,tmp_path/'a',qualifications={})
 out=tmp_path/'existing';out.mkdir()
 with pytest.raises(FileExistsError):bridge.export_readiness(cp,splits,out)


def test_final_valid_but_different_data_snapshot_refused_before_publication(tmp_path,monkeypatch):
 cp,sp,c,splits=setup(tmp_path);original=training.load_data;calls=[]
 def load(*a,**kw):
  d=original(*a,**kw);calls.append(True)
  if len(calls)==3:d.X[0,0,0]+=1
  return d
 monkeypatch.setattr(training,'load_data',load)
 with pytest.raises(ValueError,match='transaction changed'):bridge.export_readiness(cp,splits,tmp_path/'failed')
 assert not (tmp_path/'failed/readiness.json').exists()


def test_object_holdout_coverage_allows_familiar_categories_but_refuses_object_leak(tmp_path):
 cp,sp,c,splits=setup(tmp_path);d=training.load_data(cp)
 from dataclasses import replace
 schedule=d.schedule.model_dump(mode='json')
 for a in schedule['attempts']:a['category_id']='shared_familiar_category'
 d=replace(d,schedule=training.Schedule.model_validate(schedule),rows=tuple(dict(row,category_id='shared_familiar_category') for row in d.rows))
 assignments={a.attempt_id:('train','validation','test')[i//2] for i,a in enumerate(d.schedule.attempts)}
 r=bridge._coverage(d,assignments);assert r['status']=='READY'
 assignments['a1']='test'
 with pytest.raises(ValueError,match='grouping leak'):bridge._coverage(d,assignments)


def test_seven_object_fold_readiness_is_separate_and_preserves_exact_source_join(tmp_path,monkeypatch):
 cp,sp,c,splits=setup(tmp_path);d=training.load_data(cp);objects=sorted({a.object_id for a in d.schedule.attempts});folds=[];joined=[]
 for i in range(7):
  roles=dict(train=[objects[0]],validation=[objects[1]],test=[objects[2]]);fold=dict(fold_id='object_holdout_'+str(i),generalization_claim='UNSEEN_OBJECT_IN_SEEN_GEOMETRY_CATEGORIES',**{role+'_object_ids':ids for role,ids in roles.items()});folds.append(fold);assignments={a.attempt_id:next(role for role,ids in roles.items() if a.object_id in ids) for a in d.schedule.attempts};joined.append(dict(fold_id=fold['fold_id'],generalization_claim=fold['generalization_claim'],object_membership=roles,assignments=assignments))
 catalog=fixtures.put(tmp_path/'catalog.json',{'software_only':True});spec=fixtures.put(tmp_path/'folds.json',dict(folds=folds));monkeypatch.setattr(bridge,'CATALOG_SHA',training.file_sha256(catalog)[7:]);monkeypatch.setattr(bridge,'FOLD_SPEC_SHA',training.file_sha256(spec)[7:])
 value=dict(schema_version='manosim_native_pilot_catalog_seven_folds_v1',selection_uses_outcomes=False,cohort='primary120_only_matched1ms_excluded',trainer_five_category_manifest_unchanged=True,seven_fold_training_API_implemented=False,catalog=training.binding(catalog),source_fold_spec=training.binding(spec),folds=joined);manifest=fixtures.put(tmp_path/'catalog_manifest.json',value)
 results,files=bridge.catalog_readiness(d,manifest);assert len(results)==7 and all(f['status']=='READY' and not f['training_API_implemented'] for f in results)
 assert len(files)==3
 value['folds'][0]['assignments']['a0']='test';fixtures.put(manifest,value)
 with pytest.raises(ValueError,match='role join'):bridge.catalog_readiness(d,manifest)


def test_changed_source_or_input_cannot_publish_readiness(tmp_path,monkeypatch):
 cp,sp,c,splits=setup(tmp_path);original=training.file_sha256;calls=[]
 def changing(path):
  result=original(path)
  if Path(path)==Path(bridge.__file__):
   calls.append(True)
   if len(calls)>1:return 'sha256:'+'f'*64
  return result
 monkeypatch.setattr(training,'file_sha256',changing)
 with pytest.raises(ValueError,match='source/input changed'):bridge.export_readiness(cp,splits,tmp_path/'failed')
 assert not (tmp_path/'failed/readiness.json').exists()
