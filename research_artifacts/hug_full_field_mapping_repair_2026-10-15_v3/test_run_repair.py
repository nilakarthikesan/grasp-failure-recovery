"""Manufactured v3 receipt/declaration controls; no model, solve or physics."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('full_field_repair_v3_harness',Path(__file__).with_name('run_repair.py'))
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


def good_arithmetic():
    return dict(status='COMPLETED_BOUNDED_OPERAND_DIAGNOSTIC',failure=None,
        poses_completed=[[20,'original_20'],[20,'experimental_20'],[28,'original_28'],[28,'experimental_28']],
        poses_not_run_or_partial=[],actual=dict(inside_mask_disagreements=0,hull_predicate_disagreements=0,
            einsum_warnings=[],max_matmul_einsum_difference_m=5e-17,max_einsum_scalar_difference_m=5e-17,
            max_matmul_scalar_difference_m=5e-17))


def test_arithmetic_clearance_stays_scoped_and_fail_closed():
    driver.numerical_clearance(good_arithmetic())
    for key,value in [('failure',{}),('status','HARD_STOP'),('poses_completed',[]),('poses_not_run_or_partial',[[28,'original_28']])]:
        bad=good_arithmetic();bad[key]=value
        with pytest.raises(ValueError):driver.numerical_clearance(bad)
    for key,value in [('inside_mask_disagreements',1),('hull_predicate_disagreements',1),('einsum_warnings',['invalid'])]:
        bad=good_arithmetic();bad['actual'][key]=value
        with pytest.raises(ValueError):driver.numerical_clearance(bad)


@pytest.mark.parametrize('value',[np.nan,np.inf,True,-1.,1e-10,None])
def test_arithmetic_reference_comparison_cannot_be_disabled(value):
    bad=good_arithmetic();bad['actual']['max_matmul_einsum_difference_m']=value
    with pytest.raises(ValueError):driver.numerical_clearance(bad)


@pytest.fixture
def complete(tmp_path):
    shape=(2,6,3);preserved=np.ones(shape,bool);preserved.flat[0]=False
    lost=np.zeros(shape,bool);lost.flat[0]=True
    fields=dict(required=np.ones(shape,bool),preserved=preserved,definite_lost=lost,definite_added=np.zeros(shape,bool))
    state=SimpleNamespace(tip_errors_world_m=np.zeros((5,3)),original_tip_caps_m=np.full(5,.001),
        physical_joint_limits_pass=True,engineering_domain_pass=True)
    orientation=dict(valid=np.array([True,False]),passed=np.array([False,False]))
    receipts=[];full=[]
    for number,label in enumerate(('baseline','rejected_candidate','fresh_final'),1):
        path=tmp_path/f'checkpoint_{number:02d}_{label}.npz'
        margin=np.zeros(shape);distance=np.zeros((2,6))
        np.savez_compressed(path,margin_m=margin,distance_m=distance,**fields)
        digest=driver.sha(path)
        receipts.append(dict(path=str(path),sha256=digest,bytes=path.stat().st_size,label=label))
        full.append(dict(array_path=str(path),array_sha256=digest,fields=copy.deepcopy(fields),margin_m=margin,distance_m=distance,
            state=state,orientation=copy.deepcopy(orientation),summary=dict(full_strict_mapping_status='FAIL')))
    result=dict(baseline=full[0],returned=full[2],accepted_mapping=False,full_strict_mapping_accepted=False,
        accepted_relationship_repair=False,repaired_relationship_count=0,status='NO_ACCEPTED_SURFACE_IMPROVEMENT',rounds=[])
    return tmp_path,receipts,result


def test_compact_fit_indexes_all_write_time_checkpoint_hashes(complete):
    output,receipts,result=complete
    compact=driver.compact_fit(result,output,receipts)
    assert compact['checkpoint_artifacts']==receipts
    assert len(compact['checkpoint_artifacts'])==3
    assert 'rejected_candidate' in compact['checkpoint_artifacts'][1]['label']
    for key in ('baseline','returned'):
        assert 'fields' not in compact[key] and 'margin_m' not in compact[key]
        assert compact[key]['field_boolean_counts']['preserved']==35
        assert 'fields' in result[key]
    assert not compact['independent_same_part_admission_receipt']['local_relationship_repair_admitted']
    assert compact['independent_same_part_admission_receipt']['minimum_baseline_preserved_relationships']==35


def test_intermediate_tampering_cannot_receive_a_fresh_late_digest(complete):
    output,receipts,result=complete
    path=Path(receipts[1]['path']);path.write_bytes(b'changed intermediate')
    with pytest.raises(ValueError,match='Write-time checkpoint arrays changed'):driver.compact_fit(result,output,receipts)


def test_post_compaction_tampering_detected_before_publication(complete):
    output,receipts,result=complete;compact=driver.compact_fit(result,output,receipts)
    Path(receipts[1]['path']).write_bytes(b'changed after compaction')
    with pytest.raises(ValueError,match='changed'):driver.checkpoint_index(output,compact['checkpoint_artifacts'])


@pytest.mark.parametrize('change',['omitted_receipt','deleted_file','unindexed_file','noncheckpoint_file'])
def test_missing_and_extra_checkpoints_refused(complete,change):
    output,receipts,result=complete
    if change=='omitted_receipt':receipts.pop(1)
    if change=='deleted_file':Path(receipts[1]['path']).unlink()
    if change=='unindexed_file':np.savez_compressed(output/'checkpoint_04_extra.npz',x=[1])
    if change=='noncheckpoint_file':np.savez_compressed(output/'extra.npz',x=[1])
    with pytest.raises(ValueError):driver.compact_fit(result,output,receipts)


def test_baseline_and_returned_must_be_in_complete_index(complete):
    output,receipts,result=complete
    result['returned']['array_sha256']='0'*64
    with pytest.raises(ValueError,match='outside the checkpoint index'):driver.compact_fit(result,output,receipts)


def test_memory_fields_cannot_disagree_with_the_bound_npz(complete):
    output,receipts,result=complete;result['returned']['fields']['preserved'].flat[1]=False
    with pytest.raises(ValueError,match='Saved field'):driver.compact_fit(result,output,receipts)


def test_explicit_minimum_same_part_repair_requires_an_actual_new_relationship(complete):
    _,_,result=complete;result['accepted_relationship_repair']=True
    with pytest.raises(ValueError,match='Repair credit'):driver.same_part_receipt(result)


def test_one_new_relationship_has_local_credit_and_no_mapping_pass(complete):
    _,_,result=complete
    result['returned']['fields']['preserved'].flat[0]=True
    result['returned']['fields']['definite_lost'].flat[0]=False
    result['accepted_relationship_repair']=True;result['repaired_relationship_count']=1
    result['status']='IMPROVED_RELATIONSHIPS_DIAGNOSTIC'
    receipt=driver.same_part_receipt(result)
    assert receipt['local_relationship_repair_admitted'] and receipt['newly_preserved_relationships']==1
    assert receipt['policy']['minimum_newly_preserved_relationships']==1
    assert receipt['accepted_mapping'] is False and receipt['continuous_orientation_coverage']=='NOT_PROVED'


@pytest.mark.parametrize('failure',['lost','new_definite','orientation','tip','domain'])
def test_no_aggregate_tradeoff_can_admit_a_repair(complete,failure):
    _,_,result=complete;b=result['returned']
    b['fields']['preserved'].flat[0]=True;b['fields']['definite_lost'].flat[0]=False
    result['accepted_relationship_repair']=True;result['repaired_relationship_count']=1
    if failure=='lost':b['fields']['preserved'].flat[1]=False
    if failure=='new_definite':b['fields']['definite_added'].flat[1]=True
    if failure=='orientation':b['orientation']['valid'][0]=False
    if failure=='tip':b['state'].tip_errors_world_m[0,0]=.002
    if failure=='domain':b['state'].engineering_domain_pass=False
    with pytest.raises(ValueError):driver.same_part_receipt(result)


def test_refused_baseline_is_evidence_and_receives_no_local_admission(complete):
    _,_,result=complete;result['status']='REFUSED_NO_PROTECTED_START'
    result['returned']['state'].tip_errors_world_m[0,0]=.002
    receipt=driver.same_part_receipt(result)
    assert not receipt['literal_original_tips_pass'] and not receipt['local_relationship_repair_admitted']


@pytest.mark.parametrize('key',['accepted_mapping','full_strict_mapping_accepted','native_fidelity_accepted','production_promotion'])
def test_no_nested_mapping_pass_capability(complete,key):
    output,receipts,result=complete;result['rounds']=[{key:True}]
    with pytest.raises(ValueError,match='cannot issue mapping acceptance'):driver.compact_fit(result,output,receipts)


def test_hard_checkpoint_cap_and_contiguous_sequence(complete):
    output,receipts,_=complete
    for i in range(4,8):
        path=output/f'checkpoint_{i:02d}_extra.npz';np.savez_compressed(path,x=[i])
        receipts.append(dict(path=str(path),sha256=driver.sha(path),bytes=path.stat().st_size,label='extra'))
    with pytest.raises(ValueError,match='cap'):driver.checkpoint_index(output,receipts)


def test_runtime_contract_matches_core_constants_without_importing_or_solving():
    core=SimpleNamespace(MAX_ROUNDS=3,MAX_FULL_CHECKPOINTS=6,RESTORE_MAX_ITERATIONS=8,RESTORE_MAX_CALLBACKS=400,
        RESTORE_ACTIVATION_MARGIN_M=1e-7,RESTORE_INWARD_M=1e-9,MAX_LOCAL_MODEL_ROWS=512,SAME_PART_ADMISSION=driver.SAME_PART_ADMISSION.copy())
    driver.validate_core_contract(core)
    core.MAX_ROUNDS=2
    with pytest.raises(ValueError):driver.validate_core_contract(core)
    core.MAX_ROUNDS=3;core.MAX_LOCAL_MODEL_ROWS=513
    with pytest.raises(ValueError):driver.validate_core_contract(core)
    assert driver.BUDGET['maximum_local_model_rows']==512
    assert driver.BUDGET['maximum_full_field_checkpoints_per_case']==6
    assert driver.BUDGET['maximum_exchange_rounds_per_case']==3
    assert driver.BUDGET['maximum_seconds_per_case']==600
    assert driver.BUDGET['maximum_static_forwards_per_case']==driver.BUDGET['maximum_raw_callbacks_per_case']==1500
    assert driver.RESTORATION==dict(maximum_iterations=8,maximum_callbacks=400,activation_margin_m=1e-7)
    assert all(driver.BUDGET[k]==0 for k in ('physics_steps','native_inference_calls','official_dex_retarget_calls','task_outcomes','predictor_fits'))


def test_partial_failure_inventory_never_claims_verified_completion(complete):
    output,receipts,_=complete
    Path(receipts[1]['path']).write_bytes(b'partial or changed')
    inventory=driver.failure_checkpoint_inventory(output)
    assert len(inventory)==3 and not any(r['verified_write_time'] for r in inventory)
    assert inventory[1]['sha256']!=receipts[1]['sha256']


def test_actual_core_constants_agree_without_native_geometry_or_solve():
    path=Path(__file__).with_name('exchange_core.py')
    spec=importlib.util.spec_from_file_location('v3_core_contract_only',path)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    driver.validate_core_contract(core)


@pytest.mark.parametrize('broken',['forwards','raw','checkpoint','physics'])
def test_completion_budget_is_checked_against_write_time_inventory(complete,broken):
    _,receipts,_=complete
    counts=dict(static_forwards=1500,physics_step_attempts=0,inference_attempts=0,official_dex_retarget_attempts=0)
    callbacks=dict(raw_callbacks=1500,full_field_checkpoints=3)
    driver.validate_completion_budget(counts,callbacks,receipts)
    if broken=='forwards':counts['static_forwards']=1501
    if broken=='raw':callbacks['raw_callbacks']=1501
    if broken=='checkpoint':callbacks['full_field_checkpoints']=2
    if broken=='physics':counts['physics_step_attempts']=1
    with pytest.raises(ValueError):driver.validate_completion_budget(counts,callbacks,receipts)


def manufactured_visual(output):
    spec=importlib.util.spec_from_file_location('v3_synthetic_visual_fixture',Path(__file__).with_name('test_visual_transform_check.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    *_,binding=fixture.run(output/'visual_transform_check')
    return binding


def test_exact_source_bound_visual_operands_are_separately_indexed(complete):
    output,receipts,result=complete;binding=manufactured_visual(output)
    compact=driver.compact_fit(result,output,receipts,binding,20)
    assert len(compact['checkpoint_artifacts'])==3
    assert [r['kind'] for r in compact['auxiliary_artifacts']]==['visual_transform_receipt','visual_transform_operands']
    assert compact['auxiliary_artifacts'][1]['sha256']==binding['operands_and_outputs']['sha256']
    with pytest.raises(ValueError):driver.checkpoint_index(output,receipts)


@pytest.mark.parametrize('target',['json','npz'])
def test_allowed_visual_receipt_and_operand_tampering_refused(complete,target):
    output,receipts,result=complete;binding=manufactured_visual(output)
    file=Path(binding['path'] if target=='json' else binding['operands_and_outputs']['path'])
    with file.open('ab') as handle:handle.write(b'tampered')
    with pytest.raises(RuntimeError,match='bytes changed'):driver.compact_fit(result,output,receipts,binding,20)


def test_visual_binding_does_not_authorize_arbitrary_extra_npz(complete):
    output,receipts,result=complete;binding=manufactured_visual(output)
    np.savez_compressed(output/'visual_transform_check/unbound.npz',x=[0])
    with pytest.raises(ValueError,match='unindexed'):driver.compact_fit(result,output,receipts,binding,20)
    with pytest.raises(ValueError,match='exact per-case'):driver.visual_artifact_index(output,binding,28)
