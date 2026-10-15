"""Manufactured declaration/receipt controls; no model, solve, or physics."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH=Path(__file__).with_name('run_repair.py')
SPEC=importlib.util.spec_from_file_location('full_field_repair_harness',PATH)
driver=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(driver)


def good_arithmetic():
    return dict(status='COMPLETED_BOUNDED_OPERAND_DIAGNOSTIC',failure=None,
        poses_completed=[[20,'original_20'],[20,'experimental_20'],[28,'original_28'],[28,'experimental_28']],
        poses_not_run_or_partial=[],actual=dict(inside_mask_disagreements=0,hull_predicate_disagreements=0,
            einsum_warnings=[],max_matmul_einsum_difference_m=5e-17,max_einsum_scalar_difference_m=5e-17,
            max_matmul_scalar_difference_m=5e-17))


def test_arithmetic_clearance_is_complete_pose_and_predicate_bound():
    driver.numerical_clearance(good_arithmetic())
    for key,value in [('failure',{'message':'partial'}),('status','HARD_STOP'),('poses_completed',[[20,'original_20']]),
                      ('poses_not_run_or_partial',[[28,'experimental_28']])]:
        bad=good_arithmetic();bad[key]=value
        with pytest.raises(ValueError):driver.numerical_clearance(bad)
    for key,value in [('inside_mask_disagreements',1),('hull_predicate_disagreements',1),('einsum_warnings',['invalid'])]:
        bad=good_arithmetic();bad['actual'][key]=value
        with pytest.raises(ValueError):driver.numerical_clearance(bad)


@pytest.mark.parametrize('value',[np.nan,np.inf,True,-1.,1e-10,None])
def test_arithmetic_comparison_cannot_be_disabled_or_nonfinite(value):
    bad=good_arithmetic();bad['actual']['max_matmul_einsum_difference_m']=value
    with pytest.raises(ValueError):driver.numerical_clearance(bad)


def test_saved_array_compaction_keeps_exact_bound_receipt_and_counts(tmp_path):
    path=tmp_path/'fields.npz';mask=np.array([True,False,True])
    np.savez_compressed(path,preserved=mask)
    full=dict(fields=dict(preserved=mask),margin_m=np.array([.001]),distance_m=np.array([.002]),
              array_path=str(path),array_sha256=driver.sha(path),summary=dict(full_strict_mapping_status='FAIL'))
    result=dict(baseline=full,returned=full,accepted=False)
    compact=driver.compact_fit(result)
    for key in ('baseline','returned'):
        assert 'fields' not in compact[key] and 'margin_m' not in compact[key] and 'distance_m' not in compact[key]
        assert compact[key]['array_sha256']==driver.sha(path)
        assert compact[key]['field_boolean_counts']==dict(preserved=2)
        assert compact[key]['summary']['full_strict_mapping_status']=='FAIL'
    assert 'fields' in result['baseline']
    path.write_bytes(b'tampered arrays')
    with pytest.raises(ValueError,match='arrays changed'):driver.compact_fit(result)


def test_fixed_budget_and_local_search_buffers_never_add_acceptance_allowance():
    assert driver.BUDGET['physics_steps']==driver.BUDGET['native_inference_calls']==driver.BUDGET['predictor_fits']==0
    assert driver.BUDGET['maximum_static_forwards_per_case']==1500
    assert driver.BUDGET['maximum_full_field_checkpoints_per_case']==5
    assert driver.BUDGET['maximum_seconds_per_case']==600
    assert driver.LOCAL_CONFIG['tip_inward_target_m']==1e-9
