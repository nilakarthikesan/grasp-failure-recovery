import copy
import json
from pathlib import Path
import numpy as np
import pytest
import audit_raw_surfaces_v1 as r

def ledger():
 return {'sample_count':16,'records':[{'global_sample_index':g,'native_output_complete_and_finite':True,'source_limit_violations':[],'target_limit_violations':[] if g==6 else [{'joint':'THJ4'}]} for g in range(16)]}
def test_selector_never_replaces_first_two_or_filters_target_illegality():
 l=ledger();l['records'][2]['source_limit_violations']=[{'joint':'THJ4'}]
 assert r.select(l)==[0,1,3,4,5,6,7,8,9,10,11,12,13,14,15]
def test_selector_refuses_omitted_native_rows():
 l=ledger();l['records'].pop()
 with pytest.raises(ValueError):r.select(l)
def test_selector_refuses_coerced_integrity():
 l=ledger();l['records'][0]['native_output_complete_and_finite']=1
 with pytest.raises(ValueError):r.select(l)
def test_frozen_saved_poses_identical_no_angle_clipping():
 p=json.loads((r.HERE/'policy_frozen_v1.json').read_text());r.verify(p);a,records,_=r.build(p)
 with np.load(p['poses'],allow_pickle=False) as z:
  assert set(a)==set(z.files)
  for k in a:assert a[k].dtype==z[k].dtype and a[k].tobytes()==z[k].tobytes()
 assert len(records)==16 and len(a['global_sample_indices'])==11
 assert np.array_equal(a['target_joint_values'],a['source_hand_qpos'][:,9:])
 assert np.array_equal(a['object_from_source_palm'],a['object_from_target_palm'])
 assert [i for i in a['global_sample_indices'] if not records[i]['original_target_limit_violations']]==[6]
 assert all(records[i]['execution_eligible'] is False for i in range(16))
 assert records[0]['original_selected_for_bounded_mapping'] and records[1]['original_selected_for_bounded_mapping']
