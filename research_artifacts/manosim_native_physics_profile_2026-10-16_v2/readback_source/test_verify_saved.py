"""Manufactured v2 reader controls; no experiment or engine operations."""
from pathlib import Path
import importlib.util
import numpy as np
s=importlib.util.spec_from_file_location('own_independent_profile_v2',Path(__file__).with_name('verify_saved.py'));v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
def test_new_plan_and_old_math_protocol_remain_distinct():
 assert len(v.plan())==12 and sum(x['steps'] for x in v.plan())==11250
 assert sum(x['steps'] for x in v.v1.plan())==9750
 assert v.v1.BUDGET['steps']==9750 and v.BUDGET['steps']==11250

def test_approach_one_second_motion_then_half_second_settle():
 f=next(x for x in v.plan() if x['kind']=='guarded_sphere_table_approach' and x['dt_s']==.001);goal=np.r_[np.zeros(3),np.tile([1.,0,0,0],16)];start=goal.copy();start[0]=-.06;q=v.v1.refs(f,start,goal,np.zeros(3))
 np.testing.assert_array_equal(q[1000:],np.broadcast_to(goal,(501,67)))
 assert q.shape==(1501,67) and q[500,0]==-.03

def test_source_math_hashes_and_current_normalization_are_fixed():
 assert v.digest(v.HERE/'v1_saved_math.py')=='6ad02f6f671f728b7192daf23d9dc0838783b11cccdf0efcde01ad027a734688'
 assert v.numpy_scalar(np.bool_(False)) is False
