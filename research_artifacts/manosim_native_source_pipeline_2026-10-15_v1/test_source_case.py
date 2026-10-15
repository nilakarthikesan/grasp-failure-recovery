"""Tiny manufactured source-graph controls; no MANO/model/native operation."""
import importlib.util,types
from pathlib import Path
import numpy as np
import pytest
import torch

HERE=Path(__file__).parent
def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
c=module('source_graph_coord_controls','source_coordinates.py')
s=module('source_graph_case_controls','source_case.py')

def inputs():
    aa=np.linspace(.01,.15,48,dtype=np.float32);t=np.array([1.,.03125,-.125],dtype=np.float32)
    camera=np.eye(4);camera[:3,3]=[.3,-.2,.1];rest=np.tile([1.,0.,0.,0.],(16,1))
    q,_=c.source_initialized_qpos(aa,t,camera,rest)
    return aa,t,camera,rest,q

class ManufacturedLayer:
    def __init__(self):self.inputs=[]
    def __call__(self,aa,beta):
        assert aa.dtype==beta.dtype==torch.float32
        self.inputs.append((aa.clone(),beta.clone()))
        v=torch.tensor([[[2**24,0.,0.],[0.,1.,0.],[0.,0.,1.]]],dtype=torch.float32)
        v[0,1,0]=aa[0,4]
        tr=torch.eye(4,dtype=torch.float32).repeat(1,16,1,1)
        return types.SimpleNamespace(verts=v,transforms_abs=tr)

def test_exact_source_forward_receives_actualAA32_and_adds_t_before_camera64():
    aa,t,camera,rest,q=inputs();layer=ManufacturedLayer();calls=[];beta=torch.zeros((1,10),dtype=torch.float32)
    value=s.evaluate_pinned_source(layer,q,rest,camera,aa,beta,c,torch,lambda:calls.append('before'))
    assert calls==['before'] and len(layer.inputs)==1
    assert layer.inputs[0][0].numpy().tobytes()==aa[None].tobytes()
    assert value['translated_camera_vertices32'].dtype==np.float32
    assert value['vertices'][0,0]==float(np.float32(2**24)+np.float32(1.))+.3
    assert value['vertices'][0,0]!=(float(2**24)+1.)+.3
    np.testing.assert_array_equal(value['joint_anchors_world'],np.broadcast_to(t.astype(float)+camera[:3,3],(16,3)))

def test_actual_changed_joint_is_recovered_instead_of_cached_source_payload():
    aa,t,camera,rest,q=inputs();layer=ManufacturedLayer();beta=torch.zeros((1,10),dtype=torch.float32)
    first=s.evaluate_pinned_source(layer,q,rest,camera,aa,beta,c,torch,lambda:None)
    changed=q.copy();changed[7:11]=c.multiply(changed[7:11],c.exp_quaternion(np.array([0.,.02,0.])))
    second=s.evaluate_pinned_source(layer,changed,rest,camera,aa,beta,c,torch,lambda:None)
    assert not np.array_equal(layer.inputs[0][0].numpy(),layer.inputs[1][0].numpy())
    assert not np.array_equal(first['vertices'],second['vertices'])
    np.testing.assert_array_equal(q,inputs()[-1])

def test_before_source_quota_refusal_prevents_forward():
    aa,t,cam,rest,q=inputs();layer=ManufacturedLayer()
    def exhausted():raise RuntimeError('quota')
    with pytest.raises(RuntimeError,match='quota'):s.evaluate_pinned_source(layer,q,rest,cam,aa,torch.zeros((1,10)),c,torch,exhausted)
    assert layer.inputs==[]

@pytest.mark.parametrize('mutation',['delta','id','count'])
def test_exact_original_finite_grid_contract_refuses_modified109(monkeypatch,mutation):
    grid={'poses':[{'pose_id':str(i),'delta51':[0.]*51} for i in range(109)]}
    monkeypatch.setattr(s,'read',lambda path:{'motion_grid':grid})
    import copy
    changed=copy.deepcopy(grid)
    if mutation=='delta':changed['poses'][4]['delta51'][6]=.123
    elif mutation=='id':changed['poses'][4]['pose_id']='replacement'
    else:changed['poses'].pop()
    assert s.require_motion_grid(grid)
    with pytest.raises(ValueError,match='grid'):s.require_motion_grid(changed)

def test_q0_byte_guard_precedes_all_source_model_work(monkeypatch,tmp_path):
    aa,t,cam,rest,q=inputs();shape=np.zeros((1,10),dtype=np.float32)
    decoded=tmp_path/'decoded.npz';np.savez(decoded,actual_full_poses=aa,params_translation_m=t,camera_world=cam,shape=shape)
    old=tmp_path/'old.npz';np.savez(old,manufactured=np.array([0.]))
    shape_path=tmp_path/'shape.npy';np.save(shape_path,shape)
    def row(path):return {'path':str(path),'sha256':s.sha(path),'bytes':path.stat().st_size}
    grid={'poses':[{'pose_id':str(i),'delta51':[0.]*51} for i in range(109)]}
    monkeypatch.setattr(s,'require_motion_grid',lambda value:True)
    bad=types.SimpleNamespace(**{name:getattr(c,name) for name in dir(c) if not name.startswith('__')})
    def wrong(*args):
        result,receipt=c.source_initialized_qpos(*args);result[3:7]=c.multiply(result[3:7],c.exp_quaternion(np.array([1e-5,0.,0.])));return result,receipt
    bad.source_initialized_qpos=wrong
    construction=[]
    def forbidden(*args):construction.append('model/source');raise AssertionError('Native work prohibited')
    fake_bridge=types.SimpleNamespace(scene_xml=forbidden)
    monkeypatch.setattr(s,'load',lambda name,path:bad if path.name=='source_coordinates.py' else fake_bridge)
    spec={'decoded_reference':row(decoded),'original_compiled_readback':row(old),'saved_rest_quaternions':rest.tolist(),'shape_asset':row(shape_path),'expected_shape_sha256':s.sha(shape_path)}
    protocol={'budget':s.BUDGET,'gates':s.GATES,'state_domain':c.DOMAIN,'previous_GS64_states_and_profiles_reused_as_new_targets':False,'motion_grid':grid}
    with pytest.raises(ValueError,match='byte identity'):s.run_case(spec,tmp_path,protocol,lambda event:None)
    assert construction==[] and not list(tmp_path.glob('source_initialized*'))
