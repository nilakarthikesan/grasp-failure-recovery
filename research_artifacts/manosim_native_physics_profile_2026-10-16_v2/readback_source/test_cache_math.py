"""Manufactured BVH controls only; no model/assets or engine imports."""
from pathlib import Path
import importlib.util,copy
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('independent_cache_math',Path(__file__).with_name('cache_math.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)

def manufactured():
 # Two world planes, one horizontal and one90degreesabouty. Local finite
 # plane box is the pinned compiler's [0,0,-M/2,M,M,M/2].
 M=1e10;q=np.sqrt(.5);a=dict(geom_pos=np.array([[0.,0,.1],[.2,0,.15]]),geom_quat=np.array([[1.,0,0,0],[q,0,q,0]]),geom_sameframe=np.zeros(2,int),geom_aabb=np.array([[0.,0,-M/2,M,M,M/2]]*2),geom_bodyid=np.zeros(2,int),geom_contype=np.array([1,2]),geom_conaffinity=np.array([1,2]),body_bvhadr=np.array([0]),body_bvhnum=np.array([3]),body_ipos=np.zeros((1,3)),body_iquat=np.array([[1.,0,0,0]]),bvh_child=np.array([[1,2],[-1,-1],[-1,-1]]),bvh_nodeid=np.array([-1,0,1]))
 leaves=np.array([[0.,0,.1-M/2,M,M,M/2],[.2-M/2,0,.15,M/2,M,M]]);lo=np.minimum(leaves[0,:3]-leaves[0,3:],leaves[1,:3]-leaves[1,3:]);hi=np.maximum(leaves[0,:3]+leaves[0,3:],leaves[1,:3]+leaves[1,3:]);a['bvh_aabb']=np.vstack((np.r_[(lo+hi)/2,(hi-lo)/2],leaves));return a

def test_two_authored_plane_boxes_and_inventory():
 r=c.world_bvh_readback(manufactured());assert r['leaf_geom_ids']==[0,1];assert len(r['nodes'])==3
@pytest.mark.parametrize('fault',['stale_minus100','wrong_leaf','bad_child','nonfinite','world_frame'])
def test_cache_refuses_stale_or_malformed_compiler_witness(fault):
 a=manufactured()
 if fault=='stale_minus100':a['bvh_aabb'][1,2]-=100
 elif fault=='wrong_leaf':a['bvh_nodeid'][2]=0
 elif fault=='bad_child':a['bvh_child'][0,1]=0
 elif fault=='nonfinite':a['bvh_aabb'][0,0]=np.inf
 else:a['body_ipos'][0,0]=.01
 with pytest.raises(ValueError):c.world_bvh_readback(a)
def test_no_hot_geometry_mutation_is_exact_not_tolerance():
 a=manufactured();b=copy.deepcopy(a);assert c.exact_unchanged(a,b)['all_bytes_unchanged']
 b['geom_pos'][0,0]=1e-15
 with pytest.raises(ValueError):c.exact_unchanged(a,b)
