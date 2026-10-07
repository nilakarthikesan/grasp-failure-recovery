import numpy as np
import pytest
from grasp_failure_prediction.integrations.hug_preprocessing import seeded_point_cloud,tensor_content_hash

def generate(seed,count=8):
    xyz=np.arange(300,dtype=np.float32).reshape(100,3)
    colors=np.arange(300).reshape(100,3)%256
    def project(*args,**kwargs):return xyz,colors
    def sample(x,c,n,rng):
        idx=rng.choice(len(x),size=n,replace=len(x)<n);return x[idx],c[idx]
    return seeded_point_cloud(None,None,None,seed=seed,n_points=count,backproject=project,sample=sample)

def test_request_local_seed_ignores_global_rng_and_call_order():
    a=generate(42);generate(99);np.random.seed(918);np.random.rand(100)
    b=generate(42)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert not np.array_equal(a[0],generate(43)[0])

def test_resampling_retains_color_alignment():
    x,c=generate(42,150)
    assert x.shape==(150,3)
    np.testing.assert_allclose(c,(x.astype(int)%256)/255,atol=1e-7)

@pytest.mark.parametrize('seed',[-1,True,1.2])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):generate(seed)

def test_empty_crop_rejected():
    with pytest.raises(ValueError):
        seeded_point_cloud(None,None,None,seed=42,backproject=lambda *a,**k:([],[]),sample=None)

def test_content_hash_stable_and_sensitive():
    a={'xyz':generate(42)[0]}
    assert tensor_content_hash(a)==tensor_content_hash({'xyz':a['xyz'].copy()})
    assert tensor_content_hash(a)!=tensor_content_hash({'xyz':generate(43)[0]})
