#!/usr/bin/env python
"""Run pinned official HUG GraspDataset preprocessing, without model weights.
The upstream directory must contain unmodified dataloader/ and utils/ modules.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import torch
from PIL import Image
import io
import pickle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-root',type=Path,required=True)
    p.add_argument('--sample',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preprocessing-seed',type=int,default=42)
    args=p.parse_args()
    root=args.official_root.resolve()
    # Load just the official preprocessing package, not HUG's model/app imports.
    pkg=types.ModuleType('hug_preflight_official');pkg.__path__=[str(root)]
    sys.modules[pkg.__name__]=pkg
    import importlib
    loader=importlib.import_module('hug_preflight_official.dataloader.grasp_dataset')
    pcl=importlib.import_module('hug_preflight_official.utils.pcl_utils')
    from grasp_failure_prediction.integrations.hug_preprocessing import seeded_point_cloud, tensor_content_hash
    class SeededDataset(loader.GraspDataset):
        def _build_pcl(self, depth_m, rgb_np, K, point_xyz=None):
            xyz, colors = seeded_point_cloud(
                depth_m.detach().cpu().numpy(), rgb_np, K,
                seed=args.preprocessing_seed, backproject=pcl.backproject_to_pcl,
                sample=pcl.sample_fixed_n, n_points=self.n_points_input,
                center=point_xyz,
                crop_radius=self.pcl_crop_radius if point_xyz is not None else None)
            return torch.from_numpy(xyz), torch.from_numpy(colors)
    # The sample is a trusted, pinned upstream artifact, not an arbitrary upload.
    data=pickle.loads(args.sample.read_bytes())
    assert data.get('grasp') is None, 'Use an observation-only sample'
    if data.get('condition_point') is None:
        raise ValueError('Explicit condition_point required; do not randomly choose the object pixel')
    with tempfile.TemporaryDirectory(prefix='hug-preprocess-') as tmp:
        Path(tmp,'sample.pkl').write_bytes(args.sample.read_bytes())
        dataset=SeededDataset(tmp,split='eval',n_points_input=4096)
        batch=dataset[0]
        repeated=dataset[0]
        assert all(torch.equal(batch[k],repeated[k]) for k in batch if torch.is_tensor(batch[k]))
    expected={'rgb':(3,224,224),'point_uv':(3,),'camera_K':(3,3),
              'pcl_xyz':(4096,3),'pcl_rgb':(4096,3)}
    for key,shape in expected.items():
        assert tuple(batch[key].shape)==shape,(key,batch[key].shape)
        assert batch[key].dtype==torch.float32,key
        assert torch.isfinite(batch[key]).all(),key
    assert torch.any(batch['pcl_xyz']!=0),'Empty point cloud silently padded by upstream'
    assert torch.all((batch['pcl_rgb']>=0)&(batch['pcl_rgb']<=1))
    raw_rgb=loader.GraspDataset._decode_image(data['image'])
    mm=loader.GraspDataset._decode_depth_uint16(data['depth'])
    depth=dataset._depth_meters(data['depth'])
    K=np.asarray(data['camera']['K'])
    u,v,d=map(float,batch['point_uv'])
    assert np.isclose(d,mm[round(v),round(u)]/1000)
    center=pcl.pixel_to_xyz(u,v,d,K)
    # Independently confirm projected pixel geometry.
    expected_center=d*np.linalg.inv(K)@np.array([u,v,1.])
    np.testing.assert_allclose(center,expected_center,atol=1e-6)
    eligible,_=pcl.backproject_to_pcl(depth.numpy(),raw_rgb,K,center=center,crop_radius=.3)
    assert len(eligible)>0
    distances=np.linalg.norm(batch['pcl_xyz'].numpy()-center,axis=1)
    assert np.all(distances<.30001)
    mask=loader.GraspDataset._decode_mask(data['object_mask'])
    assert mask[round(v),round(u)]>0
    report={'official_preprocessing_passed':True,'model_loaded':False,
        'inference_tested':False,'simulation_tested':False,
        'sample_sha256':hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        'official_source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in root.rglob('*.py') if f.name!='__init__.py'},
        'tensor_shapes':{k:list(batch[k].shape) for k in expected},
        'selected_point_camera_m':center.tolist(),'eligible_cloud_points':len(eligible),
        'max_crop_distance_m':float(distances.max()),
        'preprocessing_seed':args.preprocessing_seed,'rng_algorithm':'PCG64',
        'numpy_version':np.__version__,'torch_version':torch.__version__,
        'preprocessing_adapter':'seeded_point_cloud_v1',
        'adapter_sha256':hashlib.sha256(Path(sys.modules[seeded_point_cloud.__module__].__file__).read_bytes()).hexdigest(),
        'repeat_identical':True,'tensor_content_hash':tensor_content_hash({k:batch[k] for k in expected}),
        'notes':['Calls official GraspDataset.__getitem__ with a narrow _build_pcl override that supplies an explicit RNG to the official sampler; no weights or MANO assets loaded.',
                 'Inference-specific get_inference_data and model execution remain untested.',
                 'Preprocessing and later model-generation seeds are separate. Repeatability verified here on CPU; model/MPS reproducibility is not established.']}
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save({k:batch[k] for k in expected},args.output/'preprocessed_tensors.pt')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
