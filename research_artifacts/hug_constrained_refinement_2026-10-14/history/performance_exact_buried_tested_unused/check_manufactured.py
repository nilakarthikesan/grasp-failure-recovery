"""Replay all old manufactured invalidity checks against both helper versions."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
import mujoco,numba

HERE=Path(__file__).parent
SOURCE=Path('/private/tmp/hug_constrained_validation_2026-10-14/test_exact_direction.py')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
old=load('buried_parity_original',SOURCE.parent/'exact_direction.py')
new=load('buried_parity_accelerated',HERE/'exact_direction_buried.py')
calls={'actual':0}
def blocked(*args,**kwargs):calls['actual']+=1;raise RuntimeError('Static manufactured check forbids integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,blocked)
numba.set_num_threads(2)
def equal(x,y):
    if isinstance(x,np.ndarray):return isinstance(y,np.ndarray) and x.dtype==y.dtype and np.array_equal(x,y,equal_nan=True)
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    return x==y

def main():
    recorded=[]
    def paired(*args,**kwargs):
        original=old.mapped_direction_fields(*args,**kwargs)
        accelerated=new.mapped_direction_fields(*args,**kwargs)
        assert equal(original,accelerated)
        recorded.append({'call':len(recorded)+1,'query_count':len(args[0]),
                         'all_fields_masks_scores_and_metadata_exact':True})
        return original
    # Execute unchanged mathematical fixture logic while replacing only the
    # module import/call dispatch and redirecting its report into this NEW dir.
    proxy=type(sys)('exact_direction')
    proxy.__dict__.update(old.__dict__)
    proxy.mapped_direction_fields=paired
    sys.modules['exact_direction']=proxy
    source=SOURCE.read_text().replace("hashlib.sha256((Path(__file__).parent/'exact_direction.py').read_bytes()).hexdigest()",
                                     "hashlib.sha256(SOURCE.parent.joinpath('exact_direction.py').read_bytes()).hexdigest()")
    scope={'__name__':'__manufactured_buried_parity__','__file__':str(HERE/'check_manufactured.py'),'SOURCE':SOURCE}
    exec(compile(source,str(SOURCE),'exec'),scope)
    assert calls['actual']==0
    # An explicit mesh-in-mesh case exercises the new inside/subset path; the
    # old buried fixture uses boxes and is intentionally left literal.
    geom=scope['geom'];far=scope['far'];p=scope['p'];on=scope['on']
    inner=geom(mujoco.mjtGeom.mjGEOM_MESH,[.002,.002,.002],gid=41)
    outer=geom(mujoco.mjtGeom.mjGEOM_MESH,[.006,.006,.006],gid=42)
    result=paired(np.array([[0.,0.,.003]]),on,[inner,outer],far,p,0)
    assert result['reason_masks']['buried_closest_point'][0]
    points=np.array([[0,0,0],[.006,0,0],[.006+1e-10,0,0],[.006+2e-10,0,0],[.03,.04,.05]],float)
    for geometry in [inner,outer]:
        signed,_,_=old.a.geom_query(geometry,points)
        assert np.array_equal(signed< -p['tie_distance_tolerance_m'],new.exact_buried_membership(geometry,points,p['tie_distance_tolerance_m']))
    report={'schema':'hug_exact_buried_manufactured_parity_v1','old_script_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'original_helper_sha256':hashlib.sha256((SOURCE.parent/'exact_direction.py').read_bytes()).hexdigest(),
            'new_helper_sha256':hashlib.sha256((HERE/'exact_direction_buried.py').read_bytes()).hexdigest(),
            'old_invalidity_case_count':13,'old_full_fixture_call_count':len(recorded)-1,'additional_mesh_buried_case':True,
            'mixed_interior_boundary_exterior_predicate_parity':True,'all_calls':recorded,
            'work_counts':new.BURIAL_AUDIT.copy(),'physics_steps':0,'actual_integration_attempts':calls['actual'],
            'new_inference':False,'new_eval_data':False}
    (HERE/'manufactured_buried_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'full_fixture_calls':len(recorded),'all_exact':True,'mesh_predicate_exact':True}))
if __name__=='__main__':main()
