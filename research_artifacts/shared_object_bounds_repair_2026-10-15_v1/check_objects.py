"""Static bounds check on ten existing objects; no episodes or native generation."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,warnings
import numpy as np
import mujoco
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.extend(['/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/hug-deps','/Users/nilakarthikesan/grasp-failure-prediction/runs/hug_integration_check/deps'])
def denied(*args,**kwargs):raise AssertionError('Physics forbidden in static bounds verification')
for name in ('mj_step','mj_step1','mj_step2','mj_integratePos'):setattr(mujoco,name,denied)
warnings.simplefilter('error',RuntimeWarning)
parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
source=Path(args.source);out=Path(args.output);assert not out.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
name='grasp_failure_prediction.evaluation._static_bounds_repair_verification'
spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
reference=Path('/private/tmp/unidex_mapping_robustness_20261015_v1/shared_loader_fix/comparison_v2.json')
original=json.loads(reference.read_text());rows=[]
for r in original['rows']:
    path=Path(r['geometry_path']);expected_hash=r['candidate']['geometry_content_hash']
    row={'case_id':r['case_id'],'path':str(path),'xml_sha256':sha(path),'expected_content_hash':expected_hash}
    try:
        geometry=module.load_object_geometry(path,expected_content_hash=expected_hash)
        m=mujoco.MjModel.from_xml_path(str(path));points=[]
        for g in range(m.ngeom):
            if not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
            mid=int(m.geom_dataid[g]);a=int(m.mesh_vertadr[mid]);b=a+int(m.mesh_vertnum[mid]);R=np.empty(9)
            mujoco.mju_quat2Mat(R,m.geom_quat[g]);R=R.reshape(3,3)
            for vertex in m.mesh_vert[a:b]:
                points.append([sum(float(R[i,j])*float(vertex[j]) for j in range(3))+float(m.geom_pos[g,i]) for i in range(3)])
        expected=np.array([np.min(points,axis=0),np.max(points,axis=0)])
        bounds=np.asarray(geometry.local_collision_bounds_m)
        residual=float(np.max(np.abs(bounds-expected)))
        assert residual<=1e-15
        row.update(status='PASS',max_scalar_bounds_error_m=residual,bounds_m=bounds.tolist(),files_sha256=geometry.files_sha256)
    except (RuntimeWarning,FloatingPointError) as e:row.update(status='FAIL',error_type=type(e).__name__,message=str(e))
    rows.append(row)
result={'source_path':str(source),'source_sha256':sha(source),'reference_report_sha256':sha(reference),'scope':'Static object compilation and scalar bounds oracle; geometry identities verified','runtime':{'numpy':np.__version__,'mujoco':mujoco.__version__},'rows':rows,'passed':sum(r['status']=='PASS' for r in rows),'physics_steps':0,'new_native_inference':False,'new_eval_data':False}
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
