"""Baseline-only source/geometry parity and timing. Never invokes an optimizer."""
import argparse,json,time
from pathlib import Path
import numpy as np
from constraint_engine import Case,plain,sha,direction,SAVED,PHYSICS_CALLS
from free_palm_kinematics import verify_jacobian

def main():
    p=argparse.ArgumentParser();p.add_argument('--case-index',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    records=[]
    for i in args.case_index:
        t=time.perf_counter();c=Case(i);loadseconds=time.perf_counter()-t;x=np.zeros(27)
        t=time.perf_counter();raw=c.raw(x);first_raw=time.perf_counter()-t
        t=time.perf_counter();repeat=c.raw(x);warm_raw=time.perf_counter()-t
        assert np.array_equal(raw['surface_margin'],repeat['surface_margin'])
        baseline_d=np.asarray([r['row']['mapped_centroid_distance_m'] for r in c.prox_rows])
        difference=float(np.max(np.abs(raw['part_distance_m'][:len(c.prox_rows)]-baseline_d),initial=0))
        assert difference<1e-12
        t=time.perf_counter();exact=c.exact(x);exactseconds=time.perf_counter()-t
        assert exact['in_domain'] and exact['tip_status']['passed']
        assert np.array_equal(raw['tip_status']['norms_m'],c.caps)
        native_report=None
        if len(c.oq):
            nf=direction.native_direction_fields(c.oq,c.on,c.native_tri,c.native_labels,c.native_normals,c.visual_tri,c.orientation_protocol)
            scoreerr=float(np.max(np.abs(nf['score']-c.oscore)));assert scoreerr<1e-12
            assert nf['valid'].all() and np.array_equal(nf['part_index'],c.op)
            native_report={'source_score_maxabs':scoreerr,'all_source_rows_valid':True,'source_part_matches':True}
        j=verify_jacobian(x,c.bounds,lambda y:c.raw(y)['task_error_vectors_m'][:5],c.tip_jacobian,step=1e-4,tolerance_m_per_state=1e-8)
        records.append({'index':i,'native_sha256':c.native_hash,'load_seconds':loadseconds,'first_raw_seconds':first_raw,'warm_raw_seconds':warm_raw,'exact_seconds':exactseconds,'baseline_distance_cache_maxabs_error_m':difference,'native_orientation':native_report,'jacobian':j,'baseline_exact':plain(exact),'bounds_scaled':c.bounds.tolist(),'domain':plain(c.domain),'counts':{'proximity_rows':len(c.prox_rows),'orientation_rows':len(c.ori_rows),'colliders':len(c.original_hand)},'collider_types':[{'part':g['part'],'kind':int(g['kind']),'triangles':0 if g['tri'] is None else len(g['tri'])} for g in c.original_hand]})
        print(json.dumps({'preflight_case':i,'warm_raw_seconds':warm_raw,'exact_seconds':exactseconds,'prox_rows':len(c.prox_rows),'ori_rows':len(c.ori_rows)}),flush=True)
    report={'schema':'hug_witness_constraint_source_preflight_v1','saved_witness_sha256':sha(SAVED),'script_sha256':sha(__file__),'engine_sha256':sha(Path(__file__).with_name('constraint_engine.py')),'core_sha256':sha(Path(__file__).with_name('constraint_core.py')),'records':records,'optimizer_called':False,'physics_called':False,'new_eval_data':False}
    assert not args.out.exists();args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
