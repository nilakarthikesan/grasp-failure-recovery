#!/usr/bin/env python3
"""Independent saved-field arithmetic recount only; no geometry or simulator imports."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'surface'
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
policy=load(ROOT/'policy_frozen_v1.json')
report_path=ROOT/'visual_results_v1/global_6_report.json';report=load(report_path)
parts=('palm','index','middle','ring','little','thumb','wrist_carrier')
allowance=policy['distance_allowance_m']; assert allowance==1e-9
assert report['global_sample_index']==6 and not report['target_limit_violations']
assert not report['full_mapping_accepted'] and not report['execution_eligible']
dependencies={str(report_path):sha(report_path),str(ROOT/'policy_frozen_v1.json'):sha(ROOT/'policy_frozen_v1.json')}
checks=0; records=[]; summary={}
def masks(d,radius,rho,band):
    lower=np.minimum(d,radius).astype(np.longdouble)
    # Infinity is right-censored above radius, not an unbounded unknown below radius.
    near=np.isfinite(d)&(d.astype(np.longdouble)+np.longdouble(rho)+np.longdouble(allowance)<=np.longdouble(band))
    far=lower-np.longdouble(rho)-np.longdouble(allowance)>np.longdouble(band)
    return near,far
for spacing,rec in report['resolutions'].items():
    h=float(spacing)/1000; area=h*h;rho=rec['covering_radius_m'];radius=rec['search_radius_m']
    assert math.isclose(area,rec['area_per_cell_m2'],abs_tol=0,rel_tol=1e-15)
    assert abs(rho-h*math.sqrt((3/5)**2+(2/3)**2))<1e-18
    path=report_path.parent/rec['query_arrays']['path'];assert sha(path)==rec['query_arrays']['sha256'];dependencies[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:
        source=z['source_distance_by_part_m'];target=z['target_distance_by_part_m'];points=z['cube_query_m'];faces=z['cube_face']
    assert source.shape==target.shape==(rec['query_count'],8)
    assert np.isfinite(points).all() and (np.abs(points)<=.025).all()
    assert (np.abs(points)==.025).sum(axis=1).min()==1
    assert not np.isnan(source).any() and not np.isnan(target).any() and (source>=0).all() and (target>=0).all()
    for bandstr,b in rec['bands'].items():
        band=float(bandstr)/1000
        for i,part in enumerate(parts):
            sn,sf=masks(source[:,i],radius,rho,band);tn,tf=masks(target[:,i],radius,rho,band)
            metrics={'native_near':(sn,~sf),'target_near':(tn,~tf),'intersection':(sn&tn,~sf&~tf),'union':(sn|tn,~sf|~tf),'lost_native':(sn&tf,~sf&~tn),'new_target':(tn&sf,~tf&~sn)}
            intervals={}
            for metric,(lo,hi) in metrics.items():
                assert np.all(~lo|hi)
                a={'lower_m2':int(lo.sum())*area,'upper_m2':int(hi.sum())*area,'unresolved_m2':int((hi&~lo).sum())*area}
                expected=b['groups'][part]['area_intervals'][metric]
                assert all(abs(a[k]-expected[k])<1e-16 for k in a),(spacing,bandstr,part,metric,a,expected)
                checks+=3;intervals[metric]=a
            # Exact Fraction check the membership predicates for selected positive lost/new witnesses.
            witnesses=[]
            for metric in ('lost_native','new_target'):
                indices=np.flatnonzero(metrics[metric][0])
                if not len(indices):continue
                k=int(indices[0]);ds=float(source[k,i]);dt=float(target[k,i]);rr=F(radius);rh=F(rho);al=F(allowance);bb=F(band)
                def exactnear(v):return math.isfinite(v) and F(v)+rh+al<=bb
                def exactfar(v):return (F(v) if math.isfinite(v) else rr)-rh-al>bb
                assert (exactnear(ds) and exactfar(dt)) if metric=='lost_native' else (exactnear(dt) and exactfar(ds))
                witnesses.append({'metric':metric,'query_index':k,'cube_point_m':points[k].tolist(),'source_distance_m':ds if math.isfinite(ds) else 'greater_than_search_radius','target_distance_m':dt if math.isfinite(dt) else 'greater_than_search_radius','exact_binary_rational_membership':True});checks+=1
            records.append({'spacing_mm':float(spacing),'band_mm':float(bandstr),'part':part,'intervals':intervals,'representative_positive_witnesses':witnesses})
for bandstr,parts_report in report['strict_encoded_nearset_acceptance']['by_band_and_part'].items():
    summary[bandstr]={}
    for part,original in parts_report.items():
        rows=[x for x in records if x['band_mm']==float(bandstr) and x['part']==part]
        combined={k:{'lower_m2':max(x['intervals'][k]['lower_m2'] for x in rows),'upper_m2':min(x['intervals'][k]['upper_m2'] for x in rows)} for k in ('lost_native','new_target')}
        assert all(combined[k]['lower_m2']<=combined[k]['upper_m2'] for k in combined)
        assert all(abs(combined[k][q]-original['bounds_intersection'][k][q])<1e-16 for k in combined for q in combined[k]);checks+=4
        status='FAIL' if any(x['lower_m2']>0 for x in combined.values()) else 'PASS' if all(x['upper_m2']==0 for x in combined.values()) else 'UNRESOLVED'
        assert status==original['status'];checks+=1
        summary[bandstr][part]={'status':status,'bounds_intersection':combined}
# Distinguish censored far from unknown, and preserve equality boundaries.
n,f=masks(np.array([math.inf,0.,1e-3]),.01,0.,.001)
assert n.tolist()==[False,True,False] and f.tolist()==[True,False,False];checks+=1
assert all(summary[b]['index']['status']=='FAIL' and summary[b]['ring']['status']=='FAIL' for b in summary)
result={'schema_version':'independent_raw6_visual_saved_array_recount_v1','status':'PASS_RECOUNT_CONFIRMS_STRICT_VISUAL_FAILURE','check_count':checks,'method':'Independent NumPy long-double masks and integer counts; selected positive witnesses checked using exact Fraction binary rational inequalities. No production geometry or acceptance code imported.','upstream_assumptions':'Saved distances and right-censor semantics are trusted only within original conditional 1nm numerical allowance. This checks arithmetic, not new distance geometry or global directed-rounding correctness.','dependencies':dependencies,'script_sha256':sha(__file__),'by_band_and_part':summary,'records':records,'full_mapping_accepted':False,'execution_eligible':False,'physics_steps':0,'surface_queries':0,'native_calls':0,'optimization_calls':0,'evaluation_episodes':0,'training_calls':0}
out=HERE/'independent_raw6_visual_saved_array_recount_v1.json'
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'checks':checks,'report':str(out),'sha256':sha(out)}))
