#!/usr/bin/env python3
"""Independent frozen candidate saved-array review. No surface query/optimizer imports."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'band_validation'
def sha(p):return 'sha256:'+hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
policy_path=ROOT/'policy_frozen_v1.json';policy=load(policy_path)
report_path=ROOT/'results_v1/report.json';full=load(report_path);case=full['case']
parts=('palm','index','middle','ring','little','thumb','wrist_carrier')
groups=parts+('whole_hand_common_runtime_coverage','whole_hand')
allowance=policy['distance_allowance_m'];assert allowance==1e-9
assert policy['query_fractions_exact']==['13/37','19/41'] and policy['query_fractions']==[13/37,19/41]
assert full['policy_sha256']==sha(policy_path)
assert case['global_sample_index']==1 and not case['full_mapping_accepted'] and not case['execution_eligible']
assert not full['independent_native_validation'] and full['original_preselected_failures_preserved']
dependencies={str(report_path):sha(report_path),str(policy_path):sha(policy_path)}
for path,digest in policy['dependencies'].items():assert sha(path)==digest
dependencies['frozen_policy_dependency_count']=len(policy['dependencies'])
checks=0;records=[];summary={};grid_checks=[]
def masks(d,radius,rho,band):
    finite=np.isfinite(d);low=np.where(finite,d,radius).astype(np.longdouble)
    near=finite&(d.astype(np.longdouble)+np.longdouble(rho)+np.longdouble(allowance)<=np.longdouble(band))
    far=low-np.longdouble(rho)-np.longdouble(allowance)>np.longdouble(band)
    return near,far
for spacing,rec in case['resolutions'].items():
    h=float(spacing)/1000; exact_h=F(spacing)/1000;side=int(round(.05/h));area=h*h;rho=rec['covering_radius_m'];radius=rec['search_radius_m']
    assert side*exact_h==F(1,20) and rec['query_count']==6*side*side
    assert math.isclose(area,rec['area_per_cell_m2'],abs_tol=0,rel_tol=1e-15)
    exact_radius_squared=exact_h**2*(F(24,37)**2+F(22,41)**2)
    assert abs(rho-h*math.sqrt((24/37)**2+(22/41)**2))<1e-18
    # Exact rational squares independently check that declared allowance covers radius rounding.
    assert (F(rho)+F(allowance))**2>=exact_radius_squared
    assert radius>=.002+rho+allowance
    path=report_path.parent/rec['query_arrays']['path'];assert sha(path)==rec['query_arrays']['sha256'];dependencies[str(path)]=sha(path)
    with np.load(path,allow_pickle=False) as z:
        source=z['source_distance_by_part_m'];target=z['target_distance_by_part_m'];points=z['cube_query_m'];faces=z['cube_face'];normals=z['cube_normals']
    assert source.shape==target.shape==(rec['query_count'],8)
    assert np.isfinite(points).all() and (np.abs(points)<=.025).all()
    assert not np.isnan(source).any() and not np.isnan(target).any() and (source>=0).all() and (target>=0).all()
    max_grid_error=0.
    for axis in range(3):
        tangents=[j for j in range(3) if j!=axis]
        for sign in (-1,1):
            mask=faces==('XYZ'[axis]+('+' if sign>0 else '-')); pp=points[mask];nn=normals[mask]
            assert len(pp)==side*side and np.all(pp[:,axis]==sign*.025)
            assert np.all(nn[:,axis]==sign) and np.all(nn[:,tangents]==0)
            # Infer original cell index independently from rounded coordinates, then bind exact fraction offset.
            cells=np.floor((pp[:,tangents]+.025)/h).astype(int)
            assert np.all(cells>=0) and np.all(cells<side) and len(np.unique(cells,axis=0))==side*side
            for col,frac in enumerate((F(13,37),F(19,41))):
                expected=np.array([float(-F(1,40)+(F(k)+frac)*exact_h) for k in range(side)])
                error=float(np.max(np.abs(pp[:,tangents[col]]-expected[cells[:,col]])));assert error<1e-16;max_grid_error=max(max_grid_error,error)
            checks+=7
    assert max_grid_error*math.sqrt(2)<allowance
    grid_checks.append({'spacing_mm':float(spacing),'query_count':len(points),'grid_offsets_exact':['13/37','19/41'],'all_face_cell_coordinates_and_normals_checked':True,'max_saved_vs_exact_grid_coordinate_error_m':max_grid_error,'covering_radius_m':rho,'radius_formula_checked_by_exact_rational_squares_including_declared_allowance':True});checks+=6
    for bandstr,b in rec['bands'].items():
        band=float(bandstr)/1000
        for part in groups:
            i=parts.index(part) if part in parts else None
            sd=source[:,i] if i is not None else source[:,:7].min(axis=1) if part=='whole_hand_common_runtime_coverage' else source.min(axis=1)
            td=target[:,i] if i is not None else target[:,:7].min(axis=1) if part=='whole_hand_common_runtime_coverage' else target.min(axis=1)
            sn,sf=masks(sd,radius,rho,band);tn,tf=masks(td,radius,rho,band)
            metrics={'native_near':(sn,~sf),'target_near':(tn,~tf),'intersection':(sn&tn,~sf&~tf),'union':(sn|tn,~sf|~tf),'lost_native':(sn&tf,~sf&~tn),'new_target':(tn&sf,~tf&~sn)}
            intervals={}
            for metric,(lo,hi) in metrics.items():
                assert np.all(~lo|hi)
                a={'lower_m2':int(lo.sum())*area,'upper_m2':int(hi.sum())*area,'unresolved_m2':int((hi&~lo).sum())*area};expected=b['groups'][part]['area_intervals'][metric]
                assert all(abs(a[k]-expected[k])<1e-16 for k in a),(spacing,bandstr,part,metric,a,expected)
                checks+=3;intervals[metric]=a
            witnesses=[]
            for metric in ('lost_native','new_target'):
                indices=np.flatnonzero(metrics[metric][0])
                if not len(indices):continue
                k=int(indices[0]);ds=float(sd[k]);dt=float(td[k]);rr=F(radius);rh=F(rho);al=F(allowance);bb=F(band)
                def near(v):return math.isfinite(v) and F(v)+rh+al<=bb
                def far(v):return (F(v) if math.isfinite(v) else rr)-rh-al>bb
                assert (near(ds) and far(dt)) if metric=='lost_native' else (near(dt) and far(ds))
                witnesses.append({'metric':metric,'query_index':k,'cube_point_m':points[k].tolist(),'source_distance_m':ds if math.isfinite(ds) else 'greater_than_search_radius','target_distance_m':dt if math.isfinite(dt) else 'greater_than_search_radius','exact_binary_rational_membership':True});checks+=1
            records.append({'spacing_mm':float(spacing),'band_mm':float(bandstr),'part':part,'intervals':intervals,'representative_positive_witnesses':witnesses})
for bandstr,parts_report in case['strict_encoded_nearset_acceptance']['by_band_and_part'].items():
    summary[bandstr]={}
    for part,original in parts_report.items():
        rows=[x for x in records if x['band_mm']==float(bandstr) and x['part']==part]
        combined={k:{'lower_m2':max(x['intervals'][k]['lower_m2'] for x in rows),'upper_m2':min(x['intervals'][k]['upper_m2'] for x in rows)} for k in ('lost_native','new_target')}
        assert all(combined[k]['lower_m2']<=combined[k]['upper_m2'] for k in combined)
        assert all(abs(combined[k][q]-original['bounds_intersection'][k][q])<1e-16 for k in combined for q in combined[k]);checks+=4
        status='FAIL' if any(x['lower_m2']>0 for x in combined.values()) else 'PASS' if all(x['upper_m2']==0 for x in combined.values()) else 'UNRESOLVED'
        assert status==original['status'];checks+=1
        summary[bandstr][part]={'status':status,'bounds_intersection':combined}
n,f=masks(np.array([math.inf,0.,1e-3]),.01,0.,.001)
assert n.tolist()==[False,True,False] and f.tolist()==[True,False,False];checks+=1
assert all(summary[b][p]['status']=='FAIL' for b in summary for p in ('index','ring','little','thumb'))
assert full['strict_visual_status']=='FAIL'
result={'schema_version':'independent_selected_band_visual_saved_array_recount_v1','status':'PASS_RECOUNT_CONFIRMS_SELECTED_CANDIDATE_STRICT_VISUAL_FAILURE','check_count':checks,'global_sample_index':1,'method':'Independent masks and integer area counts across all3 grids, all3 bands,7 matched parts and2 whole-hand aggregates. Positive witnesses independently checked using exact Fraction binary-rational inequalities. Grid fractions/face labels/normals/cell identity and coverage radius independently verified. No production geometry or acceptance code imported.','upstream_assumptions':'Saved distances and right-censor semantics remain conditional on original 1nm numerical allowance. This recount does not certify distance queries or arbitrary floating-point surface algorithms globally.','report_dependency':{'path':str(report_path),'sha256':sha(report_path)},'dependencies':dependencies,'script_sha256':sha(__file__),'grid_checks':grid_checks,'by_band_and_part':summary,'records':records,'full_mapping_accepted':False,'execution_eligible':False,'independent_native_validation':False,'physics_steps':0,'surface_queries':0,'native_calls':0,'optimization_calls':0,'evaluation_episodes':0,'training_calls':0}
out=HERE/'independent_selected_band_visual_saved_array_recount_v1.json'
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':result['status'],'checks':checks,'report':str(out),'sha256':sha(out),'one_mm_lower_bounds_mm2':{p:{k:v['lower_m2']*1e6 for k,v in x['bounds_intersection'].items()} for p,x in summary['1.0'].items()}}))
