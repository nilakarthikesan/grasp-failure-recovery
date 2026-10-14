"""Manufactured deterministic tests; not a formal numerical-error certificate."""
from pathlib import Path
import hashlib,json
import numpy as np
from cell_intervals import triangle_cells,masks,pair_intervals

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    out=Path(__file__).parent;records=[]
    # Small area can coexist with a 200 mm longestedge.
    skinny=np.array([[[0,0,0],[.2,0,0],[.1,1e-6,0]]],float)
    cell,q,area,r,diameter,_,_=triangle_cells(skinny,.002)
    assert len(q)==1 and area[0]<4 and r[0]>.099 and diameter[0]>.199
    records.append({'name':'skinny_triangle_area_density_does_not_bound_radius','area_mm2':float(area[0]),'radius_mm':float(r[0]*1000),'diameter_mm':float(diameter[0]*1000)})
    # Rigidly transformed non-axisaligned small triangle.
    axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);theta=.71;K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]);R=np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K);t=np.array([.37,-.19,.23])
    bary=np.array([[i/32,j/32,1-(i+j)/32] for i in range(33) for j in range(33-i)],float)
    moved=skinny@R.T+t
    c,q2,_,r2,_,_,_=triangle_cells(moved,.002)
    sites=bary@c[0];maximum=np.linalg.norm(sites-q2[0],axis=1).max();assert maximum<=r2[0]+1e-14;assert abs(r2[0]-r[0])<1e-14
    # Direct fixedpointdistance variation on every manufactured site.
    point=np.array([.43,-.29,.21]);dq=np.linalg.norm(q2[0]-point);ds=np.linalg.norm(sites-point,axis=1)
    assert ds.min()>=max(0,dq-r2[0])-1e-14 and ds.max()<=dq+r2[0]+1e-14
    records.append({'name':'rotated_skinny_triangle_all_561_barycentric_sites_covered','site_count':len(sites),'maximum_site_radius_mm':float(maximum*1000),'rotation_radius_difference_m':float(abs(r2[0]-r[0]))})
    # 4 mm by2mm planarrectangle. Exact native|x|<=1mm and mapped
    # |x-1.5mm|<=1mm areas andtheirsetoperations are elementary strips.
    v=np.array([[0,0,0],[.004,0,0],[.004,.002,0],[0,.002,0]])
    tris=v[np.array([[0,1,2],[0,2,3]])]@R.T+t
    cells,q,weights,r,_,_,_=triangle_cells(tris,.00025)
    local=(q-t)@R;nd=np.abs(local[:,0]);md=np.abs(local[:,0]-.0015)
    result=pair_intervals(nd,md,r,weights,.001,1e-10)
    known={'native_area':2.,'mapped_area':4.,'shared_area':1.,'union_area':5.,'lost_native_area':1.,'new_mapped_area':3.}
    for name,expected in known.items():
        assert result[name]['lower_mm2']-1e-10<=expected<=result[name]['upper_mm2']+1e-10,(name,result[name],expected)
    assert result['iou_interval']['lower']<=.2<=result['iou_interval']['upper']
    assert result['native_retention_interval']['lower']<=.5<=result['native_retention_interval']['upper']
    records.append({'name':'rotated_rectangle_analytic_region_areas_inside_intervals','query_count':len(q),'known_areas_mm2':known,'intervals':result})
    far=pair_intervals(np.full(len(q),1.),np.full(len(q),2.),r,weights,.001,1e-10)
    assert far['iou_interval']['lower'] is None and far['native_retention_interval']['lower'] is None
    assert far['native_retention_interval']['reason']=='denominator_definitely_zero'
    records.append({'name':'both_empty_proximity_denominators_null','result':far})
    d=np.array([.001]);D,P=masks(d,np.array([0.]),.001,1e-8)
    assert not D[0] and P[0]
    records.append({'name':'numeric_threshold_boundary_is_ambiguous','definite':bool(D[0]),'possible':bool(P[0])})
    # Infinite denotes empty native anatomicalpart, never invent normals/area.
    carrier=pair_intervals(np.array([np.inf]),np.array([.0001]),np.array([.00001]),np.array([1.]),.001,1e-8)
    assert carrier['native_area']['upper_mm2']==0 and carrier['new_mapped_area']['lower_mm2']==1 and carrier['native_retention_interval']['lower'] is None
    records.append({'name':'empty_native_part_explicitly_null','result':carrier})
    report={'passed':True,'script_sha256':sha(__file__),'cell_library_sha256':sha(out/'cell_intervals.py'),'test_count':len(records),'tests':records,'interpretation':'Finite manufactured tests validate intendedgeometry/empty/boundarylogic; do not establishglobalfloatingpoint numerical bound. No inputs, proposal inference, simulationmodel, dynamics or eval data used.'}
    (out/'manufactured_tests.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':True,'test_count':len(records)}))
if __name__=='__main__':main()
