"""Frozen conditional sampling bounds on existing saved HUG/current poses."""
from pathlib import Path
import argparse, importlib.util, json, pickle, time, warnings, hashlib
import numpy as np
import mujoco
from cell_intervals import triangle_cells,pair_intervals
COUNTS={'actual':0,'selftests':0}; TEST=False

def forbid(*args,**kwargs):
    COUNTS['selftests' if TEST else 'actual']+=1
    raise RuntimeError('Static diagnostic forbids all physics integration')
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbid)
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
PREV=ROOT/'research_artifacts/hug_palm_object_validation_2026-10-14/surfaces'
spec=importlib.util.spec_from_file_location('frozen_surface_geometry',PREV/'audit_other_surfaces.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
for name in ('mj_step','mj_step1','mj_step2'):setattr(mujoco,name,forbid)
TEST=True
for name in ('mj_step','mj_step1','mj_step2'):
    try:getattr(mujoco,name)(None,None)
    except RuntimeError:pass
    else:raise AssertionError('Guard failed')
TEST=False

def read(p):return json.loads(Path(p).read_text())
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def current_distances(hand,q):
    shape=(len(q),len(a.PARTS));sd=np.full(shape,np.inf);closest=np.full(shape+(3,),np.nan);normal=np.full_like(closest,np.nan);geom=np.full(shape,-1,np.int64);inside=np.zeros(len(q),bool)
    for g in hand:
        signed,cp,n=a.geom_query(g,q);inside|=signed<0;part=a.PARTS.index(g['part']);ds=np.abs(signed);update=ds<sd[:,part]
        sd[update,part]=ds[update];closest[update,part]=cp[update];normal[update,part]=n[update];geom[update,part]=g['id']
    assert np.isfinite(sd).all() and np.isfinite(closest).all() and np.isfinite(normal).all()
    return sd,closest,normal,geom,inside

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).parent);parser.add_argument('--cache-root',type=Path,default=Path('/private/tmp/hug-other-surfaces-2026-10-14'));args=parser.parse_args();out=args.out;t0=time.perf_counter()
    pfile=out/'protocol.json';ph=a.sha(pfile);p=read(pfile);assert a.sha(__file__)==p['script_sha256'];assert a.sha(out/'cell_intervals.py')==p['cell_library_sha256']
    prior=ROOT/'research_artifacts/hug_mapping_refinement_2026-10-14/saved_proposals';ip=prior/'existing_hug_inventory.json';fp=prior/'static_saved_hug_fk.json'
    for file,key in ((ip,'inventory_sha256'),(fp,'fk_sha256'),(a.ASSET,'mano_asset_sha256'),(Path(a.__file__),'previous_query_library_sha256')):assert a.sha(file)==p[key]
    inv=read(ip);fkmap={c['native_sha256']:c for c in read(fp)['cases']}
    with warnings.catch_warnings():warnings.simplefilter('ignore');asset=pickle.loads(a.ASSET.read_bytes(),encoding='latin1')
    weights=np.asarray(asset['weights']);gw=np.column_stack([weights[:,ids].sum(1) for ids in a.GROUPS.values()])
    protected=read(out/'protected_hashes.json')
    for file,h in protected.items():assert a.sha(file)==h
    ep=out/'ephemeral';ep.mkdir(exist_ok=True);records=[]
    for selected in p['cases']:
        case=inv['modern_cases'][selected['source_inventory_index']];assert case['native']['sha256']==selected['native_sha256'];meta,ntri,labels,nnorm,hand,objects,m,d,initial,objadr,pos,R=a.load_case(case,gw,fkmap);nominal=d.qpos.copy();caseid=selected['case_id'];vlist=[];excluded=0
        for v in meta['disabled_visual_geoms']:
            g=v['id'];mid=int(m.geom_dataid[g]);va=int(m.mesh_vertadr[mid]);nv=int(m.mesh_vertnum[mid]);fa=int(m.mesh_faceadr[mid]);nf=int(m.mesh_facenum[mid]);vertices=m.mesh_vert[va:va+nv].astype(float);faces=m.mesh_face[fa:fa+nf].astype(int)
            vv=(vertices@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-pos)@R;vt=vv[faces];length=np.linalg.norm(np.cross(vt[:,1]-vt[:,0],vt[:,2]-vt[:,0]),axis=1);good=length>1e-15;excluded+=int((~good).sum());vlist.append(vt[good])
        assert len(vlist)==1;vt=np.concatenate(vlist);rec={'object_id':case['object_id'],'case_id':caseid,'native_sha256':selected['native_sha256'],'source_inventory_index':selected['source_inventory_index'],'input_sha256':meta['input_sha256'],'excluded_degenerate_original_face_count':excluded,'resolutions':{}}
        for density in selected['densities_mm']:
            cache_name=f'visual_reference_surfaces/{caseid}_density{density}mm_queries.npz';cache=args.cache_root/cache_name;expected=p['cache_sha256'][cache_name];assert a.sha(cache)==expected
            with np.load(cache,allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
            cells,q,area,radius,diameter,face_ids,depths=triangle_cells(vt,density/1000)
            assert np.array_equal(q,old['query_object_m']);assert np.array_equal(area,old['area_mm2']);assert np.isclose(area.sum(),a.tri_info(vt)[1].sum()*1e6,rtol=0,atol=1e-8)
            # Reproduction of q/weights is exact. Radii are analytically valid
            # real-arithmetic bounds, conditional on stated numerical allowance.
            nd=old['native_distance_by_part_m'];tick=time.perf_counter();sd,sq,sn,sg,si=current_distances(hand,q)
            oldsd=old['shadow_distance_by_part_m'];valid=np.isfinite(oldsd);reg=float(np.max(np.abs(sd[valid]-oldsd[valid])));assert reg<1e-12
            for band in p['bands_mm']:
                assert np.array_equal(sd<=band/1000,oldsd<=band/1000)
            ephemeral=ep/f'{caseid}_density{density}mm_current.npz';tmp=ephemeral.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,query_object_m=q,area_mm2=area,cell_radius_m=radius,cell_diameter_m=diameter,source_face_id=face_ids,current_distance_by_part_m=sd,current_closest_by_part_m=sq,current_normal_by_part=sn,current_nearest_geom_by_part=sg,current_inside=si)
            tmp.rename(ephemeral);binding={'cache_sha256':expected,'protocol_sha256':ph,'query_library_sha256':a.sha(a.__file__),'file_sha256':a.sha(ephemeral),'native_sha256':selected['native_sha256'],'static_pose':'original stored nominal targets','no_new_eval_data':True};write(ephemeral.with_suffix('.json'),binding)
            print(json.dumps({'progress':'exact_current_arrays_ready','case_id':caseid,'density':density,'seconds':time.perf_counter()-tick,'path':str(ephemeral)}),flush=True)
            result={'query_count':len(q),'surface_area_mm2':float(area.sum()),'cache_sha256':expected,'centroid_and_area_exact_reproduction':True,'fixed_set_native_kind':'union of six labelled open native triangle surface patches','fixed_set_mapped_kind':'union of compatible analytic/per-geom convex collider boundary surfaces; includes buried interfaces when inside overlaps','cell_radius_mm':a.stats(radius*1000,area),'cell_diameter_mm':a.stats(diameter*1000,area),'original_face_subdivision_depth_min':min(depths),'original_face_subdivision_depth_max':max(depths),'current_cache_band_regression_max_distance_m':reg,'numerical_allowances':{}}
            for eps in p['numerical_allowances_m']:
                er={'assumed_uniform_unsigned_query_error_bound_m':eps,'conditional_on_bound':True,'bands':{}}
                for band in p['bands_mm']:
                    b=band/1000;per={part:pair_intervals(nd[:,j],sd[:,j],radius,area,b,eps) for j,part in enumerate(a.PARTS)}
                    whole=pair_intervals(nd.min(1),sd.min(1),radius,area,b,eps)
                    er['bands'][str(band)]={'whole_hand_unsigned_proximity':whole,'independent_parts':per}
                result['numerical_allowances'][str(eps)]=er
            rec['resolutions'][str(density)]=result
            assert d.time==0 and np.array_equal(d.qpos,nominal) and np.array_equal(d.qpos[objadr:objadr+7],initial)
        for file,h in meta['input_sha256'].items():assert a.sha(file)==h
        rec['static_invariants']={'physics_steps':0,'time_zero':True,'object_qpos_unchanged':True,'all_nominal_qpos_unchanged':True,'input_hashes_unchanged':True};records.append(rec)
        write(out/f'{caseid}_sampling.json',rec)
    for file,h in protected.items():assert a.sha(file)==h
    assert a.sha(pfile)==ph and a.sha(__file__)==p['script_sha256'] and a.sha(out/'cell_intervals.py')==p['cell_library_sha256'];assert COUNTS['actual']==0
    report={'schema':'hug_existing_conditional_sampling_bounds_v1','protocol_sha256':ph,'script_sha256':a.sha(__file__),'cell_library_sha256':a.sha(out/'cell_intervals.py'),'cases':records,'physics_steps':0,'guards':COUNTS,'new_eval_data':False,'new_inference':False,'training':False,'protected_hashes_unchanged':True,'runtime_seconds':time.perf_counter()-t0,'limitations':p['limitations']};write(out/'sampling_uncertainty.json',report)
    print(json.dumps({'completed':True,'report':str(out/'sampling_uncertainty.json'),'seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
