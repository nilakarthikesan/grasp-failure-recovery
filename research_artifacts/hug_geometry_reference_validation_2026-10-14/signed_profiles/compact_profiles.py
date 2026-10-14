"""Lossless compressed report plus concise, provenance-bound profile overview."""
from pathlib import Path
import gzip,hashlib,json
base=Path(__file__).resolve().parent;source=base/'signed_hand_visual_profiles.json';raw=source.read_bytes();v=json.loads(raw);compressed=gzip.compress(raw,mtime=0);assert gzip.decompress(compressed)==raw;(base/'signed_hand_visual_profiles_full.json.gz').write_bytes(compressed)
keys=('sample_count','total_weight','finite_sampled_max_penetration_mm','negative_count','negative_weight','negative_weight_fraction','conditional_negative_depth_mean_mm','signed_distance_weighted_quantiles_mm')
def short(p):return {k:p[k] for k in keys if k in p}
cases=[]
for c in v['cases']:
    r={'object_id':c['object_id'],'source_inventory_index':c['source_inventory_index'],'native_sha256':c['native_sha256'],'native_vertex_count_profile':short(c['native_unweighted_vertex_profile']['whole']),'resolutions':{}}
    for density,res in c['resolutions'].items():
        n=res['native_area_profile'];d={'native_area':short(n['whole']),'native_per_finger_miss_count':n['per_finger_miss_count'],'native_per_part':{k:short(a) for k,a in n['per_part'].items()},'methods':{}}
        for name,m in res['methods'].items():
            a=m['primary_palm_fingers_profile'];d['methods'][name]={'primary_palm_fingers':short(a['whole']),'per_finger_miss_count':a['per_finger_miss_count'],'per_part':{k:short(b) for k,b in a['per_part'].items()},'carriers':short(m['carriers_profile']),'heterogeneous_full_target':short(m['profile']['whole']),'source_physical_urdf_bounds_pass':m['source_physical_urdf_bounds_pass'],'captured_scene_max_decimal_bound_excess_rad':m['captured_scene_max_decimal_bound_excess_rad'],'buried_inside_other_area_mm2':sum(g['buried_inside_other_area_mm2'] for g in m['hand_geom_sampling']),'outward_probe_occluded_area_mm2':sum(g['outward_probe_occluded_area_mm2'] for g in m['hand_geom_sampling'])}
        r['resolutions'][density]=d
    cases.append(r)
result={'schema':v['schema']+'_compact_overview','original_report_sha256':hashlib.sha256(raw).hexdigest(),'compressed_report_sha256':hashlib.sha256(compressed).hexdigest(),'protocol_sha256':v['protocol_sha256'],'executed_script_sha256':v['script_sha256'],'compact_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime_seconds':v['runtime_seconds'],'scope':'Independent native/count and native/target area profiles. No matched-point, continuous penetration maximum, physical contact or success acceptance claim. Full report retained losslessly compressed.','physics_steps':v['physics_steps'],'new_eval_data':v['new_eval_data'],'new_inference':v['new_inference'],'optimizer_used':v['optimizer_used'],'cases':cases,'limitations':v['limitations']};(base/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(len(raw),len(compressed))
