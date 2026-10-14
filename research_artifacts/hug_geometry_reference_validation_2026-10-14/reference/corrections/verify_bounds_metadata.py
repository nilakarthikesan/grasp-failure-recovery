"""Read-only numeric source/scene bound verification, correcting legacy labels.
No optimizer, model compilation, inference, physics, or dataset generation.
Writes this supplemental JSON only; original reports/scripts/manifests unchanged.
"""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
OUT=Path(__file__).parent
BASE=OUT.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bounds_from_urdf(path):
    return {j.get('name'):(float(j.find('limit').get('lower')),float(j.find('limit').get('upper'))) for j in ET.parse(path).getroot().findall('joint') if j.find('limit') is not None and j.find('limit').get('lower') is not None}
def bounds_from_scene_xml(path):
    return {j.get('name'):tuple(map(float,j.get('range').split())) for j in ET.parse(path).getroot().findall('.//joint') if j.get('name') and j.get('range')}
def main():
    reportpath=BASE/'hug_reference_comparison.json'
    # Reproduction/publication may retain only the exact full compressed report.
    if reportpath.is_file():fullbytes=reportpath.read_bytes();report_name=str(reportpath)
    else:
        import gzip
        compressed=BASE/'hug_reference_comparison_full.json.gz';fullbytes=gzip.decompress(compressed.read_bytes());report_name=str(compressed)
    fullsha=hashlib.sha256(fullbytes).hexdigest();original_manifest=BASE/'manifest.json';protected={str(original_manifest):sha(original_manifest)}
    original=json.loads(original_manifest.read_text())
    for name,meta in original['files'].items():
        path=BASE/name;assert sha(path)==meta['sha256'];protected[str(path)]=meta['sha256']
    assert fullsha==original['full_report_uncompressed_sha256']
    r=json.loads(fullbytes);urdf=Path(r['urdf_path']);assert sha(urdf)==r['urdf_sha256'];source=bounds_from_urdf(urdf);inputs={str(urdf):sha(urdf)}
    grouped={};cases=[];nominal_source_max=0;meta_delta_max=0
    for c in r['cases']:
        scene=Path(c['scene_model_path']).with_suffix('.xml');scene_bounds=bounds_from_scene_xml(scene);inputs[str(scene)]=sha(scene);methods={}
        for method,m in c['methods'].items():
            if m['status']!='computed':continue
            names=m['joint_names'];final=m['qpos_rad'];filtered=m['qpos_filtered20_rad'];raw=m['qpos_raw20_rad'];lo=[source[n][0] for n in names];hi=[source[n][1] for n in names];sl=[scene_bounds[n][0] for n in names];sh=[scene_bounds[n][1] for n in names]
            assert m['limits']['physical_lower_rad']==sl and m['limits']['physical_upper_rad']==sh,'Legacy-table label assumption did not match actual captured XML'
            clamped=[min(max(v,a),b) for v,a,b in zip(filtered,lo,hi,strict=True)]
            assert clamped==final,'Final target differs from source-bound clamp'
            fv=[max(a-v,v-b,0.) for v,a,b in zip(final,lo,hi,strict=True)];rv=[max(a-v,v-b,0.) for v,a,b in zip(raw,lo,hi,strict=True)];flv=[max(a-v,v-b,0.) for v,a,b in zip(filtered,lo,hi,strict=True)];sv=[max(a-v,v-b,0.) for v,a,b in zip(final,sl,sh,strict=True)]
            assert max(fv)==0 and max(rv)<=1.01e-3 and max(flv)<=1.01e-3
            lower_difference=[b-a for a,b in zip(lo,sl,strict=True)];upper_difference=[b-a for a,b in zip(hi,sh,strict=True)]
            meta_delta_max=max(meta_delta_max,*map(abs,lower_difference),*map(abs,upper_difference))
            row={'joint_names':names,'source_urdf_lower_rad':lo,'source_urdf_upper_rad':hi,'captured_scene_xml_lower_rad':sl,'captured_scene_xml_upper_rad':sh,'captured_minus_source_lower_rad':lower_difference,'captured_minus_source_upper_rad':upper_difference,'original_legacy_table_equals_captured_scene_xml':True,'q_final_exactly_equals_clip_filtered_to_source_bounds':True,'q_final_source_bound_max_exceedance_rad':max(fv),'q_raw20_source_bound_max_exceedance_rad':max(rv),'q_filtered20_source_bound_max_exceedance_rad':max(flv),'q_final_strict_captured_scene_exceedance_rad':sv,'strict_captured_scene_exceeding_joint_names':[n for n,v in zip(names,sv) if v>1e-9]}
            methods[method]=row;grouped.setdefault(method,[]).append(row)
        cases.append({'native_sha256':c['native_sha256'],'object_id':c['object_id'],'seed':c['seed'],'scene_xml_path':str(scene),'scene_xml_sha256':inputs[str(scene)],'methods':methods})
    summary={}
    for method,rows in grouped.items():
        counts={}
        for row in rows:
            for n in row['strict_captured_scene_exceeding_joint_names']:counts[n]=counts.get(n,0)+1
        summary[method]={'case_count':len(rows),'all_final_source_bounds_pass':all(row['q_final_source_bound_max_exceedance_rad']==0 for row in rows),'all_final_exactly_source_bound_clamped':all(row['q_final_exactly_equals_clip_filtered_to_source_bounds'] for row in rows),'raw_source_max_exceedance_rad':max(row['q_raw20_source_bound_max_exceedance_rad'] for row in rows),'filtered_source_max_exceedance_rad':max(row['q_filtered20_source_bound_max_exceedance_rad'] for row in rows),'strict_captured_scene_exceeding_case_count':sum(bool(row['strict_captured_scene_exceeding_joint_names']) for row in rows),'strict_captured_scene_exceeding_joints_case_counts':counts,'max_strict_captured_scene_final_exceedance_rad':max(max(row['q_final_strict_captured_scene_exceedance_rad']) for row in rows)}
    for path,h in inputs.items():assert sha(path)==h
    for path,h in protected.items():assert sha(path)==h
    if reportpath.is_file():assert sha(reportpath)==fullsha
    correction={'schema':'hug_reference_limits_metadata_correction_v1','date':'2026-10-14','correction':'Original methods[].limits.physical_lower_rad/physical_upper_rad were populated from captured scene ranges and mislabeled as physical source ranges. Treat these legacy fields as captured_scene_lower_rad/captured_scene_upper_rad. Their violation fields likewise refer to captured-scene ranges. Keep original bytes unchanged; source-URDF fields below supply authoritative full-precision physical limits.','why_targets_and_metrics_are_unchanged':'Retargeting and physical safety clamping used SeqRetargeting source-URDF limits. This supplemental read-only check proves every existing final q equals clip(filtered20,full-source bounds) exactly and has zero full-source limit exceedance. No optimizer rerun or geometric recalculation.','scope':'Verification of existing metadata/target arrays only; no inference, time integration, controller execution, outcome labels, training or production changes.','source_report_name':report_name,'source_report_uncompressed_sha256':fullsha,'source_original_manifest_sha256':protected[str(original_manifest)],'source_urdf_path':str(urdf),'source_urdf_sha256':sha(urdf),'verification_script_sha256':sha(__file__),'production_preflight_interpretation':'Current proposals.execution._validate_pose checks directly compiled full URDF limits with1e-9 tolerance. Captured-scene decimal-range exceedance is not demonstrated production preflight rejection. No preflight policy is changed.','scope_of_original_safety_margin':'Original clip deltas remain actual source-bound corrections. Legacy max_physical_violation_rad is a captured-scene check, not the source margin. Source raw/filter margins independently checked here.','summary':summary,'cases':cases,'source_xml_input_sha256_before_and_after':inputs,'all_original_publication_manifest_files_unchanged':True,'original_preserved_file_sha256':protected,'maximum_source_scene_bound_difference_rad':meta_delta_max,'optimizer_calls':0,'physics_steps':0,'new_eval_data':False,'original_reports_rewritten':False}
    (OUT/'bounds_metadata_correction.json').write_text(json.dumps(correction,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'correction':str(OUT/'bounds_metadata_correction.json'),'summary':summary,'original_files_preserved':len(protected)},indent=2))
if __name__=='__main__':main()
