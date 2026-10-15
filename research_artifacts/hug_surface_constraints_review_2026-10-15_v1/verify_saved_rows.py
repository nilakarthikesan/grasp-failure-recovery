"""Independent source-pointer/cache replay; no new geometry or robot queries."""
from pathlib import Path
from collections import Counter
import datetime,gzip,hashlib,importlib.util,json
import numpy as np

OUT=Path(__file__).parent
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
SOURCE=Path('/private/tmp/hug_surface_constraints_audit_2026-10-15_v1/saved_witnesses.json')
BASE=ROOT/'research_artifacts/hug_orientation_uncertainty_acceptance_2026-10-14'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(gzip.decompress(Path(p).read_bytes()) if str(p).endswith('.gz') else Path(p).read_bytes())
COUNT=0
def check(value,label):
    global COUNT
    COUNT+=1
    if not bool(value):raise AssertionError(label)
def same(a,b,label):check(np.array_equal(np.asarray(a),np.asarray(b)),label)
def pointer(value,path):
    for token in path.lstrip('/').split('/'):
        token=token.replace('~1','/').replace('~0','~')
        value=value[int(token)] if isinstance(value,list) else value[token]
    return value

def main():
    check(not (OUT/'saved_rows_review.json').exists(),'no result overwrite')
    check(sha(SOURCE)=='0fb0898694f4f4a6f0ff3155ec18109fd95ec71f5eff4b35ada1c46abf49dec6','frozen extracted source')
    source=read(SOURCE);op=read(BASE/'orientation/protocol.json')
    spec=importlib.util.spec_from_file_location('literal_masks',BASE/'sampling/cell_intervals.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    partnames=['palm','index','middle','ring','little','thumb','forearm_carrier','wrist_carrier']
    query_paths={digest:path for path,digest in op['query_cache_sha256'].items()}
    documents={};arrays={};bound={str(SOURCE):sha(SOURCE),str(BASE/'orientation/protocol.json'):sha(BASE/'orientation/protocol.json'),str(BASE/'sampling/cell_intervals.py'):sha(BASE/'sampling/cell_intervals.py'),str(BASE/'acceptance/acceptance_rule.json'):sha(BASE/'acceptance/acceptance_rule.json')}
    def bind(path,expected):
        check(sha(path)==expected,'source file digest '+str(path));bound[str(path)]=expected
    def document(path,expected):
        bind(path,expected)
        if path not in documents:documents[path]=read(path)
        return documents[path]
    def array(path,expected):
        bind(path,expected)
        if path not in arrays:
            with np.load(path,allow_pickle=False) as archive:arrays[path]={key:archive[key] for key in archive.files}
        return arrays[path]
    groups={};cell_counts=Counter();direction_counts=Counter();case_orient=Counter({i:0 for i in [8,12,16,20,24,28]});native_dots=[]
    for entry in source['whole_cell_rows']:
        row=entry['row'];check(pointer(document(entry['source_report'],entry['source_report_sha256']),entry['source_json_pointer'])==row,'whole-cell exact source pointer')
        cache=array(query_paths[row['original_query_cache_sha256']],row['original_query_cache_sha256'])
        full=array(entry['original_full_field_cache_path'],entry['original_full_field_cache_sha256'])
        i=row['child_cell_index'];j=partnames.index(row['anatomical_part']);r=row['centroid_radius_m'];eps=row['numerical_allowance_m'];band=row['band_mm']/1000.
        same(cache['query_object_m'][i],row['centroid_object_m'],'whole-cell centroid');same(full['query_object_m'][i],row['centroid_object_m'],'whole-cell full-cache centroid')
        same(full['cell_radius_m'][i],r,'whole-cell radius');same(full['area_mm2'][i],row['cell_area_mm2'],'whole-cell area');same(full['source_face_id'][i],row['original_visual_face_index_after_degenerate_exclusion'],'whole-cell source face')
        same(cache['native_distance_by_part_m'][i,j],row['native_centroid_distance_m'],'whole-cell native distance');same(full['current_distance_by_part_m'][i,j],row['mapped_centroid_distance_m'],'whole-cell old mapped distance')
        nD,nP=module.masks(np.array([row['native_centroid_distance_m']]),np.array([r]),band,eps);mD,mP=module.masks(np.array([row['mapped_centroid_distance_m']]),np.array([r]),band,eps)
        kind=row['relationship_change'];cell_counts[kind]+=1
        if kind=='lost_native':check(nD[0] and not mP[0],'literal lost lower witness')
        elif kind=='new_mapped':check(mD[0] and not nP[0],'literal added lower witness')
        else:raise AssertionError('unknown relationship')
        check(row['cell_area_mm2']>0. and np.isfinite(r) and r>=0.,'positive-area finite covering cell')
        native=row['native_centroid_distance_m'];error=r+eps
        if kind=='lost_native':check(native+error<=band,'native itself satisfies required near relationship')
        else:check(max(0.,native-error)>band,'native itself satisfies required far relationship')
        key=(row['original_query_cache_sha256'],i,j)
        groups.setdefault(key,[]).append({'kind':kind,'band':band,'radius':r,'eps':eps,'native':native})
    for key,rows in groups.items():
        check(len({r['native'] for r in rows})==1,'one frozen native distance per cell/part')
        # The common native distance witnesses consistency across all bands.
        for r in rows:
            error=r['radius']+r['eps'];ok=r['native']+error<=r['band'] if r['kind']=='lost_native' else max(0.,r['native']-error)>r['band']
            check(ok,'all source-derived near/far conditions consistent across bands')
    for entry in source['orientation_rows']:
        row=entry['row'];check(pointer(document(entry['source_report'],entry['source_report_sha256']),entry['source_json_pointer'])==row,'orientation exact source pointer')
        cache=array(entry['original_query_cache_path'],entry['original_query_cache_sha256']);full=array(entry['original_full_field_cache_path'],entry['original_full_field_cache_sha256']);i=row['query_index'];j=partnames.index(row['native_part'])
        same(cache['query_object_m'][i],row['query_object_m'],'orientation fixed query');same(cache['visual_triangle_outward_normal'][i],row['object_outward_normal'],'orientation fixed object normal');same(full['cell_radius_m'][i],entry['cell_radius_m'],'orientation radius');same(cache['area_mm2'][i],row['area_weight_mm2'],'orientation area');same(full['source_face_id'][i],entry['source_face_id'],'orientation face')
        same(cache['native_nearest_part'][i],j,'orientation original nearest label');same(cache['native_distance_by_part_m'][i,j],row['native_unsigned_distance_m'],'orientation native distance');check(not cache['native_inside'][i],'orientation native exterior')
        check(row['native_part']==row['mapped_part'],'historical same-label witness');check(row['native_and_mapped_query_exterior'] and row['both_closest_outside_visual'] and row['excluded_tie_buried_degenerate_flags_all_false'],'historical reliable witness flags')
        native=-float(np.dot(row['native_outward_normal'],row['object_outward_normal']));mapped=-float(np.dot(row['mapped_outward_normal'],row['object_outward_normal']));native_dots.append(abs(native-row['native_facing_compatibility']))
        check(abs(native-row['native_facing_compatibility'])<=1e-14 and abs(mapped-row['mapped_facing_compatibility'])<=1e-14,'independent signed normal dot replay')
        deadzone=op['facing_dot_numerical_deadzone'];direction=entry['direction']
        if direction=='native_away_mapped_facing':check(native < -deadzone and mapped>deadzone,'native-away reversal retained')
        elif direction=='native_facing_mapped_away':check(native>deadzone and mapped < -deadzone,'native-facing reversal retained')
        else:raise AssertionError('unknown reversal')
        band=entry['band_mm']/1000.;check(op['query_boundary_tolerance_m']<row['native_unsigned_distance_m']<=band and op['query_boundary_tolerance_m']<row['mapped_unsigned_distance_m']<=band,'historical witness band')
        direction_counts[direction]+=1;case_orient[entry['source_inventory_index']]+=1
    check(len(source['whole_cell_rows'])==178 and len(source['orientation_rows'])==134,'all saved rows retained')
    for path,digest in bound.items():check(sha(path)==digest,'source unchanged '+path)
    report={'schema':'hug_constraint_saved_rows_independent_review_v1','passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(__file__),'source_sha256':sha(SOURCE),'input_sha256':bound,'checks':COUNT,'whole_cell_rows':178,'whole_cell_relationship_counts':dict(cell_counts),'orientation_rows':134,'orientation_direction_counts':dict(direction_counts),'orientation_rows_per_case':dict(case_orient),'native_dot_replay_maxabs':max(native_dots),'all_rows_and_duplicates_retained':True,'source_label_consistency':'Each original native distance satisfies every required near/far condition at that cell/part across bands; this says nothing about Shadow feasibility.','coverage':'Capped saved examples only. Mug A(index24) has no saved reliable orientation row; absence is not a pass.','no_new_geometry_queries':True,'no_new_FK':True,'new_fitting':False,'physics_steps':0,'new_eval_data':False,'scientific_acceptance_rule_sha256':sha(BASE/'acceptance/acceptance_rule.json')}
    with (OUT/'saved_rows_review.json').open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'passed':True,'checks':COUNT,'report_sha256':sha(OUT/'saved_rows_review.json')}))

if __name__=='__main__':main()
