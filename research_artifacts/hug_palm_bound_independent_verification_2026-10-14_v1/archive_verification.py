"""Archive independent checks and raw-array provenance, without more geometry."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,pickle,shutil
import numpy as np

OUT=Path(__file__).parent
DEST=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_palm_bound_independent_verification_2026-10-14_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metadata(value):
    a=np.asarray(value)
    return {'shape':list(a.shape),'dtype':a.dtype.str,'C_order_bytes_sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),'nbytes':a.nbytes}
def read(p):return json.loads(Path(p).read_bytes())
def write_new(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    local=read(OUT/'report.json');full=read(OUT/'full_wrist_v2/report.json');p=read(OUT/'protocol.json');fp=read(OUT/'full_wrist_v2/protocol.json')
    assert local['result']==full['result']=='PASS'
    assert local['protocol_sha256']==sha(OUT/'protocol.json') and full['protocol_sha256']==sha(OUT/'full_wrist_v2/protocol.json')
    for version,protocol in [(OUT,p),(OUT/'full_wrist_v2',fp)]:
        assert sha(version/'verify_midpoints.py')==protocol['script_sha256']
        for path,h in protocol['input_sha256'].items():assert sha(path)==h,path
    inventory=read(p['inventory'])['modern_cases'];case=inventory[28]
    with np.load(case['frame_files']['scene_state.npz']['path'],allow_pickle=False) as z:scene_source=metadata(z['qpos'])
    command=case['stored_shadow_command'];row=command['row_index']
    with np.load(command['path'],allow_pickle=False) as z:
        arrays={key:metadata(z[key][row]) for key in ['nominal_target_rad','root_target_position_m','root_target_quaternion_wxyz']}
    source=pickle.loads(Path(case['native']['path']).read_bytes())
    native_arrays={key:metadata(source[key]) for key in ['mesh_vertices','mesh_faces','T_camera_wrist','landmarks_3d']}
    witnesses=read(p['bound_report'])['cases'][-1]['existing_witnesses']
    assert all(w['native_sha256']==case['native']['sha256'] for w in witnesses)
    report_arrays={}
    for name,report in [('local_q0_plusminus0p35',local),('full_source_URDF_WRJ_limits',full)]:
        report_arrays[name]={'wrist_midpoints_rad':metadata([r['wrist_midpoint_rad'] for r in report['leaves']]),
            'independent_distances_m':metadata([r['independent_box_union_surface_distance_m'] for r in report['leaves']]),
            'strict_margins_m':metadata([r['independent_strict_margin_m'] for r in report['leaves']]),
            'palm_corner_arrays':{str(r['geom_id']):metadata(r['corners_palm_frame_m']) for r in report['independent_box_corner_radius_records']}}
    provenance={'schema':'hug_palm_bound_independent_array_scene_provenance_v1','source_inventory_index':28,'native_sha256':case['native']['sha256'],
        'scene_mjb_sha256':sha(case['scene_mjb']['path']),'scene_state_npz_sha256':sha(case['frame_files']['scene_state.npz']['path']),
        'stored_source_scene_qpos':scene_source,'nominal_command_npz_sha256':sha(command['path']),'nominal_command_row_index':row,'nominal_command_arrays':arrays,
        'native_proposal_arrays':native_arrays,'existing_object_query_points':metadata([w['centroid_object_m'] for w in witnesses]),
        'reconstructed_complete_nominal_scene_qpos_sha256':local['initial_scene_qpos_float64_sha256'],
        'scene_hash_encoding':'Complete reconstructed nominal scene qpos encoded little-endian float64 contiguous C-order, including free-root and object state.',
        'local_and_full_scene_identity':local['initial_scene_qpos_float64_sha256']==full['initial_scene_qpos_float64_sha256'],
        'verified_report_arrays':report_arrays,'raw_licensed_asset_arrays_not_copied':True,'physics_steps':0,'new_eval_data':False,'new_geometry_queries_during_packaging':0}
    write_new(OUT/'array_and_scene_provenance.json',provenance)
    assert not DEST.exists();DEST.mkdir()
    for src in sorted(OUT.rglob('*')):
        if not src.is_file() or src.suffix not in {'.py','.json','.md','.log','.diff'}:continue
        relative=src.relative_to(OUT);dst=DEST/relative;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
    manifest={'schema':'hug_palm_bound_independent_geometry_archive_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_directory':str(OUT),'local_protocol_sha256':sha(OUT/'protocol.json'),'full_wrist_protocol_sha256':sha(OUT/'full_wrist_v2/protocol.json'),
        'local_report_sha256':sha(OUT/'report.json'),'full_wrist_report_sha256':sha(OUT/'full_wrist_v2/report.json'),
        'file_sha256':{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        'independent_midpoint_counts':{'local_q0_plusminus0p35':149,'full_source_URDF_WRJ_limits':171},
        'local_and_full_results':'PASS','same_fixed_root_rotation_native_palm_anchor_only':True,'conditional_arithmetic_budgets_not_formal_FP_proof':True,
        'physics_steps':0,'new_eval_data':False,'new_inference':False,'new_bound_searches':0,'production_changes':False,'full_HUG_mapping_accepted':False}
    write_new(DEST/'manifest.json',manifest)
    print(json.dumps({'archive':str(DEST),'files_bound':len(manifest['file_sha256']),'manifest_sha256':sha(DEST/'manifest.json'),'local_report_sha256':manifest['local_report_sha256'],'full_wrist_report_sha256':manifest['full_wrist_report_sha256']},indent=2))
if __name__=='__main__':main()
