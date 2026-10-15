"""Lossless immutable archive of this agent's owned validation evidence only."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,shutil

OUT=Path(__file__).parent
DEST=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction/research_artifacts/hug_guarded_geometry_validation_2026-10-14_v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    assert not DEST.exists()
    summary=json.loads((OUT/'summary.json').read_text())
    expected=json.loads((OUT/'protocol.json').read_text())
    assert summary['validation_protocol_sha256']==sha(OUT/'protocol.json')
    assert sha(OUT/'validate_candidate.py')==expected['script_sha256']
    DEST.mkdir()
    source_hashes={};compressed={}
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path.suffix not in {'.py','.json','.gz','.log','.md','.diff'}:continue
        source_hashes[path.name]=sha(path)
        if path.name.startswith('hug_') and path.suffix=='.json':
            raw=path.read_bytes();name=path.name+'.gz';(DEST/name).write_bytes(gzip.compress(raw,mtime=0))
            assert gzip.decompress((DEST/name).read_bytes())==raw
            compressed[path.name]={'archive_path':name,'original_uncompressed_sha256':sha(path),'gzip_sha256':sha(DEST/name)}
        else:
            shutil.copy2(path,DEST/path.name);assert sha(DEST/path.name)==sha(path)
    manifest={'schema':'hug_guarded_geometry_independent_validation_archive_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_runtime_directory':str(OUT),'archive_directory':str(DEST),'candidate_sha256':summary['candidate_sha256'],
        'fit_protocol_sha256':summary['fit_protocol_sha256'],'validation_protocol_sha256':summary['validation_protocol_sha256'],
        'archived_source_file_sha256':source_hashes,'lossless_per_case_compression':compressed,
        'file_sha256':{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()},
        'source_preflight_PASS_count':6,'exact_original_scene_count':6,'reused_original_audit_resolutions':9,
        'new_full_grid_surface_measurements':0,'strict_status_counts':summary['strict_status_counts'],
        'pointwise_orientation_status_counts':summary['pointwise_orientation_status_counts'],
        'physics_steps':0,'new_eval_data':False,'production_mapper_changed':False,'scientific_acceptance_rule_changed':False,
        'note':'Numerical preflight PASS and archive integrity do not mean full HUG mapping fidelity PASS.'}
    write_new(DEST/'manifest.json',manifest)
    print(json.dumps({'archive':str(DEST),'files_bound':len(manifest['file_sha256']),'manifest_sha256':sha(DEST/'manifest.json')},indent=2))

if __name__=='__main__':main()
