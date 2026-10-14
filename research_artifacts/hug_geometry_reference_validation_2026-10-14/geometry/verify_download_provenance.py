"""Check existing pinned HUG asset downloads, without network or mutation."""
from pathlib import Path
import hashlib,json
BASE=Path('/Users/nilakarthikesan/grasp-failure-prediction')
OUT=Path(__file__).resolve().parent
selected=('apple','milk_box','mug')
records=[]
for manifest in (BASE/'runs/hug_benchmark_review_2026-10-09/download_manifest.json',BASE/'runs/fresh_object_scene_eval_2026-10-11/assets/download_manifest.json'):
    value=json.loads(manifest.read_text());rows=[]
    for file in value['files']:
        if not any('/'+name+'/sim_assets/' in file['dataset_path'] for name in selected):continue
        path=Path(file['local_path']);path=path if path.is_absolute() else BASE/path
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==file['sha256'],path
        rows.append({'dataset_path':file['dataset_path'],'local_path':str(path),'url':file['url'],'sha256':digest,'verified':True})
    records.append({'manifest_path':str(manifest),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'dataset':value.get('dataset'),'revision':value.get('revision'),'files':rows})
result={'scope':'Read-only pinned downloaded apple/milk-box/mug asset identity. No network or asset changes.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'records':records,'verified_file_count':sum(len(r['files']) for r in records)}
(OUT/'download_provenance_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['verified_file_count'])
