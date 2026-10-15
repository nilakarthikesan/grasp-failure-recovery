"""Build immutable compact artifact bundle after both cohorts and summaries complete."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,gzip,io,shutil
OUT=Path(__file__).parent;BUNDLE=OUT/'archive_bundle'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=json.loads((OUT/'protocol.json').read_text());summary=json.loads((OUT/'fit_summary.json').read_text());explore=json.loads((OUT/'exploration_summary.json').read_text())
    assert summary['candidate_sha256']==sha(OUT/'candidate_targets.json')
    assert not BUNDLE.exists();BUNDLE.mkdir()
    included=[]
    for pattern in ('*.py','protocol*.json','protected_hashes.json','manufactured_tests.log','cohort_*.log','candidate_cohort_*.json.gz','candidate_targets.json','fit_summary.json','exploration_summary.json','merge.log','exploration.log','README.md'):
        for path in sorted(OUT.glob(pattern)):
            if path.name in included:continue
            shutil.copyfile(path,BUNDLE/path.name);included.append(path.name)
    traces=[]
    for index in (8,12,16,20,24,28):
        directory=OUT/'full_callback_traces'/str(index);files=sorted(directory.glob('callback_*.json.gz'));assert files
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w') as tar:
            for path in files:
                data=path.read_bytes();name=f'{index}/{path.name}';info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o600;info.uid=info.gid=0;info.uname=info.gname=''
                tar.addfile(info,io.BytesIO(data))
                traces.append({'original_path':str(path),'case_index':index,'archive':f'full_callback_traces_case_{index}.tar.gz','member':name,'compressed_callback_sha256':sha(path),
                    'uncompressed_callback_sha256':hashlib.sha256(gzip.decompress(data)).hexdigest()})
        archive=BUNDLE/f'full_callback_traces_case_{index}.tar.gz';archive.write_bytes(gzip.compress(buffer.getvalue(),mtime=0))
        with tarfile.open(archive,'r:gz') as tar:
            assert len(tar.getmembers())==len(files)
            for member in tar.getmembers():
                record=next(t for t in traces if t['archive']==archive.name and t['member']==member.name)
                assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==record['compressed_callback_sha256']
    (BUNDLE/'callback_trace_archive_index.json').write_text(json.dumps({'schema':'hug_free_palm_trace_archive_index_v1','entries':traces},indent=2)+'\n')
    files=[]
    for path in sorted(BUNDLE.iterdir()):
        assert path.is_file();files.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size})
    manifest={'schema':'hug_free_palm_fit_artifact_manifest_v1','created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.json'),
        'candidate_sha256':sha(OUT/'candidate_targets.json'),'fit_summary_sha256':sha(OUT/'fit_summary.json'),
        'files':files,'file_count':len(files),'callback_traces_count':len(traces),'raw_cohorts_omitted_because_lossless_gzip_included':True,
        'native_runtime_assets_omitted_and_hash_bound':True,'new_eval_data':False,'physics_steps':0,'production_changes':False}
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bundle':str(BUNDLE),'manifest_sha256':sha(BUNDLE/'manifest.json'),'files':len(files),'bytes':sum(f['bytes'] for f in files),'callback_traces_count':len(traces)}))
if __name__=='__main__':main()
