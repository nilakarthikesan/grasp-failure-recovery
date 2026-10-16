"""Bounded pinned download for the native 30-object pilot; no model execution."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.request

DATASET = 'kevinywu/1m-hugs'
REVISION = '56ea3f100b1a4ec38bb35e986c0dd14d38977283'
MAX_BYTES = 40_000_000
MAX_FILE_BYTES = 5_000_000

def sha(data):
    return hashlib.sha256(data).hexdigest()

def plan(tree):
    files = [x for x in tree if x['type'] == 'file']
    groups = {}
    for row in files:
        path = PurePosixPath(row['path'])
        if path.is_absolute() or '..' in path.parts or len(path.parts) < 4 or path.parts[0] != 'test':
            raise ValueError('Only bounded official test paths permitted')
        groups.setdefault('/'.join(path.parts[:3]), []).append(row)
    if len(groups) != 30:
        raise ValueError('Exact 30 official test objects required')
    selected = []
    for key, rows in sorted(groups.items()):
        observations = sorted((x for x in rows if x['path'].endswith('.pkl')), key=lambda x:x['path'])
        if len(observations) != 10:
            raise ValueError('Expected ten official evaluation observations per object')
        selected.extend([observations[0], observations[-1]])
        selected.extend(x for x in rows if '/sim_assets/' in x['path'])
    if len(selected) > 500 or sum(x['size'] for x in selected) > MAX_BYTES:
        raise ValueError('Before-download aggregate quota exceeded')
    if any(not 0 < x['size'] <= MAX_FILE_BYTES for x in selected):
        raise ValueError('Before-download file quota exceeded')
    return sorted(selected,key=lambda x:x['path'])

def fetch(tree_path, output):
    raw = tree_path.read_bytes()
    rows = plan(json.loads(raw))
    output.mkdir(parents=True, exist_ok=False)
    (output/'official_test_tree.json').write_bytes(raw)
    def one(row):
        request = urllib.request.Request(
            f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{row["path"]}',
            headers={'User-Agent':'native-hug-pilot-asset-audit/1'})
        with urllib.request.urlopen(request,timeout=60) as response:
            data = response.read(MAX_FILE_BYTES+1)
        if len(data) != row['size']:
            raise ValueError('Declared byte count differs: '+row['path'])
        if 'lfs' in row:
            if sha(data) != row['lfs']['oid']:
                raise ValueError('LFS SHA-256 differs: '+row['path'])
        else:
            git_blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if git_blob != row['oid']:
                raise ValueError('Pinned Git blob differs: '+row['path'])
        path = output/row['path']
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as file:
            file.write(data)
        return {'path':row['path'],'bytes':len(data),'sha256':sha(data),
                'upstream_oid':row['oid'],'lfs_sha256':row.get('lfs',{}).get('oid')}
    with ThreadPoolExecutor(max_workers=8) as executor:
        artifacts = list(executor.map(one,rows))
    manifest = {'schema_version':'hug_native_pilot_download_v1','dataset':DATASET,
        'revision':REVISION,'metadata_sha256':sha(raw),'objects':30,
        'observation_selection':'lexicographically first and last of ten per object; before outcomes',
        'observation_count':60,'files':artifacts,'bytes':sum(x['bytes'] for x in artifacts),
        'model_calls':0,'physics_steps':0,'training_calls':0}
    with (output/'download_manifest.json').open('x') as file:
        json.dump(manifest,file,indent=2,sort_keys=True);file.write('\n')
    print(json.dumps({'objects':30,'files':len(artifacts),'bytes':manifest['bytes'],
                      'manifest_sha256':sha((output/'download_manifest.json').read_bytes())}))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();fetch(args.tree,args.output)
