"""Read-only, lossless primary-fit publication copy; invoke only after review completes.

No active primary files are modified. Historical bound source/assets remain
hash-referenced; this copies the new study, root preparation, reviewer receipts,
and witness/count preparation provenance.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gzip,hashlib,json,re
ap=argparse.ArgumentParser();ap.add_argument('--review-receipt',type=Path,nargs='+',required=True);ap.add_argument('--out',type=Path,default=Path('/private/tmp/hug_surface_constraints_archive_2026-10-15_v1'));args=ap.parse_args()
PRIMARY=Path('/private/tmp/hug_surface_constraints_2026-10-15_v1');ROOT=Path('/private/tmp/hug_surface_constraints_root_2026-10-15_v1');REVIEW=Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v1');AUDIT=Path(__file__).parent;OUT=args.out
assert not OUT.exists(), 'Never overwrite a publication archive'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
cp=read(PRIMARY/'candidate_targets.json');protocol=read(PRIMARY/'protocol.json');receipts=[read(p) for p in args.review_receipt];assert all(r.get('passed') is True for r in receipts)
assert cp['protocol_sha256']==sha(PRIMARY/'protocol.json') and sorted(c['source_inventory_index'] for c in cp['cases'])==[8,12,16,20,24,28]
assert protocol['new_eval_data'] is False and protocol['physics_steps']==0
# Bound path maps and path/SHA records provide input integrity without copying old assets.
bound={}
def collect(v):
    if isinstance(v,dict):
        for k,w in v.items():
            if isinstance(k,str) and k.startswith('/') and isinstance(w,str) and re.fullmatch('[a-f0-9]{64}',w):
                assert sha(k)==w,k;bound[k]=w
            collect(w)
        if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str) and Path(v['path']).is_file():
            assert sha(v['path'])==v['sha256'];bound[v['path']]=v['sha256']
    elif isinstance(v,list):
        for w in v:collect(w)
collect(protocol);collect(cp)
for receipt in receipts:collect(receipt)
PRIMARY_V2=Path('/private/tmp/hug_surface_constraints_2026-10-15_v2');cp2=read(PRIMARY_V2/'candidate_targets.json');p2=read(PRIMARY_V2/'protocol.json');assert cp2['protocol_sha256']==sha(PRIMARY_V2/'protocol.json');collect(cp2);collect(p2)
for path in (PRIMARY/'candidate_targets.json',PRIMARY_V2/'candidate_targets.json'):
    assert any(r.get('source_bindings',{}).get(str(path))==sha(path) for r in receipts), 'Both finalized candidate hashes require PASS receipts'
# A receipt is a required provenance artifact; human/agent invocation is only
# after numerical PASS. Exact success fields vary by independently versioned review.
selections=[]
for label,source in (('primary',PRIMARY),('primary_v2',PRIMARY_V2),('primary/root_preparation',ROOT),('review',REVIEW),('review_v2',Path('/private/tmp/hug_surface_constraints_review_2026-10-15_v2'))):
    for p in sorted(source.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and 'publication_bundle' not in p.parts:
            selections.append((p,Path(label)/p.relative_to(source)))
for name in ('saved_witnesses.json','cached_failure_row_counts.json','cached_failure_count_protocol.json','count_cached_failure_rows.py','witness_extraction_protocol.json','witness_extraction_protocol_v2.json','extract_saved_witnesses.py','extract_saved_witnesses_v2.py','extraction_v1_abort.json'):
    selections.append((AUDIT/name,Path('primary/witness_provenance')/name))
selections.append((Path(__file__),Path('primary/archive_primary_fit.py')))
for path in args.review_receipt:
    if not any(p==path for p,_ in selections):selections.append((path,Path('review/required_review_receipts')/(path.parent.name+'_'+path.name)))
OUT.mkdir();rows=[]
for source,relative in selections:
    raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest();dest=OUT/relative
    compress=source.suffix=='.json' and len(raw)>131072
    if compress:dest=Path(str(dest)+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(raw,mtime=0) if compress else raw)
    assert (gzip.decompress(dest.read_bytes()) if compress else dest.read_bytes())==raw
    assert sha(source)==h, 'Source changed during copy: '+str(source)
    rows.append({'source_absolute_path':str(source),'source_sha256':h,'source_bytes':len(raw),'archive_relative_path':str(dest.relative_to(OUT)),'archive_sha256':sha(dest),'compression':'gzip_mtime0_lossless' if compress else None})
for row in rows:assert sha(row['source_absolute_path'])==row['source_sha256'], 'Source changed during archive: '+row['source_absolute_path']
(OUT/'archive_source_bindings.json').write_text(json.dumps({'scope':'New direct witness-constraint fitting study with external preparation/review/witness provenance. Primary code and results unchanged. Full prior runtime/source closure stays hash-referenced, not copied.','candidate_sha256':sha(PRIMARY/'candidate_targets.json'),'protocol_sha256':sha(PRIMARY/'protocol.json'),'required_review_receipts':[{'path':str(p),'sha256':sha(p)} for p in args.review_receipt],'bound_input_hashes':bound,'lossless_source_checks':len(rows),'bindings':rows},indent=2)+'\n')
for group in sorted({relative.parts[0] for _,relative in selections}):
    directory=OUT/group;directory.mkdir(exist_ok=True);group_rows=[row for row in rows if Path(row['archive_relative_path']).parts[0]==group]
    (directory/'archive_source_bindings.json').write_text(json.dumps({'group':group,'scope':'Lossless publication copy; historical absolute input paths preserved as provenance. Does not assert scientific mapping acceptance.','candidate_sha256':sha(PRIMARY/'candidate_targets.json'),'protocol_sha256':sha(PRIMARY/'protocol.json'),'required_review_receipts':[{'path':str(p),'sha256':sha(p)} for p in args.review_receipt],'bindings':group_rows},indent=2)+'\n')
    group_files={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    (directory/'manifest.json').write_text(json.dumps({'schema':'hug_surface_constraints_'+group+'_publication_manifest_v1','files_sha256':group_files,'file_count':len(group_files),'physics_steps':0,'new_eval_data':False,'mapping_validation_success_not_implied':True},indent=2)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
manifest={'schema':'hug_surface_constraints_primary_publication_manifest_v1','utc':datetime.now(timezone.utc).isoformat(),'files_sha256':files,'file_count':len(files),'physics_steps':0,'new_eval_data':False,'six_existing_native_proposals_only':True,'independent_surface_audit_archived_separately':True}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for k,v in files.items():assert sha(OUT/k)==v
print(json.dumps({'archive':str(OUT),'manifest_sha256':sha(OUT/'manifest.json'),'file_count':len(files),'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),'lossless_source_checks':len(rows)},indent=2))
