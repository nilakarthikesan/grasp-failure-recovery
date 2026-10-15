"""Copy owned evidence only; never edit shared docs/source or frozen results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,gzip,shutil
ROOT=Path('/Users/nilakarthikesan/.codex/worktrees/partner-simulation-review/grasp-failure-prediction')
ART=ROOT/'research_artifacts'
FOLDERS=['hug_canonical_mcp_independent_verification_2026-10-14_v1','hug_calibrated_anchor_radial_independent_verification_2026-10-15_v1']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 for name in FOLDERS:
  source=Path('/private/tmp')/name;target=ART/name;assert not target.exists(),target
  target.mkdir();files=sorted(source.iterdir());assert all(p.is_file() for p in files)
  for p in files:
   shutil.copyfile(p,target/p.name);assert sha(p)==sha(target/p.name)
 # Byte-identical archives are alternate persisted locations, not edited protocols.
 for name in FOLDERS:
  target=ART/name;protocol=json.loads((target/'protocol.json').read_bytes());bindings=[];external=[]
  for path,h in protocol['input_sha256'].items():
   original=Path(path)
   if original.is_relative_to(Path('/private/tmp')):
    rel=original.relative_to(Path('/private/tmp'));archived=ART/rel
    if archived.is_file():
     assert sha(archived)==h,(path,archived)
     bindings.append({'frozen_original_path':path,'frozen_sha256':h,'persisted_path':str(archived),'encoding':'raw_byte_identical'})
    elif archived.with_suffix(archived.suffix+'.gz').is_file():
     compressed=archived.with_suffix(archived.suffix+'.gz');assert hashlib.sha256(gzip.decompress(compressed.read_bytes())).hexdigest()==h,(path,compressed)
     bindings.append({'frozen_original_path':path,'frozen_sha256':h,'persisted_path':str(compressed),'encoding':'gzip','encoded_sha256':sha(compressed)})
    else:external.append({'original_path':path,'sha256':h,'status':'Externally hash-bound input retained at original path; not duplicated in this small evidence bundle'})
  write(target/'archive_cross_bindings.json',{'schema':'independent_HUG_evidence_archive_cross_bindings_v1','protocol_sha256':sha(target/'protocol.json'),'protocol_absolute_paths_not_rewritten':True,'verified_persisted_bindings':bindings,'externally_retained_inputs':external})
  manifest={'schema':'independent_HUG_evidence_manifest_v1','archived_utc':datetime.now(timezone.utc).isoformat(),'files':{p.name:sha(p) for p in sorted(target.iterdir()) if p.is_file()},'scope':'Owned independent verification evidence only; no shared source/doc/Git changes'}
  write(target/'manifest.json',manifest)
  for f,h in manifest['files'].items():assert sha(target/f)==h
  print(json.dumps({'artifact_path':str(target),'file_count':len(manifest['files'])+1,'manifest_sha256':sha(target/'manifest.json'),'protocol_sha256':sha(target/'protocol.json'),'report_sha256':sha(target/'report.json'),'verified_persisted_binding_count':len(bindings),'external_temp_bindings':len(external)},indent=2))
if __name__=='__main__':main()
